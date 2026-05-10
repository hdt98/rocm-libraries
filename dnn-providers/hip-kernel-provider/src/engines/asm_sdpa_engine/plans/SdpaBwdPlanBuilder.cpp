// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaBwdPlanBuilder.hpp"
#include "HipKernelUtils.hpp"
#include "asm/AsmKernelPath.hpp"
#include "asm_fmha_v3_bwd_configs.hpp"
#include "plans/SdpaBwdPlan.hpp"

#include <cmath>
#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <utility>

// Backward CSV column dependency: codegen.py auto-derives fmha_v3_bwdConfig
// from the CSV header. The columns this builder reads are
//   dtype, hdim_q, hdim_v, mask, atomic32, pssk, pddv, mode, bf16_cvt,
//   ts_qo, ts, knl_name, co_name, arch
// If AITER renames a column, the struct field name changes and this
// translation unit fails to compile — an intended early-warning signal.

namespace asm_sdpa_engine
{
namespace
{

// Backward-local enums. Mirror the forward definitions but kept distinct so
// fwd and bwd dispatch can evolve independently.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2,
    WINDOW_GENERIC = 3
};

enum class RoundingMode : int
{
    RTNE = 0, // Round to Nearest Even (IEEE default)
    RTNA = 1, // Round to Nearest Away from zero
    RTZ = 2 // Round toward Zero
};

// CSV sentinel value the bwd registry uses in the bf16_cvt column for fp16
// rows (where rounding mode is not applicable).
constexpr int BF16_CVT_FP16_SENTINEL = 3;

enum class BatchMode : int
{
    BATCH = 0, // All sequences have same length
    GROUP = 1 // Variable sequence lengths
};

enum class AccumulatorMode : int
{
    A16 = 0, // 16-bit accumulator (atomic32 = 0)
    A32 = 1 // 32-bit accumulator (atomic32 = 1)
};

// Mask classification — ported verbatim from SdpaFwdPlanBuilder. Handles the
// modern left_bound / right_bound / diagonal_alignment trio plus the
// deprecated causal_mask* booleans.
MaskType getMaskType(const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& attrs)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    bool leftAndRightBoundsSet = attrs.left_bound().has_value() && attrs.right_bound().has_value();
    if(!leftAndRightBoundsSet)
    {
        if(attrs.causal_mask())
        {
            return MaskType::TOP_LEFT_CAUSAL;
        }
        if(attrs.causal_mask_bottom_right())
        {
            return MaskType::BOTTOM_RIGHT_CAUSAL;
        }
        return MaskType::NO_MASK;
    }

    auto left = attrs.left_bound().value();
    auto right = attrs.right_bound().value();
    if(left == -1 && right == -1)
    {
        return MaskType::NO_MASK;
    }
    if(left == -1 && right == 0)
    {
        return attrs.diagonal_alignment() == DiagonalAlignment::BOTTOM_RIGHT
                   ? MaskType::BOTTOM_RIGHT_CAUSAL
                   : MaskType::TOP_LEFT_CAUSAL;
    }
    return MaskType::WINDOW_GENERIC;
}

RoundingMode
    getRoundingMode(const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& /*attrs*/)
{
    // Rounding mode is not currently expressible in the graph; backward defaults
    // to IEEE round-to-nearest-even.
    return RoundingMode::RTNE;
}

BatchMode getBatchMode(const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& attrs)
{
    return (attrs.seq_len_q_tensor_uid().has_value() || attrs.seq_len_kv_tensor_uid().has_value())
               ? BatchMode::GROUP
               : BatchMode::BATCH;
}

// Map the seven backward tensor dtypes to a CSV dtype identifier. The backward
// graph carries Q/K/V/dO inputs and dQ/dK/dV gradient outputs that all share
// a single floating-point type per the CSV schema; FP32 stats are validated
// separately. FP8 is rejected because the backward CSV does not define FP8 rows.
std::string getDataTypeIdentifier(hipdnn_flatbuffers_sdk::data_objects::DataType qType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType kType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType vType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType doType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType dqType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType dkType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType dvType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto allEqual = [&](DataType expected) {
        return qType == expected && kType == expected && vType == expected && doType == expected
               && dqType == expected && dkType == expected && dvType == expected;
    };

    if(allEqual(DataType::BFLOAT16))
    {
        return "bf16";
    }
    if(allEqual(DataType::HALF))
    {
        return "fp16";
    }
    return "";
}

// Walk the chosen registry and return the key (arch + knl_name) of the row
// that matches the requested tuple, or an empty string when no row matches.
std::string findKey(const std::unordered_map<std::string, fmha_v3_bwdConfig>& registry,
                    const std::string& archId,
                    const std::string& dataType,
                    int hdimQ,
                    int hdimV,
                    int mask,
                    int atomic32,
                    int pssk,
                    int pddv,
                    int mode,
                    int bf16Cvt)
{
    for(const auto& el : registry)
    {
        const auto& cfg = el.second;
        if(cfg.arch != archId)
        {
            continue;
        }
        if(cfg.dtype != dataType)
        {
            continue;
        }
        if(cfg.hdim_q != hdimQ || cfg.hdim_v != hdimV)
        {
            continue;
        }
        if(cfg.mask != mask)
        {
            continue;
        }
        if(cfg.atomic32 != atomic32)
        {
            continue;
        }
        if(cfg.pssk != pssk || cfg.pddv != pddv)
        {
            continue;
        }
        if(cfg.mode != mode)
        {
            continue;
        }
        if(cfg.bf16_cvt != bf16Cvt)
        {
            continue;
        }
        return el.first;
    }
    return {};
}

// Backward kernels live in a flat layout under
//   asm_kernels/<arch>/fmha_v3_bwd/<co_name>
// Forward splits gfx942 into MI300/MI308 sub-folders; backward does not
// (AITER does not provide an MI308 backward set). The codegen-emitted co_name
// already includes the "<arch>/fmha_v3_bwd/" prefix, so this helper simply
// resolves to the absolute install path.
std::string getKernelCoPath(const std::string& /*archId*/, const std::string& coName)
{
    return asm_kernels::getAsmKernelPath(coName);
}

} // namespace

namespace bwd_dispatch
{

std::string lookupKernelNameKey(PipelineStage stage,
                                const std::string& archId,
                                const std::string& dataType,
                                int hdimQ,
                                int hdimV,
                                int mask,
                                int atomic32,
                                int pssk,
                                int pddv,
                                int mode,
                                int bf16Cvt)
{
    switch(stage)
    {
    case PipelineStage::ODO:
        return findKey(cfg_fmha_bwd_odo,
                       archId,
                       dataType,
                       hdimQ,
                       hdimV,
                       mask,
                       atomic32,
                       pssk,
                       pddv,
                       mode,
                       bf16Cvt);
    case PipelineStage::DQDKDV:
        return findKey(cfg_fmha_bwd_dqdkdv,
                       archId,
                       dataType,
                       hdimQ,
                       hdimV,
                       mask,
                       atomic32,
                       pssk,
                       pddv,
                       mode,
                       bf16Cvt);
    case PipelineStage::DQ_CONVERT:
        return findKey(cfg_fmha_bwd_dq_convert,
                       archId,
                       dataType,
                       hdimQ,
                       hdimV,
                       mask,
                       atomic32,
                       pssk,
                       pddv,
                       mode,
                       bf16Cvt);
    default:
        return {};
    }
}

int computePssk(unsigned int seqLenKv, unsigned int tsKv)
{
    if(tsKv == 0)
    {
        return 1;
    }
    return (seqLenKv % tsKv != 0) ? 1 : 0;
}

int computePddv(unsigned int headDimV)
{
    constexpr unsigned int FAST_PATH_ALIGNED = 128;
    return (headDimV != FAST_PATH_ALIGNED) ? 1 : 0;
}

} // namespace bwd_dispatch

bool SdpaBwdPlanBuilder::isApplicable(
    const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[SdpaBwdPlanBuilder::isApplicable] ";

    auto& nodeWrappers = opGraph.nodeWrappers();

    std::string deviceString;
    try
    {
        deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Could not query device string: " << e.what());
        return false;
    }

    // The codegen-generated registry contains both gfx942 and gfx950 rows;
    // only gfx942 is dispatched here.
    HIP_KERNEL_RETURN_FALSE_IF(deviceString != "gfx942",
                               "Device string does not match gfx942 (Actual value: " + deviceString
                                   + ")");

    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.size() != 1, "Graph has more than one node");
    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.front()->attributesType()
                                   != NodeAttributes::SdpaBackwardAttributes,
                               "Node attribute type is not SdpaBackwardAttributes");

    const auto& attrs = nodeWrappers.front()->attributesAs<SdpaBackwardAttributes>();

    // Graph features the engine does not currently dispatch.
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dropout_probability().has_value()
                                   && attrs.dropout_probability().value() != 0.f,
                               "dropout_probability must be unset or zero (Actual value: "
                                   + std::to_string(attrs.dropout_probability().value()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.alibi_mask(), "alibi_mask must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.padding_mask(), "padding_mask must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.attn_mask_tensor_uid(), "attn_mask tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.seed_tensor_uid(), "seed tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.offset_tensor_uid(), "offset tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dropout_mask_tensor_uid(),
                               "dropout_mask tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dbias_tensor_uid(), "dbias tensor not supported");

    // Group mode (variable sequence lengths) requires a different kernarg
    // layout than the POC; deferred to the kernarg-layout abstraction.
    HIP_KERNEL_RETURN_FALSE_IF(getBatchMode(attrs) != BatchMode::BATCH,
                               "Variable-length sequences (group mode) deferred to follow-up");

    // Validate required tensors
    const auto& tensorMap = opGraph.getTensorMap();

    int64_t qUid = attrs.q_tensor_uid();
    int64_t kUid = attrs.k_tensor_uid();
    int64_t vUid = attrs.v_tensor_uid();
    int64_t doUid = attrs.do_tensor_uid();
    int64_t statsUid = attrs.stats_tensor_uid();
    int64_t dqUid = attrs.dq_tensor_uid();
    int64_t dkUid = attrs.dk_tensor_uid();
    int64_t dvUid = attrs.dv_tensor_uid();

    auto* qTensor = tensorMap.at(qUid);
    auto* kTensor = tensorMap.at(kUid);
    auto* vTensor = tensorMap.at(vUid);
    auto* doTensor = tensorMap.at(doUid);
    auto* statsTensor = tensorMap.at(statsUid);
    auto* dqTensor = tensorMap.at(dqUid);
    auto* dkTensor = tensorMap.at(dkUid);
    auto* dvTensor = tensorMap.at(dvUid);

    HIP_KERNEL_RETURN_FALSE_IF(
        qTensor->dims()->size() != 4,
        "q tensor must be rank 4 (Actual rank: " + std::to_string(qTensor->dims()->size()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        kTensor->dims()->size() != 4,
        "k tensor must be rank 4 (Actual rank: " + std::to_string(kTensor->dims()->size()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        vTensor->dims()->size() != 4,
        "v tensor must be rank 4 (Actual rank: " + std::to_string(vTensor->dims()->size()) + ")");

    // Stats is FP32 (LSE from forward pass)
    HIP_KERNEL_RETURN_FALSE_IF(statsTensor->data_type() != DataType::FLOAT,
                               "stats tensor datatype must be FP32 (Actual type: "
                                   + EnumNameDataType(statsTensor->data_type()) + ")");

    auto dataTypeId = getDataTypeIdentifier(qTensor->data_type(),
                                            kTensor->data_type(),
                                            vTensor->data_type(),
                                            doTensor->data_type(),
                                            dqTensor->data_type(),
                                            dkTensor->data_type(),
                                            dvTensor->data_type());

    HIP_KERNEL_RETURN_FALSE_IF(
        dataTypeId.empty(),
        "All Q/K/V/dO/dQ/dK/dV tensors must share a supported dtype (BF16 or FP16). "
        "Actual: q="
            + std::string(EnumNameDataType(qTensor->data_type()))
            + ", k=" + EnumNameDataType(kTensor->data_type())
            + ", v=" + EnumNameDataType(vTensor->data_type())
            + ", do=" + EnumNameDataType(doTensor->data_type())
            + ", dq=" + EnumNameDataType(dqTensor->data_type())
            + ", dk=" + EnumNameDataType(dkTensor->data_type())
            + ", dv=" + EnumNameDataType(dvTensor->data_type()));

    auto headDimQk = static_cast<int>(qTensor->dims()->Get(3));
    auto headDimV = static_cast<int>(vTensor->dims()->Get(3));

    HIP_KERNEL_RETURN_FALSE_IF(headDimQk != headDimV,
                               "Asymmetric head dimensions not supported (D_qk = "
                                   + std::to_string(headDimQk)
                                   + ", D_v = " + std::to_string(headDimV) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        headDimQk != 64 && headDimQk != 128 && headDimQk != 192,
        "Head dimension must be 64, 128, or 192 (Actual value: " + std::to_string(headDimQk) + ")");

    auto maskType = getMaskType(attrs);
    HIP_KERNEL_RETURN_FALSE_IF(maskType != MaskType::NO_MASK,
                               "Masked attention not currently dispatched (Mask type ordinal: "
                                   + std::to_string(static_cast<int>(maskType)) + ")");

    // The dqdkdv stage hardcodes atomic32 = 1, pssk = 1, pddv = 1. Other
    // combinations live in the registry but require a different kernarg
    // layout than the engine currently builds.
    int bf16CvtValue = (dataTypeId == "fp16") ? BF16_CVT_FP16_SENTINEL
                                              : static_cast<int>(getRoundingMode(attrs));

    // Per-stage dispatch tuples differ. The odo (D-reduction) and dq_convert
    // (FP32 -> output dtype cast) kernels are not parameterised by mask/
    // accumulator/padding — every row in those CSVs has
    //   mask=0, atomic32=0, pssk=0, pddv=0
    // and odo additionally always uses the bf16_cvt=3 sentinel. The dqdkdv
    // (main backward) kernel carries the full dispatch axes.
    auto checkRegistry = [&](const char* registryName,
                             bwd_dispatch::PipelineStage stage,
                             int stageMask,
                             int stageAtomic32,
                             int stagePssk,
                             int stagePddv,
                             int stageBf16Cvt) {
        auto key = bwd_dispatch::lookupKernelNameKey(stage,
                                                     deviceString,
                                                     dataTypeId,
                                                     headDimQk,
                                                     headDimV,
                                                     stageMask,
                                                     stageAtomic32,
                                                     stagePssk,
                                                     stagePddv,
                                                     static_cast<int>(BatchMode::BATCH),
                                                     stageBf16Cvt);
        if(key.empty())
        {
            HIPDNN_PLUGIN_LOG_INFO(
                std::string{HIP_KERNEL_LOG_PREFIX} + "No matching " + registryName
                + " kernel for arch=" + deviceString + " dtype=" + dataTypeId
                + " hdim=" + std::to_string(headDimQk) + " mask=" + std::to_string(stageMask)
                + " atomic32=" + std::to_string(stageAtomic32)
                + " pssk=" + std::to_string(stagePssk) + " pddv=" + std::to_string(stagePddv)
                + " mode=batch bf16_cvt=" + std::to_string(stageBf16Cvt));
            return false;
        }
        return true;
    };

    HIP_KERNEL_RETURN_FALSE_IF(!checkRegistry("odo",
                                              bwd_dispatch::PipelineStage::ODO,
                                              /*mask=*/0,
                                              /*atomic32=*/0,
                                              /*pssk=*/0,
                                              /*pddv=*/0,
                                              /*bf16Cvt=*/BF16_CVT_FP16_SENTINEL),
                               "Failed odo registry lookup");
    HIP_KERNEL_RETURN_FALSE_IF(!checkRegistry("dqdkdv",
                                              bwd_dispatch::PipelineStage::DQDKDV,
                                              static_cast<int>(maskType),
                                              static_cast<int>(AccumulatorMode::A32),
                                              /*pssk=*/1,
                                              /*pddv=*/1,
                                              bf16CvtValue),
                               "Failed dqdkdv registry lookup");
    HIP_KERNEL_RETURN_FALSE_IF(!checkRegistry("dq_convert",
                                              bwd_dispatch::PipelineStage::DQ_CONVERT,
                                              /*mask=*/0,
                                              /*atomic32=*/0,
                                              /*pssk=*/0,
                                              /*pddv=*/0,
                                              bf16CvtValue),
                               "Failed dq_convert registry lookup");

    return true;
}

size_t SdpaBwdPlanBuilder::getMaxWorkspaceSize(
    const HipKernelHandle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const HipKernelSettings& /* executionSettings */) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const auto& attrs = opGraph.nodeWrappers().front()->attributesAs<SdpaBackwardAttributes>();
    const auto& tensorMap = opGraph.getTensorMap();
    const auto* qTensor = tensorMap.at(attrs.q_tensor_uid());

    // Q tensor layout is [B, H_q, S_q, D_qk]
    auto batch = static_cast<size_t>(qTensor->dims()->Get(0));
    auto headsQ = static_cast<size_t>(qTensor->dims()->Get(1));
    auto seqLenQ = static_cast<size_t>(qTensor->dims()->Get(2));
    auto headDim = static_cast<size_t>(qTensor->dims()->Get(3));

    return sdpaBwdDBufferSize(batch, headsQ, seqLenQ)
           + sdpaBwdDqAccBufferSize(batch, headsQ, seqLenQ, headDim);
}

void SdpaBwdPlanBuilder::initializeExecutionSettings(
    const HipKernelHandle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    HipKernelSettings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaBwdPlanBuilder::initializeExecutionSettings not implemented");
}

void SdpaBwdPlanBuilder::buildPlan(
    const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    HipKernelContext& executionContext) const
{
    std::string deviceString;
    try
    {
        deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to query device properties with error: " << e.what());
        return;
    }

    // Extract SDPA backward attributes and tensor metadata from graph
    auto& sdpaNode = opGraph.getNodeWrapper(0);
    auto& sdpaAttrs
        = sdpaNode.attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes>();
    auto& tensorMap = opGraph.getTensorMap();

    int64_t qUid = sdpaAttrs.q_tensor_uid();
    int64_t kUid = sdpaAttrs.k_tensor_uid();
    int64_t vUid = sdpaAttrs.v_tensor_uid();
    int64_t oUid = sdpaAttrs.o_tensor_uid();
    int64_t doUid = sdpaAttrs.do_tensor_uid();
    int64_t statsUid = sdpaAttrs.stats_tensor_uid();
    int64_t dqUid = sdpaAttrs.dq_tensor_uid();
    int64_t dkUid = sdpaAttrs.dk_tensor_uid();
    int64_t dvUid = sdpaAttrs.dv_tensor_uid();

    auto* qTensor = tensorMap.at(qUid);
    auto* kTensor = tensorMap.at(kUid);
    auto* vTensor = tensorMap.at(vUid);
    auto* oTensor = tensorMap.at(oUid);
    auto* doTensor = tensorMap.at(doUid);
    auto* statsTensor = tensorMap.at(statsUid);
    auto* dqTensor = tensorMap.at(dqUid);
    auto* dkTensor = tensorMap.at(dkUid);
    auto* dvTensor = tensorMap.at(dvUid);

    // Dimensions from Q: [B, H_q, S_q, D_qk]
    auto* qDims = qTensor->dims();
    auto batchSize = static_cast<unsigned int>(qDims->Get(0));
    auto numHeadsQ = static_cast<unsigned int>(qDims->Get(1));
    auto seqLenQ = static_cast<unsigned int>(qDims->Get(2));
    auto headDimQk = static_cast<unsigned int>(qDims->Get(3));

    // Dimensions from K: [B, H_kv, S_kv, D_qk]
    auto numHeadsKv = static_cast<unsigned int>(kTensor->dims()->Get(1));
    auto seqLenKv = static_cast<unsigned int>(kTensor->dims()->Get(2));

    // Dimensions from V: [B, H_kv, S_kv, D_v]
    auto headDimV = static_cast<unsigned int>(vTensor->dims()->Get(3));

    auto* qStrides = qTensor->strides();
    auto qStrideBatch = static_cast<unsigned int>(qStrides->Get(0));
    auto qStrideHead = static_cast<unsigned int>(qStrides->Get(1));
    auto qStrideSeq = static_cast<unsigned int>(qStrides->Get(2));

    auto* kStrides = kTensor->strides();
    auto kStrideBatch = static_cast<unsigned int>(kStrides->Get(0));
    auto kStrideHead = static_cast<unsigned int>(kStrides->Get(1));
    auto kStrideSeq = static_cast<unsigned int>(kStrides->Get(2));

    auto* vStrides = vTensor->strides();
    auto vStrideBatch = static_cast<unsigned int>(vStrides->Get(0));
    auto vStrideHead = static_cast<unsigned int>(vStrides->Get(1));
    auto vStrideSeq = static_cast<unsigned int>(vStrides->Get(2));

    auto* oStrides = oTensor->strides();
    auto oStrideBatch = static_cast<unsigned int>(oStrides->Get(0));
    auto oStrideHead = static_cast<unsigned int>(oStrides->Get(1));
    auto oStrideSeq = static_cast<unsigned int>(oStrides->Get(2));

    auto* doStrides = doTensor->strides();
    auto doStrideBatch = static_cast<unsigned int>(doStrides->Get(0));
    auto doStrideHead = static_cast<unsigned int>(doStrides->Get(1));
    auto doStrideSeq = static_cast<unsigned int>(doStrides->Get(2));

    auto* dqStrides = dqTensor->strides();
    auto dqStrideBatch = static_cast<unsigned int>(dqStrides->Get(0));
    auto dqStrideHead = static_cast<unsigned int>(dqStrides->Get(1));
    auto dqStrideSeq = static_cast<unsigned int>(dqStrides->Get(2));

    auto* dkStrides = dkTensor->strides();
    auto dkStrideBatch = static_cast<unsigned int>(dkStrides->Get(0));
    auto dkStrideHead = static_cast<unsigned int>(dkStrides->Get(1));
    auto dkStrideSeq = static_cast<unsigned int>(dkStrides->Get(2));

    auto* dvStrides = dvTensor->strides();
    auto dvStrideBatch = static_cast<unsigned int>(dvStrides->Get(0));
    auto dvStrideHead = static_cast<unsigned int>(dvStrides->Get(1));
    auto dvStrideSeq = static_cast<unsigned int>(dvStrides->Get(2));

    auto* statsStrides = statsTensor->strides();
    auto statsStrideBatch = static_cast<unsigned int>(statsStrides->Get(0));
    auto statsStrideHead = static_cast<unsigned int>(statsStrides->Get(1));

    // Attention scale: default to 1/sqrt(D_qk) if not provided
    float attnScale = 1.0f / std::sqrt(static_cast<float>(headDimQk));
    auto scaleValue = sdpaAttrs.attn_scale_value();
    if(scaleValue.has_value())
    {
        attnScale = scaleValue.value();
    }

    // Resolve dispatch parameters from the graph
    auto dataTypeId = getDataTypeIdentifier(qTensor->data_type(),
                                            kTensor->data_type(),
                                            vTensor->data_type(),
                                            doTensor->data_type(),
                                            dqTensor->data_type(),
                                            dkTensor->data_type(),
                                            dvTensor->data_type());
    auto maskType = getMaskType(sdpaAttrs);
    auto batchMode = getBatchMode(sdpaAttrs);
    int bf16CvtValue = (dataTypeId == "fp16") ? BF16_CVT_FP16_SENTINEL
                                              : static_cast<int>(getRoundingMode(sdpaAttrs));

    // Per-stage dispatch tuples differ — see isApplicable() for the column-by-
    // column rationale. Day-one dqdkdv selection forces atomic32=1, pssk=1,
    // pddv=1 (POC kernarg layout); odo and dq_convert always use the
    // unparameterised row.
    auto resolveStage = [&](const char* stageName,
                            bwd_dispatch::PipelineStage stage,
                            int stageMask,
                            int stageAtomic32,
                            int stagePssk,
                            int stagePddv,
                            int stageBf16Cvt,
                            const std::unordered_map<std::string, fmha_v3_bwdConfig>& registry,
                            SdpaBwdParams::KernelTiles& outTiles,
                            std::string& outCoPath,
                            std::string& outKnlName) -> bool {
        auto key = bwd_dispatch::lookupKernelNameKey(stage,
                                                     deviceString,
                                                     dataTypeId,
                                                     static_cast<int>(headDimQk),
                                                     static_cast<int>(headDimV),
                                                     stageMask,
                                                     stageAtomic32,
                                                     stagePssk,
                                                     stagePddv,
                                                     static_cast<int>(batchMode),
                                                     stageBf16Cvt);
        if(key.empty())
        {
            HIPDNN_PLUGIN_LOG_ERROR("Failed to resolve "
                                    << stageName << " kernel for arch=" << deviceString
                                    << " dtype=" << dataTypeId << " hdim=" << headDimQk);
            return false;
        }
        const auto& cfg = registry.at(key);
        outTiles.tsQO = static_cast<unsigned int>(cfg.ts_qo);
        outTiles.ts = static_cast<unsigned int>(cfg.ts);
        outCoPath = getKernelCoPath(deviceString, cfg.co_name);
        outKnlName = cfg.knl_name;
        return true;
    };

    SdpaBwdParams params{};
    std::string odoCoPath;
    std::string odoKnlName;
    std::string dqdkdvCoPath;
    std::string dqdkdvKnlName;
    std::string dqConvertCoPath;
    std::string dqConvertKnlName;

    if(!resolveStage("odo",
                     bwd_dispatch::PipelineStage::ODO,
                     /*mask=*/0,
                     /*atomic32=*/0,
                     /*pssk=*/0,
                     /*pddv=*/0,
                     /*bf16Cvt=*/BF16_CVT_FP16_SENTINEL,
                     cfg_fmha_bwd_odo,
                     params.odoTiles,
                     odoCoPath,
                     odoKnlName))
    {
        return;
    }
    if(!resolveStage("dqdkdv",
                     bwd_dispatch::PipelineStage::DQDKDV,
                     static_cast<int>(maskType),
                     static_cast<int>(AccumulatorMode::A32),
                     /*pssk=*/1,
                     /*pddv=*/1,
                     bf16CvtValue,
                     cfg_fmha_bwd_dqdkdv,
                     params.dqdkdvTiles,
                     dqdkdvCoPath,
                     dqdkdvKnlName))
    {
        return;
    }
    if(!resolveStage("dq_convert",
                     bwd_dispatch::PipelineStage::DQ_CONVERT,
                     /*mask=*/0,
                     /*atomic32=*/0,
                     /*pssk=*/0,
                     /*pddv=*/0,
                     bf16CvtValue,
                     cfg_fmha_bwd_dq_convert,
                     params.dqConvertTiles,
                     dqConvertCoPath,
                     dqConvertKnlName))
    {
        return;
    }

    HIPDNN_PLUGIN_LOG_INFO("Using bwd odo kernel: " << odoCoPath << " :: " << odoKnlName);
    HIPDNN_PLUGIN_LOG_INFO("Using bwd dqdkdv kernel: " << dqdkdvCoPath << " :: " << dqdkdvKnlName);
    HIPDNN_PLUGIN_LOG_INFO("Using bwd dq_convert kernel: " << dqConvertCoPath
                                                           << " :: " << dqConvertKnlName);

    auto odoKernel = loadKernelModule(odoCoPath, odoKnlName.c_str());
    if(!odoKernel)
    {
        return;
    }

    auto dqdkdvKernel = loadKernelModule(dqdkdvCoPath, dqdkdvKnlName.c_str());
    if(!dqdkdvKernel)
    {
        return;
    }

    auto postKernel = loadKernelModule(dqConvertCoPath, dqConvertKnlName.c_str());
    if(!postKernel)
    {
        return;
    }

    // Populate the rest of params (UIDs, dimensions, strides, scale)
    params.qUid = qUid;
    params.kUid = kUid;
    params.vUid = vUid;
    params.oUid = oUid;
    params.doUid = doUid;
    params.statsUid = statsUid;
    params.dqUid = dqUid;
    params.dkUid = dkUid;
    params.dvUid = dvUid;

    params.batchSize = batchSize;
    params.numHeadsQ = numHeadsQ;
    params.numHeadsKv = numHeadsKv;
    params.seqLenQ = seqLenQ;
    params.seqLenKv = seqLenKv;
    params.headDimQk = headDimQk;
    params.headDimV = headDimV;

    params.qStrideSeq = qStrideSeq;
    params.qStrideHead = qStrideHead;
    params.qStrideBatch = qStrideBatch;
    params.kStrideSeq = kStrideSeq;
    params.kStrideHead = kStrideHead;
    params.kStrideBatch = kStrideBatch;
    params.vStrideSeq = vStrideSeq;
    params.vStrideHead = vStrideHead;
    params.vStrideBatch = vStrideBatch;
    params.oStrideSeq = oStrideSeq;
    params.oStrideHead = oStrideHead;
    params.oStrideBatch = oStrideBatch;
    params.doStrideSeq = doStrideSeq;
    params.doStrideHead = doStrideHead;
    params.doStrideBatch = doStrideBatch;
    params.dqStrideSeq = dqStrideSeq;
    params.dqStrideHead = dqStrideHead;
    params.dqStrideBatch = dqStrideBatch;
    params.dkStrideSeq = dkStrideSeq;
    params.dkStrideHead = dkStrideHead;
    params.dkStrideBatch = dkStrideBatch;
    params.dvStrideSeq = dvStrideSeq;
    params.dvStrideHead = dvStrideHead;
    params.dvStrideBatch = dvStrideBatch;
    params.statsStrideHead = statsStrideHead;
    params.statsStrideBatch = statsStrideBatch;
    params.attnScale = attnScale;

    executionContext.setPlan(std::make_unique<SdpaBwdPlan>(
        std::move(*odoKernel), std::move(*dqdkdvKernel), std::move(*postKernel), params));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> SdpaBwdPlanBuilder::getCustomKnobs(
    const HipKernelHandle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace asm_sdpa_engine
