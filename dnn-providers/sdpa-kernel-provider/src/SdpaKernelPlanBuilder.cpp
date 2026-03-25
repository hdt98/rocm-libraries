// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelPlanBuilder.hpp"
#include "SdpaKernelHelpers.hpp"
#include "SdpaKernelPlan.hpp"
#include "asm/AsmKernelPath.hpp"

#include <cmath>
#include <format>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace sdpa_kernel_provider
{

bool SdpaKernelPlanBuilder::isApplicable(
    const SdpaKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_data_sdk::data_objects;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* SDPA_PROVIDER_LOG_PREFIX = "[SdpaKernelPlanBuilder::isApplicable] ";

    auto& nodeWrappers = opGraph.nodeWrappers();

    try
    {
        auto deviceString = getDeviceString(handle.getStream());
        SDPA_PROVIDER_RETURN_FALSE_IF(deviceString != "gfx942",
                                      "Device string does not match gfx942 (Actual value: {})",
                                      deviceString);
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Could not query device string: " << e.what());
        return false;
    }

    SDPA_PROVIDER_RETURN_FALSE_IF(nodeWrappers.size() != 1, "Graph has more than one node");
    SDPA_PROVIDER_RETURN_FALSE_IF(nodeWrappers.front()->attributesType()
                                      != NodeAttributes::SdpaAttributes,
                                  "Node attribute type is not SdpaAttributes");

    const auto& attrs = nodeWrappers.front()->attributesAs<SdpaAttributes>();
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.causal_mask(), "causal_mask must be false");
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.causal_mask_bottom_right(),
                                  "causal_mask_bottom_right must be false");
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.left_bound().has_value(), "left_bound must be unset");
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.right_bound().has_value(), "right_bound must be unset");
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.dropout_probability().has_value()
                                      && attrs.dropout_probability().value() != 0.f,
                                  "dropout_probability must be unset or zero (Actual value: {})",
                                  attrs.dropout_probability().value());
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.alibi_mask(), "alibi_mask must be false");
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.padding_mask(), "padding_mask must be false");
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.seq_len_q_tensor_uid(), "seq_len_q tensor not supported");
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.attn_mask_tensor_uid(), // Change to bias
                                  "attn_mask tensor not supported");

    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.page_table_k_tensor_uid(),
                                  "page_table_k tensor not supported");
    SDPA_PROVIDER_RETURN_FALSE_IF(attrs.page_table_v_tensor_uid(),
                                  "page_table_v tensor not supported");

    const auto& tensorMap = opGraph.getTensorMap();

    int64_t qUid = attrs.q_tensor_uid();
    int64_t kUid = attrs.k_tensor_uid();
    int64_t vUid = attrs.v_tensor_uid();
    int64_t oUid = attrs.o_tensor_uid();

    auto* qTensor = tensorMap.at(qUid);
    SDPA_PROVIDER_RETURN_FALSE_IF(qTensor->data_type() != DataType::BFLOAT16,
                                  "q tensor datatype must be BF16 (Actual type: {})",
                                  EnumNameDataType(qTensor->data_type()));
    SDPA_PROVIDER_RETURN_FALSE_IF(qTensor->dims()->size() != 4,
                                  "q tensor must be rank 4 (Actual rank: {})",
                                  qTensor->dims()->size());
    SDPA_PROVIDER_RETURN_FALSE_IF(qTensor->dims()->Get(3) != 128,
                                  "q tensor head dimension must be 128 (Actual value: {})",
                                  qTensor->dims()->Get(3));

    auto* kTensor = tensorMap.at(kUid);
    SDPA_PROVIDER_RETURN_FALSE_IF(kTensor->data_type() != DataType::BFLOAT16,
                                  "k tensor datatype must be BF16 (Actual type: {})",
                                  EnumNameDataType(kTensor->data_type()));

    auto* vTensor = tensorMap.at(vUid);
    SDPA_PROVIDER_RETURN_FALSE_IF(vTensor->data_type() != DataType::BFLOAT16,
                                  "v tensor datatype must be BF16 (Actual type: {})",
                                  EnumNameDataType(vTensor->data_type()));

    auto* oTensor = tensorMap.at(oUid);
    SDPA_PROVIDER_RETURN_FALSE_IF(oTensor->data_type() != DataType::BFLOAT16,
                                  "o tensor datatype must be BF16 (Actual type: {})",
                                  EnumNameDataType(oTensor->data_type()));

    return true;
}

size_t SdpaKernelPlanBuilder::getMaxWorkspaceSize(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const SdpaKernelSettings& /* executionSettings */) const
{
    // Forward-only kernel uses 64KB LDS internally, no external workspace needed
    // LSE (when present) is an optional output tensor, not workspace
    return 0;
}

void SdpaKernelPlanBuilder::initializeExecutionSettings(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    SdpaKernelSettings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelPlanBuilder::initializeExecutionSettings not implemented");
}

void SdpaKernelPlanBuilder::buildPlan(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    SdpaKernelContext& executionContext) const
{
    // Load kernel module
    std::string coPath
        = asm_kernels::getAsmKernelPath("gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co");

    hipModule_t module;
    hipError_t err = hipModuleLoad(&module, coPath.c_str());
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "Failed to load kernel module: " << coPath << " error: " << hipGetErrorString(err));
        return;
    }

    hipFunction_t function;
    err = hipModuleGetFunction(&function, module, "_ZN5aiter24fmha_fwd_hd128_bf16_rtneE");
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to get kernel function, error: " << hipGetErrorString(err));
        err = hipModuleUnload(module);
        if(err != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "Failed to unload kernel module on error, error: " << hipGetErrorString(err));
        }
        return;
    }

    // Extract SDPA attributes and tensor metadata
    auto& sdpaNode = opGraph.getNodeWrapper(0);
    auto& sdpaAttrs = sdpaNode.attributesAs<hipdnn_data_sdk::data_objects::SdpaAttributes>();
    auto& tensorMap = opGraph.getTensorMap();

    // Get tensor UIDs
    int64_t qUid = sdpaAttrs.q_tensor_uid();
    int64_t kUid = sdpaAttrs.k_tensor_uid();
    int64_t vUid = sdpaAttrs.v_tensor_uid();
    int64_t oUid = sdpaAttrs.o_tensor_uid();

    // Get tensor attributes
    auto* qTensor = tensorMap.at(qUid);
    auto* kTensor = tensorMap.at(kUid);
    auto* vTensor = tensorMap.at(vUid);
    auto* oTensor = tensorMap.at(oUid);

    // Extract dimensions from Q tensor: [B, H_q, S_q, D_qk]
    auto* qDims = qTensor->dims();
    auto batchSize = static_cast<unsigned int>(qDims->Get(0));
    auto numHeadsQ = static_cast<unsigned int>(qDims->Get(1));
    auto seqLenQ = static_cast<unsigned int>(qDims->Get(2));
    auto headDimQk = static_cast<unsigned int>(qDims->Get(3));

    // Extract dimensions from K tensor: [B, H_kv, S_kv, D_qk]
    auto* kDims = kTensor->dims();
    auto numHeadsKv = static_cast<unsigned int>(kDims->Get(1));
    auto seqLenKv = static_cast<unsigned int>(kDims->Get(2));

    // Extract dimensions from V tensor: [B, H_kv, S_kv, D_v]
    auto* vDims = vTensor->dims();
    auto headDimV = static_cast<unsigned int>(vDims->Get(3));

    // Extract strides (in elements) - Q: [B, H_q, S_q, D_qk]
    auto* qStrides = qTensor->strides();
    auto qStrideBatch = static_cast<unsigned int>(qStrides->Get(0));
    auto qStrideHead = static_cast<unsigned int>(qStrides->Get(1));
    auto qStrideSeq = static_cast<unsigned int>(qStrides->Get(2));
    auto qStrideRow = qStrideSeq; // Same as sequence stride

    // Extract strides - K: [B, H_kv, S_kv, D_qk]
    auto* kStrides = kTensor->strides();
    auto kStrideBatch = static_cast<unsigned int>(kStrides->Get(0));
    auto kStrideHead = static_cast<unsigned int>(kStrides->Get(1));
    auto kStrideSeq = static_cast<unsigned int>(kStrides->Get(2));

    // Extract strides - V: [B, H_kv, S_kv, D_v]
    auto* vStrides = vTensor->strides();
    auto vStrideBatch = static_cast<unsigned int>(vStrides->Get(0));
    auto vStrideHead = static_cast<unsigned int>(vStrides->Get(1));
    auto vStrideSeq = static_cast<unsigned int>(vStrides->Get(2));

    // Extract strides - O: [B, H_q, S_q, D_v]
    auto* oStrides = oTensor->strides();
    auto oStrideBatch = static_cast<unsigned int>(oStrides->Get(0));
    auto oStrideHead = static_cast<unsigned int>(oStrides->Get(1));
    auto oStrideSeq = static_cast<unsigned int>(oStrides->Get(2));

    // Get attention scale (default: 1/sqrt(D_qk) if not provided)
    float attnScale = 1.0f / std::sqrt(static_cast<float>(headDimQk));
    auto scaleValue = sdpaAttrs.attn_scale_value();
    if(scaleValue.has_value())
    {
        attnScale = scaleValue.value();
    }

    // Create plan with all metadata
    executionContext.setPlan(std::make_unique<SdpaKernelPlan>(module,
                                                              function,
                                                              qUid,
                                                              kUid,
                                                              vUid,
                                                              oUid,
                                                              batchSize,
                                                              numHeadsQ,
                                                              numHeadsKv,
                                                              seqLenQ,
                                                              seqLenKv,
                                                              headDimQk,
                                                              headDimV,
                                                              qStrideSeq,
                                                              qStrideRow,
                                                              qStrideHead,
                                                              qStrideBatch,
                                                              kStrideSeq,
                                                              kStrideHead,
                                                              kStrideBatch,
                                                              vStrideSeq,
                                                              vStrideHead,
                                                              vStrideBatch,
                                                              oStrideSeq,
                                                              oStrideHead,
                                                              oStrideBatch,
                                                              attnScale));
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> SdpaKernelPlanBuilder::getCustomKnobs(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace sdpa_kernel_provider
