// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_fwd_pagedkv/operation.hpp"
#include "ck/host/device_fmha_fwd/operation.hpp"
#include "ck/host/stringutils.hpp"

#include <unordered_map>

namespace ck {
namespace host {
namespace device_fmha_fwd_pagedkv {

static const char* const FmhaFwdPagedKVWrapperTemplate =
    "ck_tile::FmhaFwdPagedKVWrapper<${DataType}, "
    "${BM0}, ${BN0}, ${BK0}, ${BN1}, ${BK1}, ${BK0Max}, "
    "${RM0}, ${RN0}, ${RK0}, ${RM1}, ${RN1}, ${RK1}, "
    "${WM0}, ${WN0}, ${WK0}, ${WM1}, ${WN1}, ${WK1}, "
    "${IsCausal}, ${IsVRowMajor}, ${HasBias}, "
    "${PadM}, ${PadN}, ${PadK}, ${PadO}, "
    "ck_tile::FmhaPipelineTag::${PipelineTag}, "
    "${PageBlockSize}>";

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    // Reuse forward's tile table. Paged-KV typically inherits the
    // same tile shapes; page-specific constraints are checked by the
    // ck_tile kernel's static_assert at hipRTC compile time.
    auto bucket = device_fmha_fwd::GetTileConfigsForHdim(arch, prob.dtype, prob.K, prob.O);

    std::vector<Operation> result;
    for(const auto& tile : bucket.tiles)
    {
        Operation op;
        op.tile            = tile;
        op.pipeline        = "qr_pagedkv";
        op.is_causal       = prob.is_causal;
        op.is_v_rowmajor   = prob.is_v_rowmajor;
        op.has_bias        = prob.has_bias;
        op.dtype           = prob.dtype;
        op.pad_m           = (prob.M % tile.bm0) != 0;
        op.pad_n           = (prob.N % tile.bn0) != 0;
        op.pad_k           = (prob.K != bucket.bucket_hdim);
        op.pad_o           = (prob.O != bucket.bucket_hdim_v);
        op.page_block_size = prob.page_block_size > 0 ? prob.page_block_size : 64;
        result.push_back(op);
    }
    return result;
}

static std::string ToDataTypeString(DataType dtype)
{
    switch(dtype)
    {
    case DataType::Half:  return "ck_tile::fp16_t";
    case DataType::Float: return "float";
    default:              return "ck_tile::fp16_t";
    }
}

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"DataType", ToDataTypeString(dtype)},
        {"BM0", std::to_string(tile.bm0)},
        {"BN0", std::to_string(tile.bn0)},
        {"BK0", std::to_string(tile.bk0)},
        {"BN1", std::to_string(tile.bn1)},
        {"BK1", std::to_string(tile.bk1)},
        {"BK0Max", std::to_string(tile.bk0max)},
        {"RM0", std::to_string(tile.rm0)},
        {"RN0", std::to_string(tile.rn0)},
        {"RK0", std::to_string(tile.rk0)},
        {"RM1", std::to_string(tile.rm1)},
        {"RN1", std::to_string(tile.rn1)},
        {"RK1", std::to_string(tile.rk1)},
        {"WM0", std::to_string(tile.wm0)},
        {"WN0", std::to_string(tile.wn0)},
        {"WK0", std::to_string(tile.wk0)},
        {"WM1", std::to_string(tile.wm1)},
        {"WN1", std::to_string(tile.wn1)},
        {"WK1", std::to_string(tile.wk1)},
        {"IsCausal", is_causal ? "true" : "false"},
        {"IsVRowMajor", is_v_rowmajor ? "true" : "false"},
        {"HasBias", has_bias ? "true" : "false"},
        {"PadM", pad_m ? "true" : "false"},
        {"PadN", pad_n ? "true" : "false"},
        {"PadK", pad_k ? "true" : "false"},
        {"PadO", pad_o ? "true" : "false"},
        {"PipelineTag",
         pipeline == "qr_async_trload" ? "QR_ASYNC_TRLOAD"
                                       : (pipeline == "qr_async" ? "QR_ASYNC" : "QR")},
        {"PageBlockSize", std::to_string(page_block_size)},
    };

    return Solution{InterpolateString(FmhaFwdPagedKVWrapperTemplate, values), std::move(values)};
}

} // namespace device_fmha_fwd_pagedkv
} // namespace host
} // namespace ck
