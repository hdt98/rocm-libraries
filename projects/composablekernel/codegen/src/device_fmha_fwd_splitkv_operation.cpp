// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_fwd_splitkv/operation.hpp"
#include "ck/host/stringutils.hpp"

#include <cmath>
#include <unordered_map>

namespace ck {
namespace host {
namespace device_fmha_fwd_splitkv {

static const char* const WrapperTemplate =
    "ck_tile::FmhaFwdSplitKVWrapper<${DataType}, "
    "${BM0}, ${BN0}, ${BK0}, ${BN1}, ${BK1}, ${BK0Max}, "
    "${RM0}, ${RN0}, ${RK0}, ${RM1}, ${RN1}, ${RK1}, "
    "${WM0}, ${WN0}, ${WK0}, ${WM1}, ${WN1}, ${WK1}, "
    "${IsCausal}, ${IsVRowMajor}, ${HasBias}, ${HasLSE}, "
    "${PadM}, ${PadN}, ${PadK}, ${PadO}, "
    "ck_tile::FmhaPipelineTag::${PipelineTag}, ${MaxSplitsLog2}>";

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    auto bucket = device_fmha_fwd::GetTileConfigsForHdim(arch, prob.dtype, prob.K, prob.O);
    std::vector<Operation> out;
    auto log2_up = [](std::size_t n) -> std::size_t {
        std::size_t r = 0;
        while((static_cast<std::size_t>(1) << r) < n) ++r;
        return r;
    };
    for(const auto& tile : bucket.tiles)
    {
        Operation op;
        op.tile          = tile;
        op.pipeline      = "qr";
        op.is_causal     = prob.is_causal;
        op.is_v_rowmajor = prob.is_v_rowmajor;
        op.has_bias      = prob.has_bias;
        op.has_lse       = prob.has_lse;
        op.dtype         = prob.dtype;
        op.pad_m         = (prob.M % tile.bm0) != 0;
        op.pad_n         = (prob.N % tile.bn0) != 0;
        op.pad_k         = (prob.K != bucket.bucket_hdim);
        op.pad_o         = (prob.O != bucket.bucket_hdim_v);
        op.max_splits_log2 = log2_up(prob.num_splits > 1 ? prob.num_splits : 2);
        out.push_back(op);
    }
    return out;
}

static std::string ToDT(DataType d)
{
    return d == DataType::Float ? "float" : "ck_tile::fp16_t";
}

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> v = {
        {"DataType", ToDT(dtype)},
        {"BM0", std::to_string(tile.bm0)}, {"BN0", std::to_string(tile.bn0)},
        {"BK0", std::to_string(tile.bk0)}, {"BN1", std::to_string(tile.bn1)},
        {"BK1", std::to_string(tile.bk1)}, {"BK0Max", std::to_string(tile.bk0max)},
        {"RM0", std::to_string(tile.rm0)}, {"RN0", std::to_string(tile.rn0)},
        {"RK0", std::to_string(tile.rk0)}, {"RM1", std::to_string(tile.rm1)},
        {"RN1", std::to_string(tile.rn1)}, {"RK1", std::to_string(tile.rk1)},
        {"WM0", std::to_string(tile.wm0)}, {"WN0", std::to_string(tile.wn0)},
        {"WK0", std::to_string(tile.wk0)}, {"WM1", std::to_string(tile.wm1)},
        {"WN1", std::to_string(tile.wn1)}, {"WK1", std::to_string(tile.wk1)},
        {"IsCausal", is_causal ? "true" : "false"},
        {"IsVRowMajor", is_v_rowmajor ? "true" : "false"},
        {"HasBias", has_bias ? "true" : "false"},
        {"HasLSE", has_lse ? "true" : "false"},
        {"PadM", pad_m ? "true" : "false"}, {"PadN", pad_n ? "true" : "false"},
        {"PadK", pad_k ? "true" : "false"}, {"PadO", pad_o ? "true" : "false"},
        {"PipelineTag", "QR"},
        {"MaxSplitsLog2", std::to_string(max_splits_log2)},
    };
    return Solution{InterpolateString(WrapperTemplate, v), std::move(v)};
}

} // namespace device_fmha_fwd_splitkv
} // namespace host
} // namespace ck
