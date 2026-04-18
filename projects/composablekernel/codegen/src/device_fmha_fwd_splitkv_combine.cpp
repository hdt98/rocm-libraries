// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_fwd_splitkv_combine/problem.hpp"
#include "ck/host/device_fmha_fwd_splitkv_combine/operation.hpp"
#include "ck/host/stringutils.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace ck {
namespace host {
namespace device_fmha_fwd_splitkv_combine {

static const char* const WrapperTemplate =
    "ck_tile::FmhaFwdSplitKVCombineWrapper<${DataType}, ${M}, ${N1}, "
    "${MaxSplitsLog2}, ${HasLSE}, ${PadM}, ${PadN1}>";

static std::string ToDT(DataType d)
{
    return d == DataType::Float ? "float" : "ck_tile::fp16_t";
}

std::string Problem::GetIncludeHeader() const
{
    return "ck/host/device_fmha_fwd_splitkv_combine/wrapper.hpp";
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    (void)arch;
    Operation op;
    op.combine_bn1 = 32;
    op.dtype       = prob.dtype;
    op.has_lse     = prob.has_lse;
    auto log2_up = [](std::size_t n) -> std::size_t {
        std::size_t r = 0;
        while((static_cast<std::size_t>(1) << r) < n) ++r;
        return r;
    };
    op.max_splits_log2 = log2_up(prob.num_splits > 1 ? prob.num_splits : 2);
    return {op};
}

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> v = {
        {"DataType", ToDT(dtype)},
        {"M", "64"},
        {"N1", std::to_string(combine_bn1)},
        {"MaxSplitsLog2", std::to_string(max_splits_log2)},
        {"HasLSE", has_lse ? "true" : "false"},
        {"PadM", "true"},
        {"PadN1", "true"},
    };
    return Solution{InterpolateString(WrapperTemplate, v), std::move(v)};
}

std::vector<Solution> Problem::GetSolutions(const std::string& arch) const
{
    auto ops = Operation::CreateOperations(*this, arch);
    std::vector<Solution> result;
    std::transform(ops.begin(), ops.end(), std::back_inserter(result), [](const auto& op) {
        return op.ToSolution();
    });
    return result;
}

} // namespace device_fmha_fwd_splitkv_combine
} // namespace host
} // namespace ck
