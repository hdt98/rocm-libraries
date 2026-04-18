// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_bwd_dq_dk_dv/problem.hpp"
#include "ck/host/device_fmha_bwd_dq_dk_dv/operation.hpp"
#include "ck/host/stringutils.hpp"

#include <algorithm>
#include <unordered_map>

namespace ck {
namespace host {
namespace device_fmha_bwd_dq_dk_dv {

static const char* const WrapperTemplate =
    "ck_tile::FmhaBwdDqDkDvWrapper<${DataType}, "
    "${IsCausal}, ${HasBias}, ${HasDbias}, ${HasDropout}, "
    "${IsDeterministic}, ${IsStoreRandVal}, ${UseTrload}>";

std::string Problem::GetIncludeHeader() const
{
    return "ck/host/device_fmha_bwd_dq_dk_dv/wrapper.hpp";
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    (void)arch;
    Operation op;
    op.dtype             = prob.dtype;
    op.is_causal         = prob.is_causal;
    op.has_bias          = prob.has_bias;
    op.has_dbias         = prob.has_dbias;
    op.has_dropout       = prob.has_dropout;
    op.is_deterministic  = prob.is_deterministic;
    op.is_store_randval  = prob.is_store_randval;
    op.use_trload        = prob.use_trload;
    return {op};
}

static std::string ToDT(DataType d)
{
    return d == DataType::Float ? "float" : "ck_tile::fp16_t";
}

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> v = {
        {"DataType", ToDT(dtype)},
        {"IsCausal", is_causal ? "true" : "false"},
        {"HasBias", has_bias ? "true" : "false"},
        {"HasDbias", has_dbias ? "true" : "false"},
        {"HasDropout", has_dropout ? "true" : "false"},
        {"IsDeterministic", is_deterministic ? "true" : "false"},
        {"IsStoreRandVal", is_store_randval ? "true" : "false"},
        {"UseTrload", use_trload ? "true" : "false"},
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

} // namespace device_fmha_bwd_dq_dk_dv
} // namespace host
} // namespace ck
