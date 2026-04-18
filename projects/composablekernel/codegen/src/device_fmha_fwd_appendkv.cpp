// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_fwd_appendkv/problem.hpp"
#include "ck/host/device_fmha_fwd_appendkv/operation.hpp"
#include "ck/host/stringutils.hpp"

#include <algorithm>
#include <unordered_map>

namespace ck {
namespace host {
namespace device_fmha_fwd_appendkv {

static const char* const WrapperTemplate =
    "ck_tile::FmhaFwdAppendKVWrapper<${DataType}, ${IsVRowMajor}, "
    "ck_tile::AppendKVRope::${RopeType}>";

std::string Problem::GetIncludeHeader() const
{
    return "ck/host/device_fmha_fwd_appendkv/wrapper.hpp";
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    (void)arch;
    Operation op;
    op.dtype         = prob.dtype;
    op.is_v_rowmajor = prob.is_v_rowmajor;
    op.rope_type     = prob.rope_type;
    op.rotary_dim    = prob.rotary_dim;
    return {op};
}

static std::string ToDT(DataType d)
{
    return d == DataType::Float ? "float" : "ck_tile::fp16_t";
}

static std::string RopeName(RopeType r)
{
    switch(r)
    {
    case RopeType::Interleaved: return "Interleaved";
    case RopeType::HalfRotated: return "HalfRotated";
    default:                    return "None";
    }
}

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> v = {
        {"DataType", ToDT(dtype)},
        {"IsVRowMajor", is_v_rowmajor ? "true" : "false"},
        {"RopeType", RopeName(rope_type)},
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

} // namespace device_fmha_fwd_appendkv
} // namespace host
} // namespace ck
