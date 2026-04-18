// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_bwd_convert_dq/problem.hpp"
#include "ck/host/device_fmha_bwd_convert_dq/operation.hpp"
#include "ck/host/stringutils.hpp"

#include <algorithm>
#include <unordered_map>

namespace ck {
namespace host {
namespace device_fmha_bwd_convert_dq {

static const char* const WrapperTemplate =
    "ck_tile::FmhaBwdConvertDqWrapper<${DataType}>";

std::string Problem::GetIncludeHeader() const
{
    return "ck/host/device_fmha_bwd_convert_dq/wrapper.hpp";
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    (void)arch;
    Operation op;
    op.dtype = prob.dtype;
    return {op};
}

static std::string ToDT(DataType d)
{
    return d == DataType::Float ? "float" : "ck_tile::fp16_t";
}

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> v = {{"DataType", ToDT(dtype)}};
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

} // namespace device_fmha_bwd_convert_dq
} // namespace host
} // namespace ck
