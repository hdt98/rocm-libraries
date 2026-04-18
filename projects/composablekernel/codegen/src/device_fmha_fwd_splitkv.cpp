// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_fwd_splitkv/problem.hpp"
#include "ck/host/device_fmha_fwd_splitkv/operation.hpp"

#include <algorithm>

namespace ck {
namespace host {
namespace device_fmha_fwd_splitkv {

std::string Problem::GetIncludeHeader() const
{
    return "ck/host/device_fmha_fwd_splitkv/wrapper.hpp";
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

} // namespace device_fmha_fwd_splitkv
} // namespace host
} // namespace ck
