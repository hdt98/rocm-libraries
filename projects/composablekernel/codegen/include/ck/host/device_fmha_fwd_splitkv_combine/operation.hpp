// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_fwd_splitkv_combine/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd_splitkv_combine {

struct Operation
{
    std::size_t combine_bn1     = 32;
    std::size_t max_splits_log2 = 3;
    DataType dtype              = DataType::Half;
    bool has_lse                = false;

    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);
    Solution ToSolution() const;
};

} // namespace device_fmha_fwd_splitkv_combine
} // namespace host
} // namespace ck
