// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_bwd_convert_dq/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_bwd_convert_dq {

struct Operation
{
    DataType dtype = DataType::Half;
    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);
    Solution ToSolution() const;
};

} // namespace device_fmha_bwd_convert_dq
} // namespace host
} // namespace ck
