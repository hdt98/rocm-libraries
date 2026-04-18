// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_fwd_appendkv/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd_appendkv {

struct Operation
{
    DataType dtype         = DataType::Half;
    bool is_v_rowmajor     = true;
    RopeType rope_type     = RopeType::None;
    std::size_t rotary_dim = 0;
    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);
    Solution ToSolution() const;
};

} // namespace device_fmha_fwd_appendkv
} // namespace host
} // namespace ck
