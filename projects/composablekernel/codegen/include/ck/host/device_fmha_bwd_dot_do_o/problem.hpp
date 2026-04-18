// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 6 scaffolding for the bwd_dot_do_o stage. Computes
// d = rowsum(dO * O) into the workspace `d_ptr`.

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_bwd_dot_do_o {

struct Problem
{
    std::size_t M = 0;
    std::size_t O = 0;
    std::size_t batch   = 0;
    std::size_t nhead_q = 0;
    DataType dtype      = DataType::Half;
    bool is_v_rowmajor  = true;
    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_bwd_dot_do_o
} // namespace host
} // namespace ck
