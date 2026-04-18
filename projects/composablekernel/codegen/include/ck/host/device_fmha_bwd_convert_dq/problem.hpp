// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 6 scaffolding: stage 3 of backward. Converts the fp32
// accumulator `dq_acc_ptr` to the output precision `dq_ptr`.

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_bwd_convert_dq {

struct Problem
{
    std::size_t M = 0;
    std::size_t K = 0;
    std::size_t batch   = 0;
    std::size_t nhead_q = 0;
    DataType dtype = DataType::Half;
    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_bwd_convert_dq
} // namespace host
} // namespace ck
