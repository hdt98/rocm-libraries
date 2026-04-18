// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Stage 2 of split-KV. Accepts the per-split accumulators produced by
// device_fmha_fwd_splitkv and produces the final O + LSE.

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd_splitkv_combine {

struct Problem
{
    std::size_t M = 0;
    std::size_t O = 0;
    std::size_t batch   = 0;
    std::size_t nhead_q = 0;
    std::size_t num_splits = 4;
    DataType dtype = DataType::Half;
    bool has_lse   = false;
    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_fwd_splitkv_combine
} // namespace host
} // namespace ck
