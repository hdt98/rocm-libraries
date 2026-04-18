// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 5 scaffolding for the FwdSplitKv family. FmhaDispatcher::plan()
// for FwdSplitKv produces a 2-stage plan (split + combine); this is
// the "split" half. See device_fmha_fwd_splitkv_combine/ for the combine
// kernel.

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd_splitkv {

struct Problem
{
    std::size_t M = 0;
    std::size_t N = 0;
    std::size_t K = 0;
    std::size_t O = 0;
    std::size_t batch   = 0;
    std::size_t nhead_q = 0;
    std::size_t nhead_k = 0;
    DataType dtype      = DataType::Half;
    bool is_v_rowmajor  = true;
    bool is_causal      = false;
    bool has_bias       = false;
    std::size_t num_splits = 4;
    std::size_t page_block_size = 0; // 0 = non-paged split-kv
    bool has_lse      = false;

    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_fwd_splitkv
} // namespace host
} // namespace ck
