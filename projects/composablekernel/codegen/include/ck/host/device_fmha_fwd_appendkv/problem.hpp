// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 5 scaffolding for FwdAppendKv. Memory-copy primitive (not
// compute): appends new K/V tokens into an existing paged cache,
// optionally applying rotary positional embeddings.

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd_appendkv {

enum class RopeType : int { None = 0, Interleaved = 1, HalfRotated = 2 };

struct Problem
{
    std::size_t seqlen_q    = 0;
    std::size_t seqlen_knew = 0;
    std::size_t hdim_q      = 0;
    std::size_t hdim_v      = 0;
    std::size_t batch       = 0;
    std::size_t nhead_q     = 0;
    std::size_t nhead_k     = 0;
    DataType dtype          = DataType::Half;
    bool is_v_rowmajor      = true;
    RopeType rope_type      = RopeType::None;
    std::size_t rotary_dim  = 0;
    std::size_t page_block_size = 0;

    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_fwd_appendkv
} // namespace host
} // namespace ck
