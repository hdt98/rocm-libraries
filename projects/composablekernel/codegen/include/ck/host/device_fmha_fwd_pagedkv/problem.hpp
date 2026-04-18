// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 5 scaffolding for the FwdPagedKv family. Mirrors
// device_fmha_fwd/problem.hpp but adds paged-KV-specific fields:
// page_block_size, block_table, cache_batch_idx, is_gappy.
//
// The Wrapper (device_fmha_fwd_pagedkv/wrapper.hpp) composes directly
// onto ck_tile::FmhaFwdPagedKVKernel.

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd_pagedkv {

struct Problem
{
    std::size_t M = 0;  // seqlen_q
    std::size_t N = 0;  // seqlen_k
    std::size_t K = 0;  // hdim_q
    std::size_t O = 0;  // hdim_v

    std::size_t batch   = 0;
    std::size_t nhead_q = 0;
    std::size_t nhead_k = 0;

    DataType dtype = DataType::Half;

    bool is_v_rowmajor = true;
    bool is_causal     = false;
    bool has_bias      = false;

    // PagedKV-specific
    std::size_t page_block_size = 0;
    bool is_gappy               = false;
    bool has_cache_batch_idx    = false;

    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_fwd_pagedkv
} // namespace host
} // namespace ck
