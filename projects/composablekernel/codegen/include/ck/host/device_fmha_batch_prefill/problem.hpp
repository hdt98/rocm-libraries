// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 5 scaffolding for BatchPrefill. Fused prefill + paged-KV
// forward; typically used by inference serving stacks.

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_batch_prefill {

enum class KVMemoryLayout : int { Vectorized = 0, Linear = 1 };
enum class KVLookupTable  : int { SGLang = 0, Vllm = 1, PagedAttention = 2 };

struct Problem
{
    std::size_t M = 0, N = 0, K = 0, O = 0;
    std::size_t batch = 0, nhead_q = 0, nhead_k = 0;
    DataType dtype = DataType::Half;
    bool is_v_rowmajor = true;
    bool is_causal     = false;
    bool has_bias      = false;
    bool has_lse       = false;
    std::size_t page_block_size        = 64;
    KVMemoryLayout kv_memory_layout    = KVMemoryLayout::Vectorized;
    KVLookupTable kv_lookup_table      = KVLookupTable::SGLang;

    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_batch_prefill
} // namespace host
} // namespace ck
