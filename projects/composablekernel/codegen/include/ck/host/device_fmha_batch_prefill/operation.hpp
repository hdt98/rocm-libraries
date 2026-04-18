// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_fwd/operation.hpp"
#include "ck/host/device_fmha_batch_prefill/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_batch_prefill {

using device_fmha_fwd::TileConfig;

struct Operation
{
    TileConfig tile = {};
    std::string pipeline = "qr_async";
    bool is_causal     = false;
    bool is_v_rowmajor = true;
    bool has_bias      = false;
    bool has_lse       = false;
    DataType dtype     = DataType::Half;
    bool pad_m = true, pad_n = true, pad_k = true, pad_o = true;
    std::size_t page_block_size    = 64;
    KVMemoryLayout kv_memory_layout = KVMemoryLayout::Vectorized;
    KVLookupTable kv_lookup_table   = KVLookupTable::SGLang;

    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);
    Solution ToSolution() const;
};

} // namespace device_fmha_batch_prefill
} // namespace host
} // namespace ck
