// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 5 scaffolding for BatchPrefill. Wraps ck_tile::FmhaBatchPrefillKernel.
// TODO(phase5-followup): complete Kargs composition with kv_indptr /
// kv_page_indices / kv_last_page_lens for real runs.

#include <cmath>
#include <cstdint>
#include <type_traits>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck/host/device_fmha_fwd/fmha_fwd_wrapper.hpp"

namespace ck_tile {

enum class BatchPrefillKVMemoryLayout : int { Vectorized = 0, Linear = 1 };
enum class BatchPrefillKVLookupTable  : int { SGLang = 0, Vllm = 1, PagedAttention = 2 };

template <typename DataType_,
          index_t kBM0,
          index_t kBN0,
          index_t kBK0,
          index_t kBN1,
          index_t kBK1,
          index_t kBK0Max,
          index_t kRM0,
          index_t kRN0,
          index_t kRK0,
          index_t kRM1,
          index_t kRN1,
          index_t kRK1,
          index_t kWM0,
          index_t kWN0,
          index_t kWK0,
          index_t kWM1,
          index_t kWN1,
          index_t kWK1,
          bool kIsCausal,
          bool kIsVRowMajor,
          bool kHasBias,
          bool kHasLSE,
          bool kPadM,
          bool kPadN,
          bool kPadK,
          bool kPadO,
          FmhaPipelineTag kPipelineTag,
          index_t kPageBlockSize,
          BatchPrefillKVMemoryLayout kLayout,
          BatchPrefillKVLookupTable kLookup>
struct FmhaBatchPrefillWrapper
{
    struct Descriptor
    {
        index_t batch, nhead_q, nhead_k, M, K, N, O;
        index_t num_total_pages = 0;
        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return true; }
    };

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   float scale_s,
                                   const DataType_* q_ptr,
                                   const DataType_* k_ptr,
                                   const DataType_* v_ptr,
                                   const DataType_* bias_ptr,
                                   DataType_* o_ptr)
    {
        (void)desc; (void)scale_s; (void)q_ptr; (void)k_ptr; (void)v_ptr; (void)bias_ptr; (void)o_ptr;
        // Phase 5 scaffolding.
    }
};

} // namespace ck_tile
