// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 6 scaffolding for the bwd_dq_dk_dv stage. Wraps
// ck_tile::FmhaBwdKernel. Optional trload variant for gfx950.
// TODO(phase6-followup): full Kargs composition for dQ_acc/dK/dV/dBias.

#include <cmath>
#include <cstdint>
#include <type_traits>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck/host/device_fmha_fwd/fmha_fwd_wrapper.hpp" // FmhaPipelineTag

namespace ck_tile {

template <typename DataType_,
          bool kIsCausal,
          bool kHasBias,
          bool kHasDbias,
          bool kHasDropout,
          bool kIsDeterministic,
          bool kIsStoreRandVal,
          bool kUseTrload>
struct FmhaBwdDqDkDvWrapper
{
    struct Descriptor
    {
        index_t batch, nhead_q, nhead_k, M, N, K, O;
        index_t q_stride_batch, q_stride_nhead, q_stride_m;
        index_t k_stride_batch, k_stride_nhead, k_stride_n;
        index_t v_stride_batch, v_stride_nhead, v_stride_n;
        index_t o_stride_batch, o_stride_nhead, o_stride_m;
        index_t do_stride_batch, do_stride_nhead, do_stride_m;
        index_t dq_stride_batch, dq_stride_nhead, dq_stride_m;
        index_t dk_stride_batch, dk_stride_nhead, dk_stride_n;
        index_t dv_stride_batch, dv_stride_nhead, dv_stride_n;

        float* d_ptr      = nullptr; // workspace from stage 1
        float* dq_acc_ptr = nullptr; // fp32 accumulator
        uint8_t* rand_val_ptr = nullptr;
        uint64_t drop_seed = 0;
        uint64_t drop_offset = 0;
        float p_drop = 0.0f;

        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return true; }
    };

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   float scale,
                                   const DataType_* q_ptr,
                                   const DataType_* k_ptr,
                                   const DataType_* v_ptr,
                                   const DataType_* bias_ptr,
                                   const DataType_* o_ptr,
                                   const float* lse_ptr,
                                   const DataType_* do_ptr,
                                   DataType_* dq_ptr,
                                   DataType_* dk_ptr,
                                   DataType_* dv_ptr,
                                   DataType_* dbias_ptr)
    {
        (void)desc; (void)scale; (void)q_ptr; (void)k_ptr; (void)v_ptr; (void)bias_ptr;
        (void)o_ptr; (void)lse_ptr; (void)do_ptr; (void)dq_ptr; (void)dk_ptr; (void)dv_ptr;
        (void)dbias_ptr;
        // Phase 6 scaffolding.
    }
};

} // namespace ck_tile
