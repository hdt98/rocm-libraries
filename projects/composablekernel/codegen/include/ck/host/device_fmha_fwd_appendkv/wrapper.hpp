// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 5 scaffolding for FwdAppendKv. Tile shape is fixed by the
// AppendKV primitive itself (it's a memory op); only dtype and RoPE
// variant are parameterised.
// TODO(phase5-followup): bind FmhaFwdAppendKVKernel properly.

#include <cmath>
#include <cstdint>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

namespace ck_tile {

enum class AppendKVRope : int { None = 0, Interleaved = 1, HalfRotated = 2 };

template <typename DataType_,
          bool kIsVRowMajor,
          AppendKVRope kRopeType>
struct FmhaFwdAppendKVWrapper
{
    struct Descriptor
    {
        index_t batch, nhead_q, nhead_k;
        index_t seqlen_q, seqlen_knew;
        index_t hdim_q, hdim_v;
        index_t rotary_dim = 0;
        index_t page_block_size = 0;
        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return true; }
    };

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   DataType_* q_ptr,
                                   DataType_* k_ptr,
                                   const DataType_* knew_ptr,
                                   DataType_* v_ptr,
                                   const DataType_* vnew_ptr)
    {
        (void)desc; (void)q_ptr; (void)k_ptr; (void)knew_ptr; (void)v_ptr; (void)vnew_ptr;
        // Phase 5 scaffolding.
    }
};

} // namespace ck_tile
