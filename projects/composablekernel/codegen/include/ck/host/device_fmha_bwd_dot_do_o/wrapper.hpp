// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 6 scaffolding: bwd_dot_do_o kernel. Computes per-element
// d[b,h,m] = sum_v O[b,h,m,v] * dO[b,h,m,v], writing into the workspace
// `d_ptr` ahead of the bwd_dq_dk_dv stage.
// TODO(phase6-followup): bind ck_tile::BlockFmhaBwdDotDoOKernel properly.

#include <cmath>
#include <cstdint>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

namespace ck_tile {

template <typename DataType_,
          index_t kBM0,
          index_t kBN0,
          bool kPadM,
          bool kPadO>
struct FmhaBwdDotDoOWrapper
{
    struct Descriptor
    {
        index_t batch, nhead_q, M, O;
        index_t o_stride_batch, o_stride_nhead, o_stride_m;
        index_t do_stride_batch, do_stride_nhead, do_stride_m;
        float* d_ptr;
        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return true; }
    };

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   const DataType_* o_ptr,
                                   const DataType_* do_ptr)
    {
        (void)desc; (void)o_ptr; (void)do_ptr;
        // Phase 6 scaffolding.
    }
};

} // namespace ck_tile
