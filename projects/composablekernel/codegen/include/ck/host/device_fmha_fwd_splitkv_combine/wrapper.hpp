// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 5 scaffolding for the FwdSplitKvCombine family (stage 2).
// TODO(phase5-followup): full Kargs composition for
// FmhaFwdSplitKVCombineKernel lands with a correctness test.

#include <cmath>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

namespace ck_tile {

template <typename DataType_,
          index_t kM,
          index_t kN1, // hdim_v tile
          index_t kMaxSplitsLog2,
          bool kHasLSE,
          bool kPadM,
          bool kPadN1>
struct FmhaFwdSplitKVCombineWrapper
{
    struct Descriptor
    {
        index_t batch, nhead_q, M, O;
        index_t num_splits;
        index_t stride_o, nhead_stride_o, batch_stride_o;
        // Scaffolding: Phase 5 follow-up completes.
        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return true; }
    };

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   float* lse_acc_ptr,
                                   float* o_acc_ptr,
                                   float* lse_ptr,
                                   DataType_* o_ptr)
    {
        (void)desc; (void)lse_acc_ptr; (void)o_acc_ptr; (void)lse_ptr; (void)o_ptr;
        // Placeholder; Phase 5 follow-up fills this.
    }
};

} // namespace ck_tile
