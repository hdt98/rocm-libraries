// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 6 scaffolding for fp32 -> out-dtype conversion of the dQ
// accumulator. Simple element-wise kernel.

#include <cmath>
#include <cstdint>
#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename DataType_>
struct FmhaBwdConvertDqWrapper
{
    struct Descriptor
    {
        index_t batch, nhead_q, M, K;
        float* dq_acc_ptr = nullptr;
        index_t stride_batch, stride_nhead, stride_m;
        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return true; }
    };

    CK_TILE_DEVICE static void Run(const Descriptor& desc, DataType_* dq_ptr)
    {
        (void)desc; (void)dq_ptr;
        // Phase 6 scaffolding. Real impl: dq_ptr[i] = type_convert<DataType_>(dq_acc_ptr[i])
    }
};

} // namespace ck_tile
