#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/// Transpose a batch of 16 matrices, each 4x4, stored in MatrixLayout<4,4,16,__half> format.
///
/// Each lane holds 2 x uint32_t registers (packing 2 x __half each), representing one row
/// of one 4x4 matrix.  Groups of 4 consecutive lanes (sharing the same batch index) form
/// one matrix.
///
/// The algorithm is a two-step butterfly using wave shuffle instructions:
///
///   Step 1 — XOR-2:  Swap off-diagonal 2x2 blocks.
///     Lanes 0,1 exchange their reg1 with reg0 of lanes 2,3 (and vice-versa).
///
///   Step 2 — XOR-1:  2x2 transpose within each pair of adjacent lanes.
///     Even lanes keep their low half-word and take the other lane's low half-word.
///     Odd  lanes take the other lane's high half-word and keep their own high half-word.
///
/// After both steps, element (row, col) of each matrix has moved to (col, row).
__device__ inline void transpose_4x4_batch16(uint32_t& reg0, uint32_t& reg1)
{
    const int r = threadIdx.x & 3;

    // Step 1: swap off-diagonal 2x2 blocks (xor with 2 within groups of 4 lanes).
    uint32_t xor2_0 = __shfl_xor(static_cast<int>(reg0), 2, 4);
    uint32_t xor2_1 = __shfl_xor(static_cast<int>(reg1), 2, 4);
    if(r < 2)
        reg1 = xor2_0; // lanes 0,1: replace reg1 with reg0 from lanes 2,3
    else
        reg0 = xor2_1; // lanes 2,3: replace reg0 with reg1 from lanes 0,1

    // Step 2: 2x2 transpose within each pair (xor with 1 within groups of 4 lanes).
    uint32_t xor1_0 = __shfl_xor(static_cast<int>(reg0), 1, 4);
    uint32_t xor1_1 = __shfl_xor(static_cast<int>(reg1), 1, 4);
    if(r & 1)
    {
        // Odd lane: take hi half from other, keep own hi half.
        reg0 = (xor1_0 >> 16) | (reg0 & 0xFFFF0000u);
        reg1 = (xor1_1 >> 16) | (reg1 & 0xFFFF0000u);
    }
    else
    {
        // Even lane: keep own lo half, take lo half from other.
        reg0 = (reg0 & 0xFFFFu) | (xor1_0 << 16);
        reg1 = (reg1 & 0xFFFFu) | (xor1_1 << 16);
    }
}
