#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/// Load a 16x16 row-major fp16 matrix into registers in MFMA
/// MatrixLayout<16,16,1,__half> format, transposed.
///
/// Each lane reads 4 elements from column r (= lane % 16), rows 4g..4g+3
/// (where g = lane / 16), and packs them into two uint32_t registers.
/// The resulting register contents represent M^T: the MFMA position
/// [r][4g+j] holds src[4g+j][r].
///
/// @param src        Pointer to the 16x16 row-major source matrix.
/// @param row_stride Stride between consecutive rows (in __half units). Default 16
///                   for a contiguous 16x16 matrix; set to kh*kw*group_size for
///                   weight tensors laid out as wei[k][kh*kw*group_size].
/// @param reg0       Output: first register (columns 4g+0, 4g+1 packed as lo/hi half).
/// @param reg1       Output: second register (columns 4g+2, 4g+3 packed as lo/hi half).
__device__ inline void
load_16x16_transposed(const __half* src, int row_stride, uint32_t& reg0, uint32_t& reg1)
{
    const int lane = threadIdx.x % 64;
    const int r    = lane % 16; // MFMA row (outer dimension)
    const int g    = lane / 16; // group index (0-3)

    // Read 4 elements from column r, rows 4g..4g+3 of the source matrix.
    const __half v0 = src[(4 * g + 0) * row_stride + r];
    const __half v1 = src[(4 * g + 1) * row_stride + r];
    const __half v2 = src[(4 * g + 2) * row_stride + r];
    const __half v3 = src[(4 * g + 3) * row_stride + r];

    // Pack two halves into each uint32_t (lo half in bits [0:15], hi in [16:31]).
    union
    {
        __half h[2];
        uint32_t u;
    } p0, p1;
    p0.h[0] = v0;
    p0.h[1] = v1;
    p1.h[0] = v2;
    p1.h[1] = v3;
    reg0    = p0.u;
    reg1    = p1.u;
}
