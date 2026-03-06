// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>

namespace mint_gemm {

using namespace mint;
using namespace mint::tensor;
using namespace mint::tile;

/**
 * @brief MINT-based GEMM kernel
 *
 * Computes C[M, N] = A[M, K] * B[N, K] (note: B is transposed)
 * Uses MINT's tensor views and warp-level operations
 *
 * @tparam MPerBlock Rows of C per block
 * @tparam NPerBlock Cols of C per block
 * @tparam KPerBlock Inner dimension tile size
 * @tparam T Data type (fp16_t, bf16_t, etc.)
 * @tparam AccT Accumulator type (float)
 */
template <index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          typename T,
          typename AccT = float>
__global__ void MintGemmKernel(const T* __restrict__ p_a,
                                const T* __restrict__ p_b,
                                T* __restrict__ p_c,
                                index_t M,
                                index_t N,
                                index_t K,
                                index_t stride_a,
                                index_t stride_b,
                                index_t stride_c)
{
    // Block tile coordinates
    const index_t block_m = blockIdx.x * MPerBlock;
    const index_t block_n = blockIdx.y * NPerBlock;

    // Early exit if out of bounds
    if(block_m >= M || block_n >= N)
        return;

    // Create tensor views for global memory
    // A: [M, K] with stride_a
    const auto a_view = make_global_packed_tensor_view(
        p_a, nd_index<2>{M, K});

    // B: [N, K] with stride_b (transposed)
    const auto b_view = make_global_packed_tensor_view(
        p_b, nd_index<2>{N, K});

    // C: [M, N] with stride_c
    auto c_view = make_global_packed_tensor_view(
        p_c, nd_index<2>{M, N});

    // Shared memory for block tiles
    __shared__ T a_shared[MPerBlock][KPerBlock];
    __shared__ T b_shared[NPerBlock][KPerBlock];

    // Accumulator in registers
    AccT acc[MPerBlock / blockDim.x][NPerBlock / blockDim.y];

    // Initialize accumulator
    #pragma unroll
    for(index_t i = 0; i < MPerBlock / blockDim.x; ++i)
    {
        #pragma unroll
        for(index_t j = 0; j < NPerBlock / blockDim.y; ++j)
        {
            acc[i][j] = static_cast<AccT>(0);
        }
    }

    // Thread indices
    const index_t tid_x = threadIdx.x;
    const index_t tid_y = threadIdx.y;
    const index_t tid = tid_y * blockDim.x + tid_x;

    // Main K loop
    for(index_t k_block = 0; k_block < K; k_block += KPerBlock)
    {
        // Cooperatively load A tile [MPerBlock, KPerBlock] into shared memory
        for(index_t idx = tid; idx < MPerBlock * KPerBlock; idx += blockDim.x * blockDim.y)
        {
            const index_t i = idx / KPerBlock;
            const index_t j = idx % KPerBlock;
            const index_t a_row = block_m + i;
            const index_t a_col = k_block + j;

            if(a_row < M && a_col < K)
            {
                a_shared[i][j] = a_view.element(nd_index<2>{a_row, a_col});
            }
            else
            {
                a_shared[i][j] = static_cast<T>(0);
            }
        }

        // Cooperatively load B tile [NPerBlock, KPerBlock] into shared memory
        for(index_t idx = tid; idx < NPerBlock * KPerBlock; idx += blockDim.x * blockDim.y)
        {
            const index_t i = idx / KPerBlock;
            const index_t j = idx % KPerBlock;
            const index_t b_row = block_n + i;
            const index_t b_col = k_block + j;

            if(b_row < N && b_col < K)
            {
                b_shared[i][j] = b_view.element(nd_index<2>{b_row, b_col});
            }
            else
            {
                b_shared[i][j] = static_cast<T>(0);
            }
        }

        __syncthreads();

        // Compute: each thread computes a subset of the output tile
        #pragma unroll
        for(index_t m = 0; m < MPerBlock / blockDim.x; ++m)
        {
            #pragma unroll
            for(index_t n = 0; n < NPerBlock / blockDim.y; ++n)
            {
                const index_t i = m * blockDim.x + tid_x;
                const index_t j = n * blockDim.y + tid_y;

                if(i < MPerBlock && j < NPerBlock)
                {
                    AccT sum = static_cast<AccT>(0);
                    #pragma unroll
                    for(index_t k = 0; k < KPerBlock; ++k)
                    {
                        sum += static_cast<AccT>(a_shared[i][k]) *
                               static_cast<AccT>(b_shared[j][k]);
                    }
                    acc[m][n] += sum;
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for(index_t m = 0; m < MPerBlock / blockDim.x; ++m)
    {
        #pragma unroll
        for(index_t n = 0; n < NPerBlock / blockDim.y; ++n)
        {
            const index_t i = m * blockDim.x + tid_x;
            const index_t j = n * blockDim.y + tid_y;
            const index_t c_row = block_m + i;
            const index_t c_col = block_n + j;

            if(c_row < M && c_col < N && i < MPerBlock && j < NPerBlock)
            {
                c_view.element(nd_index<2>{c_row, c_col}) = static_cast<T>(acc[m][n]);
            }
        }
    }
}

} // namespace mint_gemm
