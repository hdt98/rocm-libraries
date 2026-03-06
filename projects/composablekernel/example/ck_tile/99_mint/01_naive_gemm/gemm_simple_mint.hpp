// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>

namespace mint_gemm_simple {

using namespace mint;
using namespace mint::tensor;

// Simple MINT GEMM kernel (working version)
// This demonstrates a basic blocked GEMM without the full hierarchical structure
// For hierarchical implementation attempt, see the files in warp_level/, block_level/, host_level/
// (Note: Full hierarchical version has compilation issues with current MINT API limitations)

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          index_t kMPerBlock,
          index_t kNPerBlock,
          index_t kKPerBlock,
          index_t kBlockSize>
__device__ void gemm_simple_impl(const ADataType* __restrict__ p_a,
                                const BDataType* __restrict__ p_b,
                                CDataType* __restrict__ p_c,
                                index_t M,
                                index_t N,
                                index_t K)
{
    // Calculate block indices
    const index_t num_n_blocks = (N + kNPerBlock - 1) / kNPerBlock;
    const index_t m_block_idx = blockIdx.x / num_n_blocks;
    const index_t n_block_idx = blockIdx.x % num_n_blocks;

    const index_t m_block_start = m_block_idx * kMPerBlock;
    const index_t n_block_start = n_block_idx * kNPerBlock;

    if(m_block_start >= M || n_block_start >= N)
        return;

    // Shared memory for A and B tiles
    __shared__ ADataType a_smem[kMPerBlock * kKPerBlock];
    __shared__ BDataType b_smem[kNPerBlock * kKPerBlock];

    // Each thread computes kMPerThread x kNPerThread elements
    constexpr index_t kMPerThread = kMPerBlock / (kBlockSize / (kNPerBlock / 4));
    constexpr index_t kNPerThread = 4;

    // Thread mapping: distribute threads across M and N dimensions
    const index_t tid = threadIdx.x;
    const index_t tid_n = tid % (kNPerBlock / kNPerThread);
    const index_t tid_m = tid / (kNPerBlock / kNPerThread);

    // Thread-local accumulator
    AccDataType c_acc[kMPerThread][kNPerThread];
    for(index_t i = 0; i < kMPerThread; ++i)
        for(index_t j = 0; j < kNPerThread; ++j)
            c_acc[i][j] = static_cast<AccDataType>(0);

    // Main K loop: iterate over K dimension in blocks
    for(index_t k_block = 0; k_block < K; k_block += kKPerBlock)
    {
        // Load A tile [kMPerBlock, kKPerBlock] from global to shared memory
        for(index_t idx = tid; idx < kMPerBlock * kKPerBlock; idx += blockDim.x)
        {
            const index_t m_local = idx / kKPerBlock;
            const index_t k_local = idx % kKPerBlock;
            const index_t m_global = m_block_start + m_local;
            const index_t k_global = k_block + k_local;

            a_smem[idx] = (m_global < M && k_global < K)
                            ? p_a[m_global * K + k_global]
                            : static_cast<ADataType>(0);
        }

        // Load B tile [kNPerBlock, kKPerBlock] from global to shared memory
        for(index_t idx = tid; idx < kNPerBlock * kKPerBlock; idx += blockDim.x)
        {
            const index_t n_local = idx / kKPerBlock;
            const index_t k_local = idx % kKPerBlock;
            const index_t n_global = n_block_start + n_local;
            const index_t k_global = k_block + k_local;

            b_smem[idx] = (n_global < N && k_global < K)
                            ? p_b[n_global * K + k_global]
                            : static_cast<BDataType>(0);
        }

        __syncthreads();

        // Compute: C += A * B for this thread's tile
        for(index_t k_local = 0; k_local < kKPerBlock; ++k_local)
        {
            for(index_t i = 0; i < kMPerThread; ++i)
            {
                const index_t m_local = tid_m * kMPerThread + i;
                if(m_local < kMPerBlock)
                {
                    const AccDataType a_val = static_cast<AccDataType>(a_smem[m_local * kKPerBlock + k_local]);

                    for(index_t j = 0; j < kNPerThread; ++j)
                    {
                        const index_t n_local = tid_n * kNPerThread + j;
                        if(n_local < kNPerBlock)
                        {
                            const AccDataType b_val = static_cast<AccDataType>(b_smem[n_local * kKPerBlock + k_local]);
                            c_acc[i][j] += a_val * b_val;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // Store results to global memory
    for(index_t i = 0; i < kMPerThread; ++i)
    {
        const index_t m_local = tid_m * kMPerThread + i;
        const index_t m_global = m_block_start + m_local;

        if(m_global < M)
        {
            for(index_t j = 0; j < kNPerThread; ++j)
            {
                const index_t n_local = tid_n * kNPerThread + j;
                const index_t n_global = n_block_start + n_local;

                if(n_global < N)
                {
                    p_c[m_global * N + n_global] = static_cast<CDataType>(c_acc[i][j]);
                }
            }
        }
    }
}

} // namespace mint_gemm_simple
