// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Tutorial: CK Tile Distribution Encoding — Scenario 2
 *
 * Production GEMM: A-Matrix DRAM Load (RowMajor)
 *
 * Source: gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp
 *         MakeADramTileDistribution(), RowMajor branch
 *
 * Demonstrates how lane_id splits across BOTH M and K dimensions to ensure
 * coalesced 128-bit reads along the contiguous K-dimension in row-major data.
 *
 * No compute is performed — this is purely about data movement.
 *
 * Note: Comments and values assume CDNA (warp_size=64). On RDNA (warp_size=32),
 * the thread-to-data mapping will differ.
 */

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include <cstdio>

using namespace ck_tile;

// ============================================================================
// THE GOAL
// ============================================================================
// We have a matrix A of size M=64 rows × K=32 columns stored in DRAM (row-major).
// We want to load this entire tile into registers using 128 threads (2 warps).
// Each thread should hold exactly 16 elements.
//
// For coalesced memory access, we want consecutive lanes to read consecutive
// K-elements (since K is the contiguous dimension in row-major). Each lane
// reads 8 contiguous K-values (128 bits for fp16), so 4 consecutive lanes
// cover all 32 K-columns of one row:
//
//      lane 0: K=0..7    lane 1: K=8..15    lane 2: K=16..23    lane 3: K=24..31
//      └──────────────── one row of 32 K-columns ──────────────────────────────┘
//
// The next group of 4 lanes handles the next M-row, and so on:
//
//      lanes  0- 3: row 0    (4 lanes × 8 K-values = 32 columns)
//      lanes  4- 7: row 1
//      lanes  8-11: row 2
//      ...
//      lanes 60-63: row 15
//
// So each warp (64 lanes) covers 16 M-rows × 32 K-columns.
// With 2 warps: warp 0 → rows 0-15, warp 1 → rows 16-31.
// That's only 32 rows, but we need 64. Each thread iterates twice (stride=32)
// to cover the remaining rows:
//
//   Iteration 0: rows 0-31  (warp 0: 0-15,  warp 1: 16-31)
//   Iteration 1: rows 32-63 (warp 0: 32-47, warp 1: 48-63)
//
// Per-thread buffer = 2 iterations × 8 K-values = 16 elements.
//
// Visually for warp 0:
//
//       A matrix (64×32)           lane_id decomposition
//       ────────────────           ──────────────────────
//       row  0: [ K=0..7 | 8..15 | 16..23 | 24..31 ]
//                  L0       L1      L2       L3       ← iter 0
//       row  1: [ K=0..7 | 8..15 | 16..23 | 24..31 ]
//                  L4       L5      L6       L7
//       ...
//       row 15: [ K=0..7 | 8..15 | 16..23 | 24..31 ]
//                  L60      L61     L62      L63
//       ────── stride of 32 rows ──────
//       row 32: [ K=0..7 | 8..15 | 16..23 | 24..31 ]
//                  L0       L1      L2       L3       ← iter 1
//       ...
//       row 47: same lanes as iter 0
//
// ============================================================================
// THE SOLUTION: tile_distribution_encoding
// ============================================================================
//
//   Step 1 — Hierarchical dimensions (Hs): factor each axis.
//
//      Hs[0] = sequence<2, 2, 16>  → M = 2 × 2 × 16 = 64
//      Hs[1] = sequence<4, 8>      → K = 4 × 8 = 32
//
//              Hs[0]                    Hs[1]
//         ┌─────┼─────┐              ┌───┴───┐
//      level 0  level 1  level 2   level 0  level 1
//        = 2     = 2      = 16       = 4      = 8
//
//   Step 2 — Parallel dimensions (Ps): NDimP=2 (P0=warp_id, P1=lane_id).
//
//      P0 = warp_id  → Hs[0][1] = 2  (which warp → which M-group)
//      P1 = lane_id  → Hs[0][2]=16 AND Hs[1][0]=4  (merged, total=64)
//
//      The merge transform decomposes lane_id:
//        row_in_group = lane_id / 4   (0..15, outer)
//        k_chunk      = lane_id % 4   (0..3,  inner → coalesced!)
//
//      Ps_major = tuple<sequence<1>, sequence<1, 2>>
//      Ps_minor = tuple<sequence<1>, sequence<2, 0>>
//
//   Step 3 — Yield dimensions (Ys): what each thread owns.
//
//      Y0 = Hs[0][0] = 2  (M-iterations)
//      Y1 = Hs[1][1] = 8  (vector load width)
//
//      Ys_major = sequence<1, 2>
//      Ys_minor = sequence<0, 1>
//
//   Step 4 — Replicate: Rs = sequence<> (none).
//
//   Complete tree:
//
//              Hs[0]                    Hs[1]
//         ┌─────┼─────┐              ┌───┴───┐
//       [Y0]   [P0]  [P1]          [P1]     [Y1]
//        = 2    = 2   = 16          = 4      = 8
//      (iter) (warp) (row)        (K-chunk) (vec load)
//
//   Buffer size = Y0 × Y1 = 2 × 8 = 16 elements per thread.
//
// ============================================================================

static constexpr index_t kM = 64;
static constexpr index_t kK = 32;

struct TileDistKernel2
{
    static constexpr index_t kBlockSize = 128;

    CK_TILE_DEVICE void operator()(const int32_t* p_data) const
    {
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            p_data, make_tuple(kM, kK), make_tuple(kK, 1), number<1>{}, number<1>{});

        constexpr auto distribution = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<2, 2, 16>, sequence<4, 8>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});

        auto window = make_tile_window(
            a_tensor, make_tuple(number<kM>{}, number<kK>{}), {0, 0}, distribution);

        const auto tile = load_tile(window);

        const auto& buf             = tile.get_thread_buffer();
        constexpr index_t warp_size = get_warp_size();
        constexpr index_t kBufSize  = 16; // 2 iterations × 8 K-values

        int32_t local_buf[kBufSize];
        static_for<0, kBufSize, 1>{}([&](auto i) { local_buf[i] = static_cast<int32_t>(buf[i]); });

        auto print_thread = [&](int tid) {
            if(static_cast<int>(threadIdx.x) == tid)
            {
                int lane       = tid % static_cast<int>(warp_size);
                int warp       = tid / static_cast<int>(warp_size);
                int row_in_grp = lane / 4;
                int k_chunk    = lane % 4;

                printf("Thread %3d  (warp %d, lane %2d)  row_in_grp=%2d  k_chunk=%d\n",
                       tid,
                       warp,
                       lane,
                       row_in_grp,
                       k_chunk);

                for(int iter = 0; iter < 2; iter++)
                {
                    int row = iter * 32 + warp * 16 + row_in_grp;
                    int col = k_chunk * 8;
                    printf("  iter %d: A[%2d][%2d..%2d] =", iter, row, col, col + 7);
                    for(int k = 0; k < 8; k++)
                        printf(" %4d", local_buf[iter * 8 + k]);
                    printf("\n");
                }
            }
        };

        if(blockIdx.x == 0)
        {
            if(threadIdx.x == 0)
            {
                printf("\n=== Scenario 2: GEMM A-Matrix RowMajor DRAM Load ===\n");
                printf("Source: MakeADramTileDistribution (RowMajor branch)\n");
                printf("Tile: %dx%d  BlockSize: %d  WarpSize: %d  Warps: 2\n",
                       kM,
                       kK,
                       kBlockSize,
                       static_cast<int>(warp_size));
                printf("Each thread: 2 iterations x 8 K-values = 16 elements\n\n");
                printf("Coalescing: lanes 0-3 read K=0..31 of the same row\n");
                printf("            (4 x 8 = 32 K-values = one full row)\n\n");
            }
            __syncthreads();

            // Lane 0: row_in_grp=0, k_chunk=0 → rows {0, 32}, K=0..7
            print_thread(0);
            __syncthreads();
            // Lane 1: row_in_grp=0, k_chunk=1 → rows {0, 32}, K=8..15 (coalesced!)
            print_thread(1);
            __syncthreads();
            // Lane 4: row_in_grp=1, k_chunk=0 → rows {1, 33}, K=0..7 (next row)
            print_thread(4);
            __syncthreads();
            // Lane 63: row_in_grp=15, k_chunk=3 → rows {15, 47}, K=24..31
            print_thread(63);
            __syncthreads();

            if(threadIdx.x == 0)
                printf("\n--- Warp 1 ---\n");
            __syncthreads();
            // Warp 1, Lane 0: rows {16, 48}, K=0..7
            print_thread(64);
            __syncthreads();
            // Warp 1, Lane 63: rows {31, 63}, K=24..31
            print_thread(127);
            __syncthreads();
        }
    }
};

int main()
{
    printf("=== CK Tile Distribution Tutorial — Scenario 2 ===\n");
    printf("=== Production GEMM A-Matrix RowMajor DRAM Load ===\n\n");

    HostTensor<int32_t> h_tensor({kM, kK});
    for(int i = 0; i < kM * kK; i++)
        h_tensor.mData[i] = i;

    printf("Host matrix A[%d x %d], row-major, A[m][k] = m*%d + k\n\n", kM, kK, kK);

    DeviceMem d_data(h_tensor);

    launch_kernel(stream_config{},
                  make_kernel<1>(TileDistKernel2{},
                                 dim3(1),
                                 dim3(TileDistKernel2::kBlockSize),
                                 0,
                                 static_cast<const int32_t*>(d_data.GetDeviceBuffer())));
    hip_check_error(hipDeviceSynchronize());

    printf("Done.\n");
    return 0;
}
