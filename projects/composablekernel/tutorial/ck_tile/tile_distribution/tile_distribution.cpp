// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Tutorial: CK Tile Distribution Encoding
 *
 * Demonstrates how tile_distribution_encoding maps threads to data elements.
 * Uses a simplified A-matrix DRAM tile distribution from the production GEMM pipeline.
 * Source: gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp
 *         MakeADramTileDistribution(), RowMajor branch
 *
 * Host initialises A with sequential values 0, 1, 2, ... (row-major).
 * A[m][k] = m * K + k, so the printed value directly gives the linear index.
 * GPU kernel loads A using the distribution, then prints per-thread buffer
 * contents so the reader can verify which elements each thread received.
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
// We have a matrix A of size M=32 rows × K=8 columns stored in DRAM (row-major).
// We want to load this entire tile into thread registers using 128 threads
// (2 warps on CDNA, 4 warps on RDNA).
//
// The desired thread-to-data mapping is:
//
//   - Split the 32 M-rows into 2 equal halves (16 rows each).
//   - Assign each half to a different group of threads.
//   - Each thread in a group holds all 16 rows × 8 K-columns = 128 elements.
//   - Multiple threads in the same group hold identical copies of the data.
//     (This redundancy is intentional — in production GEMM, the data is
//      written to LDS next, where each thread reads a different subset.)
//
//   Group 0 (even lanes: lane 0, 2, 4, ...):  M-rows  0..15, K-cols 0..7
//   Group 1 (odd  lanes: lane 1, 3, 5, ...):  M-rows 16..31, K-cols 0..7
//
// Visually:
//
//       A matrix (32×8)                Thread assignment
//       ────────────────               ─────────────────
//       row  0: [  0   1  ...   7]  ─┐
//       row  1: [  8   9  ...  15]   │  Group 0 (even lanes)
//       ...                           │  128 elements per thread
//       row 15: [120 121  ... 127]  ─┘
//       row 16: [128 129  ... 135]  ─┐
//       row 17: [136 137  ... 143]   │  Group 1 (odd lanes)
//       ...                           │  128 elements per thread
//       row 31: [248 249  ... 255]  ─┘
//
// Now try to write a tile_distribution_encoding that achieves this mapping
// before reading the solution below!
//
// ============================================================================
// THE SOLUTION: tile_distribution_encoding
// ============================================================================
// CK's tile_distribution_encoding has 6 template parameters:
//
//   tile_distribution_encoding<Rs, Hs, Ps_major, Ps_minor, Ys_major, Ys_minor>
//
// Every level of Hs is assigned to exactly one role:
//   P = "parallel" — thread ID selects which slice
//   Y = "yield"    — each thread owns the entire range in its buffer
//   R = "replicate"— identical data broadcast to multiple thread groups
//
// Here is how we build it for our goal:
//
//   Step 1 — Hierarchical dimensions (Hs): factor each axis of the tile.
//
//      Hs[0] = sequence<2, 16>  → M = 2 groups × 16 rows/group = 32
//      Hs[1] = sequence<8>      → K = 8 columns (kept flat)
//
//              Hs[0]              Hs[1]
//           ┌────┴────┐            │
//       level 0    level 1      level 0
//        = 2        = 16          = 8
//
//   Step 2 — Parallel dimensions (Ps): which levels are selected by thread ID?
//
//      Ps_major = tuple<sequence<1>>   →  P0 points to Hs[0]  (1-indexed)
//      Ps_minor = tuple<sequence<0>>   →  P0 points to level 0 of Hs[0] = the "2"
//
//      Mark it on the tree:
//
//              Hs[0]              Hs[1]
//           ┌────┴────┐            │
//          [P]       [?]          [?]
//        = 2        = 16          = 8
//
//   Step 3 — Yield dimensions (Ys): what does each thread own?
//
//      Ys_major = sequence<1, 2>   →  Y0 → Hs[0],  Y1 → Hs[1]  (1-indexed)
//      Ys_minor = sequence<1, 0>   →  Y0 → level 1, Y1 → level 0
//
//      Complete tree:
//
//              Hs[0]              Hs[1]
//           ┌────┴────┐            │
//          [P]       [Y0]         [Y1]
//        = 2        = 16          = 8
//       (group)   (rows/grp)    (columns)
//
//      Buffer size = Y0 × Y1 = 16 × 8 = 128 elements per thread.
//
//   Step 4 — Replicate (Rs): sequence<> — none.
//
// ============================================================================
// HOW CK ASSIGNS THREADS TO P-GROUPS
// ============================================================================
// The number of P-dimensions = number of elements in the Ps_major tuple.
//
//   tuple<sequence<1>>              → 1 element  → NDimP = 1
//   tuple<sequence<1>, sequence<2>> → 2 elements → NDimP = 2
//
// CK uses NDimP to decide what thread identity feeds P:
//   NDimP == 1:  P0 = lane_id                   (0..63 on CDNA)
//   NDimP == 2:  P0 = warp_id,  P1 = lane_id
//
// Our encoding has NDimP = 1, so P0 = lane_id.
// P0 selects from Hs[0][0] which has size 2.
// With 64 lanes mapping to 2 groups → lane_id % 2 picks the group:
//
//   lane  0 → 0 % 2 = 0 → group 0 (M=0..15)
//   lane  1 → 1 % 2 = 1 → group 1 (M=16..31)
//   lane  2 → 2 % 2 = 0 → group 0
//   ...
//   lane 63 → 63 % 2 = 1 → group 1
//
// 32 lanes per group, each holding identical 128-element buffers.
// Both warps have lanes 0..63, so warp 0 and warp 1 hold identical data.
// ============================================================================

static constexpr index_t kM = 32;
static constexpr index_t kK = 8;

struct TileDistKernel1
{
    static constexpr index_t kBlockSize = 128;

    CK_TILE_DEVICE void operator()(const int32_t* p_data) const
    {
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            p_data, make_tuple(kM, kK), make_tuple(kK, 1), number<1>{}, number<1>{});

        constexpr auto distribution = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<2, 16>, sequence<8>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{});

        auto window = make_tile_window(
            a_tensor, make_tuple(number<kM>{}, number<kK>{}), {0, 0}, distribution);

        const auto tile = load_tile(window);

        const auto& buf             = tile.get_thread_buffer();
        constexpr index_t warp_size = get_warp_size();

        constexpr index_t kBufSize       = 16 * 8; // 16 rows × 8 cols = 128
        constexpr index_t kRowsPerThread = 16;
        constexpr index_t kColsPerThread = 8;

        // Copy compile-time-indexed buffer into a plain array for runtime printing
        int32_t local_buf[kBufSize];
        static_for<0, kBufSize, 1>{}([&](auto i) { local_buf[i] = static_cast<int32_t>(buf[i]); });

        auto print_thread = [&](int tid) {
            if(static_cast<int>(threadIdx.x) == tid)
            {
                printf("Thread %3d  (warp %d, lane %2d)  buf_size=%d  (%d rows x %d cols)\n",
                       tid,
                       tid / static_cast<int>(warp_size),
                       tid % static_cast<int>(warp_size),
                       static_cast<int>(kBufSize),
                       static_cast<int>(kRowsPerThread),
                       static_cast<int>(kColsPerThread));

                for(index_t m = 0; m < kRowsPerThread; m++)
                {
                    int val0    = local_buf[m * kColsPerThread];
                    int decoded = val0 / kColsPerThread;
                    printf("  row %2d: (M=%2d) ", static_cast<int>(m), decoded);
                    for(index_t k = 0; k < kColsPerThread; k++)
                    {
                        printf("%5d ", local_buf[m * kColsPerThread + k]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        };

        if(blockIdx.x == 0)
        {
            if(threadIdx.x == 0)
            {
                printf("\n=== Tile Distribution: A-Matrix DRAM Load (RowMajor) ===\n");
                printf("Tile: %d x %d,  BlockSize: %d,  WarpSize: %d\n",
                       static_cast<int>(kM),
                       static_cast<int>(kK),
                       static_cast<int>(kBlockSize),
                       static_cast<int>(warp_size));
                printf("Each thread owns: 16 rows x 8 cols = 128 elements\n");
                printf("NDimP=1 => P0 = lane_id. P0 maps to Hs[0][0] (size 2)\n");
                printf("  => even lanes get M=0..15, odd lanes get M=16..31\n");
                printf("  => warp 0 and warp 1 hold identical data (same lane IDs)\n\n");
            }
            __syncthreads();

            // Warp 0, lane 0 (even → M=0..15)
            print_thread(0);
            __syncthreads();

            // Warp 0, lane 1 (odd → M=16..31)
            print_thread(1);
            __syncthreads();

            // Warp 1, lane 0 (even → M=0..15, same as T0)
            print_thread(static_cast<int>(warp_size));
            __syncthreads();

            // Warp 1, lane 1 (odd → M=16..31, same as T1)
            print_thread(static_cast<int>(warp_size) + 1);
            __syncthreads();
        }
    }
};

int main()
{
    printf("=== CK Tile Distribution Tutorial ===\n\n");

    HostTensor<int32_t> h_tensor({kM, kK});
    for(int i = 0; i < kM * kK; i++)
        h_tensor.mData[i] = i;

    printf("Host matrix A[%d x %d], row-major, values 0..%d\n", kM, kK, kM * kK - 1);
    printf("  A[0][0]=%d  A[0][7]=%d  A[1][0]=%d  A[31][7]=%d\n\n",
           h_tensor(0, 0),
           h_tensor(0, 7),
           h_tensor(1, 0),
           h_tensor(31, 7));

    DeviceMem d_data(h_tensor);

    launch_kernel(stream_config{},
                  make_kernel<1>(TileDistKernel1{},
                                 dim3(1),
                                 dim3(TileDistKernel1::kBlockSize),
                                 0,
                                 static_cast<const int32_t*>(d_data.GetDeviceBuffer())));
    hip_check_error(hipDeviceSynchronize());

    printf("Done.\n");
    return 0;
}
