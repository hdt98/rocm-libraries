// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Tutorial: CK Tile Distribution Encoding — Scenario 4
 *
 * Demonstrates the R (replicate) dimension for inter-warp data replication.
 *
 * Matrix A: 128 rows × 8 columns, row-major.
 * BlockSize: 256 threads (4 warps of 64 on CDNA).
 * Each thread holds exactly 8 elements (one full row).
 *
 * The desired thread-to-data mapping is:
 *
 *   Warp 0 and Warp 1 load the SAME top half (rows 0-63):
 *     T0   (warp 0, lane  0) → row   0     T64  (warp 1, lane  0) → row   0
 *     T1   (warp 0, lane  1) → row   1     T65  (warp 1, lane  1) → row   1
 *     ...                                   ...
 *     T63  (warp 0, lane 63) → row  63     T127 (warp 1, lane 63) → row  63
 *
 *   Warp 2 and Warp 3 load the SAME bottom half (rows 64-127):
 *     T128 (warp 2, lane  0) → row  64     T192 (warp 3, lane  0) → row  64
 *     T129 (warp 2, lane  1) → row  65     T193 (warp 3, lane  1) → row  65
 *     ...                                   ...
 *     T191 (warp 2, lane 63) → row 127     T255 (warp 3, lane 63) → row 127
 *
 * Key insight: Ps_major can reference rh_major=0 to link the R dimension
 * to a thread identity (warp_id). The warp_id is then decomposed across
 * BOTH the H-partition and the R-replicate dimension via the merge transform.
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
// Matrix A: M=128 rows × K=8 columns, 256 threads (4 warps).
// We want 2 unique data partitions replicated across 2 warp pairs:
//
//   Warp 0 and Warp 1 both load rows 0-63 (identical data):
//       row   0: [  0   1  ...   7]  ← warp 0 lane 0  AND  warp 1 lane 0
//       row   1: [  8   9  ...  15]  ← warp 0 lane 1  AND  warp 1 lane 1
//       ...
//       row  63: [504 505  ... 511]  ← warp 0 lane 63 AND  warp 1 lane 63
//
//   Warp 2 and Warp 3 both load rows 64-127 (identical data):
//       row  64: [512 513  ... 519]  ← warp 2 lane 0  AND  warp 3 lane 0
//       ...
//       row 127: [1016 1017 .. 1023] ← warp 2 lane 63 AND  warp 3 lane 63
//
// The key new concept is the R (replicate) dimension. To link R to warp_id,
// Ps_major uses rh_major=0 (which references Rs, not Hs). The warp_id is
// then decomposed across BOTH the H-partition and the R-replicate dimension.
//
// ============================================================================
// THE SOLUTION: tile_distribution_encoding with R
// ============================================================================
//
//   Hs[0] = sequence<2, 64>   → M = 2 × 64 = 128
//   Hs[1] = sequence<8>       → K = 8
//   Rs    = sequence<2>        → 2 replicate groups
//
//   P0 = warp_id maps to BOTH Hs[0][0]=2 AND Rs[0]=2:
//
//     Ps_major = tuple<sequence<1, 0>, sequence<1>>
//                               ^  ^
//                               |  └── rh_major=0 → Rs dimension (size 2)
//                               └──── rh_major=1 → Hs[0] (1-indexed)
//
//     Ps_minor = tuple<sequence<0, 0>, sequence<1>>
//                               ^  ^
//                               |  └── Rs[0] (level 0)
//                               └──── Hs[0][0] (level 0, size 2)
//
//   P0 total size = Hs[0][0] × Rs[0] = 2 × 2 = 4 → matches 4 warps.
//
//   The merge transform decomposes warp_id across [Hs[0][0]=2, Rs[0]=2]:
//
//     M_half = warp_id / 2   (outermost, first in sequence)
//     R_idx  = warp_id % 2   (innermost, last in sequence)
//
//     warp 0: M_half=0, R=0  →  rows 0-63
//     warp 1: M_half=0, R=1  →  rows 0-63   (replicate of warp 0)
//     warp 2: M_half=1, R=0  →  rows 64-127
//     warp 3: M_half=1, R=1  →  rows 64-127  (replicate of warp 2)
//
//   P1 = lane_id maps to Hs[0][1]=64 (one lane per row within the half).
//
//   Y0 = Hs[1][0] = 8 → each thread owns all 8 K-columns.
//   Buffer size = 8 elements per thread.
//
// ============================================================================
// GOLDEN RULE: Ordering in Ps_major controls which warps are replicates.
// ============================================================================
// The merge transform decomposes warp_id like a row-major flat index:
//   - FIRST entry in the sequence = outermost (divided first, changes slowly)
//   - LAST  entry in the sequence = innermost (modulo, changes every step)
//
// sequence<1, 0>  →  [H=2, R=2]  →  H outer, R inner
//   warp 0: H=0, R=0   (rows 0-63)
//   warp 1: H=0, R=1   (rows 0-63)     ← consecutive warps are replicates
//   warp 2: H=1, R=0   (rows 64-127)
//   warp 3: H=1, R=1   (rows 64-127)
//
// sequence<0, 1>  →  [R=2, H=2]  →  R outer, H inner
//   warp 0: R=0, H=0   (rows 0-63)
//   warp 1: R=0, H=1   (rows 64-127)   ← consecutive warps are DIFFERENT
//   warp 2: R=1, H=0   (rows 0-63)
//   warp 3: R=1, H=1   (rows 64-127)   ← alternating warps are replicates
//
// Rule: put the dimension you want to cycle fastest LAST in the sequence.
// ============================================================================

static constexpr index_t kM = 128;
static constexpr index_t kK = 8;

struct TileDistKernel4
{
    static constexpr index_t kBlockSize = 256;

    CK_TILE_DEVICE void operator()(const int32_t* p_data) const
    {
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            p_data, make_tuple(kM, kK), make_tuple(kK, 1), number<1>{}, number<1>{});

        constexpr auto distribution = make_static_tile_distribution(
            tile_distribution_encoding<sequence<2>,
                                       tuple<sequence<2, 64>, sequence<8>>,
                                       tuple<sequence<1, 0>, sequence<1>>,
                                       tuple<sequence<0, 0>, sequence<1>>,
                                       sequence<2>,
                                       sequence<0>>{});

        auto window = make_tile_window(
            a_tensor, make_tuple(number<kM>{}, number<kK>{}), {0, 0}, distribution);

        const auto tile = load_tile(window);

        const auto& buf             = tile.get_thread_buffer();
        constexpr index_t warp_size = get_warp_size();
        constexpr index_t kBufSize  = kK;

        int32_t local_buf[kBufSize];
        static_for<0, kBufSize, 1>{}([&](auto i) { local_buf[i] = static_cast<int32_t>(buf[i]); });

        auto print_thread = [&](int tid) {
            if(static_cast<int>(threadIdx.x) == tid)
            {
                int lane = tid % static_cast<int>(warp_size);
                int warp = tid / static_cast<int>(warp_size);

                printf("Thread %3d  (warp %d, lane %2d)  |", tid, warp, lane);

                for(index_t k = 0; k < kBufSize; k++)
                    printf(" %4d", local_buf[k]);
                printf("\n");
            }
        };

        if(blockIdx.x == 0)
        {
            if(threadIdx.x == 0)
            {
                printf(
                    "\n=== Tile Distribution Scenario 4: R Dimension (Inter-Warp Replicate) ===\n");
                printf("Tile: %d x %d,  BlockSize: %d,  WarpSize: %d,  Warps: 4\n",
                       static_cast<int>(kM),
                       static_cast<int>(kK),
                       static_cast<int>(kBlockSize),
                       static_cast<int>(warp_size));
                printf("Each thread owns: 1 row x %d cols = %d elements\n",
                       static_cast<int>(kK),
                       static_cast<int>(kBufSize));
                printf("R=2: warp0=warp1 (rows 0-63), warp2=warp3 (rows 64-127)\n\n");
            }
            __syncthreads();

            // Warp 0 (M_half=0, R=0)
            if(threadIdx.x == 0)
                printf("--- Warp 0 (M_half=0, R=0) ---\n");
            __syncthreads();
            print_thread(0);
            __syncthreads();
            print_thread(63);
            __syncthreads();

            // Warp 1 (M_half=0, R=1) — should match Warp 0
            if(threadIdx.x == 0)
                printf("\n--- Warp 1 (M_half=0, R=1) — should match Warp 0 ---\n");
            __syncthreads();
            print_thread(64);
            __syncthreads();
            print_thread(127);
            __syncthreads();

            // Warp 2 (M_half=1, R=0)
            if(threadIdx.x == 0)
                printf("\n--- Warp 2 (M_half=1, R=0) ---\n");
            __syncthreads();
            print_thread(128);
            __syncthreads();
            print_thread(191);
            __syncthreads();

            // Warp 3 (M_half=1, R=1) — should match Warp 2
            if(threadIdx.x == 0)
                printf("\n--- Warp 3 (M_half=1, R=1) — should match Warp 2 ---\n");
            __syncthreads();
            print_thread(192);
            __syncthreads();
            print_thread(255);
            __syncthreads();
        }
    }
};

int main()
{
    printf("=== CK Tile Distribution Tutorial — Scenario 4 (R Dimension) ===\n\n");

    HostTensor<int32_t> h_tensor({kM, kK});
    for(int i = 0; i < kM * kK; i++)
        h_tensor.mData[i] = i;

    printf("Host matrix A[%d x %d], row-major, values 0..%d\n", kM, kK, kM * kK - 1);
    printf("  A[0][0..7]   = {0, 1, 2, 3, 4, 5, 6, 7}\n");
    printf("  A[63][0..7]  = {504, 505, 506, 507, 508, 509, 510, 511}\n");
    printf("  A[64][0..7]  = {512, 513, 514, 515, 516, 517, 518, 519}\n");
    printf("  A[127][0..7] = {1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023}\n\n");

    DeviceMem d_data(h_tensor);

    launch_kernel(stream_config{},
                  make_kernel<1>(TileDistKernel4{},
                                 dim3(1),
                                 dim3(TileDistKernel4::kBlockSize),
                                 0,
                                 static_cast<const int32_t*>(d_data.GetDeviceBuffer())));
    hip_check_error(hipDeviceSynchronize());

    printf("Done.\n");
    return 0;
}
