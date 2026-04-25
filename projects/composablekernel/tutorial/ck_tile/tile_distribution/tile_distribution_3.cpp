// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Tutorial: CK Tile Distribution Encoding — Scenario 3
 *
 * Production GEMM: A-Matrix DRAM Load (RowMajor, small K)
 *
 * Source: gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp
 *         MakeADramTileDistribution(), RowMajor branch
 *
 * Same function as Scenario 2, but with K=8 (small enough to fit in one
 * vector load). This eliminates K-splitting — lane_id maps entirely to M.
 * Serves as a stepping stone between the simple Scenario 1 and the
 * coalesced Scenario 2.
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
// Matrix A: M=128 rows × K=8 columns, stored in DRAM (row-major).
// Load the entire tile using 128 threads (2 warps).
// Each thread loads exactly 8 elements (one complete row).
//
// Since K=8 fits in a single 128-bit vector load (8 × fp16 = 16 bytes),
// there is no need to split K across lanes. Each lane loads ALL K-columns
// of its assigned row. lane_id selects only which M-row to load.
//
//       A matrix (128×8)              Thread assignment
//       ────────────────              ─────────────────
//       row   0: [  0   1  ...   7]   ← warp 0, lane  0
//       row   1: [  8   9  ...  15]   ← warp 0, lane  1
//       row   2: [ 16  17  ...  23]   ← warp 0, lane  2
//       ...
//       row  63: [504 505  ... 511]   ← warp 0, lane 63
//       row  64: [512 513  ... 519]   ← warp 1, lane  0
//       row  65: [520 521  ... 527]   ← warp 1, lane  1
//       ...
//       row 127: [1016 1017 .. 1023]  ← warp 1, lane 63
//
// This is the simplest NDimP=2 distribution:
//   P0 = warp_id → which M-half  (2 warps)
//   P1 = lane_id → which M-row within that half  (64 lanes)
//   Y  = all 8 K-columns  (vector load, each thread owns the full row)
//
// Compare with Scenario 2 (K=32):
//   When K grows beyond the vector load width, K must also be split across
//   lanes for coalescing, making P1 map to both M and K. Here K is small
//   enough that P1 maps only to M.
//
// ============================================================================
// THE SOLUTION: tile_distribution_encoding
// ============================================================================
//
//   Production code derives (fp16, BlockSize=128, MPerBlock=128, KPerBlock=8):
//     K1 = 16/sizeof(fp16) = 8   → vector load = 8 K-values = entire K
//     K0 = KPerBlock/K1 = 1      → only 1 K-chunk (no K-splitting!)
//     M2 = warp_size/K0 = 64     → all 64 lanes go to M
//     M1 = BlockSize/warp_size = 2
//     M0 = MPerBlock/(M2*M1) = 1
//
//   We simplify by dropping trivial levels (M0=1, K0=1, Rs=1):
//
//   Step 1 — Hierarchical dimensions (Hs):
//
//      Hs[0] = sequence<2, 64>  → M = 2 × 64 = 128
//      Hs[1] = sequence<8>      → K = 8
//
//              Hs[0]              Hs[1]
//           ┌────┴────┐            │
//         [P0]      [P1]         [Y0]
//          = 2      = 64          = 8
//        (warp)    (lane)      (vec load)
//
//   Step 2 — Parallel (Ps): NDimP=2.
//
//      P0 = warp_id → Hs[0][0] = 2
//      P1 = lane_id → Hs[0][1] = 64  (maps only to M, no K-splitting)
//
//      Ps_major = tuple<sequence<1>, sequence<1>>
//      Ps_minor = tuple<sequence<0>, sequence<1>>
//
//   Step 3 — Yield (Ys):
//
//      Y0 = Hs[1][0] = 8  (K-values, the vector load)
//
//      Ys_major = sequence<2>
//      Ys_minor = sequence<0>
//
//   Step 4 — Replicate: Rs = sequence<> (none).
//
//   Buffer size = Y0 = 8 elements per thread.
//
// ============================================================================

static constexpr index_t kM = 128;
static constexpr index_t kK = 8;

struct TileDistKernel3
{
    static constexpr index_t kBlockSize = 128;

    CK_TILE_DEVICE void operator()(const int32_t* p_data) const
    {
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            p_data, make_tuple(kM, kK), make_tuple(kK, 1), number<1>{}, number<1>{});

        constexpr auto distribution = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<2, 64>, sequence<8>>,
                                       tuple<sequence<1>, sequence<1>>,
                                       tuple<sequence<0>, sequence<1>>,
                                       sequence<2>,
                                       sequence<0>>{});

        auto window = make_tile_window(
            a_tensor, make_tuple(number<kM>{}, number<kK>{}), {0, 0}, distribution);

        const auto tile = load_tile(window);

        const auto& buf             = tile.get_thread_buffer();
        constexpr index_t warp_size = get_warp_size();
        constexpr index_t kBufSize  = 8;

        int32_t local_buf[kBufSize];
        static_for<0, kBufSize, 1>{}([&](auto i) { local_buf[i] = static_cast<int32_t>(buf[i]); });

        auto print_thread = [&](int tid) {
            if(static_cast<int>(threadIdx.x) == tid)
            {
                int lane = tid % static_cast<int>(warp_size);
                int warp = tid / static_cast<int>(warp_size);
                int row  = warp * 64 + lane;

                printf("Thread %3d  (warp %d, lane %2d)  row=%3d  |", tid, warp, lane, row);
                for(int k = 0; k < kBufSize; k++)
                    printf(" %4d", local_buf[k]);
                printf("\n");
            }
        };

        if(blockIdx.x == 0)
        {
            if(threadIdx.x == 0)
            {
                printf("\n=== Scenario 3: GEMM A-Matrix RowMajor (Small K) ===\n");
                printf("Source: MakeADramTileDistribution (RowMajor, K=%d)\n",
                       static_cast<int>(kK));
                printf("Tile: %dx%d  BlockSize: %d  WarpSize: %d  Warps: 2\n",
                       kM,
                       kK,
                       kBlockSize,
                       static_cast<int>(warp_size));
                printf("K=%d fits in one vector load → no K-splitting\n", kK);
                printf("P1=lane_id maps entirely to M (cf. Scenario 2: M+K)\n");
                printf("Each thread: 1 row x %d cols = %d elements\n\n", kK, kBufSize);
            }
            __syncthreads();

            if(threadIdx.x == 0)
                printf("--- Warp 0 (rows 0-63) ---\n");
            __syncthreads();
            print_thread(0); // lane 0  → row 0
            __syncthreads();
            print_thread(1); // lane 1  → row 1
            __syncthreads();
            print_thread(63); // lane 63 → row 63
            __syncthreads();

            if(threadIdx.x == 0)
                printf("\n--- Warp 1 (rows 64-127) ---\n");
            __syncthreads();
            print_thread(64); // lane 0  → row 64
            __syncthreads();
            print_thread(127); // lane 63 → row 127
            __syncthreads();
        }
    }
};

int main()
{
    printf("=== CK Tile Distribution Tutorial — Scenario 3 ===\n");
    printf("=== Production GEMM A-Matrix RowMajor (Small K) ===\n\n");

    HostTensor<int32_t> h_tensor({kM, kK});
    for(int i = 0; i < kM * kK; i++)
        h_tensor.mData[i] = i;

    printf("Host matrix A[%d x %d], row-major, A[m][k] = m*%d + k\n\n", kM, kK, kK);

    DeviceMem d_data(h_tensor);

    launch_kernel(stream_config{},
                  make_kernel<1>(TileDistKernel3{},
                                 dim3(1),
                                 dim3(TileDistKernel3::kBlockSize),
                                 0,
                                 static_cast<const int32_t*>(d_data.GetDeviceBuffer())));
    hip_check_error(hipDeviceSynchronize());

    printf("Done.\n");
    return 0;
}
