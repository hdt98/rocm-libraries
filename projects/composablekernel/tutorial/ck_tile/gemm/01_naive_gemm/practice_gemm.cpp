// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <iostream>
#include "ck_tile/host.hpp"
#include "practice_gemm.hpp"
#include "../reference_gemm.hpp"

/*
 * Naive GEMM implementation (no optimizations)
 * A [M, K]  -- M rows, K cols, row-major
 * B [N, K]  -- N rows, K cols, row-major (B is stored TRANSPOSED: N as leading dim, K contiguous)
 * C [M, N]  -- M rows, N cols, row-major output
 *
 * Conceptually computes C = A * B^T, but since B is already stored in [N,K] layout
 * (B-transposed), the kernel reads B rows directly as K-vectors without any transposition.
 */

// CElementFunction: identity epilogue applied to each C element after the GEMM.
// Using a struct (rather than a lambda or std::identity) makes the type explicit and
// allows it to be passed as a template parameter and included in the kernel argument list.
// In production this could be replaced with a bias-add, ReLU, GELU, etc.
struct CElementFunction
{
    template <typename X>
    CK_TILE_HOST_DEVICE auto operator()(const X& x) const
    {
        return x;
    }
};

int main(int argc, char* argv[])
{
    // Input matrices A and B use FP16 (half_t) for memory efficiency and MFMA throughput.
    // Accumulation uses FP32 (float) to maintain numerical precision during the dot products.
    // Output matrix C uses FP16 matching the input types; it is cast from the FP32 accumulator
    // in grid_gemm.hpp via type_convert<CDataType>.
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;

    ck_tile::index_t verification = 0;
    // Default problem sizes: large enough to exercise multiple blocks and K iterations.
    ck_tile::index_t M = 3328;
    ck_tile::index_t N = 4096;
    ck_tile::index_t K = 4096;

    if(argc == 2)
    {
        verification = std::stoi(argv[1]);
    }
    if(argc == 5)
    {
        verification = std::stoi(argv[1]);
        M            = std::stoi(argv[2]);
        N            = std::stoi(argv[3]);
        K            = std::stoi(argv[4]);
    }

    printf("*** Naive implementation test ***\n");

    // Strides define the row-major memory layout.
    // Lda = K: to advance one row in A, skip K elements.
    // Ldb = K: B is stored as [N, K]; to advance one "B row" (i.e., one N slice), skip K elements.
    // Ldc = N: to advance one row in C, skip N elements.
    const ck_tile::index_t Lda = K;
    const ck_tile::index_t Ldb = K;
    const ck_tile::index_t Ldc = N;

    const auto a_lengths = std::array<ck_tile::index_t, 2>{M, K};
    const auto a_strides = std::array<ck_tile::index_t, 2>{Lda, 1};

    // B is stored as [N, K] -- N rows of K elements each.
    // N as the leading dimension means that for a fixed K column, N values are
    // non-contiguous (stride=K), but for a fixed N row, K values ARE contiguous (stride=1).
    // This layout enables vectorized loads (K is innermost) and coalesced accesses simultaneously.
    const auto b_lengths = std::array<ck_tile::index_t, 2>{N, K};
    const auto b_strides = std::array<ck_tile::index_t, 2>{Ldb, 1};

    const auto c_lengths = std::array<ck_tile::index_t, 2>{M, N};
    const auto c_strides = std::array<ck_tile::index_t, 2>{Ldc, 1};

    // Host tensors for reference computation and result verification.
    ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
    ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);
    ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);

    // FillUniformDistributionIntegerValue: fills with integer-valued fp16 in [-5, 5].
    // Using integer values simplifies CPU reference verification (no floating-point rounding
    // differences between GPU and CPU implementations).
    ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    // DeviceMem: allocates GPU (DRAM) memory of the given byte size.
    // Explicit two-step pattern: first allocate, then copy host data to device.
    // (This differs from the tutorial copy which uses DeviceMem(host_tensor) convenience
    // constructor.)
    ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_buf(c_host_dev.get_element_space_size_in_bytes());

    a_buf.ToDevice(a_host.mData.data());
    b_buf.ToDevice(b_host.mData.data());

    // Alignment = 8 means the load/store path guarantees 8 fp16 elements (= 128 bits = 16 bytes)
    // can be loaded/stored in a single instruction (global_load_dwordx4 / ds_write_b128).
    // This value must match the DRAM tile distribution's K1 parameter (K1 = 16/sizeof(fp16) = 8).
    constexpr ck_tile::index_t kAAlignment = 8;
    constexpr ck_tile::index_t kBAlignment = 8;
    constexpr ck_tile::index_t kCAlignment = 8;

    // kBlockSize: threads per block. Must equal M1*M2*K0 = 4*16*4 = 256 (see DRAM distribution).
    constexpr ck_tile::index_t kBlockSize = 256;

    // Block tile dimensions: the region of C each thread block computes in one launch.
    // kGemmMPerBlock=256 rows of A, kGemmNPerBlock=128 rows of B, kGemmKPerBlock=32 K elements
    // per K-loop iteration.
    constexpr ck_tile::index_t kGemmMPerBlock = 256;
    constexpr ck_tile::index_t kGemmKPerBlock = 32;
    constexpr ck_tile::index_t kGemmNPerBlock = 128;

    // Grid size: number of thread blocks = number of C tiles.
    // Assumes M and N are exact multiples of the tile sizes (no boundary handling).
    // For M=3328, N=4096: kGridSize = (3328/256) * (4096/128) = 13 * 32 = 416 blocks.
    ck_tile::index_t kGridSize = (M / kGemmMPerBlock) * (N / kGemmNPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    // Occupancy calculation (AMD CDNA architecture):
    // kWarpSize=64: AMD GPU wavefront has 64 threads.
    // kWarpPerCu=8: each Compute Unit has 4 SIMDs, each SIMD runs 2 wavefronts concurrently
    //              when register pressure allows (2 warps/SIMD × 4 SIMDs/CU = 8 warps/CU).
    // kWarpPerBlock=kBlockSize/kWarpSize=256/64=4: each block contains 4 warps.
    // kBlockPerCu=kWarpPerCu/kWarpPerBlock=8/4=2: 2 blocks can run simultaneously per CU.
    // make_kernel<kBlockPerCu>: this hint enables occupancy-based register allocation.
    constexpr ck_tile::index_t kWarpSize     = 64; // AMD GPU warp size
    constexpr ck_tile::index_t kWarpPerCu    = 8;  // 2 warps per SIMD * 4 SIMDs per CU
    constexpr ck_tile::index_t kWarpPerBlock = kBlockSize / kWarpSize;
    constexpr ck_tile::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    // gemm_kernel: the complete, fully specialized kernel type.
    // All template parameters are compile-time constants -- the kernel is fully specialized
    // before launch, with no runtime dispatch.
    using gemm_kernel = ck_tile::Gemm<ADataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      CElementFunction,
                                      kAAlignment,
                                      kBAlignment,
                                      kCAlignment,
                                      kBlockSize,
                                      kGemmMPerBlock,
                                      kGemmNPerBlock,
                                      kGemmKPerBlock>;

    // launch_kernel: CK Tile wrapper around the HIP runtime kernel launch.
    // Returns the average execution time in milliseconds across all timed runs.
    //
    // stream_config{nullptr, true, 0, 5, 1000}:
    //   nullptr  -- use the default HIP stream
    //   true     -- enable timing (measures GPU wall time)
    //   0        -- log verbosity level (0 = no extra output)
    //   5        -- number of warmup iterations (not included in timing)
    //   1000     -- number of timed iterations (average is reported)
    //
    // make_kernel<kBlockPerCu>: wraps the kernel functor with occupancy hint.
    // kBlockPerCu tells the HIP runtime how many blocks to co-schedule per CU,
    // which influences register file spill thresholds during compilation.
    float ave_time = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, 5, 1000},
        ck_tile::make_kernel<kBlockPerCu>(gemm_kernel{},
                                          kGridSize,
                                          kBlockSize,
                                          0,
                                          static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                                          static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                                          static_cast<CDataType*>(c_buf.GetDeviceBuffer()),
                                          M,
                                          N,
                                          K,
                                          Lda,
                                          Ldb,
                                          Ldc,
                                          CElementFunction{}));
    auto pass = true;

    if(verification)
    {
        // CPU reference: naive triple-loop GEMM for correctness checking.
        // Uses AccDataType (float) for accumulation, same as the GPU kernel.
        ck_tile::HostTensor<CDataType> c_host_ref(c_lengths, c_strides);
        reference_basic_gemm<ADataType, ADataType, AccDataType, CDataType>(
            a_host, b_host, c_host_ref);
        c_buf.FromDevice(c_host_dev.mData.data());
        pass &= ck_tile::check_err(c_host_dev, c_host_ref);
        std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
    }

    // Performance metrics:
    // flop = 2*M*N*K: each output element requires K multiply-adds (each counts as 2 ops).
    // num_btype: total bytes transferred (A read + B read + C write), ignoring caching effects.
    // tflops: TFlop/s = (2MNK ops) / (1e9 * time_ms) gives effective throughput.
    // gb_per_sec: GB/s bandwidth = bytes / (1e6 * time_ms).
    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return !pass;
}
