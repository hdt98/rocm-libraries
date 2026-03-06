// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <hip/hip_runtime.h>

// Use CK_Tile host utilities for tensor management and verification
#include "ck_tile/host.hpp"
#include "gemm_mint.hpp"

// Reference GEMM implementation for verification
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
void reference_gemm_mint(const ck_tile::HostTensor<ADataType>& a,
                         const ck_tile::HostTensor<BDataType>& b,
                         ck_tile::HostTensor<CDataType>& c)
{
    auto M = a.mDesc.get_lengths()[0];
    auto K = a.mDesc.get_lengths()[1];
    auto N = b.mDesc.get_lengths()[0];

    for(ck_tile::index_t m = 0; m < M; ++m)
    {
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            AccDataType acc = 0;
            for(ck_tile::index_t k = 0; k < K; ++k)
            {
                acc += static_cast<AccDataType>(a(m, k)) * static_cast<AccDataType>(b(n, k));
            }
            c(m, n) = static_cast<CDataType>(acc);
        }
    }
}

int main(int argc, char* argv[])
{
    // Data types
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;

    // Problem size
    ck_tile::index_t M = 2048;
    ck_tile::index_t N = 2048;
    ck_tile::index_t K = 2048;
    ck_tile::index_t verification = 0;

    // Parse command line arguments
    if(argc == 2)
    {
        verification = std::stoi(argv[1]);
    }
    else if(argc == 5)
    {
        verification = std::stoi(argv[1]);
        M            = std::stoi(argv[2]);
        N            = std::stoi(argv[3]);
        K            = std::stoi(argv[4]);
    }

    std::cout << "=== MINT Naive GEMM Example ===" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Matrix strides (row-major)
    const ck_tile::index_t stride_a = K;
    const ck_tile::index_t stride_b = K;
    const ck_tile::index_t stride_c = N;

    // Tensor dimensions
    const auto a_lengths = std::array<ck_tile::index_t, 2>{M, K};
    const auto a_strides = std::array<ck_tile::index_t, 2>{stride_a, 1};

    const auto b_lengths = std::array<ck_tile::index_t, 2>{N, K};
    const auto b_strides = std::array<ck_tile::index_t, 2>{stride_b, 1};

    const auto c_lengths = std::array<ck_tile::index_t, 2>{M, N};
    const auto c_strides = std::array<ck_tile::index_t, 2>{stride_c, 1};

    // Create host tensors
    ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
    ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);
    ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);

    // Initialize with random data
    std::cout << "Initializing input matrices..." << std::endl;
    ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    // Allocate device memory
    ck_tile::DeviceMem a_device(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_device(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_device(c_host_dev.get_element_space_size_in_bytes());

    // Transfer data to device
    a_device.ToDevice(a_host.mData.data());
    b_device.ToDevice(b_host.mData.data());

    // Kernel configuration
    constexpr ck_tile::index_t kMPerBlock  = 128;
    constexpr ck_tile::index_t kNPerBlock  = 128;
    constexpr ck_tile::index_t kKPerBlock  = 16;

    // Grid dimensions
    const ck_tile::index_t grid_m = (M + kMPerBlock - 1) / kMPerBlock;
    const ck_tile::index_t grid_n = (N + kNPerBlock - 1) / kNPerBlock;

    std::cout << "Launching kernel with grid (" << grid_m << ", " << grid_n
              << "), block MxN tile (" << kMPerBlock << ", " << kNPerBlock << ")" << std::endl;

    // Launch configuration
    dim3 grid_size(grid_m, grid_n, 1);
    dim3 block_size(16, 16, 1);  // 256 threads total

    // Warmup
    hipLaunchKernelGGL((mint_gemm::MintGemmKernel<kMPerBlock,
                                                    kNPerBlock,
                                                    kKPerBlock,
                                                    ADataType,
                                                    AccDataType>),
                       grid_size,
                       block_size,
                       0,
                       nullptr,
                       static_cast<const ADataType*>(a_device.GetDeviceBuffer()),
                       static_cast<const BDataType*>(b_device.GetDeviceBuffer()),
                       static_cast<CDataType*>(c_device.GetDeviceBuffer()),
                       M,
                       N,
                       K,
                       stride_a,
                       stride_b,
                       stride_c);

    hipDeviceSynchronize();

    // Timing runs
    const int num_runs = 10;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for(int i = 0; i < num_runs; ++i)
    {
        hipLaunchKernelGGL((mint_gemm::MintGemmKernel<kMPerBlock,
                                                        kNPerBlock,
                                                        kKPerBlock,
                                                        ADataType,
                                                        AccDataType>),
                           grid_size,
                           block_size,
                           0,
                           nullptr,
                           static_cast<const ADataType*>(a_device.GetDeviceBuffer()),
                           static_cast<const BDataType*>(b_device.GetDeviceBuffer()),
                           static_cast<CDataType*>(c_device.GetDeviceBuffer()),
                           M,
                           N,
                           K,
                           stride_a,
                           stride_b,
                           stride_c);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float elapsed_ms = 0;
    hipEventElapsedTime(&elapsed_ms, start, stop);
    float ave_time = elapsed_ms / num_runs;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    // Verification
    bool pass = true;
    if(verification)
    {
        std::cout << "Running CPU reference..." << std::endl;
        ck_tile::HostTensor<CDataType> c_host_ref(c_lengths, c_strides);
        reference_gemm_mint<ADataType, BDataType, AccDataType, CDataType>(a_host, b_host, c_host_ref);

        // Copy results back from device
        c_device.FromDevice(c_host_dev.mData.data());

        // Check results
        pass = ck_tile::check_err(c_host_dev, c_host_ref);
        std::cout << "Verification: " << (pass ? "PASSED" : "FAILED") << std::endl;
    }

    // Performance metrics
    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_bytes =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops     = static_cast<float>(flop) / 1.0e9 / ave_time;
    float gb_per_sec = static_cast<float>(num_bytes) / 1.0e6 / ave_time;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Time: " << ave_time << " ms" << std::endl;
    std::cout << "  TFlops: " << tflops << std::endl;
    std::cout << "  Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    return pass ? 0 : 1;
}
