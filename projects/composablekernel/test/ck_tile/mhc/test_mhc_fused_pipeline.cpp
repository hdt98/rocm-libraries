// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/mhc.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_mhc.hpp"
#include "ck_tile/host/check_err.hpp"

// Include the invoker from the example
#include "../../../example/ck_tile/42_mhc/mhc_fused_pipeline_invoker.hpp"

// Test fixture for MHC fused pipeline tests
template <typename TestConfig>
class TestMHCFusedPipeline : public ::testing::Test
{
    public:
    using XDataType       = std::tuple_element_t<0, TestConfig>;
    using PhiDataType     = std::tuple_element_t<1, TestConfig>;
    using YDataType       = std::tuple_element_t<2, TestConfig>;
    using ComputeDataType = std::tuple_element_t<3, TestConfig>;

    static constexpr ck_tile::index_t MTile = std::tuple_element_t<4, TestConfig>::value;

    // Simple test with small dimensions
    template <int B = 16, int n = 4, int C = 64>
    void RunBasicTest()
    {
        const int nC         = n * C;
        const int output_dim = 2 * n + n * n;

        std::cout << "\n--- Testing MHC Fused Pipeline: B=" << B << ", n=" << n << ", C=" << C
                  << ", MTile=" << MTile << " ---" << std::endl;

        // Allocate host tensors
        ck_tile::HostTensor<XDataType> h_x({B, nC});
        ck_tile::HostTensor<PhiDataType> h_phi({nC, output_dim});
        ck_tile::HostTensor<YDataType> h_output({B, output_dim});

        // Initialize with random data
        ck_tile::FillUniformDistribution<XDataType>{-1.0f, 1.0f}(h_x);
        ck_tile::FillUniformDistribution<PhiDataType>{-0.5f, 0.5f}(h_phi);
        h_output.SetZero();

        // Allocate device memory
        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_phi_mem(h_phi.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_output_mem(h_output.get_element_space_size_in_bytes());

        d_x_mem.ToDevice(h_x.data());
        d_phi_mem.ToDevice(h_phi.data());
        d_output_mem.ToDevice(h_output.data());

        // Use invoker for kernel configuration
        using Invoker         = ck_tile::MHCFusedPipelineInvoker<XDataType,
                                                                 PhiDataType,
                                                                 YDataType,
                                                                 ComputeDataType,
                                                                 ck_tile::element_wise::Sigmoid,
                                                                 MTile>;
        using GemmKernel      = typename Invoker::GemmKernel;
        using ReductionKernel = typename Invoker::ReductionKernel;

        const ck_tile::index_t kBlockSize = Invoker::GetGemmBlockSize();
        auto grid_size                    = Invoker::GetGridSize(B, output_dim, nC);
        const ck_tile::index_t grid_m     = grid_size.at(ck_tile::number<0>{});
        const ck_tile::index_t grid_n     = grid_size.at(ck_tile::number<1>{});
        const ck_tile::index_t grid_k     = grid_size.at(ck_tile::number<2>{});
        const ck_tile::index_t kGridSize  = grid_m * grid_n * grid_k;

        // Allocate workspace for split-K
        const std::size_t workspace_size     = grid_k * B * output_dim * sizeof(ComputeDataType);
        const std::size_t partial_norms_size = grid_k * B * sizeof(ComputeDataType);

        ck_tile::DeviceMem d_workspace_mem(workspace_size);
        ck_tile::DeviceMem d_partial_norms_mem(partial_norms_size);

        (void)hipMemset(d_workspace_mem.GetDeviceBuffer(), 0, workspace_size);
        (void)hipMemset(d_partial_norms_mem.GetDeviceBuffer(), 0, partial_norms_size);

        const ck_tile::index_t reduction_threads = ReductionKernel::BlockSize();
        const ck_tile::index_t reduction_blocks =
            (B * output_dim + reduction_threads - 1) / reduction_threads;

        constexpr ck_tile::index_t kBlockPerCu = 1;
        constexpr ck_tile::index_t kSmemSize   = 8192; // Conservative estimate

        const float r = 1.0f, alpha_pre = 1.0f, alpha_post = 1.0f, alpha_res = 1.0f, bias = 0.0f;

        // Launch GEMM kernel
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false},
            ck_tile::make_kernel<kBlockPerCu>(
                GemmKernel{},
                kGridSize,
                kBlockSize,
                kSmemSize,
                static_cast<XDataType*>(d_x_mem.GetDeviceBuffer()),
                static_cast<PhiDataType*>(d_phi_mem.GetDeviceBuffer()),
                static_cast<ComputeDataType*>(d_workspace_mem.GetDeviceBuffer()),
                static_cast<ComputeDataType*>(d_partial_norms_mem.GetDeviceBuffer()),
                B,
                nC,
                output_dim,
                n,
                r,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias));

        // Launch reduction kernel
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false},
            ck_tile::make_kernel<kBlockPerCu>(
                ReductionKernel{},
                reduction_blocks,
                reduction_threads,
                0,
                static_cast<ComputeDataType*>(d_workspace_mem.GetDeviceBuffer()),
                static_cast<ComputeDataType*>(d_partial_norms_mem.GetDeviceBuffer()),
                static_cast<YDataType*>(d_output_mem.GetDeviceBuffer()),
                B,
                nC,
                output_dim,
                n,
                grid_k,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
                0));

        d_output_mem.FromDevice(h_output.data());

        // Compute reference
        ck_tile::HostTensor<YDataType> h_output_ref({B, output_dim});
        h_output_ref.SetZero();
        ck_tile::reference_mhc<XDataType,
                               PhiDataType,
                               YDataType,
                               ComputeDataType,
                               ck_tile::element_wise::Sigmoid>(
            h_x, h_phi, h_output_ref, n, C, r, alpha_pre, alpha_post, alpha_res, bias, 0);

        // Validate with appropriate tolerance
        float rtol = std::is_same_v<XDataType, ck_tile::bf16_t> ? 1e-2f : 1e-3f;
        float atol = std::is_same_v<XDataType, ck_tile::bf16_t> ? 1e-2f : 1e-3f;

        bool pass = ck_tile::check_err(
            h_output, h_output_ref, "Error: MHC fused pipeline output mismatch!", rtol, atol);

        std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        EXPECT_TRUE(pass);
    }
};

// Test configurations
// Tuple format: <XDataType, PhiDataType, YDataType, ComputeDataType, MTile>
using TestConfig_F32_M32 =
    std::tuple<float, float, float, float, std::integral_constant<ck_tile::index_t, 32>>;

using TestConfig_BF16_M32 = std::tuple<ck_tile::bf16_t,
                                       ck_tile::bf16_t,
                                       float,
                                       float,
                                       std::integral_constant<ck_tile::index_t, 32>>;

using TestConfig_BF16_M64 = std::tuple<ck_tile::bf16_t,
                                       ck_tile::bf16_t,
                                       float,
                                       float,
                                       std::integral_constant<ck_tile::index_t, 64>>;

using TestTypes = ::testing::Types<TestConfig_F32_M32, TestConfig_BF16_M32, TestConfig_BF16_M64>;

TYPED_TEST_SUITE(TestMHCFusedPipeline, TestTypes);

// Basic tests with different batch sizes
TYPED_TEST(TestMHCFusedPipeline, TestBatchSize16) { this->template RunBasicTest<16, 4, 64>(); }

TYPED_TEST(TestMHCFusedPipeline, TestBatchSize32) { this->template RunBasicTest<32, 4, 64>(); }

TYPED_TEST(TestMHCFusedPipeline, TestBatchSize8) { this->template RunBasicTest<8, 4, 128>(); }
