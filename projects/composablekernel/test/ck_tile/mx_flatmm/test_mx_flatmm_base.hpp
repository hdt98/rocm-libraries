// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"

// GEMM config with 16x16 warp tile for FP4
struct MXfp4_FlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 512;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

// GEMM config with 16x16 warp tile for FP6
struct MXfp6_FlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

// GEMM config with 16x16 warp tile for FP8
struct MXfp8_FlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

// GEMM config for FP8 activation x FP4 weight
struct MXf8f4_FlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

// GEMM config for FP4 activation x FP8 weight
struct MXf4f8_FlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

// Helper: is_row_major
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename Layout>
using is_row_major_t = ck_tile::bool_constant<
    std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>;

// Helper: preShuffleWeight
template <ck_tile::index_t N_Warp_Tile, typename dtype>
auto preShuffleWeight(ck_tile::HostTensor<dtype>& src)
{
    auto src_lengths          = src.get_lengths();
    const int K               = src_lengths[0];
    const int N               = src_lengths[1];
    constexpr int packed_size = ck_tile::numeric_traits<dtype>::PackedSize;
    int KPack =
        std::is_same_v<dtype, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size;
    int NLane = N_Warp_Tile;
    int KLane = 64 / NLane;
    int K0    = K / (KLane * KPack);

    ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor({N * K}, {1}));

    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; k += packed_size)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0    = k / (KLane * KPack);
            int tempk = k % (KLane * KPack);
            int k1    = tempk / KPack;
            int k2    = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            shuffled(outputIndex) = src(k, n);
        }
    }
    return shuffled;
}

// Helper: preShuffleScale
template <class FlatmmConfig, bool KLast, typename dtype>
auto preShuffleScale(ck_tile::HostTensor<dtype>& src)
{
    auto src_lengths = src.get_lengths();
    const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
    const auto K     = KLast ? src_lengths[1] : src_lengths[0];

    size_t MNXdlPack   = 2;
    size_t KXdlPack    = 2;
    size_t XdlMNThread = FlatmmConfig::N_Warp_Tile;
    size_t XdlKThread  = 64 / XdlMNThread;

    const auto MN_Paded = ck_tile::integer_least_multiple(MN, XdlMNThread * MNXdlPack);

    ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor({MN_Paded * K}, {1}));

    size_t K0 = K / KXdlPack / XdlKThread;

    for(size_t n = 0; n < MN_Paded; ++n)
    {
        for(size_t k = 0; k < K; ++k)
        {
            auto n0    = n / (XdlMNThread * MNXdlPack);
            auto tempn = n % (XdlMNThread * MNXdlPack);
            auto n1    = tempn % XdlMNThread;
            auto n2    = tempn / XdlMNThread;

            auto k0    = k / (XdlKThread * KXdlPack);
            auto tempk = k % (XdlKThread * KXdlPack);
            auto k1    = tempk % XdlKThread;
            auto k2    = tempk / XdlKThread;

            auto outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                               k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                               k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                               k2 * MNXdlPack + n2;

            if constexpr(KLast)
                shuffled(outputIndex) = n < MN ? src(n, k) : dtype{};
            else
                shuffled(outputIndex) = n < MN ? src(k, n) : dtype{};
        }
    }
    return shuffled;
}

// Base test fixture for MX FLATMM tests
template <typename Tuple>
class TestCkTileMXFlatmm : public ::testing::Test
{
    protected:
    using ADataType    = std::tuple_element_t<0, Tuple>;
    using BDataType    = std::tuple_element_t<1, Tuple>;
    using CDataType    = std::tuple_element_t<2, Tuple>;
    using FlatmmConfig = std::tuple_element_t<3, Tuple>;

    using AccDataType = float;
    using ScaleType   = ck_tile::e8m0_t;

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    static constexpr int ScaleGranularityM = 1;
    static constexpr int ScaleGranularityN = 1;
    static constexpr int ScaleGranularityK = 32;

    using ScaleM =
        ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>;
    using ScaleN =
        ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>;

    void run_test(ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t kbatch = 1)
    {
        ASSERT_EQ(K % ScaleGranularityK, 0) << "K must be multiple of ScaleGranularityK";
        ASSERT_EQ(K % ck_tile::numeric_traits<ADataType>::PackedSize, 0)
            << "K must be multiple of A packed size";
        ASSERT_EQ(K % ck_tile::numeric_traits<BDataType>::PackedSize, 0)
            << "K must be multiple of B packed size";

        ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
        ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
        ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));

        auto scale_stride_A = ck_tile::get_default_stride(
            M / ScaleGranularityM, K / ScaleGranularityK, 0, is_row_major(ALayout{}));
        auto scale_stride_B = ck_tile::get_default_stride(
            K / ScaleGranularityK, N / ScaleGranularityN, 0, is_row_major(BLayout{}));

        ck_tile::HostTensor<ADataType> a_host(
            ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_origin_host(
            ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
        ck_tile::HostTensor<CDataType> c_rslt_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

        ck_tile::HostTensor<ScaleType> scale_a(ck_tile::host_tensor_descriptor(
            M / ScaleGranularityM,
            K / ScaleGranularityK,
            scale_stride_A,
            is_row_major(ALayout{})));
        ck_tile::HostTensor<ScaleType> scale_b(ck_tile::host_tensor_descriptor(
            K / ScaleGranularityK,
            N / ScaleGranularityN,
            scale_stride_B,
            is_row_major(BLayout{})));

        // Initialize data
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_fp6x16_t>)
        {
            auto a_buffer_bytes = a_host.get_element_space_size_in_bytes();
            auto b_buffer_bytes = b_origin_host.get_element_space_size_in_bytes();
            ck_tile::FillUniformDistribution<>{-1.f, 1.f}(scale_a);
            ck_tile::FillUniformDistribution<>{-1.f, 1.f}(scale_b);
            std::vector<int8_t> random_bufA(a_buffer_bytes);
            std::vector<int8_t> random_bufB(b_buffer_bytes);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(1, 4);

            for(size_t i = 0; i < a_buffer_bytes; ++i)
                random_bufA[i] = static_cast<int8_t>(dis(gen));

            for(size_t i = 0; i < b_buffer_bytes; ++i)
                random_bufB[i] = static_cast<int8_t>(dis(gen));

            memcpy(a_host.data(), random_bufA.data(), a_buffer_bytes);
            memcpy(b_origin_host.data(), random_bufB.data(), b_buffer_bytes);
        }
        else
        {
            ck_tile::FillUniformDistribution<>{0.0f, 1.0f}(a_host);
            ck_tile::FillUniformDistribution<>{-.5f, .5f}(b_origin_host);
            ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_a);
            ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_b);
        }

        // Pre-shuffle weight and scale tensors
        const auto b_shuffled_host =
            preShuffleWeight<FlatmmConfig::N_Warp_Tile>(b_origin_host);
        const auto scale_a_shuffled = preShuffleScale<FlatmmConfig, true>(scale_a);
        const auto scale_b_shuffled = preShuffleScale<FlatmmConfig, false>(scale_b);

        // Allocate device memory
        ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_shuffled_dev_buf(
            b_shuffled_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem scale_a_dev_buf(
            scale_a_shuffled.get_element_space_size_in_bytes());
        ck_tile::DeviceMem scale_b_dev_buf(
            scale_b_shuffled.get_element_space_size_in_bytes());

        // Copy to device
        a_dev_buf.ToDevice(a_host.data());
        b_shuffled_dev_buf.ToDevice(b_shuffled_host.data());
        c_rslt_host.SetZero();
        if(kbatch > 1)
        {
            c_dev_buf.SetZero();
        }
        scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
        scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

        auto scale_a_dev_ptr = ScaleM{
            static_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()),
            M / ScaleGranularityM};
        auto scale_b_dev_ptr = ScaleN{
            static_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()),
            N / ScaleGranularityN};

        // Create host args
        ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN> args = {
            a_dev_buf.GetDeviceBuffer(),
            b_shuffled_dev_buf.GetDeviceBuffer(),
            {},
            c_dev_buf.GetDeviceBuffer(),
            kbatch,
            M,
            N,
            K,
            stride_A,
            stride_B,
            {},
            stride_C,
            scale_a_dev_ptr,
            scale_b_dev_ptr};

        // Launch kernel directly (inline pipeline construction, like gemm_block_scale tests)
        launch_mx_flatmm_kernel(args);

        // Copy result back to host
        c_dev_buf.FromDevice(c_rslt_host.data());

        // Compute reference on CPU
        ck_tile::HostTensor<CDataType> c_ref_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_ref_host.SetZero();

        ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
            a_host, b_origin_host, c_ref_host, scale_a, scale_b);

        const float rtol = 1e-2;
        const float atol = 1e-2;

        bool pass = ck_tile::check_err(
            c_rslt_host, c_ref_host, "Error: Incorrect results!", rtol, atol);

        EXPECT_TRUE(pass) << "MX FLATMM test failed for M=" << M << " N=" << N << " K=" << K
                          << " kbatch=" << kbatch << " rtol=" << rtol << " atol=" << atol;
    }

    private:
    // Directly construct and launch the MX FLATMM kernel pipeline
    // (follows the gemm_block_scale test pattern — no extern template instantiation)
    void launch_mx_flatmm_kernel(
        const ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN>& args)
    {
        using FlatmmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
            ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
            ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                              FlatmmConfig::N_Warp_Tile,
                              FlatmmConfig::K_Warp_Tile>>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                       FlatmmConfig::TileParitionerGroupNum,
                                                       FlatmmConfig::TileParitionerM01>;

        using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                               FlatmmConfig::kPadN,
                                               FlatmmConfig::kPadK,
                                               ALayout,
                                               BLayout,
                                               CLayout,
                                               FlatmmConfig::NumWaveGroups>;
        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, FlatmmShape, Traits>;

        using BaseFlatmmPipeline =
            ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

        const ck_tile::index_t k_grain  = args.k_batch * FlatmmConfig::K_Tile;
        const ck_tile::index_t k_split  = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(k_split);
        const bool has_hot_loop         = BaseFlatmmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseFlatmmPipeline::GetBlockLoopTailNum(num_loop);

        using MXGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                               FlatmmConfig::kPadN,
                                                               FlatmmConfig::kPadK,
                                                               FlatmmConfig::DoubleSmemBuffer,
                                                               ALayout,
                                                               BLayout,
                                                               CLayout,
                                                               FlatmmConfig::TransposeC,
                                                               FlatmmConfig::UseStructuredSparsity,
                                                               false, // persistent
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>;

        using ComputeDataType                  = ADataType;
        constexpr auto scheduler               = FlatmmConfig::Scheduler;
        constexpr int BlockedXDLN_PerWarp      = 2;
        using CDEElementWise                   = ck_tile::element_wise::PassThrough;

        const auto Run = [&](const auto has_hot_loop_, const auto tail_num_) {
            constexpr auto has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_num_v     = tail_num_.value;

            using MXPipelineProblem =
                ck_tile::MXFlatmmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  FlatmmShape,
                                                  MXGemmTraits,
                                                  scheduler,
                                                  has_hot_loop_v,
                                                  tail_num_v>;

            using MXFlatmmPipeline =
                ck_tile::MXFlatmmPipelineAGmemBGmemCRegV1<MXPipelineProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                                  ComputeDataType,
                                                  ck_tile::tuple<>,
                                                  AccDataType,
                                                  CDataType,
                                                  ck_tile::tuple<>,
                                                  CLayout,
                                                  CDEElementWise,
                                                  TilePartitioner::MPerBlock,
                                                  TilePartitioner::NPerBlock,
                                                  FlatmmConfig::M_Warp,
                                                  FlatmmConfig::N_Warp,
                                                  FlatmmConfig::M_Warp_Tile,
                                                  FlatmmConfig::N_Warp_Tile,
                                                  FlatmmConfig::K_Warp_Tile,
                                                  MXPipelineProblem::TransposeC,
                                                  FlatmmConfig::NumWaveGroups,
                                                  false, // FixedVectorSize
                                                  1,     // VectorSizeC
                                                  FlatmmConfig::TiledMMAPermuteN,
                                                  BlockedXDLN_PerWarp>>;

            using Kernel =
                ck_tile::MXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, GemmEpilogue>;

            auto kargs = Kernel::MakeKernelArgs(args);

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error(
                    "Wrong! Arguments not supported for MX FLATMM kernel!\n");
            }

            const dim3 grids      = Kernel::GridSize(kargs);
            constexpr dim3 blocks = Kernel::BlockSize();

            auto clear_output = [&]() {
                if(args.k_batch > 1)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.M * args.N * sizeof(CDataType), nullptr));
            };

            ck_tile::stream_config s{};
            clear_output();
            ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(
                    Kernel{}, grids, blocks, 0, kargs));
        };

        BaseFlatmmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};
