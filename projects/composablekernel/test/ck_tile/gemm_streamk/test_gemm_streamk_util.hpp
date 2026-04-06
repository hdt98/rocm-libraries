// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

enum struct GemmPipelineType
{
    Mem,
    CompV3,
    CompV4
};

template <GemmPipelineType PT, typename Problem>
struct GemmPipelineTypeSelector;

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::Mem, Problem>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrMem<Problem>;
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV3, Problem>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV4, Problem>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrCompV4<Problem>;
};

template <typename A0DataType,
          typename B0DataType,
          typename D0DataType,
          typename AccDataType,
          typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(A0DataType) < sizeof(B0DataType), A0DataType, B0DataType>;

    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(D0DataType), ComputeTypeAB, D0DataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // The logic below may need to become more advanced once bugs in Stream-K Tile Partitioner are
    // resolved. Because the number of WGs contributing to a macro tile in C may not be the same for
    // all macro tiles in C.

    // Calculate error due to more than 1 WG contributing to the same macro tile in C
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

ck_tile::index_t get_cu_count();

template <typename Tuple>
class TestCkTileStreamK : public ::testing::Test
{
    protected:
    // Original elements from develop (positions 0-15)
    using A0Layout                                = std::tuple_element_t<0, Tuple>; // ALayout
    using B0Layout                                = std::tuple_element_t<1, Tuple>; // BLayout
    using CLayout                                 = std::tuple_element_t<2, Tuple>;
    using A0DataType                              = std::tuple_element_t<3, Tuple>; // ADataType
    using B0DataType                              = std::tuple_element_t<4, Tuple>; // BDataType
    using AccDataType                             = std::tuple_element_t<5, Tuple>;
    using CDataType                               = std::tuple_element_t<6, Tuple>;
    static constexpr ck_tile::index_t M_Tile      = std::tuple_element_t<7, Tuple>::value;
    static constexpr ck_tile::index_t N_Tile      = std::tuple_element_t<8, Tuple>::value;
    static constexpr ck_tile::index_t K_Tile      = std::tuple_element_t<9, Tuple>::value;
    static constexpr ck_tile::index_t M_Warp_Tile = std::tuple_element_t<10, Tuple>::value;
    static constexpr ck_tile::index_t N_Warp_Tile = std::tuple_element_t<11, Tuple>::value;
    static constexpr ck_tile::index_t K_Warp_Tile = std::tuple_element_t<12, Tuple>::value;
    static constexpr bool Persistent              = std::tuple_element_t<13, Tuple>::value;
    static constexpr auto PipelineType            = std::tuple_element_t<14, Tuple>::value;
    static constexpr auto ReductionStrategy       = std::tuple_element_t<15, Tuple>::value;

    // Multi-ABD elements (positions 16-26)
    using A1Layout        = std::tuple_element_t<16, Tuple>;
    using A1DataType      = std::tuple_element_t<17, Tuple>;
    using B1Layout        = std::tuple_element_t<18, Tuple>;
    using B1DataType      = std::tuple_element_t<19, Tuple>;
    using D0Layout        = std::tuple_element_t<20, Tuple>;
    using D0DataType      = std::tuple_element_t<21, Tuple>;
    using D1Layout        = std::tuple_element_t<22, Tuple>;
    using D1DataType      = std::tuple_element_t<23, Tuple>;
    using AElementWiseFn  = std::tuple_element_t<24, Tuple>;
    using BElementWiseFn  = std::tuple_element_t<25, Tuple>;
    using CDElementWiseFn = std::tuple_element_t<26, Tuple>;

    using AsLayout   = ck_tile::tuple<A0Layout, A1Layout>;
    using AsDataType = ck_tile::tuple<A0DataType, A1DataType>;
    using BsLayout   = ck_tile::tuple<B0Layout, B1Layout>;
    using BsDataType = ck_tile::tuple<B0DataType, B1DataType>;
    using DsLayout   = ck_tile::tuple<D0Layout, D1Layout>;
    using DsDataType = ck_tile::tuple<D0DataType, D1DataType>;

    template <bool PadM       = true,
              bool PadN       = true,
              bool PadK       = true,
              bool Preshuffle = false,
              bool TransposeC = false>
    ck_tile::index_t invoke_streamk(
        const ck_tile::StreamKHostArgs<AsDataType::size(), BsDataType::size(), DsDataType::size()>&
            args,
        const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr bool kPadM      = PadM;
        constexpr bool kPadN      = PadN;
        constexpr bool kPadK      = PadK;
        constexpr bool preshuffle = Preshuffle;

        constexpr bool DoubleSmemBuffer = (PipelineType == GemmPipelineType::CompV4) ? true : false;
        constexpr int kBlockPerCu       = 1;
        constexpr bool StructuredSparsity = false;
        constexpr bool NumWaveGroup       = 1;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner =
            ck_tile::StreamKTilePartitioner<GemmShape, ReductionStrategy, Persistent>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     AsLayout,
                                                                     BsLayout,
                                                                     CLayout,
                                                                     TransposeC,
                                                                     StructuredSparsity,
                                                                     Persistent,
                                                                     NumWaveGroup,
                                                                     preshuffle>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        // We create the GEMM pipeline without specifying has_hot_loop or tail_num.
        // This is because num_loop can vary (a) per WG and (b) per iteration of the Stream-K
        // while loop. Instead, has_hot_loop and tail_num are determined in the Stream-K
        // Kernel's RunGemm function. This is a similar pattern used by grouped GEMM.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<AsDataType,
                                                                           BsDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           AElementWiseFn,
                                                                           BElementWiseFn>;

        using GemmPipeline = GemmPipelineTypeSelector<PipelineType, UniversalGemmProblem>::pipeline;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<AsDataType,
                                             BsDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             CDElementWiseFn,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC>>;

        using Kernel = ck_tile::StreamKKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs                = Kernel::MakeKernelArgs(args);
        const auto workspace_size = Kernel::GetWorkSpaceSize(kargs);
        ck_tile::DeviceMem workspace_data(workspace_size);
        workspace_data.SetZero();
        kargs.workspace_ptr = workspace_data.GetDeviceBuffer();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            // Since IsSupportedArgument only logs with an enviroment variable set, it's best to
            // throw when we hit an unsupported case.
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        dim3 grid_dims  = Kernel::GridSize(kargs.tile_partitioner);
        dim3 block_dims = Kernel::BlockSize();

        ck_tile::ignore = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid_dims, block_dims, 0, kargs));

        return kargs.tile_partitioner.estimate_num_wgs_per_tile();
    }

    public:
    void Run(ck_tile::index_t M,
             ck_tile::index_t N,
             ck_tile::index_t K,
             ck_tile::index_t stride_A0 = 0,
             ck_tile::index_t stride_A1 = 0,
             ck_tile::index_t stride_B0 = 0,
             ck_tile::index_t stride_B1 = 0,
             ck_tile::index_t stride_D0 = 0,
             ck_tile::index_t stride_D1 = 0,
             ck_tile::index_t stride_C  = 0)
    {
        // Since M, N, and K will vary depending on the number of CUs, we print it here to
        // facilitate test output readability.
        std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

        using namespace ck_tile::literals;

        auto f_host_tensor_descriptor = [](std::size_t row,
                                           std::size_t col,
                                           std::size_t stride,
                                           auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

        auto f_get_default_stride =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(stride == 0)
                {
                    if constexpr(std::is_same_v<decltype(layout),
                                                ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        return col;
                    }
                    else
                    {
                        return row;
                    }
                }
                else
                    return stride;
            };

        stride_A0 = f_get_default_stride(M, K, stride_A0, A0Layout{});
        stride_A1 = f_get_default_stride(M, K, stride_A1, A1Layout{});
        stride_B0 = f_get_default_stride(K, N, stride_B0, B0Layout{});
        stride_B1 = f_get_default_stride(K, N, stride_B1, B1Layout{});
        stride_D0 = f_get_default_stride(M, N, stride_D0, D0Layout{});
        stride_D1 = f_get_default_stride(M, N, stride_D1, D1Layout{});
        stride_C  = f_get_default_stride(M, N, stride_C, CLayout{});

        ck_tile::HostTensor<A0DataType> a0_m_k(
            f_host_tensor_descriptor(M, K, stride_A0, A0Layout{}));
        ck_tile::HostTensor<A1DataType> a1_m_k(
            f_host_tensor_descriptor(M, K, stride_A1, A1Layout{}));
        ck_tile::HostTensor<B0DataType> b0_k_n(
            f_host_tensor_descriptor(K, N, stride_B0, B0Layout{}));
        ck_tile::HostTensor<B1DataType> b1_k_n(
            f_host_tensor_descriptor(K, N, stride_B1, B1Layout{}));
        ck_tile::HostTensor<D0DataType> d0_m_n(
            f_host_tensor_descriptor(M, N, stride_D0, D0Layout{}));
        ck_tile::HostTensor<D1DataType> d1_m_n(
            f_host_tensor_descriptor(M, N, stride_D1, D1Layout{}));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));

        ck_tile::FillUniformDistribution<A0DataType>{-1.f, 1.f}(a0_m_k);
        ck_tile::FillUniformDistribution<A1DataType>{-1.f, 1.f}(a1_m_k);
        ck_tile::FillUniformDistribution<B0DataType>{-1.f, 1.f}(b0_k_n);
        ck_tile::FillUniformDistribution<B1DataType>{-1.f, 1.f}(b1_k_n);
        ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_m_n);
        ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n);

        ck_tile::DeviceMem a0_m_k_dev_buf(a0_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem a1_m_k_dev_buf(a1_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b0_k_n_dev_buf(b0_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b1_k_n_dev_buf(b1_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a0_m_k_dev_buf.ToDevice(a0_m_k.data());
        a1_m_k_dev_buf.ToDevice(a1_m_k.data());
        b0_k_n_dev_buf.ToDevice(b0_k_n.data());
        b1_k_n_dev_buf.ToDevice(b1_k_n.data());
        d0_m_n_dev_buf.ToDevice(d0_m_n.data());
        d1_m_n_dev_buf.ToDevice(d1_m_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        std::array<const void*, AsDataType::size()> as_ptr = {a0_m_k_dev_buf.GetDeviceBuffer(),
                                                              a1_m_k_dev_buf.GetDeviceBuffer()};
        std::array<const void*, BsDataType::size()> bs_ptr = {b0_k_n_dev_buf.GetDeviceBuffer(),
                                                              b1_k_n_dev_buf.GetDeviceBuffer()};
        std::array<const void*, DsDataType::size()> ds_ptr = {d0_m_n_dev_buf.GetDeviceBuffer(),
                                                              d1_m_n_dev_buf.GetDeviceBuffer()};
        std::array<ck_tile::index_t, AsDataType::size()> stride_As = {stride_A0, stride_A1};
        std::array<ck_tile::index_t, BsDataType::size()> stride_Bs = {stride_B0, stride_B1};
        std::array<ck_tile::index_t, DsDataType::size()> stride_Ds = {stride_D0, stride_D1};

        ck_tile::StreamKHostArgs<AsDataType::size(), BsDataType::size(), DsDataType::size()> args{
            as_ptr,
            bs_ptr,
            ds_ptr,
            c_m_n_dev_buf.GetDeviceBuffer(),
            M,
            N,
            K,
            stride_As,
            stride_Bs,
            stride_Ds,
            stride_C};

        ck_tile::index_t num_accumulations_per_tile =
            invoke_streamk<>(args, ck_tile::stream_config{nullptr, false, 0, 0, 1});

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

        // Calculate reference GEMM on the host
        ck_tile::HostTensor<A0DataType> a_m_k_host_ref_element_result(
            f_host_tensor_descriptor(M, K, stride_A0, A0Layout{}));
        ck_tile::HostTensor<B0DataType> b_k_n_host_ref_element_result(
            f_host_tensor_descriptor(K, N, stride_B0, B0Layout{}));
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));

        a_m_k_host_ref_element_result.SetZero();
        b_k_n_host_ref_element_result.SetZero();
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm_multiple_abd<AsDataType,
                                             BsDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             AElementWiseFn,
                                             BElementWiseFn,
                                             CDElementWiseFn>({a0_m_k, a1_m_k},
                                                              {b0_k_n, b1_k_n},
                                                              {d0_m_n, d1_m_n},
                                                              a_m_k_host_ref_element_result,
                                                              b_k_n_host_ref_element_result,
                                                              c_m_n_host_ref);

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());

        const auto rtol_atol =
            calculate_rtol_atol<A0DataType, B0DataType, D0DataType, AccDataType, CDataType>(
                K, num_accumulations_per_tile, max_accumulated_value);

        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass);
    };
};
