// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/sinkhorn_knopp.hpp"
#include "ck_tile/host/kernel_launch.hpp"

template <typename Tuple>
class TestCkTileSinkHorn : public ::testing::Test
{
    protected:
    using XDataType       = std::tuple_element_t<0, Tuple>;
    using ComputeDataType = std::tuple_element_t<1, Tuple>;
    using YDataType       = std::tuple_element_t<2, Tuple>;
    using BatchSize       = std::tuple_element_t<3, Tuple>;
    using N               = std::tuple_element_t<4, Tuple>;

    using TestSinkhornShape = ck_tile::SinkhornKnoppShape<BatchSize, N>;

    void RunGenericTest(const std::vector<ck_tile::index_t>& input_shape, const int max_iterations)
    {
        auto batchsize      = input_shape[0];
        auto input_n        = input_shape[1];
        auto default_stride = {input_n * input_n, input_n, 1};

        ck_tile::HostTensor<XDataType> h_x(input_shape, default_stride);
        ck_tile::HostTensor<YDataType> h_y(input_shape, default_stride);

        ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(h_x);

        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_y_mem(h_y.get_element_space_size_in_bytes());

        ck_tile::SinkhornKnoppArgs args{static_cast<void*>(d_y_mem.GetDeviceBuffer()),
                                        static_cast<void*>(d_x_mem.GetDeviceBuffer()),
                                        input_shape,
                                        max_iterations};

        d_x_mem.ToDevice(h_x.data());
        d_y_mem.ToDevice(h_y.data());

        using Problem =
            ck_tile::SinkhornKnoppProblem<XDataType, YDataType, TestSinkhornShape, ComputeDataType>;
        using Kernel =
            ck_tile::SinkhornKnoppNaiveKernel<Problem, ck_tile::SinkhornKnoppDefaultPolicy>;

        // Launch configuration
        const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = 1;

        ck_tile::index_t kGridSize = 1;

        if(!Kernel::IsSupportedArgument(args))
        {
            throw std::runtime_error("Wrong! Arguments not supported!\n");
        }

        auto timer = ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, true, 0},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, kGridSize, kBlockSize, 0, args));

        printf("Average time: %f ms\n", timer);

        // Reference computation
        ck_tile::HostTensor<YDataType> h_y_ref(input_shape, default_stride);
        sinkhorn_knopp_naive_ref<XDataType, ComputeDataType, YDataType>(
            h_x, h_y_ref, max_iterations);

        // TODO: Refine tolerances. The naive algorithm is remarkably inaccurate compared to CPU
        const float rtol = 1e-3;
        const float atol = 1e-2;

        // Check that reference result is doubly stochastic
        ck_tile::HostTensor<YDataType> unit_n({batchsize, input_n}, {input_n, 1});
        ck_tile::FillConstant<YDataType>{1.0}(unit_n);

        bool result = true;

        // NOTE: As the iteration happens first on rows, then columns, only the latter
        // can be expected to be exactly normalized
        // auto rows_ref = row_sum_ref(h_y_ref);
        // result &= ck_tile::check_err(rows_ref,
        //                              unit_n,
        //                              "Error: Reference computation result rows do not sum to 1!",
        //                              rtol,
        //                              atol);

        auto cols_ref = col_sum_ref(h_y_ref);
        result &= ck_tile::check_err(cols_ref,
                                     unit_n,
                                     "Error: Reference computation result cols do not sum to 1!",
                                     rtol,
                                     atol);

        // Transfer data from device and check that it matches reference
        d_y_mem.FromDevice(h_y.data());

        // NOTE: As the iteration happens first on rows, then columns, only the latter
        // can be expected to be exactly normalized
        // auto rows = row_sum_ref(h_y);
        // result &=
        //     ck_tile::check_err(rows, unit_n, "Error: Result rows do not sum to 1!", rtol, atol);

        auto cols = col_sum_ref(h_y);
        result &=
            ck_tile::check_err(cols, unit_n, "Error: Result cols do not sum to 1!", rtol, atol);

        result &= ck_tile::check_err(
            h_y, h_y_ref, "Error: Sinkhorn-Knopp doesn't match CPU reference!", rtol, atol);

        EXPECT_TRUE(result);
    }
};
