// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Standalone: grouped conv bwd data via ck_tile GroupedConvolutionBackwardDataKernel.
// Same problem as grouped_conv2d_bwd_data_miopen_driver_kernel.cpp for performance comparison.
// Build: see standalone/CMakeLists.txt (add this source; requires example/ path for invoker).
//
// Set to 1 to use COMPUTE_ASYNC pipeline (amd_async_buffer_load, gfx950). Requires CKTile's
// GemmPipelineAgBgCrCompAsync to provide GetName() and the 6-arg operator() used by
// GroupedConvolutionBackwardDataKernel; otherwise the build will fail. If the run hangs at 100%
// GPU, build with USE_ASYNC_GROUPED_CONV_BWD_DATA=OFF to use the V3 pipeline instead.
#ifndef USE_ASYNC_GROUPED_CONV_BWD_DATA
#define USE_ASYNC_GROUPED_CONV_BWD_DATA 1
#endif

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

// Pull in conv types in correct order (ConvolutionSpecialization, tensor_layout, then utils/kernel)
// TileGemmTraits and get_device_name() must be visible before grouped_convolution.hpp
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/host/device_memory.hpp"

#include "conv_configs.hpp"
#include "grouped_convolution_backward_data_invoker.hpp"

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;

static constexpr ck_tile::index_t NDimSpatial = 2;

// Same problem as grouped_conv2d_bwd_data_miopen_driver_kernel.cpp (MIOpenDriver: convfp16 -n 32 -c 256 -H 100 -W 100 -k 2376 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1)
static constexpr ck_tile::index_t G  = 1;
static constexpr ck_tile::index_t N  = 32;
static constexpr ck_tile::index_t K  = 2376;
static constexpr ck_tile::index_t C  = 256;
static constexpr ck_tile::index_t Y  = 3;
static constexpr ck_tile::index_t X  = 3;
static constexpr ck_tile::index_t Hi = 100;
static constexpr ck_tile::index_t Wi = 100;
static constexpr ck_tile::index_t Ho = 100;
static constexpr ck_tile::index_t Wo = 100;

using NHWGC = ck_tile::tensor_layout::convolution::NHWGC;
using GKYXC = ck_tile::tensor_layout::convolution::GKYXC;
using NHWGK = ck_tile::tensor_layout::convolution::NHWGK;

int main()
{
    std::vector<ck_tile::index_t> filter_spatial = {Y, X};
    std::vector<ck_tile::index_t> input_spatial  = {Hi, Wi};
    std::vector<ck_tile::index_t> strides       = {1, 1};
    std::vector<ck_tile::index_t> dilations     = {1, 1};
    std::vector<ck_tile::index_t> left_pads     = {1, 1};
    std::vector<ck_tile::index_t> right_pads    = {1, 1};

    ck_tile::conv::ConvParam conv_param{NDimSpatial,
                                        G,
                                        N,
                                        K,
                                        C,
                                        filter_spatial,
                                        input_spatial,
                                        strides,
                                        dilations,
                                        left_pads,
                                        right_pads};

    const ck_tile::index_t split_k = 1;
    const std::size_t in_bytes =
        static_cast<std::size_t>(G) * N * C * Hi * Wi * sizeof(InDataType);
    const std::size_t wei_bytes =
        static_cast<std::size_t>(G) * K * C * Y * X * sizeof(WeiDataType);
    const std::size_t out_bytes =
        static_cast<std::size_t>(G) * N * K * Ho * Wo * sizeof(OutDataType);

    ck_tile::DeviceMem in_dev(in_bytes);
    ck_tile::DeviceMem wei_dev(wei_bytes);
    ck_tile::DeviceMem out_dev(out_bytes);

    in_dev.SetZero();
    wei_dev.SetZero();
    out_dev.SetZero();

    ck_tile::GroupedConvBwdDataHostArgs args(conv_param,
                                             in_dev.GetDeviceBuffer(),
                                             wei_dev.GetDeviceBuffer(),
                                             {},
                                             out_dev.GetDeviceBuffer(),
                                             split_k);

    using ConvConfig = std::conditional_t<USE_ASYNC_GROUPED_CONV_BWD_DATA != 0,
                                          ConvConfigComputeAsync<ck_tile::half_t>,
                                          ConvConfigComputeV3<ck_tile::half_t>>;
    using Invoker    = GroupedConvolutionBackwardDataInvoker;

    try
    {
        // Match grouped_conv2d_bwd_data_miopen_driver_kernel.cpp for fair perf comparison
        const int n_warmup = 5;
        const int n_iter   = 1000;  // reduce if run hangs or is too slow (e.g. 20)
        float avg_ms       = Invoker::template grouped_conv_bwd_data<NDimSpatial,
                                                              ConvConfig,
                                                              InDataType,
                                                              WeiDataType,
                                                              float,
                                                              OutDataType,
                                                              NHWGC,
                                                              GKYXC,
                                                              NHWGK>(
            args, ck_tile::stream_config{nullptr, true, 0, n_warmup, n_iter});

        std::size_t flop   = args.GetFlops();
        std::size_t bytes  = args.template GetByte<InDataType, WeiDataType, OutDataType>();
        float tflops       = static_cast<float>(flop) / 1.E9f / avg_ms;
        float gb_per_sec   = bytes / 1.E6f / avg_ms;

#if USE_ASYNC_GROUPED_CONV_BWD_DATA
        std::cout << "ck_tile GroupedConvBwdData (ConvConfigComputeAsync fp16, amd_async_buffer_load)"
                  << std::endl;
#else
        std::cout << "ck_tile GroupedConvBwdData (ConvConfigComputeV3 fp16)" << std::endl;
#endif
        std::cout << "Problem: G=" << G << " N=" << N << " K=" << K << " C=" << C << " Hi=" << Hi
                  << " Wi=" << Wi << " Ho=" << Ho << " Wo=" << Wo << " Y=" << Y << " X=" << X
                  << " (same as grouped_conv2d_bwd_data_miopen_driver_kernel)" << std::endl;
        std::cout << "Time:   " << std::fixed << std::setprecision(4) << avg_ms << " ms (avg over "
                  << n_iter << " iterations)" << std::endl;
        std::cout << "Perf:   " << std::fixed << std::setprecision(2) << tflops << " TFlops, "
                  << gb_per_sec << " GB/s" << std::endl;
        std::cout << "Done." << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
