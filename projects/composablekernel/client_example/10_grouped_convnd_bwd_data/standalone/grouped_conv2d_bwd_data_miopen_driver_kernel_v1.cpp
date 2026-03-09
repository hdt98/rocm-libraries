// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Standalone example: runs the exact CK kernel selected by MIOpenDriver for conv bwd data (V1).
// DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<...>.
// Use same problem dimensions as MIOpenDriver to reproduce and debug. No GTest, no utility library.
// Build: see CMakeLists.txt in this directory (mkdir build && cd build && cmake .. && make).

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto ConvBwdDataDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

// MIOpen uses NHWC (channel-last): same layout types as in log ShaderName (NHWGK, NHWGC).
using InLayout  = ck::tensor_layout::convolution::NHWGC;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using OutLayout = ck::tensor_layout::convolution::NHWGK;

static constexpr ck::index_t NDimSpatial = 2;

// Problem dimensions matching MIOpenDriver: convfp16 -n 32 -c 256 -H 100 -W 100 -k 2376 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 1
// (bwd data: output_grad N,K,Ho,Wo; weight K,C,Y,X; input_grad N,C,Hi,Wi; G=1)
static constexpr ck::index_t G  = 1;
static constexpr ck::index_t N  = 32;
static constexpr ck::index_t K  = 2376;
static constexpr ck::index_t C  = 256;
static constexpr ck::index_t Y  = 3;
static constexpr ck::index_t X  = 3;
static constexpr ck::index_t Hi  = 100;
static constexpr ck::index_t Wi  = 100;
static constexpr ck::index_t Ho  = 100;
static constexpr ck::index_t Wo  = 100;

// Exact MIOpenDriver kernel: DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 256, 32, 8, 8, Default, 32, 32, 2, 4, 8, 4, 1, 1>
static constexpr char kExpectedMIOpenKernelType[] =
    "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 256, 32, 8, 8, Default, 32, 32, 2, 4, 8, 4, 1, 1>";
using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<
    NDimSpatial,
    OutLayout,
    WeiLayout,
    ck::Tuple<>,
    InLayout,
    OutDataType,
    WeiDataType,
    float,
    OutDataType,
    ck::Tuple<>,
    InDataType,
    PassThrough,
    PassThrough,
    PassThrough,
    ConvBwdDataDefault,
    true,
    true,
    1,
    256,
    128,
    256,
    32,
    8,
    8,
    32,
    32,
    2,
    4,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    1,
    S<4, 64, 1>,
    S<0, 2, 1>,
    S<0, 2, 1>,
    1,
    4,
    4,
    0,
    1,
    1,
    S<1, 32, 1, 8>,
    8>;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;
    explicit SimpleDeviceMem(std::size_t mem_size) : p_mem_(nullptr)
    {
        if(mem_size > 0)
            (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }
    void* GetDeviceBuffer() { return p_mem_; }
    ~SimpleDeviceMem()
    {
        if(p_mem_)
            (void)hipFree(p_mem_);
    }
    void* p_mem_;
};

static int run_conv()
{
    std::array<ck::index_t, NDimSpatial + 3> in_lengths{G, N, C, Hi, Wi};
    std::array<ck::index_t, NDimSpatial + 3> out_lengths{G, N, K, Ho, Wo};
    std::array<ck::index_t, NDimSpatial + 3> wei_lengths{G, K, C, Y, X};

    std::array<ck::index_t, NDimSpatial + 3> in_strides{
        static_cast<ck::index_t>(C),
        static_cast<ck::index_t>(Hi * Wi * G * C),
        1,
        static_cast<ck::index_t>(Wi * G * C),
        static_cast<ck::index_t>(G * C)};
    std::array<ck::index_t, NDimSpatial + 3> out_strides{
        static_cast<ck::index_t>(K),
        static_cast<ck::index_t>(Ho * Wo * G * K),
        1,
        static_cast<ck::index_t>(Wo * G * K),
        static_cast<ck::index_t>(G * K)};
    std::array<ck::index_t, NDimSpatial + 3> wei_strides{
        static_cast<ck::index_t>(K * Y * X * C),
        static_cast<ck::index_t>(Y * X * C),
        1,
        static_cast<ck::index_t>(X * C),
        static_cast<ck::index_t>(C)};

    std::array<ck::index_t, NDimSpatial> filter_strides{1, 1};
    std::array<ck::index_t, NDimSpatial> filter_dilations{1, 1};
    std::array<ck::index_t, NDimSpatial> input_left_pads{1, 1};
    std::array<ck::index_t, NDimSpatial> input_right_pads{1, 1};

    const std::size_t out_bytes = sizeof(OutDataType) * G * N * Ho * Wo * K;
    const std::size_t wei_bytes = sizeof(WeiDataType) * G * K * Y * X * C;
    const std::size_t in_bytes  = sizeof(InDataType) * G * N * Hi * Wi * C;

    SimpleDeviceMem out_dev(out_bytes);
    SimpleDeviceMem wei_dev(wei_bytes);
    SimpleDeviceMem in_dev(in_bytes);

    const ck::index_t split_k = 1;

    DeviceOp op;
    auto argument_ptr = op.MakeArgumentPointer(out_dev.GetDeviceBuffer(),
                                               wei_dev.GetDeviceBuffer(),
                                               std::array<const void*, 0>{},
                                               in_dev.GetDeviceBuffer(),
                                               out_lengths,
                                               out_strides,
                                               wei_lengths,
                                               wei_strides,
                                               {},
                                               {},
                                               in_lengths,
                                               in_strides,
                                               filter_strides,
                                               filter_dilations,
                                               input_left_pads,
                                               input_right_pads,
                                               PassThrough{},
                                               PassThrough{},
                                               PassThrough{},
                                               split_k);

    std::size_t workspace_size = op.GetWorkSpaceSize(argument_ptr.get());
    SimpleDeviceMem workspace_dev(workspace_size);
    if(workspace_size > 0)
        op.SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

    if(!op.IsSupportedArgument(argument_ptr.get()))
    {
        std::cerr << "Kernel does not support this problem." << std::endl;
        return EXIT_FAILURE;
    }

    const char* ck_args_tag = "Standalone-V1";
    const int C1 = C, K1 = K;
    std::cout << "[CKArgs " << ck_args_tag << "] G=" << G << " N=" << N << " K=" << K << " C=" << C
              << " C1=" << C1 << " K1=" << K1 << " Hi=" << Hi << " Wi=" << Wi << " Ho=" << Ho
              << " Wo=" << Wo << " Y=" << Y << " X=" << X << std::endl;
    std::cout << "[CKArgs " << ck_args_tag << "] input={" << in_lengths[0] << "," << in_lengths[1]
              << "," << in_lengths[2] << "," << in_lengths[3] << "," << in_lengths[4] << "}"
              << std::endl;
    std::cout << "[CKArgs " << ck_args_tag << "] in_strides={" << in_strides[0] << ","
              << in_strides[1] << "," << in_strides[2] << "," << in_strides[3] << ","
              << in_strides[4] << "}" << std::endl;
    std::cout << "[CKArgs " << ck_args_tag << "] output={" << out_lengths[0] << "," << out_lengths[1]
              << "," << out_lengths[2] << "," << out_lengths[3] << "," << out_lengths[4] << "}"
              << std::endl;
    std::cout << "[CKArgs " << ck_args_tag << "] out_strides={" << out_strides[0] << ","
              << out_strides[1] << "," << out_strides[2] << "," << out_strides[3] << ","
              << out_strides[4] << "}" << std::endl;
    std::cout << "[CKArgs " << ck_args_tag << "] weight={" << wei_lengths[0] << ","
              << wei_lengths[1] << "," << wei_lengths[2] << "," << wei_lengths[3] << ","
              << wei_lengths[4] << "}" << std::endl;
    std::cout << "[CKArgs " << ck_args_tag << "] wei_strides={" << wei_strides[0] << ","
              << wei_strides[1] << "," << wei_strides[2] << "," << wei_strides[3] << ","
              << wei_strides[4] << "}" << std::endl;
    std::cout << "[CKArgs " << ck_args_tag << "] strides={" << filter_strides[0] << ","
              << filter_strides[1] << "} dilation={" << filter_dilations[0] << ","
              << filter_dilations[1] << "} lPadding={" << input_left_pads[0] << ","
              << input_left_pads[1] << "} rPadding={" << input_right_pads[0] << ","
              << input_right_pads[1] << "}" << std::endl;
    std::cout << "[Workspace " << ck_args_tag << "] workspace_size=" << workspace_size
              << " bytes, ptr_set=" << (workspace_size > 0 ? "yes" : "no") << std::endl;

    const std::string ck_type_str = op.GetTypeString();
    const bool same_as_miopen     = (ck_type_str == kExpectedMIOpenKernelType);
    std::cout << "CK kernel type:   " << ck_type_str << std::endl;
    std::cout << "MIOpen expected:  " << kExpectedMIOpenKernelType
              << " (from MIOPEN_LOG_LEVEL=6 \"kernel_name = ...\")" << std::endl;
    std::cout << "Same CK kernel?   " << (same_as_miopen ? "YES" : "NO") << std::endl;

    auto invoker_ptr = op.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

    const int n_warmup = 5;
    const int n_iter   = 1000;
    float avg_time_ms   = invoker_ptr->Run(
        argument_ptr.get(), StreamConfig{nullptr, true, 0, n_warmup, n_iter});

    std::size_t flop      = static_cast<std::size_t>(2) * G * N * K * C * Ho * Wo * Y * X;
    std::size_t num_bytes = out_bytes + wei_bytes + in_bytes;
    float tflops          = static_cast<float>(flop) / 1.E9f / avg_time_ms;
    float gb_per_sec      = num_bytes / 1.E6f / avg_time_ms;

    std::cout << "Time:   " << std::fixed << std::setprecision(4) << avg_time_ms << " ms (avg over "
              << n_iter << " iterations)" << std::endl;
    std::cout << "Perf:   " << std::fixed << std::setprecision(2) << tflops << " TFlops, "
              << gb_per_sec << " GB/s" << std::endl;
    std::cout << "Done." << std::endl;

    return EXIT_SUCCESS;
}

int main()
{
    if(!ck::is_xdl_supported())
    {
        std::cerr << "XDL not supported on this device." << std::endl;
        return EXIT_FAILURE;
    }
    return run_conv();
}
