// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Split-K forward convolution example for 3D convolutions with large GEMM-K dimension.
// Uses SplitKFactor=-1 (auto-deduce k_batch from GPU occupancy heuristic).
// Layouts: NDHWGC input, GKZYXC weights, NDHWGK output (channels-last, 3D).

#include "common.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_fwd_gpu.hpp"
#include "ck/library/utility/gpu_verification.hpp"
#include <iostream>

// kernel data types
using InKernelDataType  = BF16;
using WeiKernelDataType = BF16;
using AccDataType       = FP32;
using CShuffleDataType  = BF16;
using OutKernelDataType = BF16;

// tensor data types
using InUserDataType  = InKernelDataType;
using WeiUserDataType = WeiKernelDataType;
using OutUserDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

// Split-K 3D conv instance (Filter3x3Stride1Pad0, SplitKFactor=-1 = auto-deduce).
// This instance is good for cases where K=96 and C=96
// template <ck::index_t NDimSpatial, int SplitKFactor>
// using ConvFwdSplitK =
//     ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
//         NDimSpatial,
//         InputLayout<NDimSpatial>,
//         WeightLayout<NDimSpatial>,
//         ck::Tuple<>,
//         OutputLayout<NDimSpatial>,
//         InKernelDataType,
//         WeiKernelDataType,
//         AccDataType,
//         CShuffleDataType,
//         ck::Tuple<>,
//         OutKernelDataType,
//         InElementOp,
//         WeiElementOp,
//         OutElementOp,
//         //ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
//         ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
//         GemmSpec,
//         1,               // NumGemmKPrefetchStage
//         256,             // BlockSize
//         256, 96,  32,   // MPerBlock=256, NPerBlock=32, KPerBlock=64
//           8,   8,        // AK1, BK1
//          32,  32,        // MPerXDL, NPerXDL
//           4,   3,        // MXdlPerWave=2, NXdlPerWave=1
//                          //   → MWave=256/(2×32)=4, NWave=32/(1×32)=1, WaveSize=64 ✓ (Rule 1)
//         S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // A: AK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
//         S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // B: BK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
//         1, 1,
//         S<1, 64, 1, 4>,  // CDE: CShuffleTile=128×32; B=32/8=4, A=64; product=256 ✓ (Rule 6)
//         8,               // ScalarPerVector: NWave×NPerXDL=32, 32%8=0 ✓; K=96, 96%8=0 ✓
//         BF16,            // AComputeDataType (explicit to avoid unpack default complexity)
//         BF16,            // BComputeDataType
//         ck::LoopScheduler::Default,
//         1,               // NumGroupsToMerge
//         false,           // DoubleBuffer
//         SplitKFactor>;             // SplitKFactor = -1 (auto-deduce k_batch)

// K = 192 instance
// template <ck::index_t NDimSpatial, int SplitKFactor>
// using ConvFwdSplitK =
//     ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
//         NDimSpatial,
//         InputLayout<NDimSpatial>,
//         WeightLayout<NDimSpatial>,
//         ck::Tuple<>,
//         OutputLayout<NDimSpatial>,
//         InKernelDataType,
//         WeiKernelDataType,
//         AccDataType,
//         CShuffleDataType,
//         ck::Tuple<>,
//         OutKernelDataType,
//         InElementOp,
//         WeiElementOp,
//         OutElementOp,
//         ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
//         //ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
//         GemmSpec,
//         1,               // NumGemmKPrefetchStage
//         256,             // BlockSize
//         128, 192,  32,   // MPerBlock=256, NPerBlock=32, KPerBlock=64
//           8,   8,        // AK1, BK1
//          32,  32,        // MPerXDL, NPerXDL
//           2,   3,        // MXdlPerWave=2, NXdlPerWave=1
//                          //   → MWave=256/(2×32)=4, NWave=32/(1×32)=1, WaveSize=64 ✓ (Rule 1)
//         S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // A: AK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
//         S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // B: BK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
//         1, 1,
//         S<1, 32, 1, 8>,  // CDE: CShuffleTile=128×32; B=32/8=4, A=64; product=256 ✓ (Rule 6)
//         8,               // ScalarPerVector: NWave×NPerXDL=32, 32%8=0 ✓; K=96, 96%8=0 ✓
//         BF16,            // AComputeDataType (explicit to avoid unpack default complexity)
//         BF16,            // BComputeDataType
//         ck::LoopScheduler::Default,
//         1,               // NumGroupsToMerge
//         false,           // DoubleBuffer
//         SplitKFactor>;             // SplitKFactor = -1 (auto-deduce k_batch)

// K = 384 instance
// template <ck::index_t NDimSpatial, int SplitKFactor>
// using ConvFwdSplitK =
//     ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
//         NDimSpatial,
//         InputLayout<NDimSpatial>,
//         WeightLayout<NDimSpatial>,
//         ck::Tuple<>,
//         OutputLayout<NDimSpatial>,
//         InKernelDataType,
//         WeiKernelDataType,
//         AccDataType,
//         CShuffleDataType,
//         ck::Tuple<>,
//         OutKernelDataType,
//         InElementOp,
//         WeiElementOp,
//         OutElementOp,
//         ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
//         //ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
//         GemmSpec,
//         2,               // NumGemmKPrefetchStage
//         256,             // BlockSize
//         128, 192,  32,   // MPerBlock=256, NPerBlock=32, KPerBlock=64
//           8,   8,        // AK1, BK1
//          32,  32,        // MPerXDL, NPerXDL
//           2,   3,        // MXdlPerWave=2, NXdlPerWave=1
//                          //   → MWave=256/(2×32)=4, NWave=32/(1×32)=1, WaveSize=64 ✓ (Rule 1)
//         S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // A: AK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
//         S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // B: BK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
//         1, 1,
//         S<1, 32, 1, 8>,  // CDE: CShuffleTile=128×32; B=32/8=4, A=64; product=256 ✓ (Rule 6)
//         8,               // ScalarPerVector: NWave×NPerXDL=32, 32%8=0 ✓; K=96, 96%8=0 ✓
//         BF16,            // AComputeDataType (explicit to avoid unpack default complexity)
//         BF16,            // BComputeDataType
//         ck::LoopScheduler::Default,
//         1,               // NumGroupsToMerge
//         true,           // DoubleBuffer
//         SplitKFactor>;             // SplitKFactor = -1 (auto-deduce k_batch)

// C = 384, Split-K double buffer instance
// template <ck::index_t NDimSpatial, int SplitKFactor>
// using ConvFwdSplitK =
//     ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
//         NDimSpatial,
//         InputLayout<NDimSpatial>,
//         WeightLayout<NDimSpatial>,
//         ck::Tuple<>,
//         OutputLayout<NDimSpatial>,
//         InKernelDataType,
//         WeiKernelDataType,
//         AccDataType,
//         CShuffleDataType,
//         ck::Tuple<>,
//         OutKernelDataType,
//         InElementOp,
//         WeiElementOp,
//         OutElementOp,
//         //ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
//         ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
//         GemmSpec,
//         2,               // NumGemmKPrefetchStage
//         256,             // BlockSize
//         128, 192,  32,   // MPerBlock=256, NPerBlock=32, KPerBlock=64
//           8,   8,        // AK1, BK1
//          32,  32,        // MPerXDL, NPerXDL
//           2,   3,        // MXdlPerWave=2, NXdlPerWave=1
//                          //   → MWave=256/(2×32)=4, NWave=32/(1×32)=1, WaveSize=64 ✓ (Rule 1)
//         S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // A: AK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
//         S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // B: BK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
//         1, 1,
//         S<1, 32, 1, 8>,  // CDE: CShuffleTile=128×32; B=32/8=4, A=64; product=256 ✓ (Rule 6)
//         8,               // ScalarPerVector: NWave×NPerXDL=32, 32%8=0 ✓; K=96, 96%8=0 ✓
//         BF16,            // AComputeDataType (explicit to avoid unpack default complexity)
//         BF16,            // BComputeDataType
//         ck::LoopScheduler::Default,
//         1,               // NumGroupsToMerge
//         true,           // DoubleBuffer
//         SplitKFactor>;             // SplitKFactor = -1 (auto-deduce k_batch)

// C = 384, Split-K
template <ck::index_t NDimSpatial, int SplitKFactor>
using ConvFwdSplitK =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InputLayout<NDimSpatial>,
        WeightLayout<NDimSpatial>,
        ck::Tuple<>,
        OutputLayout<NDimSpatial>,
        InKernelDataType,
        WeiKernelDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>,
        OutKernelDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        //ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        GemmSpec,
        2,               // NumGemmKPrefetchStage
        256,             // BlockSize
        256, 192,  32,   // MPerBlock=256, NPerBlock=32, KPerBlock=64
          8,   8,        // AK1, BK1
         32,  32,        // MPerXDL, NPerXDL
          4,   3,        // MXdlPerWave=2, NXdlPerWave=1
                         //   → MWave=256/(2×32)=4, NWave=32/(1×32)=1, WaveSize=64 ✓ (Rule 1)
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // A: AK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,  // B: BK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
        1, 1,
        S<1, 64, 1, 4>,  // CDE: CShuffleTile=128×32; B=32/8=4, A=64; product=256 ✓ (Rule 6)
        8,               // ScalarPerVector: NWave×NPerXDL=32, 32%8=0 ✓; K=96, 96%8=0 ✓
        BF16,            // AComputeDataType (explicit to avoid unpack default complexity)
        BF16,            // BComputeDataType
        ck::LoopScheduler::Default,
        1,               // NumGroupsToMerge
        true,           // DoubleBuffer
        SplitKFactor>;             // SplitKFactor = -1 (auto-deduce k_batch)

/*
// This instance is good for cases where K=3 and C=96
template <ck::index_t NDimSpatial, int SplitKFactor>
using ConvFwdSplitK =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InputLayout<NDimSpatial>,
        WeightLayout<NDimSpatial>,
        ck::Tuple<>,
        OutputLayout<NDimSpatial>,
        InKernelDataType,
        WeiKernelDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>,
        OutKernelDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
        //ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        GemmSpec,
        1,               // NumGemmKPrefetchStage
        256,             // BlockSize
        256, 96,  32,   // MPerBlock=256, NPerBlock=32, KPerBlock=64
          1,   1,        // AK1, BK1
         32,  32,        // MPerXDL, NPerXDL
          4,   3,        // MXdlPerWave=2, NXdlPerWave=1
                         //   → MWave=256/(2×32)=4, NWave=32/(1×32)=1, WaveSize=64 ✓ (Rule 1)
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 1, 1,  // A: AK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 1, 1,  // B: BK0=64/8=8, S<8,32,1>=256 ✓ (Rule 2)
        1, 1,
        S<1, 64, 1, 4>,  // CDE: CShuffleTile=128×32; B=32/8=4, A=64; product=256 ✓ (Rule 6)
        8,               // ScalarPerVector: NWave×NPerXDL=32, 32%8=0 ✓; K=96, 96%8=0 ✓
        BF16,            // AComputeDataType (explicit to avoid unpack default complexity)
        BF16,            // BComputeDataType
        ck::LoopScheduler::Default,
        1,               // NumGroupsToMerge
        false,           // DoubleBuffer
        SplitKFactor>;             // SplitKFactor = -1 (auto-deduce k_batch)
*/

template <ck::index_t NDimSpatial>
using HostConvFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                         InUserDataType,
                                                                         WeiUserDataType,
                                                                         CShuffleDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         PassThrough>;

// Default: G=1, N=1, K=96, C=96, 3×3×3 filter, stride=1, no pad.
// GEMM-K = 96×3×3×3 = 2592 — large enough to benefit from split-K.
// Input spatial: D=6, H=1106, W=834
#undef DefaultConvParam
#define DefaultConvParam                                                                           \
    ck::utils::conv::ConvParam                                                                    \
    {                                                                                             \
        3, 1, 1, 96, 96, {3, 3, 3}, {6, 1106, 834}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, { 0, 0, 0 } \
    }

template <ck::index_t NDimSpatial, int SplitKFactor>
bool run_grouped_conv_fwd_splitk(const ExecutionConfig& config,
                                 const ck::utils::conv::ConvParam& conv_param)
{
    static_assert(NDimSpatial == 3, "This example only supports 3D spatial convolutions");

    const auto in_g_n_c_wis_desc  = make_input_descriptor(conv_param);
    const auto wei_g_k_c_xs_desc  = make_weight_descriptor(conv_param);
    const auto out_g_n_k_wos_desc = make_output_descriptor(conv_param);

    Tensor<InUserDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiUserDataType> wei(wei_g_k_c_xs_desc);
    Tensor<OutUserDataType> out_host(out_g_n_k_wos_desc);
    Tensor<OutKernelDataType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InUserDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiUserDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InUserDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiUserDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InKernelDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiKernelDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutKernelDataType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    auto conv     = ConvFwdSplitK<NDimSpatial, SplitKFactor>{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      out_device_buf.GetDeviceBuffer(),
                                      a_g_n_c_wis_lengths,
                                      a_g_n_c_wis_strides,
                                      b_g_k_c_xs_lengths,
                                      b_g_k_c_xs_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      e_g_n_k_wos_lengths,
                                      e_g_n_k_wos_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementOp{},
                                      WeiElementOp{},
                                      OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    out_device_buf.SetZero();

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InUserDataType, WeiUserDataType, OutUserDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
              << " GB/s, " << conv.GetTypeString() << std::endl;

    if(config.do_verification)
    {
        // Use GPU reference with GPU verification
        std::cout << "Using GPU reference with GPU verification" << std::endl;

        // Allocate GPU reference buffer
        DeviceMem gpu_ref_out_buf(sizeof(OutKernelDataType) * out_device.mDesc.GetElementSpaceSize());

        // Call GPU reference with ConvParam directly
        ck::ref::naive_conv_fwd<InputLayout<NDimSpatial>,
                            WeightLayout<NDimSpatial>,
                            OutputLayout<NDimSpatial>,
                            InKernelDataType,
                            WeiKernelDataType,
                            OutKernelDataType,
                            InElementOp,
                            WeiElementOp,
                            OutElementOp>(
            reinterpret_cast<const InKernelDataType*>(in_device_buf.GetDeviceBuffer()),
            reinterpret_cast<const WeiKernelDataType*>(wei_device_buf.GetDeviceBuffer()),
            reinterpret_cast<OutKernelDataType*>(gpu_ref_out_buf.GetDeviceBuffer()),
            conv_param,
            InElementOp{},
        WeiElementOp{},
        OutElementOp{});

        // GPU verification path
        // Calculate number of accumulations (C * filter spatial dimensions)
        std::size_t filter_spatial_size = 1;
        for(auto len : conv_param.filter_spatial_lengths_)
        {
            filter_spatial_size *= len;
        }
        const int num_accums = static_cast<int>(conv_param.C_ * filter_spatial_size);

        // Perform GPU verification (max value computed internally on GPU)
        const std::size_t tensor_size = out_device.mDesc.GetElementSpaceSize();
        auto gpu_result = ck::profiler::gpu_verify<OutKernelDataType, BF16, OutKernelDataType>(
            out_device_buf.GetDeviceBuffer(),
            gpu_ref_out_buf.GetDeviceBuffer(),
            num_accums,
            tensor_size);

        if(!gpu_result)
        {
            // GPU verification failed - print detailed error summary
            gpu_result.print_error_summary();
            return false;
        }
        else 
        {
            std::cout << "Verification PASSED\n";
        }
    }

    return true;
}

bool run_grouped_conv3d_fwd_splitk_example(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return false;
    }

    // Last param is the split-K value.
    const int split_k = (argc > 26) ? std::stoi(argv[argc - 1]) : -1;

    if(conv_param.num_dim_spatial_ != 3)
    {
        std::cerr << "This example only supports 3D spatial convolutions\n";
        return false;
    }

    //return run_grouped_conv_fwd_splitk<3, 1>(config, conv_param);

    if (split_k == -1)
    {
        std::cout << "Using auto-deduced Split-K factor based on GPU occupancy heuristic\n";
        return run_grouped_conv_fwd_splitk<3, -1>(config, conv_param);
    }
    else if (split_k == 1)
    {
        return run_grouped_conv_fwd_splitk<3, 1>(config, conv_param);
    }
    else if (split_k == 2)
    {
        std::cout << "Using Split-K factor of 2\n";
        return run_grouped_conv_fwd_splitk<3, 2>(config, conv_param);
    }
    else if (split_k == 4)
    {
        std::cout << "Using Split-K factor of 4\n";
        return run_grouped_conv_fwd_splitk<3, 4>(config, conv_param);
    }
    else if (split_k == 8)
    {
        std::cout << "Using Split-K factor of 8\n";
        return run_grouped_conv_fwd_splitk<3, 8>(config, conv_param);
    }
    else if (split_k == 16)
    {
        std::cout << "Using Split-K factor of 16\n";
        return run_grouped_conv_fwd_splitk<3, 16>(config, conv_param);
    }
    else
    {
        std::cerr << "Invalid Split-K factor specified. Supported values: 1, 2, 4, 8, 16, or -1 for auto-deduce.\n";
        return false;
    }

    
}

int main(int argc, char* argv[]) { return !run_grouped_conv3d_fwd_splitk_example(argc, argv); }
