// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

#include "ck_grouped_conv_common.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>

#include "ck/ck.hpp"
#include "miopen/conv/device_grouped_conv_fwd.hpp"

// ---------------------------------------------------------------------------
// CK type aliases for depthwise convolution forward
// ---------------------------------------------------------------------------

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

template <ck::index_t... Is>
using S                           = ck::Sequence<Is...>;
using InElementOp                 = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp                = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp                = ck::tensor_operation::element_wise::PassThrough;
using InType                      = ck::half_t;
using WeiType                     = ck::half_t;
using AccType                     = float;
using OutType                     = ck::half_t;
constexpr ck::index_t NDimSpatial = 2;
constexpr ck::index_t BlockSize   = 64;
constexpr bool RequirePadding     = false;

// Tuple of potential device CK kernels. Shapes taken to target fp16 Pytorch EfficientNet B0 model:
// https://docs.pytorch.org/vision/main/models/efficientnet.html
using DeviceConvFwdFactory =
    std::tuple<ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<7, 7>,                              // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   1,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<14, 14>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<28, 28>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   4,  // InScalarPerVector
                   4,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<14, 14>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<28, 28>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   4,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<56, 56>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8, // NBatch
                   4, // SubTileH
                   4, // SubTileW
                   8, // InScalarPerVector
                   4, // OutScalarPerVector
                   RequirePadding>

               ,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<7, 7>,                              // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   1,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<14, 14>,                            // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<56, 56>,                            // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8, // NBatch
                   7, // SubTileH
                   8, // SubTileW
                   8, // InScalarPerVector
                   8, // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<112, 112>,                          // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   2,  // NBatch
                   14, // SubTileH
                   16, // SubTileW
                   8,  // InScalarPerVector
                   8,  // OutScalarPerVector
                   RequirePadding>

               ,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<28, 28>,                            // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   4,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<112, 112>,                          // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8, // NBatch
                   7, // SubTileH
                   8, // SubTileW
                   8, // InScalarPerVector
                   8, // OutScalarPerVector
                   RequirePadding>>;

// ---------------------------------------------------------------------------
// CKArgs — extracts convolution dimensions for CK argument construction.
// ---------------------------------------------------------------------------
struct CKArgs
{
    explicit CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channel per group
        K  = K1 / G; // Number of output Channel per group
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);

        input_lengths = {G, N, C, Hi, Wi}; // input
        out_lens      = {G, N, K, Ho, Wo}; // output
        wei_lens      = {G, K, C, Y, X};   // filter = wei
        bias_lens     = {G, 1, K, 1, 1};
        bias_strides  = {K, 0, 1, 0, 0};

        in_strides  = {Hi * Wi * C, G * Hi * Wi * C, 1, Wi * C, C};
        out_strides = {Ho * Wo * K, G * Ho * Wo * K, 1, Wo * K, K};
        wei_strides = {Y * X * C, G * Y * X * C, 1, X * C, C};

        filter_stride   = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding        = {ProblemInterpreter::GetInputLeftPadH(problem),
                           ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding        = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                           ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&)            = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgument(const ConvPtr& conv_ptr, ConstData_t in, ConstData_t w, Data_t out) const
    {
        return conv_ptr.MakeArgument(in,
                                     w,
                                     std::array<const void*, 0>{},
                                     out,
                                     input_lengths,
                                     in_strides,
                                     wei_lens,
                                     wei_strides,
                                     std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                     std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                     out_lens,
                                     out_strides,
                                     filter_stride,
                                     filter_dilation,
                                     lPadding,
                                     rPadding,
                                     InElementOp{},
                                     WeiElementOp{},
                                     OutElementOp{});
    }

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    std::array<ck::index_t, 5> input_lengths;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> out_lens;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> wei_lens;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 5> bias_lens;
    std::array<ck::index_t, 5> bias_strides;
    std::array<ck::index_t, 2> filter_stride;
    std::array<ck::index_t, 2> filter_dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};

} // anonymous namespace

// ===========================================================================
// Depthwise FWD direction functions
// ===========================================================================

extern "C" CKKernelListHandle* ckgrpconv_depthwise_fwd_fill_valid_kernels(
    const miopen::conv::ProblemDescription* problem,
    miopenDataType_t /*data_type*/,
    bool /*use_tf32*/)
{
    try
    {
        if(!problem)
            return nullptr;

        const auto& ck_args            = CKArgs{*problem};
        constexpr uint32_t kernelCount = std::tuple_size_v<DeviceConvFwdFactory>;

        auto result = std::make_unique<CKKernelListHandle>();

        ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
            const auto conv_ptr = std::get<i>(DeviceConvFwdFactory{});
            auto argument       = ck_args.MakeArgument(conv_ptr, nullptr, nullptr, nullptr);
            if(conv_ptr.IsSupportedArgument(argument))
            {
                result->kernels.push_back(std::move(conv_ptr.GetTypeString()));
            }
        });

        return result.release();
    }
    catch(...)
    {
        return nullptr;
    }
}

extern "C" bool ckgrpconv_depthwise_fwd_is_applicable(
    const miopen::conv::ProblemDescription* problem,
    miopenDataType_t /*data_type*/,
    bool /*use_tf32*/)
{
    try
    {
        if(!problem)
            return false;

        const auto& ck_args            = CKArgs{*problem};
        constexpr uint32_t kernelCount = std::tuple_size_v<DeviceConvFwdFactory>;
        bool found                     = false;

        ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
            if(found)
                return;
            const auto conv_ptr = std::get<i>(DeviceConvFwdFactory{});
            auto argument       = ck_args.MakeArgument(conv_ptr, nullptr, nullptr, nullptr);
            if(conv_ptr.IsSupportedArgument(argument))
            {
                found = true;
            }
        });

        return found;
    }
    catch(...)
    {
        return false;
    }
}

extern "C" bool ckgrpconv_depthwise_fwd_is_args_supported(
    const miopen::conv::ProblemDescription* problem,
    const char* kernel_id,
    miopenDataType_t /*data_type*/,
    bool /*use_tf32*/)
{
    try
    {
        if(!problem || !kernel_id)
            return false;

        const std::string kid(kernel_id);
        const auto& ck_args            = CKArgs{*problem};
        constexpr uint32_t kernelCount = std::tuple_size_v<DeviceConvFwdFactory>;
        bool found                     = false;

        ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
            if(found)
                return;
            const auto conv_ptr = std::get<i>(DeviceConvFwdFactory{});
            if(conv_ptr.GetTypeString() == kid)
            {
                auto argument = ck_args.MakeArgument(conv_ptr, nullptr, nullptr, nullptr);
                if(conv_ptr.IsSupportedArgument(argument))
                {
                    found = true;
                }
            }
        });

        return found;
    }
    catch(...)
    {
        return false;
    }
}

extern "C" size_t
ckgrpconv_depthwise_fwd_get_workspace_size(const miopen::conv::ProblemDescription* /*problem*/,
                                           miopenDataType_t /*data_type*/)
{
    return 0;
}

extern "C" miopen::solver::ConvSolution*
ckgrpconv_depthwise_fwd_get_solution(const miopen::ExecutionContext* /*ctx*/,
                                     const miopen::conv::ProblemDescription* problem,
                                     const char* kernel_id,
                                     bool /*use_tf32*/)
{
    try
    {
        if(!problem || !kernel_id)
            return nullptr;

        const std::string kid(kernel_id);
        auto sol = std::make_unique<miopen::solver::ConvSolution>();

        ck::static_for<0, std::tuple_size_v<DeviceConvFwdFactory>, 1>{}([&](auto i) -> void {
            const auto device_conv_fwd_instance = std::get<i>(DeviceConvFwdFactory{});
            using DeviceConvFwdInstance = ck::remove_cvref_t<decltype(device_conv_fwd_instance)>;
            auto conv_ptr               = std::make_shared<DeviceConvFwdInstance>();

            if(conv_ptr->GetTypeString() == kid)
            {
                sol->invoker_factory = [conv_ptr = std::move(conv_ptr),
                                         ck_args  = CKArgs{*problem}](const std::vector<Kernel>&) {
                    return [conv_ptr = std::move(conv_ptr),
                            ck_args](const Handle& handle, const AnyInvokeParams& primitive_params) {
                        const auto& fwd_ctx =
                            primitive_params.CastTo<miopen::conv::DataInvokeParams>();
                        auto invoker  = conv_ptr->MakeInvoker();
                        auto argument = ck_args.MakeArgument(*conv_ptr.get(),
                                                             fwd_ctx.tensors.in,
                                                             fwd_ctx.tensors.w,
                                                             fwd_ctx.tensors.out);

                        {
                            {
                                WorkAroundHipEventProfiler prf(handle);
                                invoker.Run(argument, StreamConfig{nullptr, false});
                            }
                            if(handle.IsProfilingEnabled())
                            {
                                float avg_time = handle.GetKernelTime();
                                handle.ResetKernelTime();
                                handle.AccumKernelTime(avg_time);
                            }
                        }
                    };
                };
            }
        });

        return sol.release();
    }
    catch(...)
    {
        return nullptr;
    }
}
