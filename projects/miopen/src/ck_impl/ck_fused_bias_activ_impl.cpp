// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <cassert>

#include "ck_grouped_conv_common.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>

#include <ck/tensor_operation/gpu/device/device_conv_fwd_bias_activation.hpp>

// ---------------------------------------------------------------------------
// CK type aliases for fused bias+ReLU convolution forward
// ---------------------------------------------------------------------------

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

using DeviceConvFwdBiasReluPtr = ck::tensor_operation::device::DeviceConvFwdBiasActivationPtr<
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::AddRelu>;

// Forward declare CK's instance function.
namespace ck_inst = ck::tensor_operation::device::instance;

void GetInstances(std::vector<DeviceConvFwdBiasReluPtr>& ptrs)
{
    ck_inst::add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(ptrs);
}

// ---------------------------------------------------------------------------
// CKArgs — extracts convolution dimensions for CK argument construction.
// ---------------------------------------------------------------------------
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        N        = ProblemInterpreter::GetBatchN(problem);
        K        = ProblemInterpreter::GetOutputChannelK(problem);
        C        = ProblemInterpreter::GetInputChannelC(problem);
        input    = {ProblemInterpreter::GetInputHeightHi(problem),
                    ProblemInterpreter::GetInputWidthWi(problem)};
        output   = {ProblemInterpreter::GetOutputHeightHo(problem),
                    ProblemInterpreter::GetOutputWidthWo(problem)};
        filter   = {ProblemInterpreter::GetFilterHeightY(problem),
                    ProblemInterpreter::GetFilterWidthX(problem)};
        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }
    int N;
    int K;
    int C;
    std::vector<int> input;
    std::vector<int> output;
    std::vector<int> filter;
    std::vector<int> strides;
    std::vector<int> dilation;
    std::vector<int> lPadding;
    std::vector<int> rPadding;
};

} // anonymous namespace

// ===========================================================================
// Fused bias+ReLU direction functions
// ===========================================================================

extern "C" CKKernelListHandle* ckgrpconv_fused_biasrelu_fill_valid_kernels(
    const miopen::conv::ProblemDescription* problem,
    miopenDataType_t /*data_type*/,
    bool /*use_tf32*/)
{
    try
    {
        if(!problem)
            return nullptr;

        const auto& args = CKArgs{*problem};
        std::vector<DeviceConvFwdBiasReluPtr> conv_ptrs;
        GetInstances(conv_ptrs);
        assert(!conv_ptrs.empty());

        auto result = std::make_unique<CKKernelListHandle>();

        for(const auto& it : conv_ptrs)
        {
            auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        args.N,
                                                        args.K,
                                                        args.C,
                                                        args.input,
                                                        args.filter,
                                                        args.output,
                                                        args.strides,
                                                        args.dilation,
                                                        args.lPadding,
                                                        args.rPadding,
                                                        {},
                                                        {},
                                                        {});
            if(it->IsSupportedArgument(argument_ptr.get()))
            {
                result->kernels.push_back(it->GetTypeString());
            }
        }

        return result.release();
    }
    catch(...)
    {
        return nullptr;
    }
}

extern "C" bool ckgrpconv_fused_biasrelu_is_applicable(
    const miopen::conv::ProblemDescription* problem,
    miopenDataType_t /*data_type*/,
    bool /*use_tf32*/)
{
    try
    {
        if(!problem)
            return false;

        std::vector<DeviceConvFwdBiasReluPtr> conv_ptrs;
        GetInstances(conv_ptrs);
        assert(!conv_ptrs.empty());

        const auto& args = CKArgs{*problem};
        for(const auto& it : conv_ptrs)
        {
            auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        args.N,
                                                        args.K,
                                                        args.C,
                                                        args.input,
                                                        args.filter,
                                                        args.output,
                                                        args.strides,
                                                        args.dilation,
                                                        args.lPadding,
                                                        args.rPadding,
                                                        {},
                                                        {},
                                                        {});
            if(it->IsSupportedArgument(argument_ptr.get()))
                return true;
        }
        return false;
    }
    catch(...)
    {
        return false;
    }
}

extern "C" bool ckgrpconv_fused_biasrelu_is_args_supported(
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
        const auto& args = CKArgs{*problem};
        std::vector<DeviceConvFwdBiasReluPtr> conv_ptrs;
        GetInstances(conv_ptrs);

        for(const auto& it : conv_ptrs)
        {
            if(it->GetTypeString() == kid)
            {
                auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            args.N,
                                                            args.K,
                                                            args.C,
                                                            args.input,
                                                            args.filter,
                                                            args.output,
                                                            args.strides,
                                                            args.dilation,
                                                            args.lPadding,
                                                            args.rPadding,
                                                            {},
                                                            {},
                                                            {});
                return it->IsSupportedArgument(argument_ptr.get());
            }
        }
        return false;
    }
    catch(...)
    {
        return false;
    }
}

extern "C" size_t
ckgrpconv_fused_biasrelu_get_workspace_size(const miopen::conv::ProblemDescription* /*problem*/,
                                            miopenDataType_t /*data_type*/)
{
    return 0;
}

extern "C" miopen::solver::ConvSolution*
ckgrpconv_fused_biasrelu_get_solution(const miopen::ExecutionContext* /*ctx*/,
                                      const miopen::conv::ProblemDescription* problem,
                                      const char* kernel_id,
                                      bool /*use_tf32*/)
{
    try
    {
        if(!problem || !kernel_id)
            return nullptr;

        const std::string kid(kernel_id);
        const auto args = CKArgs{*problem};

        auto sol = std::make_unique<miopen::solver::ConvSolution>();

        sol->invoker_factory = [kid, args](const std::vector<Kernel>&) {
            return [kid, args](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                std::vector<DeviceConvFwdBiasReluPtr> conv_ptrs;
                GetInstances(conv_ptrs);

                int id = 0;
                for(; id < static_cast<int>(conv_ptrs.size()); id++)
                {
                    if(conv_ptrs[id]->GetTypeString() == kid)
                    {
                        break;
                    }
                }
                assert(id < static_cast<int>(conv_ptrs.size()));
                auto& conv_ck = conv_ptrs.at(id);

                const auto& invoke_ctx =
                    primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
                const auto& wei_buf =
                    dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(
                        *invoke_ctx.op_args.params[0])
                        .weights;
                const auto& bias_buf =
                    dynamic_cast<miopen::fusion::BiasOpInvokeParam&>(*invoke_ctx.op_args.params[1])
                        .bdata;

                auto argument_ptr = conv_ck->MakeArgumentPointer(
                    const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
                        static_cast<const void*>(invoke_ctx.in)),
                    const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
                        static_cast<const void*>(wei_buf)),
                    invoke_ctx.out,
                    const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
                        static_cast<const void*>(bias_buf)),
                    args.N,
                    args.K,
                    args.C,
                    args.input,
                    args.filter,
                    args.output,
                    args.strides,
                    args.dilation,
                    args.lPadding,
                    args.rPadding,
                    {},
                    {},
                    {});
                auto invoker_ptr            = conv_ck->MakeInvokerPointer();
                const auto enable_profiling = handle.IsProfilingEnabled();

                float elapsed_time =
                    invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
                if(enable_profiling)
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed_time);
                }
            };
        };

        return sol.release();
    }
    catch(...)
    {
        return nullptr;
    }
}
