#include <miopen/conv/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>

#include <hipconv/hipconv.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_HIPCONV)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

static hipconv::Conv2dParams ToHipconvParams(const ProblemDescription& problem)
{
    hipconv::Conv2dParams par{};

    if(problem.IsDirectionForward())
        par.direction = hipconv::Direction::Fprop;
    else if(problem.IsDirectionBackwardData())
        par.direction = hipconv::Direction::Dgrad;
    else
        par.direction = hipconv::Direction::Wgrad;

    par.n  = ProblemInterpreter::GetBatchN(problem);
    par.c  = ProblemInterpreter::GetInputChannelC(problem);
    par.h  = ProblemInterpreter::GetInputHeightHi(problem);
    par.w  = ProblemInterpreter::GetInputWidthWi(problem);
    par.k  = ProblemInterpreter::GetOutputChannelK(problem);
    par.kh = ProblemInterpreter::GetFilterHeightY(problem);
    par.kw = ProblemInterpreter::GetFilterWidthX(problem);

    par.pad_h      = ProblemInterpreter::GetInputLeftPadH(problem);
    par.pad_w      = ProblemInterpreter::GetInputLeftPadW(problem);
    par.stride_h   = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    par.stride_w   = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    par.dilation_h = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    par.dilation_w = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    par.groups     = ProblemInterpreter::GetGroupCountG(problem);

    par.p = ProblemInterpreter::GetOutputHeightHo(problem);
    par.q = ProblemInterpreter::GetOutputWidthWo(problem);

    if(problem.IsFp16())
    {
        par.in_type  = hipconv::DataType::fp16;
        par.wei_type = hipconv::DataType::fp16;
        par.out_type = hipconv::DataType::fp16;
    }
    else if(problem.IsBfp16())
    {
        par.in_type  = hipconv::DataType::bf16;
        par.wei_type = hipconv::DataType::bf16;
        par.out_type = hipconv::DataType::bf16;
    }

    if(problem.IsLayoutNHWC())
        par.order = hipconv::TensorOrder::NHWC;
    else
        par.order = hipconv::TensorOrder::NCHW;

    return par;
}

bool ConvHipConv::IsApplicable(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_HIPCONV))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    if(!StartsWith(ctx.GetStream().GetDeviceName(), "gfx12") &&
       !StartsWith(ctx.GetStream().GetDeviceName(), "gfx9"))
        return false;

    // Initially only fp16 is supported.
    if(!problem.IsFp16())
        return false;

    const auto par = ToHipconvParams(problem);
    const auto cfg = hipconv::find_config(par);
    return cfg.has_value();
}

ConvSolution ConvHipConv::GetSolution(const ExecutionContext&,
                                      const ProblemDescription& problem) const
{
    ConvSolution result;

    const auto par = ToHipconvParams(problem);
    const auto cfg = hipconv::find_config(par).value();

    result.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;

            {
                const HipEventProfiler profiler(handle);
                hipconv::launch(cfg, par, tensors.in, tensors.w, tensors.out, handle.GetStream());
            }
        };
    };

    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
