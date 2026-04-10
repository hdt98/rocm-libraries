#include <miopen/conv/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/tensor_ops.hpp>

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

    // The wgrad kernel outputs fp32.
    if(par.direction == hipconv::Direction::Wgrad)
        par.out_type = hipconv::DataType::fp32;

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
    const auto arch = hipconv::parse_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        return false;

    // The wgrad kernel uses atomicAdd and is non-deterministic.
    if(problem.IsDirectionBackwardWrW() && problem.GetConv().attribute.deterministic)
        return false;

    // Initially only fp16 is supported.
    if(!problem.IsFp16())
        return false;

    const auto par = ToHipconvParams(problem);
    const auto cfg = hipconv::find_config(*arch, par);
    return cfg.has_value();
}

size_t ConvHipConv::GetWorkspaceSize(const ExecutionContext&,
                                     const ProblemDescription& problem) const
{
    if(!problem.IsDirectionBackwardWrW())
        return 0;

    // The wgrad kernel outputs fp32, but MIOpen expects dw in the same type as the weights (fp16).
    // We need a fp32 workspace to hold the kernel output, then cast to fp16.
    const auto k           = ProblemInterpreter::GetOutputChannelK(problem);
    const auto c           = ProblemInterpreter::GetInputChannelC(problem);
    const auto y           = ProblemInterpreter::GetFilterHeightY(problem);
    const auto x           = ProblemInterpreter::GetFilterWidthX(problem);
    const auto group       = ProblemInterpreter::GetGroupCountG(problem);
    const auto c_per_group = c / group;
    return static_cast<size_t>(k) * y * x * c_per_group * sizeof(float);
}

ConvSolution ConvHipConv::GetSolution(const ExecutionContext& ctx,
                                      const ProblemDescription& problem) const
{
    ConvSolution result;

    const auto arch = hipconv::parse_arch(ctx.GetStream().GetDeviceName()).value();
    const auto par  = ToHipconvParams(problem);
    const auto cfg  = hipconv::find_config(arch, par).value();

    if(problem.IsDirectionBackwardWrW())
    {
        const auto workspace_size = GetWorkspaceSize(ctx, problem);
        result.workspace_sz       = workspace_size;

        const auto lowp_quant = problem.GetConv().lowp_quant;

        // Build a TensorDescriptor for the fp32 intermediate buffer (same shape as weights).
        const TensorDescriptor cast_desc(
            miopenFloat, problem.GetWeights().GetLengths(), problem.GetWeights().GetStrides());

        result.invoker_factory = [=](const std::vector<Kernel>&) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) wrw_ctx =
                    primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
                const auto& tensors       = wrw_ctx.tensors;
                const auto& workSpace     = wrw_ctx.workSpace;
                const auto& workSpaceSize = wrw_ctx.workSpaceSize;

                if(workSpace == nullptr || workSpaceSize < workspace_size)
                    MIOPEN_THROW("Not enough workspace for ConvHipConv wgrad.");

                {
                    const HipEventProfiler profiler(handle);

                    // Launch the hipconv wgrad kernel writing fp32 into workspace.
                    hipconv::launch(arch,
                                    cfg,
                                    par,
                                    tensors.x,
                                    tensors.dy,
                                    workSpace,
                                    nullptr,
                                    handle.GetStream());

                    // Cast fp32 workspace -> fp16 dw.
                    CastTensor(handle,
                               &lowp_quant,
                               false,
                               cast_desc,
                               workSpace,
                               tensors.dwDesc,
                               tensors.dw,
                               0,
                               0);
                }
            };
        };
    }
    else
    {
        result.invoker_factory = [=](const std::vector<Kernel>&) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx =
                    primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
                const auto& tensors = data_ctx.tensors;

                {
                    const HipEventProfiler profiler(handle);
                    hipconv::launch(arch,
                                    cfg,
                                    par,
                                    tensors.in,
                                    tensors.w,
                                    tensors.out,
                                    nullptr,
                                    handle.GetStream());
                }
            };
        };
    }

    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
