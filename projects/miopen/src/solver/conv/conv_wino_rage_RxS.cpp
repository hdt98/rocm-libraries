/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#define CONV_WINO_RAGE_RXS_CPP

#include <cstdint>
#include <cstdlib>
#include <miopen/kernel_build_params.hpp>
#include <miopen/conv/invokers/gcn_asm_wino.hpp>
#include <miopen/conv/kernel_interface/winograd_kernel_interface.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/utils.hpp>
#include <miopen/solver/solver_utils.hpp>

namespace miopen {

namespace solver {

using ProblemDescription = miopen::conv::ProblemDescription;
using WinoShaderArgs     = miopen::WinoShaderArgsV2;

namespace {

// Divide two non-negative integers and return ceil of the quotient
constexpr uint64_t DivCeil(uint64_t numer, uint64_t denom) { return (numer + denom - 1) / denom; }

constexpr uint64_t maxNGroups = WinoShaderArgs::PowOf2<16>() - 1;

template <uint32_t Winodata, uint32_t Winofilter>
struct ConvWinoRageRxSCommon
{
    static bool IsApplicable(const ExecutionContext&, const ProblemDescription&);
    static float GetWti(const ExecutionContext&, const ProblemDescription&);
    static ConvSolution GetSolution(const ExecutionContext&,
                                    const ProblemDescription&,
                                    bool fused                        = false,
                                    bool do_bias                      = false,
                                    miopenActivationMode_t activ_mode = miopenActivationPASTHRU);

private:
    static int64_t getNGroups(const ExecutionContext& ctx)
    {
        return std::min(ctx.GetStream().GetMaxHardwareComputeUnits(), maxNGroups);
    }
};

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoRageRxSCommon<Winodata, Winofilter>::IsApplicable(const ExecutionContext& ctx,
                                                               const ProblemDescription& problem)
{
    std::string context = "ConvWinoRageRxSCommon"; // for logging

    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(
        !ctx.use_asm_kernels, context, inapplicable_msg::UseAsmKernels);

    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(
        problem.IsTensorsCasted(), context, inapplicable_msg::IsTensorsCasted);

    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(!problem.IsFp16(), context, inapplicable_msg::DataType);

    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(
        problem.HasNonPackedTensors(), context, inapplicable_msg::HasNonPackedTensors);

    const auto devName = ctx.GetStream().GetDeviceName();
    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(
        !(devName == "gfx942"), context, inapplicable_msg::UnsupportedDevice);

    const auto& target = ctx.GetStream().GetTargetProperties();
    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(
        target.isXnackEnabled(), context, inapplicable_msg::isXnackEnabled);

    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(
        !(problem.GetKernelStrideH() == 1 && problem.GetKernelStrideW() == 1),
        context,
        "Kernel stride must be 1.");

    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(
        !(problem.GetDilationH() == 1 && problem.GetDilationW() == 1),
        context,
        "Dilation must be 1.");

    WinoShaderArgs args;
    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(
        !args.SetConvParams(problem), context, "Convolution parameters not set.");

    args.n_groups = getNGroups(ctx);

    // clang-format off
    MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(!(args.dimsFit16bit()
        && args.R_S_fit16bit()
        && args.batchTensorSizesFit31bits()
        && args.paddedSizesFit16bits()
        && DivCeil(args.Kg, 32) <= args.n_groups), context, inapplicable_msg::NoKernelForConfig);
    // clang-format on
    return true;
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoRageRxSCommon<Winodata, Winofilter>::GetWti(const ExecutionContext& ctx,
                                                          const ProblemDescription& problem)
{
    std::ignore = ctx;
    std::ignore = problem;
    return -2.0f;
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoRageRxSCommon<Winodata, Winofilter>::GetSolution(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem,
                                                         bool fused,
                                                         bool do_bias,
                                                         miopenActivationMode_t activ_mode)
{
    // Kernel args

    WinoShaderArgsV2 args;
    if(!args.SetConvParams(problem))
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }
    args.SetStrides(problem);
    args.SetActivParams(activ_mode);

    auto flags = WinoShaderFlagsV2::F_NKCHR_STRIDES | WinoShaderFlagsV2::F_TENSOR_OFFSETS |
                 WinoShaderFlagsV2::F_USE_ACTIVATION_MODE |
                 WinoShaderFlagsV2::F_DENORMS_RND_ENABLE |
                 WinoShaderFlagsV2::F_USE_EXTENDED_FLAGS_64;
    if(args.G != 1)
        flags |= WinoShaderFlagsV2::F_GROUPED_CONVOLUTION;
    if(problem.IsDirectionBackwardData())
        flags |= WinoShaderFlagsV2::F_REVERSE_R | WinoShaderFlagsV2::F_REVERSE_S;
    if(do_bias)
        flags |= WinoShaderFlagsV2::F_BIAS;

    auto nGroups = getNGroups(ctx);
    args.SetShaderParams(nGroups, flags, 0, 0);

    // Kernel name and file

    std::string kernelVersion;
    if(args.R_S_fit3x3())
    {
        kernelVersion = "_v4_6_0";
    }
    else
    {
        kernelVersion = "_v4_7_0";
    }
    std::string kernelName = "miopenSp3AsmConvRage" + kernelVersion;
    std::string kernelFile = "Conv_Winograd_Rage" + kernelVersion;

    /// \todo Add case for gfx12 kernels
    kernelName += "_gfx9";

    std::string kernelPostfix;

    if(problem.IsFp16())
    {
        kernelPostfix += "_fp16_fp32acc";
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    if constexpr(Winodata == 2 && Winofilter == 3)
    {
        kernelPostfix += "_f2x3";
    }
    else
    {
        static_assert(Winodata == 2 && Winofilter == 3);
    }

    kernelPostfix += "_stride1";

    kernelName += kernelPostfix;
    kernelFile += kernelPostfix;
    kernelFile += ".s";

    // Kernel info

    KernelInfo kernelInfo;

    /// Kernel doesn't need ROCM_METADATA_VERSION, but AmdgcnAssemble()
    /// uses it to find out required CO version (hack).
    /// \todo Delete when COv2 support is removed.
    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernelInfo.comp_options = options.GenerateFor(kbp::GcnAsm{});
    kernelInfo.comp_options += std::string(" -mcumode -mwavefrontsize64");

    /// \todo Add case for gfx12 wgp size of 384U
    uint64_t wgSize = 768U;

    kernelInfo.l_wk.push_back(wgSize);
    kernelInfo.l_wk.push_back(1);
    kernelInfo.l_wk.push_back(1);

    kernelInfo.g_wk.push_back(wgSize * nGroups * args.G);
    kernelInfo.g_wk.push_back(1);
    kernelInfo.g_wk.push_back(1);

    kernelInfo.kernel_file = kernelFile;
    kernelInfo.kernel_name = kernelName;

    // Solution

    ConvSolution result;
    result.construction_params.push_back(kernelInfo);
    result.invoker_factory =
        miopen::MakeGcnAsmWinoV2InvokerFactory(args, problem.GetDirection(), 0U, fused);
    result.workspace_sz = 0U;

    return result;
}

} // namespace

namespace conv {

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoRageRxS<Winodata, Winofilter>::IsApplicable(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem) const
{
    return ConvWinoRageRxSCommon<Winodata, Winofilter>::IsApplicable(ctx, problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoRageRxS<Winodata, Winofilter>::GetWti(const ExecutionContext& ctx,
                                                    const ProblemDescription& problem) const
{
    return ConvWinoRageRxSCommon<Winodata, Winofilter>::GetWti(ctx, problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoRageRxS<Winodata, Winofilter>::GetSolution(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    return ConvWinoRageRxSCommon<Winodata, Winofilter>::GetSolution(ctx, problem);
}

template struct MIOPEN_INTERNALS_EXPORT ConvWinoRageRxS<2, 3>;

} // namespace conv

namespace fusion {

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoRageRxSFused<Winodata, Winofilter>::IsApplicable(
    const FusionContext& ctx, const FusionDescription& problem) const
{
    const auto& desc = *problem.fusion_plan_desc;

    if(desc.op_map.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    MIOPEN_SOLVER_INAPPLICABLE_IF((desc.op_map.size() > 3), "Too many ops in fusion plan.");

    MIOPEN_SOLVER_INAPPLICABLE_IF((desc.op_map[0]->kind() != miopenFusionOpConvForward),
                                  inapplicable_msg::NotFWDFusion);

    if(desc.op_map.size() == 2)
    {
        const auto prim = desc.op_map[1]->kind();
        MIOPEN_SOLVER_INAPPLICABLE_IF(
            !(prim == miopenFusionOpBiasForward || prim == miopenFusionOpActivForward),
            "Second op is neither bias nor activation.");
    }
    if(desc.op_map.size() == 3)
    {
        MIOPEN_SOLVER_INAPPLICABLE_IF((desc.op_map[1]->kind() != miopenFusionOpBiasForward),
                                      "Second op is not bias.");
        MIOPEN_SOLVER_INAPPLICABLE_IF((desc.op_map[2]->kind() != miopenFusionOpActivForward),
                                      "Third op is not activation.");
    }

    const int activ_idx = GetOpIdx(desc.op_map, miopenFusionOpActivForward);
    if(activ_idx != -1)
    {
        const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
        switch(activ_op.activMode)
        {
        case miopenActivationPASTHRU:
        case miopenActivationLOGISTIC:
        case miopenActivationTANH:
        case miopenActivationRELU:
        case miopenActivationLEAKYRELU: break;
        default: MIOPEN_SOLVER_INAPPLICABLE_IF(true, "Unsupported activation mode.");
        }
    }

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    MIOPEN_SOLVER_INAPPLICABLE_IF(
        !(ConvWinoRageRxSCommon<Winodata, Winofilter>::IsApplicable(ctx, conv_problem)),
        inapplicable_msg::NoKernelForConfig);
    return true;
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoRageRxSFused<Winodata, Winofilter>::GetWti(const FusionContext& ctx,
                                                         const FusionDescription& problem) const
{
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    return ConvWinoRageRxSCommon<Winodata, Winofilter>::GetWti(ctx, conv_problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoRageRxSFused<Winodata, Winofilter>::GetSolution(const FusionContext& ctx,
                                                        const FusionDescription& problem) const
{
    const auto& desc    = *problem.fusion_plan_desc;
    const int bias_idx  = GetOpIdx(desc.op_map, miopenFusionOpBiasForward);
    const int activ_idx = GetOpIdx(desc.op_map, miopenFusionOpActivForward);

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);

    const bool do_bias = (bias_idx != -1);
    auto activ_mode    = miopenActivationPASTHRU;
    if(activ_idx != -1)
    {
        const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
        activ_mode           = activ_op.activMode;
    }

    return ConvWinoRageRxSCommon<Winodata, Winofilter>::GetSolution(
        ctx, conv_problem, true, do_bias, activ_mode);
}

template struct MIOPEN_INTERNALS_EXPORT ConvWinoRageRxSFused<2, 3>;

} // namespace fusion

} // namespace solver

} // namespace miopen
