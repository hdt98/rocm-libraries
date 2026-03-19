// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/conv/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_info.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_HIPCONV_DIRECT_FWD)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvHipconvDirectFwd::IsApplicable(const ExecutionContext& ctx,
                                        const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_HIPCONV_DIRECT_FWD))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // Must be gfx950 (CDNA4) — kernel uses CDNA4-specific intrinsics
    if(ctx.GetStream().GetDeviceName() != "gfx950")
        return false;

    // Forward direction, 2D only
    if(!problem.IsDirectionForward() || !problem.Is2d())
        return false;

    // NHWC layout only
    if(!problem.IsLayoutNHWC())
        return false;

    // FP16 or BF16 (all tensors same type, no mixed types)
    if(!((problem.IsFp16() || problem.IsBfp16()) && !problem.HasMixedDataTypes()))
        return false;

    // 3x3 filter only
    if(problem.GetWeightsHeight() != 3 || problem.GetWeightsWidth() != 3)
        return false;

    // Stride 1x1 only
    if(problem.GetKernelStrideH() != 1 || problem.GetKernelStrideW() != 1)
        return false;

    // Dilation 1x1 only
    if(problem.GetDilationH() != 1 || problem.GetDilationW() != 1)
        return false;

    // No grouped convolution
    if(problem.GetGroupCount() != 1)
        return false;

    // Channel divisibility: C % 64 == 0, K % 256 == 0
    if(problem.GetInChannels() % 64 != 0)
        return false;
    if(problem.GetOutChannels() % 256 != 0)
        return false;

    return true;
}

ConvSolution ConvHipconvDirectFwd::GetSolution(const ExecutionContext& /*ctx*/,
                                               const ProblemDescription& problem) const
{
    ConvSolution result;

    // Extract problem parameters
    const int N  = problem.GetBatchSize();
    const int C  = problem.GetInChannels();
    const int K  = problem.GetOutChannels();
    const int hi = problem.GetInHeight();
    const int wi = problem.GetInWidth();
    const int ho = problem.GetOutHeight();
    const int wo = problem.GetOutWidth();
    const int fy = problem.GetWeightsHeight(); // = 3
    const int fx = problem.GetWeightsWidth();  // = 3
    const int sy = problem.GetKernelStrideH(); // = 1
    const int sx = problem.GetKernelStrideW(); // = 1
    const int dy = problem.GetDilationH();     // = 1
    const int dx = problem.GetDilationW();     // = 1
    const int py = problem.GetPadH();
    const int px = problem.GetPadW();

    // Kernel tile parameters (must match direct_conv_nhwc_cdna4.cpp Config)
    constexpr int tile_size_k = 256;
    constexpr int tile_size_h = 16;
    constexpr int tile_size_w = 16;
    // 8 waves (tiles_h=4 * tiles_w=1 * tiles_k=2), wave_size=64 => block_size=512
    constexpr int block_size = 512;

    const int blocks_k   = (K + tile_size_k - 1) / tile_size_k;
    const int blocks_w   = (wo + tile_size_w - 1) / tile_size_w;
    const int blocks_h_n = ((ho + tile_size_h - 1) / tile_size_h) * N;

    KernelInfo kernel;
    kernel.kernel_file = "hipconv/direct_conv_nhwc_cdna4.cpp";

    // Select kernel name based on data type
    if(problem.IsFp16())
        kernel.kernel_name = "conv2d_direct_nhwc_cdna4_fwd_fp16";
    else
        kernel.kernel_name = "conv2d_direct_nhwc_cdna4_fwd_bf16";

    kernel.g_wk = {static_cast<size_t>(blocks_k) * block_size,
                   static_cast<size_t>(blocks_w),
                   static_cast<size_t>(blocks_h_n)};
    kernel.l_wk = {block_size, 1, 1};
    kernel.comp_options = "";

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx =
                primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors = data_ctx.tensors;

            // Kernel uses alpha=1.0, beta=0.0 (no bias accumulation)
            handle.Run(kern)(tensors.in,
                             tensors.w,
                             static_cast<double>(1.0),
                             static_cast<double>(0.0),
                             tensors.out,
                             N,
                             /*groups=*/1,
                             C,
                             K,
                             hi,
                             wi,
                             ho,
                             wo,
                             fy,
                             fx,
                             sy,
                             sx,
                             dy,
                             dx,
                             py,
                             px);
        };
    };

    result.construction_params.push_back(kernel);
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
