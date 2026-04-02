// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/env.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/solver/ck_grouped_conv_lib_loader.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

void PerformanceConfigConvDepthwiseFwd2D::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";

    const auto& loader = CKGroupedConvLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return;

    auto data_type = problem.GetInDataType();
    if(data_type != miopenHalf)
        return;

    valid_kernels =
        loader.fill_valid_kernels(CKSolverSlot::DepthwiseFwd, problem, data_type, false);

    if(valid_kernels.empty())
        return;

    index     = 0;
    kernel_id = valid_kernels[index];
}

bool PerformanceConfigConvDepthwiseFwd2D::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        HeuristicInit({}, problem);
        return true;
    }
    if(index + 1 < valid_kernels.size())
    {
        index++;
        kernel_id = valid_kernels[index];
    }
    else
    {
        return false;
    }
    return true;
}

bool PerformanceConfigConvDepthwiseFwd2D::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigConvDepthwiseFwd2D::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
    return IsValidValue();
}

bool PerformanceConfigConvDepthwiseFwd2D::operator==(
    const PerformanceConfigConvDepthwiseFwd2D& other) const
{
    return kernel_id == other.kernel_id;
}

bool ConvDepthwiseFwd2D::IsApplicable(const ExecutionContext& ctx,
                                      const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // Kernel requires a wavefront size of 64
    if(64 != ctx.GetStream().GetWavefrontWidth())
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    if(!problem.IsFp16())
        return false;

    if(!problem.IsDirectionForward())
        return false;

    // Only depthwise convolution is supported
    if((problem.GetGroupCount() != problem.GetOutChannels()) ||
       (problem.GetGroupCount() != problem.GetInChannels()))
        return false;

    if(GetSupportedSolutionCount(ctx, problem) == 0)
    {
        return false;
    }

    return true;
}

uint32_t
ConvDepthwiseFwd2D::GetSupportedSolutionCount([[maybe_unused]] const ExecutionContext& ctx,
                                              const miopen::conv::ProblemDescription& problem) const
{
    const auto& loader = CKGroupedConvLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return 0;

    auto data_type = problem.GetInDataType();
    auto kernels =
        loader.fill_valid_kernels(CKSolverSlot::DepthwiseFwd, problem, data_type, false);
    return static_cast<uint32_t>(kernels.size());
}

PerformanceConfigConvDepthwiseFwd2D ConvDepthwiseFwd2D::GetDefaultPerformanceConfig(
    [[maybe_unused]] const ExecutionContext& ctx,
    const miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigConvDepthwiseFwd2D pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvDepthwiseFwd2D::IsValidPerformanceConfig(
    const ExecutionContext&,
    const miopen::conv::ProblemDescription& problem,
    const PerformanceConfigConvDepthwiseFwd2D& config) const
{
    return config.IsValid((problem));
}

PerformanceConfigConvDepthwiseFwd2D
ConvDepthwiseFwd2D::Search(const ExecutionContext& ctx,
                           const miopen::conv::ProblemDescription& problem,
                           const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution
ConvDepthwiseFwd2D::GetSolution(const ExecutionContext& ctx,
                                const miopen::conv::ProblemDescription& problem,
                                const PerformanceConfigConvDepthwiseFwd2D& config) const
{
    const auto& loader = CKGroupedConvLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return {};

    return loader.get_solution(
        CKSolverSlot::DepthwiseFwd, ctx, problem, config.kernel_id, false);
}

} // namespace conv
} // namespace solver
} // namespace miopen
