/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/solver/solver_utils.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvDirectNaiveConvFwd::IsApplicable(const ExecutionContext& ctx,
                                          const ProblemDescription& problem) const
{
    if(!miopen::debug::AlwaysEnableConvDirectNaive)
    {
        MIOPEN_SOLVER_INAPPLICABLE_IF(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD),
                                      inapplicable_msg::EnvDisabled);
        MIOPEN_SOLVER_INAPPLICABLE_IF(!ctx.use_hip_kernels, inapplicable_msg::HIPDisabled);
    }

    MIOPEN_SOLVER_INAPPLICABLE_IF(!ConvDirectNaiveConvIsApplicableByKernelType(ctx, problem),
                                  inapplicable_msg::Generic);

    MIOPEN_SOLVER_INAPPLICABLE_IF(!problem.IsLayoutDefault() && !problem.IsLayoutNHWC(),
                                  inapplicable_msg::Layout);

    MIOPEN_SOLVER_INAPPLICABLE_IF(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16() ||
                                    problem.IsInt8() || problem.IsFp8() || problem.IsBfp8()),
                                  inapplicable_msg::DataType);

    MIOPEN_SOLVER_INAPPLICABLE_IF(!problem.IsDirectionForward(), inapplicable_msg::Direction);

    MIOPEN_SOLVER_INAPPLICABLE_IF(!problem.AllTensorsLengthsFitIntoInt(),
                                  inapplicable_msg::AllTensorsDimsFitIntoInt);

    if(problem.IsTensorsCasted())
    {
        auto test_cast = [&](const TensorDescriptor& desc) {
            if(desc.GetCastType())
            {
                const auto cast_type = *desc.GetCastType();
                if(cast_type == miopenFloat8_fnuz || cast_type == miopenBFloat8_fnuz)
                    return false;
            }
            // all tested tensors must have cast type set
            return true;
        };

        MIOPEN_SOLVER_INAPPLICABLE_IF(test_cast(problem.GetIn()), "Input tensor missing cast type");
        MIOPEN_SOLVER_INAPPLICABLE_IF(test_cast(problem.GetWeights()),
                                      "Weight tensor missing cast type");
    }

    return true;
}

ConvSolution ConvDirectNaiveConvFwd::GetSolution(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem) const
{
    ConvSolution result;

    if(problem.Is2d())
    {
        result = conv_internal::GetConv2DFWDSolution(ctx, problem);
    }
    else
    {
        result = conv_internal::GetConv3DFWDSolution(ctx, problem);
    }
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
