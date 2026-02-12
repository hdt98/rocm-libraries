/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/solver/gemm_common.hpp>
#include <boost/range/adaptors.hpp>
#include <numeric>

// Workaround for ALMIOPEN-1044: Temporary workspace size limit for GEMM solvers
// This can be removed once the underlying issue is resolved
#define MIOPEN_WORKAROUND_ALMIOPEN_1044 1

#if MIOPEN_WORKAROUND_ALMIOPEN_1044
// temporary workaround, essentially reverting #2393.
// PR 2393 removed this limit with this comment attached:
// "Workaround for MLOpen issue 1430. Vega20 fails to access GPU memory
//  larger than the return value of GetMaxMemoryAllocSize() of Vega10.
//  Due to historical reasons, this W/A is applied to all targets.
//  We are going to keep it as is until the new GEMM backend
//  is used instead of rocBLAS. See also issue #2809."
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_CONV_GEMM_MAX_WORKSPACE_SIZE, 7287183769)
#endif

namespace miopen {
namespace solver {
namespace conv {
namespace gemm {

bool IsAnyBufferBf16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenBFloat16    //
           || yDesc.GetType() == miopenBFloat16 //
           || wDesc.GetType() == miopenBFloat16;
}

bool IsAnyBufferFp16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenHalf    //
           || yDesc.GetType() == miopenHalf //
           || wDesc.GetType() == miopenHalf;
}

double SlowdownFactor(const int n_oper, const double oper_factor, const double multiple_oper_factor)
{
    if(n_oper > 0)
    {
        auto rv = oper_factor;
        if(n_oper > 1)
            rv *= multiple_oper_factor;
        return rv;
    }
    else
        return 1.0;
}

#if MIOPEN_USE_GEMM
bool IsGEMMProblemTooLarge(const miopen::conv::ProblemDescription& problem)
{
#if MIOPEN_WORKAROUND_ALMIOPEN_1044
    const std::size_t max_size = env::value(MIOPEN_DEBUG_CONV_GEMM_MAX_WORKSPACE_SIZE);
    // 0 means no limit
    if(max_size == 0)
        return false;

    const auto& conv  = problem.GetConv();
    const auto& wDesc = problem.GetWeights();
    const auto& yDesc = problem.GetOut();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto wei_c       = wDesc.GetLengths()[1];

    // Calculate workspace size using the same formula as GemmFwdRest::GetWorkspaceSize()
    // workspace = C × filter_spatial × output_spatial × type_size × groups
    const auto workspace_size = wei_c *
                                std::accumulate(wei_spatial.begin(),
                                                wei_spatial.end(),
                                                std::size_t(1),
                                                std::multiplies<std::size_t>()) *
                                std::accumulate(out_spatial.begin(),
                                                out_spatial.end(),
                                                std::size_t(1),
                                                std::multiplies<std::size_t>()) *
                                GetTypeSize(wDesc.GetType()) * conv.group_count;

    // For Int8, workspace is doubled (for transpose operations)
    const auto ws_sz = (wDesc.GetType() == miopenInt8 ? 2 * workspace_size : workspace_size);

    // Workspace is within limit
    if(ws_sz <= max_size)
    {
        return false;
    }

    MIOPEN_LOG_I2("GEMMSolverFinder disabled for workspace size "
                  << ws_sz << " bytes > " << max_size
                  << " bytes (MIOPEN_DEBUG_CONV_GEMM_MAX_WORKSPACE_SIZE)");
    return true;
#else
    // Workaround disabled - no size limit
    (void)problem; // Suppress unused parameter warning
    return false;
#endif
}
#endif

} // namespace gemm
} // namespace conv
} // namespace solver
} // namespace miopen
