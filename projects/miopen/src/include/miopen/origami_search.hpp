/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_ORIGAMI_SEARCH_HPP_
#define GUARD_MIOPEN_ORIGAMI_SEARCH_HPP_

#include <origami/origami.hpp>
#include <origami/types.hpp>

namespace miopen {
namespace solver {

template <typename PerformanceConfig, typename Problem>
std::vector<PerformanceConfig>
GetOrigamiPerformanceConfig(const Problem& problem,
                            const std::vector<PerformanceConfig>& all_configs)
{
    auto hardware = origami::hardware_t::get_hardware_for_device(0);

    auto GetOriDataType =
        []() {
            switch(problem.GetInDataType())
            {
            case miopenHalf: return origami::data_type_t::Half;
            case miopenFloat: return origami::data_type_t::Float;
            case miopenInt8: return origami::data_type_t::Int8;
            case miopenBFloat16: return origami::data_type_t::BFloat16;
            case miopenInt64: return origami::data_type_t::Int64;
            case miopenInt32: return origami::data_type_t::Int32;
            case miopenFloat8_fnuz: return origami::data_type_t::Float8_fnuz;
            case miopenBFloat8_fnuz: return origami::data_type_t::BFloat8_fnuz;
            case miopenDouble: return origami::data_type_t::Double;
            }
        }
    // Create a problem description
    origami::problem_t ori_prob;
    ori_prob.size.m = ProblemInterpreter::GetBatchN(problem); // M dimension // batch size
    ori_prob.size.n =
        ProblemInterpreter::GetOutputChannelK(problem); // N dimension // output channels
    ori_prob.size.k =
        ProblemInterpreter::GetInputChannelC(problem); // K dimension // input channels
    // TBD
    ori_prob.batch = 1;
    // TransA T and TransB N is both K Contiguous (Ks next to each other in memory)
    // TransA N and TransB T is M/N contiguous (M and N next to each other in memory)
    ori_prob.a_transpose = origami::transpose_t::T;
    ori_prob.b_transpose = origami::transpose_t::N;
    ori_prob.a_dtype     = GetOriDataType();
    ori_prob.b_dtype     = GetOriDataType();
    ori_prob.c_dtype     = GetOriDataType();
    ori_prob.d_dtype     = GetOriDataType();
    ori_prob.mi_dtype    = GetOriDataType();
    // TBD
    ori_prob.a_mx_block_size = 0;
    ori_prob.b_mx_block_size = 0;

    // Create candidate configurations
    std::vector<origami::config_t> ori_cfgs;
    std::map<size_t, std::vector<PerformanceConfig>> ori_to_perf_cfg;

    for(perf_cfg : all_configs)
    {
        origami::config_t ori_cfg = GetOrigamiConfig(perf_cfg);
        if(ori_cfg.is_valid())
        {
            ori_cfgs.push_back(ori_cfg);

            if(ori_to_perf_cfg.contains(ori_cfg.hash()))
                ori_to_perf_cfg[ori_cfg.hash()].push_back(perf_cfg);
            else
                ori_to_perf_cfg[ori_cfg.hash()] = {perf_cfg};
        }
    }

    // Rank all configurations by performance
    std::vector<PerformanceConfig> ret;

    if(!ori_cfgs.empty())
    {
        auto ranked_configs = origami::rank_configs(ori_prob, hardware, ori_cfgs);

        for(prediction : ranked_configs)
            for(perf_cfg : ori_to_perf_cfg[prediction.config.hash()])
                ret.push_back(perf_cfg);
    }

    return ret;
}

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_ORIGAMI_SEARCH_HPP_
