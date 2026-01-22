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

#include <miopen/solver/problem_description_interpreter.hpp>

#include <origami/origami.hpp>
#include <origami/types.hpp>

namespace miopen {
namespace solver {

origami::data_type_t GetOriDataType(const miopen::conv::ProblemDescription& problem);

std::string SerializeOrigamiConfig(const origami::config_t config);

template <class Solver, class PerformanceConfig>
std::vector<PerformanceConfig>
GetOrigamiPerformanceConfig(const Solver s,
                            const miopen::conv::ProblemDescription& problem,
                            const std::vector<PerformanceConfig>& all_configs)
{
    auto hardware = origami::hardware_t::get_hardware_for_device(0);

    // Create a problem description
    auto in_chan = ProblemInterpreter::GetInputChannelC(problem);
    auto fil_h = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    auto fil_w = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    auto fil_d = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
    auto batch = ProblemInterpreter::GetBatchN(problem);
    auto out_h = ProblemInterpreter::GetOutputHeightHo(problem);
    auto out_w = ProblemInterpreter::GetOutputWidthWo(problem);
    auto out_d = ProblemInterpreter::GetOutputDepthDo(problem);
    origami::problem_t ori_prob;
    ori_prob.size.m = 1; // number of filters 
    ori_prob.size.n = batch * out_h * out_w * out_d; // batch x out height x out width x out depth 
    ori_prob.size.k = in_chan * fil_h * fil_w * fil_d; // channels x filter height x filter width x filter depth 
    // TBD
    ori_prob.batch = 1;
    // TransA T and TransB N is both K Contiguous (Ks next to each other in memory)
    // TransA N and TransB T is M/N contiguous (M and N next to each other in memory)
    ori_prob.a_transpose = origami::transpose_t::T;
    ori_prob.b_transpose = origami::transpose_t::N;
    ori_prob.a_dtype     = GetOriDataType(problem);
    ori_prob.b_dtype     = GetOriDataType(problem);
    ori_prob.c_dtype     = GetOriDataType(problem);
    ori_prob.d_dtype     = GetOriDataType(problem);
    ori_prob.mi_dtype    = GetOriDataType(problem);
    // TBD
    ori_prob.a_mx_block_size = 0;
    ori_prob.b_mx_block_size = 0;

    // Create candidate configurations
    std::vector<origami::config_t> ori_cfgs;
    std::map<std::string, std::vector<PerformanceConfig>> ori_to_perf_cfg;

    for(auto perf_cfg : all_configs)
    {
        origami::config_t ori_cfg = s.GetOrigamiConfig(problem, perf_cfg);
        if(ori_cfg.is_valid())
        {
            auto ori_cfg_str = SerializeOrigamiConfig(ori_cfg);
            if(!ori_to_perf_cfg.contains(ori_cfg_str))
                ori_cfgs.push_back(ori_cfg);

            if(ori_to_perf_cfg.contains(ori_cfg_str))
                ori_to_perf_cfg[ori_cfg_str].push_back(perf_cfg);
            else
                ori_to_perf_cfg[ori_cfg_str] = {perf_cfg};
        }
    }

    // Rank all configurations by performance
    std::vector<PerformanceConfig> ret{};
    if(!ori_cfgs.empty())
    {
        MIOPEN_LOG_I2("Ranking PerformanceConfig");
        auto ranked_configs = origami::rank_configs(ori_prob, hardware, ori_cfgs);

        bool first = true;
        for(auto prediction : ranked_configs)
        {
            if(first)
            {
                auto ori_cfg = prediction.config;
                MIOPEN_LOG_T("Rank 1 configs: "
                             << "MT.M(" << ori_cfg.mt.m << ") MT.N(" << ori_cfg.mt.n << ") MT.K("
                             << ori_cfg.mt.k << ") MI.M(" << ori_cfg.mi.m << ") MI.N("
                             << ori_cfg.mi.n << ") MI.K(" << ori_cfg.mi.k << ")");
            }
            for(auto perf_cfg : ori_to_perf_cfg[SerializeOrigamiConfig(prediction.config)])
            {
                ret.push_back(perf_cfg);
                if(first)
                    MIOPEN_LOG_T(perf_cfg);
            }
            first = false;
        }
    }
    MIOPEN_LOG_I2("return perf_cfg: " << ret.size());

    return ret;
}

template <class Solver, class PerformanceConfig, class Problem>
std::vector<PerformanceConfig> GetOrigamiPerformanceConfig(
    const Solver s, const Problem& problem, const std::vector<PerformanceConfig>& all_configs)
{
    return {};
}

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_ORIGAMI_SEARCH_HPP_
