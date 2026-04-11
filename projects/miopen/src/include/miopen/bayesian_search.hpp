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

#ifndef GUARD_MIOPEN_BAYESIAN_SEARCH_HPP_
#define GUARD_MIOPEN_BAYESIAN_SEARCH_HPP_

#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/utility/modified_z.hpp>

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <sstream>
#include <cassert>

// 0 = brute force (default, existing GenericSearch)
// 1 = Bayesian optimization
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_TUNING_SEARCH_METHOD, 0)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_TUNING_BO_INITIAL, 3)

// Must come after env var declarations (generic_search.hpp routing uses MIOPEN_TUNING_SEARCH_METHOD)
#include <miopen/generic_search.hpp>

namespace miopen {
namespace solver {
namespace bayesian {

// -------------------------------------------------------------------
// Gaussian Process with Matern 5/2 kernel
// -------------------------------------------------------------------
class GaussianProcess
{
public:
    void Fit(const std::vector<double>& X_obs,
             const std::vector<double>& y_obs,
             std::size_t n_obs,
             std::size_t n_features);

    // Returns mu and sigma in ORIGINAL y-space (un-normalized)
    void Predict(const std::vector<double>& X_cand,
                 std::size_t n_cand,
                 std::vector<double>& out_mu,
                 std::vector<double>& out_sigma) const;

private:
    double Matern52(const double* x1, const double* x2, std::size_t d) const;
    void CholeskySolve(const std::vector<double>& L,
                       const std::vector<double>& b,
                       std::vector<double>& x,
                       std::size_t n) const;

    std::vector<double> X_obs_;
    std::vector<double> y_norm_;
    std::vector<double> L_;      // lower Cholesky of (K + noise*I)
    std::vector<double> alpha_;  // K^{-1} * y_norm
    std::size_t n_obs_     = 0;
    std::size_t n_features_ = 0;
    double y_mean_       = 0.0;
    double y_std_        = 1.0;
    double length_scale_ = 1.0;
    double noise_        = 1e-4;
};

// Expected Improvement: returns index of candidate with highest EI.
// mu, sigma are in original y-space; best_observed is the best (lowest) time seen.
// We MINIMIZE time, so EI = E[max(best_observed - f(x), 0)].
std::size_t SelectNextByEI(const std::vector<double>& mu,
                           const std::vector<double>& sigma,
                           double best_observed,
                           std::size_t n_cand);

// Two-pass feature extraction from config strings.
// Pass 1: discover variant names and max param count across all configs.
// Pass 2: build uniform-dimension feature vectors (shorter configs zero-padded).
// Handles CK "Variant<p0,p1,...>", CK WrW "Variant<...>+split_k",
// ASM "val0,val1,...[gks]", and simple numeric configs.
// Returns the feature dimension (all vectors in out_features have this size).
std::size_t BuildFeatureMatrix(const std::vector<std::string>& config_strings,
                               std::vector<std::vector<double>>& out_features);

struct BayesOptTracker
{
    int solver_count = 0;
    float best_time  = std::numeric_limits<float>::max();
    std::string best_solver;
    int best_solver_num = 0;
};

MIOPEN_INTERNALS_EXPORT BayesOptTracker& GetBayesOptTracker();

} // namespace bayesian

// -------------------------------------------------------------------
// BayesianSearch: drop-in replacement for GenericSearch
// -------------------------------------------------------------------
template <class Solver, class Context, class Problem>
auto BayesianSearch(const Solver s,
                    const Context& context_,
                    const Problem& problem,
                    const AnyInvokeParams& invoke_ctx_)
    -> decltype(s.GetDefaultPerformanceConfig(context_, problem))
{
    auto context                  = context_;
    context.is_for_generic_search = true;

    using PerformanceConfig = decltype(s.GetDefaultPerformanceConfig(context, problem));

    const auto default_solution =
        s.GetSolution(context, problem, s.GetDefaultPerformanceConfig(context, problem));
    const auto invoke_ctx = [invoke_ctx_]() {
        auto copy = invoke_ctx_;
        copy.SetInvokeType(InvokeType::AutoTune);
        return copy;
    }();

    auto& profile_h = context.GetStream();
    const AutoEnableProfiling enableProfiling{profile_h};

    auto tmp_all_configs = GetAllConfigs(s, context, problem);
    std::vector<PerformanceConfig> all_configs;
    std::copy(tmp_all_configs.begin(), tmp_all_configs.end(), std::back_inserter(all_configs));

    const std::size_t n_configs = all_configs.size();
    if(n_configs == 0)
        MIOPEN_THROW("BayesianSearch: solver " + s.SolverDbId() + " has no valid configs");

    auto& bo_tracker     = bayesian::GetBayesOptTracker();
    bo_tracker.solver_count++;
    const int solver_num = bo_tracker.solver_count;

    MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning, false, "BayesOpt",
        MIOPEN_GET_FN_NAME,
        ">> [Solver #" << std::to_string(solver_num) << "] "
        << s.SolverDbId() << " (" << std::to_string(n_configs) << " configs)");

    // --- Collect config strings ---
    std::vector<std::string> config_strings(n_configs);
    for(std::size_t i = 0; i < n_configs; ++i)
    {
        std::ostringstream oss;
        oss << all_configs[i];
        config_strings[i] = oss.str();
    }

    // --- Build uniform feature matrix (two-pass: discover schema, then parse) ---
    std::vector<std::vector<double>> features;
    const std::size_t n_features = bayesian::BuildFeatureMatrix(config_strings, features);

    if(n_features == 0)
    {
        MIOPEN_LOG_W("BayesianSearch: cannot parse features, falling back to brute force");
        return GenericSearch(s, context_, problem, invoke_ctx_);
    }

    // --- Evaluate one config on GPU, return mean time or -1 on failure ---
    auto evaluate = [&](std::size_t idx) -> float {
        auto solution = s.GetSolution(context, problem, all_configs[idx]);
        try
        {
            if(default_solution.workspace_sz != solution.workspace_sz)
                return -1.0f;

            auto invoker = profile_h.PrepareInvoker(*solution.invoker_factory,
                                                    solution.construction_params);
            invoker(profile_h, invoke_ctx);
            profile_h.ResetKernelTime();

            constexpr int N_RUNS = 10;
            std::vector<float> samples;
            samples.reserve(N_RUNS);
            for(int r = 0; r < N_RUNS; ++r)
            {
                invoker(profile_h, invoke_ctx);
                samples.push_back(profile_h.GetKernelTime());
                profile_h.ResetKernelTime();
            }

            float elapsed = miopen::removeHighOutliersAndGetMean(samples, 2.0f);

            for(const auto& ki : solution.construction_params)
                profile_h.ClearProgram(ki.kernel_file, ki.comp_options);

            return elapsed;
        }
        catch(const std::exception& e)
        {
            MIOPEN_LOG_E("BayesianSearch: eval #" << idx << " error: " << e.what());
            for(const auto& ki : solution.construction_params)
                profile_h.ClearProgram(ki.kernel_file, ki.comp_options);
            return -1.0f;
        }
    };

    // --- State ---
    std::vector<bool> evaluated(n_configs, false);
    std::vector<double> observed_y;
    std::vector<std::size_t> observed_idx;
    PerformanceConfig best_config;
    float best_time = std::numeric_limits<float>::max();
    std::size_t n_best = 0;
    bool is_passed = false;

    // --- Phase 1: Random initial probes ---
    const std::size_t n_initial =
        std::min(static_cast<std::size_t>(env::value(MIOPEN_DEBUG_TUNING_BO_INITIAL)), n_configs);

    std::vector<std::size_t> perm(n_configs);
    std::iota(perm.begin(), perm.end(), 0);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(perm.begin(), perm.end(), rng);

    MIOPEN_LOG_I("BayesianSearch: phase1 random_init=" << std::to_string(n_initial));
    for(std::size_t i = 0; i < n_initial; ++i)
    {
        std::size_t idx = perm[i];
        float t = evaluate(idx);
        if(t < 0)
            continue;

        is_passed = true;
        evaluated[idx] = true;
        observed_y.push_back(static_cast<double>(t));
        observed_idx.push_back(idx);

        if(t < best_time)
        {
            best_time   = t;
            best_config = all_configs[idx];
            n_best      = idx;
        }
        MIOPEN_LOG_I2("BayesianSearch: init #" << i << " idx=" << idx
                      << " t=" << t << " best=" << best_time);
    }

    if(!is_passed)
    {
        MIOPEN_LOG_W("BayesianSearch: all initial probes failed, falling back to brute force");
        return GenericSearch(s, context_, problem, invoke_ctx_);
    }

    // --- Phase 2: BO guided by GP + EI ---
    // Budget: at most half of remaining unevaluated configs
    const std::size_t remaining = n_configs - observed_idx.size();
    const std::size_t bo_budget = std::max<std::size_t>(1, remaining / 2);

    MIOPEN_LOG_I("BayesianSearch: phase2 bo_budget=" << std::to_string(bo_budget));

    bayesian::GaussianProcess gp;

    for(std::size_t iter = 0; iter < bo_budget; ++iter)
    {
        // Build candidate list
        std::vector<std::size_t> cands;
        for(std::size_t i = 0; i < n_configs; ++i)
            if(!evaluated[i])
                cands.push_back(i);
        if(cands.empty())
            break;

        const std::size_t n_obs  = observed_idx.size();
        const std::size_t n_cand = cands.size();

        // Flatten observed X
        std::vector<double> X_obs(n_obs * n_features);
        for(std::size_t i = 0; i < n_obs; ++i)
            for(std::size_t j = 0; j < n_features; ++j)
                X_obs[i * n_features + j] = features[observed_idx[i]][j];

        gp.Fit(X_obs, observed_y, n_obs, n_features);

        // Flatten candidate X
        std::vector<double> X_cand(n_cand * n_features);
        for(std::size_t i = 0; i < n_cand; ++i)
            for(std::size_t j = 0; j < n_features; ++j)
                X_cand[i * n_features + j] = features[cands[i]][j];

        std::vector<double> mu, sigma;
        gp.Predict(X_cand, n_cand, mu, sigma);

        std::size_t pick = bayesian::SelectNextByEI(mu, sigma,
                                                    static_cast<double>(best_time), n_cand);
        std::size_t next = cands[pick];

        float t = evaluate(next);
        evaluated[next] = true;
        if(t < 0)
            continue;

        observed_y.push_back(static_cast<double>(t));
        observed_idx.push_back(next);

        if(t < best_time)
        {
            best_time   = t;
            best_config = all_configs[next];
            n_best      = next;
            MIOPEN_LOG_I("BayesianSearch: iter=" << std::to_string(iter)
                         << " NEW BEST #" << std::to_string(next) << " t=" << t);
        }
        else
        {
            MIOPEN_LOG_I2("BayesianSearch: iter=" << iter << " #" << next
                          << " t=" << t << " best=" << best_time);
        }
    }

    // --- Done ---
    if(!is_passed)
        MIOPEN_THROW("BayesianSearch failed: no config succeeded");

    const auto& invoker = profile_h.PrepareInvoker(*default_solution.invoker_factory,
                                                   default_solution.construction_params);
    invoker(profile_h, invoke_ctx);
    const auto default_time = profile_h.GetKernelTime();
    const auto score        = (best_time > 0.0f) ? default_time / best_time : 0.0f;

    MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning, false, "BayesOpt",
        MIOPEN_GET_FN_NAME,
        "     evaluated: " << std::to_string(observed_idx.size())
        << "/" << std::to_string(n_configs)
        << " | best: #" << std::to_string(n_best)
        << " | time: " << best_time << "ms"
        << " | score: " << score << "x");
    MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning, false, "BayesOpt",
        MIOPEN_GET_FN_NAME,
        "     config: " << best_config);

    if(best_time < bo_tracker.best_time)
    {
        bo_tracker.best_time       = best_time;
        bo_tracker.best_solver     = s.SolverDbId();
        bo_tracker.best_solver_num = solver_num;
    }
    MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning, false, "BayesOpt",
        MIOPEN_GET_FN_NAME,
        ">> [BEST] Solver #" << std::to_string(bo_tracker.best_solver_num)
        << " " << bo_tracker.best_solver
        << " | time: " << bo_tracker.best_time << "ms");

    return best_config;
}

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_BAYESIAN_SEARCH_HPP_
