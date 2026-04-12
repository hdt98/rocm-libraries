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
// EI convergence threshold: stop BO when max EI < best_time * threshold.
// 0 = disabled (run full budget). Default 0.001 = stop when <0.1% improvement expected.
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_TUNING_BO_EI_THRESH_E4, 10)
// Skip full 10-run measurement if single probe > best_time * (skip_pct/100).
// 0 = disabled (always run 10). Default 150 = skip if probe is >50% worse than best.
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_TUNING_BO_SKIP_PCT, 150)
// Cap observations fed to GP (O(n^3) Cholesky safety).
// 0 = disabled (use all). Default 128 = keep best + most recent 127.
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_TUNING_BO_GP_CAP, 128)

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
// If out_max_ei is non-null, stores the max EI value (for convergence detection).
std::size_t SelectNextByEI(const std::vector<double>& mu,
                           const std::vector<double>& sigma,
                           double best_observed,
                           std::size_t n_cand,
                           double* out_max_ei = nullptr);

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

    // --- State ---
    std::vector<bool> evaluated(n_configs, false);
    std::vector<double> observed_y;
    std::vector<std::size_t> observed_idx;
    PerformanceConfig best_config;
    float best_time = std::numeric_limits<float>::max();
    std::size_t n_best = 0;
    bool is_passed = false;

    // Skip threshold: skip full 10-run if probe > best * (skip_pct/100). 0 = disabled.
    const float skip_pct = static_cast<float>(env::value(MIOPEN_DEBUG_TUNING_BO_SKIP_PCT));

    // --- Evaluate one config on GPU, return mean time or -1 on failure ---
    auto evaluate = [&](std::size_t idx) -> float {
        auto solution = s.GetSolution(context, problem, all_configs[idx]);
        try
        {
            if(default_solution.workspace_sz != solution.workspace_sz)
                return -1.0f;

            auto invoker = profile_h.PrepareInvoker(*solution.invoker_factory,
                                                    solution.construction_params);
            // Warm-up run
            invoker(profile_h, invoke_ctx);
            profile_h.ResetKernelTime();

            // Single probe
            invoker(profile_h, invoke_ctx);
            float probe = profile_h.GetKernelTime();
            profile_h.ResetKernelTime();

            // Skip full measurement if probe is far worse than current best
            if(skip_pct > 0.0f && best_time < std::numeric_limits<float>::max() &&
               probe > best_time * (skip_pct / 100.0f))
            {
                for(const auto& ki : solution.construction_params)
                    profile_h.ClearProgram(ki.kernel_file, ki.comp_options);
                return probe;
            }

            // Full measurement: 9 more runs (total 10 including probe)
            constexpr int N_RUNS = 10;
            std::vector<float> samples;
            samples.reserve(N_RUNS);
            samples.push_back(probe);
            for(int r = 1; r < N_RUNS; ++r)
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

    // --- Phase 1: Random initial probes ---
    // Scale with config space: max(env_min, sqrt(n_configs)), capped at n_configs
    const std::size_t env_min = static_cast<std::size_t>(env::value(MIOPEN_DEBUG_TUNING_BO_INITIAL));
    const std::size_t scaled  = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n_configs))));
    const std::size_t n_initial = std::min(std::max(env_min, scaled), n_configs);

    std::vector<std::size_t> perm(n_configs);
    std::iota(perm.begin(), perm.end(), 0);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(perm.begin(), perm.end(), rng);

    MIOPEN_LOG_I("BayesianSearch: phase1 random_init=" << std::to_string(n_initial));
    std::size_t init_successes = 0;
    for(std::size_t i = 0; i < n_configs && init_successes < n_initial; ++i)
    {
        std::size_t idx = perm[i];
        float t = evaluate(idx);
        evaluated[idx] = true;
        if(t < 0)
            continue;

        is_passed = true;
        init_successes++;
        observed_y.push_back(static_cast<double>(t));
        observed_idx.push_back(idx);

        if(t < best_time)
        {
            best_time   = t;
            best_config = all_configs[idx];
            n_best      = idx;
        }
        MIOPEN_LOG_I2("BayesianSearch: init #" << init_successes << " idx=" << idx
                      << " t=" << t << " best=" << best_time);
    }

    if(!is_passed)
    {
        MIOPEN_LOG_W("BayesianSearch: all initial probes failed, falling back to brute force");
        return GenericSearch(s, context_, problem, invoke_ctx_);
    }

    // --- Phase 2: BO guided by GP + EI ---
    // Count actually remaining (unevaluated) configs
    std::size_t remaining = 0;
    for(std::size_t i = 0; i < n_configs; ++i)
        if(!evaluated[i])
            remaining++;

    const std::size_t bo_budget = std::max<std::size_t>(1, remaining / 2);

    // EI convergence: stop when max EI < best_time * threshold.
    // Env var is in units of 1e-4 (integer). Default 10 → 10/10000 = 0.001.
    // Set to 0 to disable.
    const double ei_thresh_raw =
        static_cast<double>(env::value(MIOPEN_DEBUG_TUNING_BO_EI_THRESH_E4)) / 10000.0;

    const std::size_t gp_cap =
        static_cast<std::size_t>(env::value(MIOPEN_DEBUG_TUNING_BO_GP_CAP));

    const std::size_t max_attempts = std::min(remaining, bo_budget * 3);

    MIOPEN_LOG_I("BayesianSearch: phase2 bo_budget=" << std::to_string(bo_budget)
                 << " max_attempts=" << std::to_string(max_attempts)
                 << " ei_thresh=" << ei_thresh_raw
                 << " gp_cap=" << std::to_string(gp_cap));

    bayesian::GaussianProcess gp;

    std::size_t successful_iters = 0;
    std::size_t total_attempts   = 0;

    while(successful_iters < bo_budget && total_attempts < max_attempts)
    {
        // Build candidate list
        std::vector<std::size_t> cands;
        for(std::size_t i = 0; i < n_configs; ++i)
            if(!evaluated[i])
                cands.push_back(i);
        if(cands.empty())
            break;

        const std::size_t n_cand = cands.size();

        // Select which observations to feed the GP (cap for O(n^3) safety)
        std::vector<std::size_t> gp_sel_idx;
        std::vector<double> gp_sel_y;

        const std::size_t n_total_obs = observed_idx.size();
        if(gp_cap == 0 || n_total_obs <= gp_cap)
        {
            gp_sel_idx = observed_idx;
            gp_sel_y   = observed_y;
        }
        else
        {
            // Find which position holds the best observation
            std::size_t best_pos = 0;
            for(std::size_t i = 1; i < n_total_obs; ++i)
                if(observed_y[i] < observed_y[best_pos])
                    best_pos = i;

            // Take most recent (gp_cap - 1), always include best
            const std::size_t tail_start = n_total_obs - (gp_cap - 1);
            bool best_in_tail = (best_pos >= tail_start);

            if(!best_in_tail)
            {
                gp_sel_idx.push_back(observed_idx[best_pos]);
                gp_sel_y.push_back(observed_y[best_pos]);
            }
            for(std::size_t i = tail_start; i < n_total_obs; ++i)
            {
                gp_sel_idx.push_back(observed_idx[i]);
                gp_sel_y.push_back(observed_y[i]);
            }
        }

        const std::size_t n_obs = gp_sel_idx.size();

        // Flatten observed X
        std::vector<double> X_obs(n_obs * n_features);
        for(std::size_t i = 0; i < n_obs; ++i)
            for(std::size_t j = 0; j < n_features; ++j)
                X_obs[i * n_features + j] = features[gp_sel_idx[i]][j];

        gp.Fit(X_obs, gp_sel_y, n_obs, n_features);

        // Flatten candidate X
        std::vector<double> X_cand(n_cand * n_features);
        for(std::size_t i = 0; i < n_cand; ++i)
            for(std::size_t j = 0; j < n_features; ++j)
                X_cand[i * n_features + j] = features[cands[i]][j];

        std::vector<double> mu, sigma;
        gp.Predict(X_cand, n_cand, mu, sigma);

        double max_ei = 0.0;
        std::size_t pick = bayesian::SelectNextByEI(mu, sigma,
                                                    static_cast<double>(best_time), n_cand,
                                                    &max_ei);

        // EI convergence check
        if(ei_thresh_raw > 0.0 && best_time < std::numeric_limits<float>::max())
        {
            const double ei_threshold = static_cast<double>(best_time) * ei_thresh_raw;
            if(max_ei < ei_threshold)
            {
                MIOPEN_LOG_I("BayesianSearch: EI converged at iter="
                             << std::to_string(successful_iters)
                             << " (attempts=" << std::to_string(total_attempts)
                             << ") max_ei=" << max_ei
                             << " < thresh=" << ei_threshold);
                break;
            }
        }

        std::size_t next = cands[pick];

        float t = evaluate(next);
        evaluated[next] = true;
        total_attempts++;
        if(t < 0)
            continue;

        successful_iters++;
        observed_y.push_back(static_cast<double>(t));
        observed_idx.push_back(next);

        if(t < best_time)
        {
            best_time   = t;
            best_config = all_configs[next];
            n_best      = next;
            MIOPEN_LOG_I("BayesianSearch: iter=" << std::to_string(successful_iters)
                         << " (attempts=" << std::to_string(total_attempts)
                         << ") NEW BEST #" << std::to_string(next) << " t=" << t);
        }
        else
        {
            MIOPEN_LOG_I2("BayesianSearch: iter=" << successful_iters
                          << " (attempts=" << total_attempts
                          << ") #" << next
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
