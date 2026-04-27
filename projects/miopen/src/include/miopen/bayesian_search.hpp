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
#include <unordered_map>
#include <set>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <sstream>
#include <fstream>
#include <cassert>

// 0 = brute force (default, existing GenericSearch)
// 1 = Bayesian optimization
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_TUNING_SEARCH_METHOD, 0)
// Override env vars: 0 = use algorithmic defaults. Non-zero = use the env value.
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_TUNING_BO_INITIAL, 0)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_TUNING_BO_EI_THRESH_E4, 0)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_TUNING_BO_GP_CAP, 0)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_TUNING_BO_DUMP, 0)
// Fixed seed for reproducible benchmarks. 0 = use std::random_device (non-deterministic).
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_TUNING_BO_SEED, 0)

// Must come after env var declarations (generic_search.hpp routing uses
// MIOPEN_TUNING_SEARCH_METHOD)
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

    double GetNoise() const { return noise_; }
    double GetLengthScale() const { return length_scale_; }

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
    std::vector<double> L_;     // lower Cholesky of (K + noise*I)
    std::vector<double> alpha_; // K^{-1} * y_norm
    std::size_t n_obs_      = 0;
    std::size_t n_features_ = 0;
    double y_mean_          = 0.0;
    double y_std_           = 1.0;
    double length_scale_    = 1.0;
    double noise_           = 1e-4;

    static double EstimateNoise(const std::vector<double>& y_obs, std::size_t n_obs);
};

// v25 — PCA feature reduction: projects high-dimensional feature space
// to top-k principal components capturing >= 95% variance.
struct PCAState
{
    std::vector<double> mean;
    std::vector<double> components; // k × d_orig, row-major
    std::size_t d_orig = 0;
    std::size_t k      = 0;
    bool active() const { return k > 0 && k < d_orig && !components.empty(); }
};

PCAState
PCAFit(const std::vector<double>& X, std::size_t n, std::size_t d, double variance_ratio = 0.95);
void PCATransform(const PCAState& pca,
                  const std::vector<double>& X_in,
                  std::size_t n,
                  std::vector<double>& X_out);

// Expected Improvement: returns index of candidate with highest EI.
// mu, sigma are in original y-space; best_observed is the best (lowest) time seen.
// We MINIMIZE time, so EI = E[max(best_observed - f(x), 0)].
// If out_max_ei is non-null, stores the max EI value (for convergence detection).
std::size_t SelectNextByEI(const std::vector<double>& mu,
                           const std::vector<double>& sigma,
                           double best_observed,
                           std::size_t n_cand,
                           double* out_max_ei = nullptr);

// Thompson Sampling: sample from GP posterior N(mu, sigma^2) per candidate,
// return index of candidate with lowest sample. Naturally balances exploration
// (high sigma → wide samples → occasional optimistic draws) and exploitation
// (low mu → likely selected). No tunable parameters.
std::size_t SelectNextByTS(const std::vector<double>& mu,
                           const std::vector<double>& sigma,
                           std::size_t n_cand,
                           std::mt19937& rng);

// Two-pass feature extraction from config strings.
// Pass 1: discover variant names and max param count across all configs.
// Pass 2: build uniform-dimension feature vectors (shorter configs zero-padded).
// Handles CK "Variant<p0,p1,...>", CK WrW "Variant<...>+split_k",
// ASM "val0,val1,...[gks]", and simple numeric configs.
// Returns the feature dimension (all vectors in out_features have this size).
// out_variant_ids: per-config variant group (0..K-1 for CK, -1 for non-CK).
std::size_t BuildFeatureMatrix(const std::vector<std::string>& config_strings,
                               std::vector<std::vector<double>>& out_features,
                               std::vector<int>& out_variant_ids);

struct BayesOptTracker
{
    int solver_count = 0;
    float best_time  = std::numeric_limits<float>::max();
    std::string best_solver;
    int best_solver_num = 0;
};

// Unified BO parameter computation via exploration ratio R = sqrt(N)/N.
// ALL parameters are derived from R and N — no scattered magic numbers.
// R represents the fraction of configs BO should evaluate:
//   R = 1.0 → brute force (evaluate all), R = 0.3 → evaluate ~30%.
//
// Budget split derivation (v20, paper §9.28.12):
//   n_initial  = max(sqrt(N), 2·V)   — GP needs O(sqrt(N)) for surrogate,
//                                       plus 2 per variant for ranking.
//                                       Capped at budget/2 (safety).
//   topk_budget = min(V, budget/5)   — need 1 per variant for verification,
//                                       /5 ensures >50% goes to BO loop.
//   bo_budget  = budget - n_initial - topk_budget  (the residual).
//
// The old comment "30%/50%/20%" was approximate; the actual split is
// data-dependent via sqrt(N) and V. For our benchmark suite (N ~ 50–700,
// V ~ 1–8) the realised split is typically 25–40% init, 45–60% BO,
// 5–15% verify — close to the 30/50/20 heuristic but derived from
// coverage and GP convergence requirements, not arbitrary.
struct BayesOptParams
{
    double R;                   // Exploration ratio (0..1], 1.0 = brute force
    std::size_t total_budget;   // R * N: total eval budget
    std::size_t n_initial;      // Phase 1: initial random probes (30% of budget)
    std::size_t probes_per_var; // Phase 2a: extra probes per variant
    std::size_t bo_budget;      // Phase 2c: max successful BO iterations (50% of budget)
    std::size_t max_attempts;   // Phase 2c: max total attempts (incl. failures)
    std::size_t min_bo_iters;   // Phase 2c: min iters before EI convergence allowed
    double ei_thresh;           // Phase 2c: EI convergence threshold = 1/sqrt(N)
    std::size_t gp_cap;         // GP: max observations for Cholesky O(n^3) safety
    std::size_t topk_budget;    // Phase 3: verification budget (20% of budget)
    std::size_t topk_per_var;   // Phase 3: per-variant verification budget
    std::size_t sk_sweep_limit; // Phase 4: split_k sweep limit

    // Exploration ratio R = how much of the config space to evaluate.
    // R = 1.0 → brute force. R = 0.3 → evaluate ~30%.
    // Formula: R = sqrt(N_eff) / sqrt(N) where N_eff = effective config count.
    // For N_eff we factor in variant coverage: N_eff = max(N, V * sqrt(N/V)).
    // This gives: R ≈ 1/N^(1/4) for single-variant, higher with many variants.
    // No direction-specific or batch-specific thresholds.
    //
    // Note: dimensionality-aware scaling was tested (v22, paper §9.28.15)
    // but rejected: R_dim = sqrt(d/N) increased budget by 20% for medium
    // cases (N=200-500, d≈20) but didn't improve kernel quality (noise).
    // Root cause: 20D is too high for any affordable GP-BO sample size.
    static double ComputeR(std::size_t n_configs, std::size_t n_variants)
    {
        const double N = static_cast<double>(std::max<std::size_t>(1, n_configs));
        const double V = static_cast<double>(std::max<std::size_t>(1, n_variants));

        // Base: R = sqrt(sqrt(N)) / sqrt(N) = N^(-1/4)
        // N=16 → 0.5, N=81 → 0.33, N=256 → 0.25, N=625 → 0.2, N=2500 → 0.14
        double R = 1.0 / std::pow(N, 0.25);

        // Variant coverage: each variant needs sqrt(N/V) probes minimum
        // Total min probes = V * sqrt(N/V) = sqrt(N*V)
        // R_var = sqrt(N*V) / N = sqrt(V/N)
        double R_var = std::sqrt(V / N);
        R            = std::max(R, R_var);

        return std::min(R, 1.0);
    }

    static BayesOptParams
    Compute(std::size_t n_configs, std::size_t n_variants, std::size_t /*n_features*/)
    {
        BayesOptParams p;
        const double N = static_cast<double>(std::max<std::size_t>(1, n_configs));

        p.R = ComputeR(n_configs, n_variants);

        // Total evaluation budget = R * N
        p.total_budget = (p.R >= 1.0)
                             ? n_configs
                             : std::max<std::size_t>(n_variants + 1,
                                                     static_cast<std::size_t>(std::ceil(p.R * N)));

        // n_initial: GP surrogate convergence requires O(sqrt(N)) space-filling
        // probes (Srinivas et al., 2012 GP-UCB regret bound), plus ≥ 2 per
        // solver variant so Phase 2b ranking is non-degenerate.
        // Safety cap at budget/2: ensures ≥ 50% left for BO + verification.
        //
        // Note: dimension-aware scaling (n_initial ≥ d+1) was tested in v22
        // (paper §9.28.15) but rejected — extra initial probes didn't help
        // because 20D is too high for any affordable GP sample size.
        const auto env_init = env::value(MIOPEN_DEBUG_TUNING_BO_INITIAL);
        const std::size_t gp_min =
            std::max<std::size_t>(5, static_cast<std::size_t>(std::ceil(std::sqrt(N))));
        const std::size_t init_need = std::max(gp_min, static_cast<std::size_t>(2) * n_variants);
        p.n_initial                 = (env_init > 0) ? static_cast<std::size_t>(env_init)
                                                     : std::min(init_need, p.total_budget / 2);

        p.probes_per_var =
            std::max<std::size_t>(1, p.n_initial / std::max<std::size_t>(1, n_variants));

        // topk_budget: verification needs 1 eval per variant to confirm the top
        // candidates; min 3 to avoid degenerate top-K. Cap at budget/5 so ≥ 50%
        // is still available for the BO loop (the highest-leverage phase).
        const std::size_t verify_need = std::max<std::size_t>(3, n_variants);
        p.topk_budget                 = std::min(verify_need, p.total_budget / 5);
        p.topk_per_var =
            std::max<std::size_t>(1, p.topk_budget / std::max<std::size_t>(1, n_variants));

        p.bo_budget = std::max<std::size_t>(1, p.total_budget - p.n_initial - p.topk_budget);

        // max_attempts: allow bo_budget + sqrt(bo_budget) extra for failures
        p.max_attempts = p.bo_budget + static_cast<std::size_t>(
                                           std::ceil(std::sqrt(static_cast<double>(p.bo_budget))));

        // Min BO iters before EI convergence is allowed.
        //
        // Principled cap: log2(bo_budget) gives a "doubling" exploration
        // floor that grows slowly (5 for bo_budget=32, 7 for bo_budget=128)
        // and still ensures the EI threshold isn't hit prematurely from a
        // few unlucky early observations. This is the same scaling we use
        // for K_RESTARTS, so no new magic number.
        {
            const std::size_t log2_bo =
                (p.bo_budget < 2) ? 1
                                  : static_cast<std::size_t>(
                                        std::ceil(std::log2(static_cast<double>(p.bo_budget))));
            p.min_bo_iters = std::min(p.n_initial, std::max<std::size_t>(5, log2_bo));
        }

        // EI convergence threshold = 1/sqrt(N)
        {
            const auto env_ei = env::value(MIOPEN_DEBUG_TUNING_BO_EI_THRESH_E4);
            p.ei_thresh = (env_ei > 0) ? static_cast<double>(env_ei) / 10000.0 : 1.0 / std::sqrt(N);
        }

        // GP cap: O(n^3) Cholesky — use total_budget as natural cap
        {
            const auto env_cap = env::value(MIOPEN_DEBUG_TUNING_BO_GP_CAP);
            p.gp_cap           = (env_cap > 0) ? static_cast<std::size_t>(env_cap)
                                               : std::min(n_configs, p.total_budget);
        }

        // split_k sweep = half of verification budget
        p.sk_sweep_limit = std::max<std::size_t>(1, p.topk_budget / 2);

        return p;
    }

    std::string ToString() const
    {
        std::ostringstream oss;
        oss << "R=" << R << " total_budget=" << total_budget << " n_initial=" << n_initial
            << " probes_per_var=" << probes_per_var << " bo_budget=" << bo_budget
            << " max_attempts=" << max_attempts << " min_bo_iters=" << min_bo_iters
            << " ei_thresh=" << ei_thresh << " gp_cap=" << gp_cap << " topk_budget=" << topk_budget
            << " topk_per_var=" << topk_per_var << " sk_sweep_limit=" << sk_sweep_limit;
        return oss.str();
    }
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

    auto& bo_tracker = bayesian::GetBayesOptTracker();
    bo_tracker.solver_count++;
    const int solver_num = bo_tracker.solver_count;

    MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning,
                         false,
                         "BayesOpt",
                         MIOPEN_GET_FN_NAME,
                         ">> [Solver #" << std::to_string(solver_num) << "] " << s.SolverDbId()
                                        << " (" << std::to_string(n_configs) << " configs)");

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
    std::vector<int> variant_ids;
    const std::size_t n_features =
        bayesian::BuildFeatureMatrix(config_strings, features, variant_ids);

    if(n_features == 0)
    {
        MIOPEN_LOG_W("BayesianSearch: cannot parse features, falling back to brute force");
        return GenericSearch(s, context_, problem, invoke_ctx_, nullptr, true);
    }

    // Count variants for parameter computation
    std::set<int> unique_variants(variant_ids.begin(), variant_ids.end());
    const std::size_t n_variants_total = unique_variants.size();

    // --- Unified bypass: compute R and check if BO is worthwhile ---
    // R is purely algorithmic: f(N, V). No direction or batch thresholds.
    const auto params = bayesian::BayesOptParams::Compute(n_configs, n_variants_total, n_features);

    if(params.R >= 1.0)
    {
        MIOPEN_LOG_W("BayesianSearch: R=" << params.R << " (n=" << n_configs << " v="
                                          << n_variants_total << "), falling back to brute force");
        return GenericSearch(s, context_, problem, invoke_ctx_, nullptr, true);
    }

    MIOPEN_LOG_I("BayesianSearch: " << params.ToString());

    // --- State ---
    std::vector<bool> evaluated(n_configs, false);
    std::vector<double> observed_y;
    std::vector<std::size_t> observed_idx;
    PerformanceConfig best_config;
    float best_time    = std::numeric_limits<float>::max();
    std::size_t n_best = 0;
    bool is_passed     = false;

    // Log-transform (v24, v27) and other GP improvements were tested in the
    // extended experiment sweep (paper §9.28.21). v25 PCA remains production.

    // Get provided workspace size (match GenericSearch behavior)
    std::size_t provided_workspace = std::numeric_limits<std::size_t>::max();
    try
    {
        provided_workspace = invoke_ctx.GetWorkspaceSize();
    }
    catch(const miopen::Exception&)
    {
    }

    // --- Evaluate one config on GPU, return mean time or -1 on failure ---
    // Note: multi-fidelity (N_RUNS=3 for screening) was tested in v23
    // (paper §9.28.17) but rejected: -1.6pp kernel quality regression.
    // Measurement noise with 3 runs degrades GP surrogate accuracy.
    auto evaluate = [&](std::size_t idx) -> float {
        auto solution = s.GetSolution(context, problem, all_configs[idx]);
        try
        {
            if(solution.workspace_sz > provided_workspace)
                return -1.0f;

            auto invoker =
                profile_h.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
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

    // --- Dump mode: evaluate ALL configs, write CSV for offline replay ---
    if(env::value(MIOPEN_DEBUG_TUNING_BO_DUMP) != 0)
    {
        std::string solver_tag = s.SolverDbId();
        for(auto& c : solver_tag)
            if(c == '/' || c == ' ')
                c = '_';
        std::string csv_path = "/tmp/bo_dump_" + solver_tag + ".csv";
        std::ofstream csv(csv_path);
        csv << "idx,variant_id,config,time_ms";
        for(std::size_t f = 0; f < n_features; ++f)
            csv << ",f" << f;
        csv << "\n";

        MIOPEN_LOG_W("BayesianSearch DUMP mode: evaluating all " << n_configs << " configs → "
                                                                 << csv_path);

        for(std::size_t i = 0; i < n_configs; ++i)
        {
            auto solution = s.GetSolution(context, problem, all_configs[i]);
            float t       = -1.0f;
            try
            {
                if(solution.workspace_sz <= provided_workspace)
                {
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
                    t = miopen::removeHighOutliersAndGetMean(samples, 2.0f);
                }
            }
            catch(...)
            {
                t = -1.0f;
            }

            csv << i << "," << variant_ids[i] << ",\"" << config_strings[i] << "\"," << t;
            for(std::size_t f = 0; f < n_features; ++f)
                csv << "," << features[i][f];
            csv << "\n";

            if(t > 0 && t < best_time)
            {
                best_time   = t;
                best_config = all_configs[i];
                n_best      = i;
                is_passed   = true;
            }

            if(i % 50 == 0)
                MIOPEN_LOG_W("  DUMP progress: " << i << "/" << n_configs << " best=" << best_time);

            for(const auto& ki : solution.construction_params)
                profile_h.ClearProgram(ki.kernel_file, ki.comp_options);
        }
        csv.close();
        MIOPEN_LOG_W("BayesianSearch DUMP done: " << n_configs << " configs → " << csv_path
                                                  << " best=#" << n_best << " " << best_time
                                                  << "ms");

        if(!is_passed)
            MIOPEN_THROW("BayesianSearch dump: no config succeeded");

        auto& bo_tracker = bayesian::GetBayesOptTracker();
        bo_tracker.solver_count++;
        if(best_time < bo_tracker.best_time)
        {
            bo_tracker.best_time       = best_time;
            bo_tracker.best_solver     = s.SolverDbId();
            bo_tracker.best_solver_num = bo_tracker.solver_count;
        }

        const auto& invoker_final = profile_h.PrepareInvoker(*default_solution.invoker_factory,
                                                             default_solution.construction_params);
        invoker_final(profile_h, invoke_ctx);
        const auto default_time = profile_h.GetKernelTime();
        const auto score        = (best_time > 0.0f) ? default_time / best_time : 0.0f;
        MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning,
                             false,
                             "BayesOpt",
                             MIOPEN_GET_FN_NAME,
                             "     evaluated: " << n_configs << "/" << n_configs << " | best: #"
                                                << n_best << " | time: " << best_time << "ms"
                                                << " | score: " << score << "x");
        MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning,
                             false,
                             "BayesOpt",
                             MIOPEN_GET_FN_NAME,
                             "     config: " << best_config);
        MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning,
                             false,
                             "BayesOpt",
                             MIOPEN_GET_FN_NAME,
                             ">> [BEST] Solver #" << bo_tracker.solver_count << " "
                                                  << bo_tracker.best_solver
                                                  << " | time: " << bo_tracker.best_time << "ms");
        return best_config;
    }

    // --- Phase 1: Multi-restart variant-stratified initial probes ---
    // Run K independent restarts of Phase 1, each with a different seed.
    // Each restart probes n_initial/K configs. Keep the restart whose initial
    // probing found the best (lowest) time, then continue BO from there.
    // K=1 is equivalent to classic single-start BO.
    const std::size_t n_initial = params.n_initial;

    // Seed: fixed from env var (reproducible) or random_device (default)
    const auto env_seed = env::value(MIOPEN_DEBUG_TUNING_BO_SEED);
    std::random_device rd;
    const std::uint64_t base_seed = (env_seed > 0) ? env_seed : rd();

    // Build per-variant config lists
    std::map<int, std::vector<std::size_t>> variant_configs;
    for(std::size_t i = 0; i < n_configs; ++i)
        variant_configs[variant_ids[i]].push_back(i);

    const std::size_t n_variants = variant_configs.size();

    // Derive K_RESTARTS from problem structure:
    //   - More configs per variant → more diverse landscape → more restarts
    //   - But cap by budget (each restart needs ≥ n_variants+1 probes)
    const std::size_t cpv =
        (n_variants > 0) ? (n_configs + n_variants - 1) / n_variants : n_configs;
    const std::size_t k_log = static_cast<std::size_t>(
        std::ceil(std::log2(static_cast<double>(std::max<std::size_t>(cpv, 2)))));
    const std::size_t k_budget = n_initial / std::max<std::size_t>(n_variants + 1, 2);
    const std::size_t K_RESTARTS =
        std::max<std::size_t>(1, std::min({k_log, k_budget, std::size_t(5)}));

    // Per-restart budget: split n_initial across K restarts
    const std::size_t per_restart =
        std::max<std::size_t>(n_variants + 1, (n_initial + K_RESTARTS - 1) / K_RESTARTS);

    MIOPEN_LOG_I("BayesianSearch: phase1 n_initial="
                 << std::to_string(n_initial) << " n_variants=" << std::to_string(n_variants)
                 << " K_RESTARTS=" << std::to_string(K_RESTARTS)
                 << " per_restart=" << std::to_string(per_restart) << " seed=" << base_seed);

    // Track best restart
    float overall_best_time  = std::numeric_limits<float>::max();
    std::size_t best_restart = 0;

    struct RestartState
    {
        std::vector<std::size_t> obs_idx;
        std::vector<double> obs_y;
        std::vector<bool> eval_mask;
        PerformanceConfig best_cfg;
        float best_t;
        std::size_t best_n;
    };
    std::vector<RestartState> restarts(K_RESTARTS);

    for(std::size_t k = 0; k < K_RESTARTS; ++k)
    {
        std::mt19937 rng_k(base_seed + k);
        auto vc_copy = variant_configs;
        for(auto& kv : vc_copy)
            std::shuffle(kv.second.begin(), kv.second.end(), rng_k);

        auto& rs = restarts[k];
        rs.eval_mask.assign(n_configs, false);
        rs.best_t = std::numeric_limits<float>::max();
        rs.best_n = 0;

        std::size_t successes = 0;

        // Phase 1a: ensure at least 1 success per variant
        if(n_variants > 1)
        {
            for(auto& kv : vc_copy)
            {
                for(auto idx : kv.second)
                {
                    if(evaluated[idx] || rs.eval_mask[idx])
                        continue;
                    float t           = evaluate(idx);
                    evaluated[idx]    = true;
                    rs.eval_mask[idx] = true;
                    if(t < 0)
                        continue;
                    is_passed = true;
                    successes++;
                    rs.obs_y.push_back(static_cast<double>(t));
                    rs.obs_idx.push_back(idx);
                    if(t < rs.best_t)
                    {
                        rs.best_t   = t;
                        rs.best_cfg = all_configs[idx];
                        rs.best_n   = idx;
                    }
                    break;
                }
            }
        }

        // Phase 1b: fill remaining budget with random probes
        // Note: maximin space-filling (v28) was tested but rejected — O(n²)
        // overhead added ~80% wall-clock time with no quality improvement.
        std::vector<std::size_t> perm(n_configs);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_k);

        for(std::size_t i = 0; i < n_configs && successes < per_restart; ++i)
        {
            std::size_t idx = perm[i];
            if(evaluated[idx] || rs.eval_mask[idx])
                continue;
            float t           = evaluate(idx);
            evaluated[idx]    = true;
            rs.eval_mask[idx] = true;
            if(t < 0)
                continue;
            is_passed = true;
            successes++;
            rs.obs_y.push_back(static_cast<double>(t));
            rs.obs_idx.push_back(idx);
            if(t < rs.best_t)
            {
                rs.best_t   = t;
                rs.best_cfg = all_configs[idx];
                rs.best_n   = idx;
            }
        }

        MIOPEN_LOG_I("BayesianSearch: restart " << k << " probed " << successes
                                                << " configs, best_t=" << rs.best_t);

        if(rs.best_t < overall_best_time)
        {
            overall_best_time = rs.best_t;
            best_restart      = k;
        }
    }

    if(!is_passed)
    {
        MIOPEN_LOG_W("BayesianSearch: all initial probes failed, falling back to brute force");
        return GenericSearch(s, context_, problem, invoke_ctx_, nullptr, true);
    }

    // Merge all restart observations into the main state
    for(std::size_t k = 0; k < K_RESTARTS; ++k)
    {
        auto& rs = restarts[k];
        for(std::size_t j = 0; j < rs.obs_idx.size(); ++j)
        {
            observed_y.push_back(rs.obs_y[j]);
            observed_idx.push_back(rs.obs_idx[j]);
        }
    }
    // Use the best restart's best config
    {
        auto& rs    = restarts[best_restart];
        best_time   = rs.best_t;
        best_config = rs.best_cfg;
        n_best      = rs.best_n;
    }
    // Also check if any other restart found something even better across all obs
    for(std::size_t i = 0; i < observed_idx.size(); ++i)
    {
        float t = static_cast<float>(observed_y[i]);
        if(t < best_time)
        {
            best_time   = t;
            best_config = all_configs[observed_idx[i]];
            n_best      = observed_idx[i];
        }
    }

    MIOPEN_LOG_I("BayesianSearch: phase1 done, best_restart=" << best_restart << " total_obs="
                                                              << observed_idx.size()
                                                              << " best_t=" << best_time);

    // --- Phase 1.5: Diversity probing via farthest-point sampling ---
    // After random initial probes, the GP may have biased coverage of the
    // feature space — some regions (e.g., pipeline v3 sub-clusters) may be
    // covered only by slow configs, or missed entirely. Farthest-point
    // probing selects configs MAXIMALLY FAR from all already-evaluated configs,
    // guaranteeing diverse coverage regardless of categorical boundaries.
    // Budget: sqrt(n_observed) probes — proportional to existing observations.
    // Additional diversity is injected periodically in Phase 2c.
    {
        const std::size_t n_diversity = static_cast<std::size_t>(
            std::ceil(std::sqrt(static_cast<double>(observed_idx.size()))));
        std::size_t div_probed = 0;

        for(std::size_t di = 0; di < n_diversity; ++di)
        {
            // Find unevaluated config with maximum min-distance to any evaluated config
            double best_min_dist  = -1.0;
            std::size_t best_cand = 0;
            bool found            = false;

            for(std::size_t i = 0; i < n_configs; ++i)
            {
                if(evaluated[i])
                    continue;

                double min_d2 = std::numeric_limits<double>::max();
                for(auto eidx : observed_idx)
                {
                    double d2 = 0.0;
                    for(std::size_t f = 0; f < n_features; ++f)
                    {
                        double diff = features[i][f] - features[eidx][f];
                        d2 += diff * diff;
                    }
                    min_d2 = std::min(min_d2, d2);
                }

                if(min_d2 > best_min_dist)
                {
                    best_min_dist = min_d2;
                    best_cand     = i;
                    found         = true;
                }
            }

            if(!found)
                break;

            float t              = evaluate(best_cand);
            evaluated[best_cand] = true;
            if(t < 0)
                continue;
            div_probed++;
            observed_y.push_back(static_cast<double>(t));
            observed_idx.push_back(best_cand);
            if(t < best_time)
            {
                best_time   = t;
                best_config = all_configs[best_cand];
                n_best      = best_cand;
            }
        }
        if(div_probed > 0)
            MIOPEN_LOG_I("BayesianSearch: phase1.5 diversity probing probed="
                         << div_probed << " total_obs=" << observed_idx.size());
    }

    // --- Phase 2a: per-variant exploration probes ---
    const std::size_t probes_per_variant = params.probes_per_var;
    {
        std::mt19937 rng_2a(base_seed + K_RESTARTS);
        for(auto& kv : variant_configs)
            std::shuffle(kv.second.begin(), kv.second.end(), rng_2a);
    }

    if(n_variants > 1)
    {
        std::map<int, std::size_t> variant_successes;
        for(std::size_t i = 0; i < observed_idx.size(); ++i)
            variant_successes[variant_ids[observed_idx[i]]]++;

        for(auto& kv : variant_configs)
        {
            int vid          = kv.first;
            std::size_t have = variant_successes[vid];
            if(have >= probes_per_variant)
                continue;
            std::size_t need = probes_per_variant - have;
            for(auto idx : kv.second)
            {
                if(need == 0)
                    break;
                if(evaluated[idx])
                    continue;
                float t        = evaluate(idx);
                evaluated[idx] = true;
                if(t < 0)
                    continue;
                need--;
                observed_y.push_back(static_cast<double>(t));
                observed_idx.push_back(idx);
                if(t < best_time)
                {
                    best_time   = t;
                    best_config = all_configs[idx];
                    n_best      = idx;
                }
            }
        }

        MIOPEN_LOG_I("BayesianSearch: phase2a probing done, total obs="
                     << std::to_string(observed_idx.size()));
    }

    // --- Phase 2b: variant ranking (informational, no filtering) ---
    // GP's EI naturally balances exploration across variants; manual filtering
    // caused quality regressions when initial probes were unlucky.
    if(n_variants > 1)
    {
        struct VarStat
        {
            int vid;
            double best_t;
        };
        std::vector<VarStat> vstats;
        for(auto& kv : variant_configs)
        {
            int vid      = kv.first;
            double vbest = std::numeric_limits<double>::max();
            for(std::size_t i = 0; i < observed_idx.size(); ++i)
                if(variant_ids[observed_idx[i]] == vid && observed_y[i] < vbest)
                    vbest = observed_y[i];
            vstats.push_back({vid, vbest});
        }
        std::sort(vstats.begin(), vstats.end(), [](const VarStat& a, const VarStat& b) {
            return a.best_t < b.best_t;
        });

        std::string vstr;
        for(auto& vs : vstats)
            vstr += "v" + std::to_string(vs.vid) + "=" + std::to_string(vs.best_t) + " ";
        MIOPEN_LOG_I("BayesianSearch: phase2b variant ranking: " << vstr);
    }

    // --- Phase 2c: BO guided by GP + EI (all variants) ---
    std::size_t remaining = 0;
    for(std::size_t i = 0; i < n_configs; ++i)
        if(!evaluated[i])
            remaining++;

    // Respect total_budget: subtract what Phase 1+2a already used
    const std::size_t already_used = observed_idx.size();
    const std::size_t budget_left =
        (params.total_budget > already_used) ? params.total_budget - already_used : 0;
    const std::size_t bo_budget    = std::min(remaining, std::min(params.bo_budget, budget_left));
    const double ei_thresh_raw     = params.ei_thresh;
    const std::size_t gp_cap       = params.gp_cap;
    const std::size_t max_attempts = std::min(
        remaining,
        bo_budget + static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(bo_budget)))));
    const std::size_t min_bo_iters = params.min_bo_iters;

    // v21 — Cross-solver early termination.
    //
    // BayesOptTracker.best_time holds the best kernel time found by any
    // *prior* solver for this same convolution. If our Phase-1 best is
    // already much worse, there's little point running a full BO loop —
    // even if we close the gap by the typical BO improvement (~5-15%),
    // we'd still lose to the prior solver's result.
    //
    // Gate: skip the BO loop (but keep Phase 3 verification of current
    // best) when our init best > 2× the cross-solver global best. The
    // 2× margin is generous — it only fires when we're clearly outclassed.
    // No new magic: 2× is the natural "same order of magnitude" threshold,
    // and matches the BO literature's typical recommendation for when to
    // restart vs continue.
    //
    // This is safe because MIOpen's framework runs ALL applicable solvers
    // and picks the overall best — we're not discarding the solver, just
    // saving its BO budget. Phase 1 + Phase 3 still run.
    const float cross_solver_best = bo_tracker.best_time;
    bool skip_bo_loop             = false;
    if(cross_solver_best < std::numeric_limits<float>::max() &&
       best_time > 2.0f * cross_solver_best)
    {
        MIOPEN_LOG_I("BayesianSearch: cross-solver skip — init best="
                     << best_time << "ms > 2× global best=" << cross_solver_best
                     << "ms from solver #" << bo_tracker.best_solver_num << " "
                     << bo_tracker.best_solver);
        skip_bo_loop = true;
    }

    MIOPEN_LOG_I("BayesianSearch: phase2c remaining="
                 << std::to_string(remaining) << " bo_budget=" << std::to_string(bo_budget)
                 << " max_attempts=" << std::to_string(max_attempts)
                 << " min_bo_iters=" << std::to_string(min_bo_iters)
                 << " ei_thresh=" << ei_thresh_raw << " gp_cap=" << std::to_string(gp_cap)
                 << " cross_skip=" << (skip_bo_loop ? "YES" : "no"));

    bayesian::GaussianProcess gp;
    std::size_t successful_iters = 0;
    std::size_t total_attempts   = 0;
    std::mt19937 rng_bo(base_seed + K_RESTARTS + 2);

    // Note: TS injection was tested at 20-33% rates but didn't improve
    // cases like F25 where the GP posterior is overconfident in wrong
    // regions. Only pure TS (100%) helped F25 but caused regressions
    // elsewhere. PCA feature reduction (v25) partially mitigates this.

    // v20 — Stale-best early stopping (P3).
    //
    // In addition to the EI-threshold convergence check below, track how
    // many successful iterations have passed since the best observation
    // improved. When that streak exceeds a "patience" window *and* the
    // min-iteration floor has been satisfied, exit the BO loop early —
    // the GP has stopped finding improvements, so any further iterations
    // are wasted budget.
    //
    // v20a used patience = sqrt(bo_budget) which saved 32% wall-clock
    // but cost −1.45 pp kernel quality — too aggressive, cutting
    // exploration before GP-driven late-stage improvements could land.
    //
    // v20b (this): patience = ceil(2 * sqrt(bo_budget)). This gives the
    // GP about twice as long to break through a plateau, while still
    // halting on genuinely-stale searches.
    //   bo_budget = 4 → 4,  16 → 8,  64 → 16,  256 → 32.
    //
    // Guard: only arm stale-best after `min_bo_iters` successful iters
    // so we don't halt before the GP has had a chance to explore.
    const std::size_t stale_patience = std::max<std::size_t>(
        4, static_cast<std::size_t>(std::ceil(2.0 * std::sqrt(static_cast<double>(bo_budget)))));
    std::size_t iters_since_best = 0;

    while(!skip_bo_loop && successful_iters < bo_budget && total_attempts < max_attempts)
    {
        std::vector<std::size_t> cands;
        for(std::size_t i = 0; i < n_configs; ++i)
        {
            if(evaluated[i])
                continue;
            cands.push_back(i);
        }
        if(cands.empty())
            break;

        const std::size_t n_cand = cands.size();

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
            std::size_t best_pos = 0;
            for(std::size_t i = 1; i < n_total_obs; ++i)
                if(observed_y[i] < observed_y[best_pos])
                    best_pos = i;

            const std::size_t tail_start = n_total_obs - (gp_cap - 1);
            bool best_in_tail            = (best_pos >= tail_start);

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

        std::vector<double> X_obs(n_obs * n_features);
        for(std::size_t i = 0; i < n_obs; ++i)
            for(std::size_t j = 0; j < n_features; ++j)
                X_obs[i * n_features + j] = features[gp_sel_idx[i]][j];

        // v25 PCA: reduce feature dimensionality before GP fitting
        auto pca = bayesian::PCAFit(X_obs, n_obs, n_features);
        std::vector<double> X_obs_pca;
        bayesian::PCATransform(pca, X_obs, n_obs, X_obs_pca);
        const std::size_t n_feat_gp = pca.active() ? pca.k : n_features;

        gp.Fit(X_obs_pca, gp_sel_y, n_obs, n_feat_gp);
        MIOPEN_LOG_I2("BayesianSearch: GP noise=" << gp.GetNoise() << " ls=" << gp.GetLengthScale()
                                                  << " n_obs=" << n_obs << " pca_k=" << n_feat_gp
                                                  << "/" << n_features);

        std::vector<double> X_cand(n_cand * n_features);
        for(std::size_t i = 0; i < n_cand; ++i)
            for(std::size_t j = 0; j < n_features; ++j)
                X_cand[i * n_features + j] = features[cands[i]][j];

        // PCA-transform candidates with same projection
        std::vector<double> X_cand_pca;
        bayesian::PCATransform(pca, X_cand, n_cand, X_cand_pca);

        std::vector<double> mu, sigma;
        gp.Predict(X_cand_pca, n_cand, mu, sigma);

        double max_ei = 0.0;
        std::size_t ei_pick =
            bayesian::SelectNextByEI(mu, sigma, static_cast<double>(best_time), n_cand, &max_ei);

        // EI convergence check
        if(ei_thresh_raw > 0.0 && best_time < std::numeric_limits<float>::max() &&
           successful_iters >= min_bo_iters)
        {
            const double ei_threshold = static_cast<double>(best_time) * ei_thresh_raw;
            if(max_ei < ei_threshold)
            {
                MIOPEN_LOG_I("BayesianSearch: EI converged at iter="
                             << std::to_string(successful_iters) << " max_ei=" << max_ei
                             << " < thresh=" << ei_threshold);
                break;
            }
        }

        std::size_t pick = ei_pick;

        std::size_t next = cands[pick];
        float t          = evaluate(next);
        evaluated[next]  = true;
        total_attempts++;
        if(t < 0)
            continue;

        successful_iters++;
        observed_y.push_back(static_cast<double>(t));
        observed_idx.push_back(next);

        if(t < best_time)
        {
            best_time        = t;
            best_config      = all_configs[next];
            n_best           = next;
            iters_since_best = 0;
            MIOPEN_LOG_I("BayesianSearch: iter=" << std::to_string(successful_iters)
                                                 << " (attempts=" << std::to_string(total_attempts)
                                                 << ") NEW BEST #" << std::to_string(next)
                                                 << " t=" << t);
        }
        else
        {
            ++iters_since_best;
            MIOPEN_LOG_I2("BayesianSearch: iter="
                          << successful_iters << " (attempts=" << total_attempts << ") #" << next
                          << " t=" << t << " best=" << best_time << " stale=" << iters_since_best
                          << "/" << stale_patience);
        }

        // Stale-best early stop: no improvement for `stale_patience` iters
        if(iters_since_best >= stale_patience && successful_iters >= min_bo_iters)
        {
            MIOPEN_LOG_I("BayesianSearch: stale-best early stop at iter="
                         << std::to_string(successful_iters)
                         << " stale=" << std::to_string(iters_since_best)
                         << " patience=" << std::to_string(stale_patience));
            break;
        }
    }

    // --- Phase 3: Variant-aware Top-K verification ---
    // GP can be biased toward heavily-explored variants. Instead of global top-K,
    // pick top-K PER VARIANT GROUP so under-explored variants get verified too.
    // For non-CK (variant_id == -1) and single-variant solvers, this degenerates
    // to a standard top-K.
    {
        // Group unevaluated configs by variant
        std::map<int, std::vector<std::size_t>> uneval_by_variant;
        for(std::size_t i = 0; i < n_configs; ++i)
            if(!evaluated[i])
                uneval_by_variant[variant_ids[i]].push_back(i);

        const std::size_t n_variants = uneval_by_variant.size();
        if(n_variants > 0 && observed_idx.size() >= 3)
        {
            // Cap Phase 3 by remaining total budget
            const std::size_t phase3_used = observed_idx.size();
            const std::size_t phase3_left =
                (params.total_budget > phase3_used) ? params.total_budget - phase3_used : 0;
            const std::size_t total_budget = std::min(params.topk_budget, phase3_left);
            const std::size_t per_variant =
                (total_budget > 0)
                    ? std::max<std::size_t>(1, total_budget / std::max<std::size_t>(1, n_variants))
                    : 0;

            // Re-fit GP on all observations (v25: with PCA reduction)
            std::vector<double> X_all(observed_idx.size() * n_features);
            for(std::size_t i = 0; i < observed_idx.size(); ++i)
                for(std::size_t j = 0; j < n_features; ++j)
                    X_all[i * n_features + j] = features[observed_idx[i]][j];
            auto pca3 = bayesian::PCAFit(X_all, observed_idx.size(), n_features);
            std::vector<double> X_all_pca;
            bayesian::PCATransform(pca3, X_all, observed_idx.size(), X_all_pca);
            const std::size_t n_feat_p3 = pca3.active() ? pca3.k : n_features;
            gp.Fit(X_all_pca, observed_y, observed_idx.size(), n_feat_p3);

            std::size_t topk_improved = 0;
            std::size_t topk_total    = 0;

            for(auto& [vid, vidx_list] : uneval_by_variant)
            {
                const std::size_t n_vcand = vidx_list.size();
                const std::size_t k       = std::min(per_variant, n_vcand);
                if(k == 0)
                    continue;

                // GP predict for this variant's unevaluated configs (PCA-projected)
                std::vector<double> X_v(n_vcand * n_features);
                for(std::size_t i = 0; i < n_vcand; ++i)
                    for(std::size_t j = 0; j < n_features; ++j)
                        X_v[i * n_features + j] = features[vidx_list[i]][j];
                std::vector<double> X_v_pca;
                bayesian::PCATransform(pca3, X_v, n_vcand, X_v_pca);

                std::vector<double> mu_v, sigma_v;
                gp.Predict(X_v_pca, n_vcand, mu_v, sigma_v);

                // Sort by predicted time within this variant
                std::vector<std::size_t> rank(n_vcand);
                std::iota(rank.begin(), rank.end(), 0);
                std::sort(rank.begin(), rank.end(), [&mu_v](std::size_t a, std::size_t b) {
                    return mu_v[a] < mu_v[b];
                });

                for(std::size_t ki = 0; ki < k; ++ki)
                {
                    std::size_t idx = vidx_list[rank[ki]];
                    float t         = evaluate(idx);
                    evaluated[idx]  = true;
                    topk_total++;
                    if(t < 0)
                        continue;
                    observed_y.push_back(static_cast<double>(t));
                    observed_idx.push_back(idx);
                    if(t < best_time)
                    {
                        best_time   = t;
                        best_config = all_configs[idx];
                        n_best      = idx;
                        topk_improved++;
                        MIOPEN_LOG_I("BayesianSearch: topK variant=" << vid << " #" << idx
                                                                     << " t=" << t << " NEW BEST");
                    }
                }
            }
            MIOPEN_LOG_I("BayesianSearch: phase3 variant-aware topK="
                         << topk_total << " variants=" << n_variants
                         << " improved=" << topk_improved);
        }
    }

    // --- Phase 4: split_k sweep — try split_k=-1 sibling of top configs ---
    {
        // Build map: base config (without +N suffix) -> index of the split_k=-1 variant
        std::unordered_map<std::string, std::size_t> base_to_auto;
        for(std::size_t i = 0; i < n_configs; ++i)
        {
            const auto& cs = config_strings[i];
            auto pos       = cs.rfind("+-1");
            if(pos != std::string::npos && pos + 3 == cs.size())
            {
                auto base          = cs.substr(0, pos);
                base_to_auto[base] = i;
            }
        }

        if(!base_to_auto.empty())
        {
            // Sort observed by time (ascending)
            std::vector<std::pair<std::size_t, double>> sorted_obs;
            sorted_obs.reserve(observed_idx.size());
            for(std::size_t i = 0; i < observed_idx.size(); ++i)
                sorted_obs.emplace_back(observed_idx[i], observed_y[i]);
            std::sort(sorted_obs.begin(), sorted_obs.end(), [](const auto& a, const auto& b) {
                return a.second < b.second;
            });

            int sk_tried             = 0;
            const int SK_SWEEP_LIMIT = static_cast<int>(params.sk_sweep_limit);
            for(const auto& [obs_idx, obs_time] : sorted_obs)
            {
                if(sk_tried >= SK_SWEEP_LIMIT)
                    break;
                const auto& cs = config_strings[obs_idx];
                auto plus_pos  = cs.rfind('+');
                if(plus_pos == std::string::npos)
                    continue;
                if(cs.substr(plus_pos) == "+-1")
                    continue; // already auto
                auto base = cs.substr(0, plus_pos);
                auto it   = base_to_auto.find(base);
                if(it == base_to_auto.end())
                    continue;
                auto auto_idx = it->second;
                if(evaluated[auto_idx])
                    continue;

                float t             = evaluate(auto_idx);
                evaluated[auto_idx] = true;
                sk_tried++;
                if(t < 0)
                    continue;
                is_passed = true;
                observed_y.push_back(static_cast<double>(t));
                observed_idx.push_back(auto_idx);
                if(t < best_time)
                {
                    best_time   = t;
                    best_config = all_configs[auto_idx];
                    n_best      = auto_idx;
                    MIOPEN_LOG_I("BayesianSearch: split_k sweep #" << auto_idx << " t=" << t
                                                                   << " NEW BEST");
                }
            }
            if(sk_tried > 0)
                MIOPEN_LOG_I("BayesianSearch: phase4 split_k sweep tried=" << sk_tried);
        }
    }

    // --- Phase 5: Feature-space neighborhood search ---
    // The GP can miss optimal configs separated by categorical features
    // (e.g., different pipeline versions). Search neighborhoods of multiple
    // promising observed configs to bridge these gaps.
    {
        const std::size_t phase5_used = observed_idx.size();
        const std::size_t phase5_left =
            (params.total_budget > phase5_used) ? params.total_budget - phase5_used : 0;
        const std::size_t nb_budget =
            std::min<std::size_t>(phase5_left, std::max<std::size_t>(3, params.topk_budget));

        if(nb_budget > 0 && observed_idx.size() >= 2)
        {
            // Pick diverse seed configs: top-K observed configs that are
            // at least min_sep apart in feature space (avoids clustering
            // all neighborhood probes in the same region)
            std::vector<std::pair<double, std::size_t>> sorted_obs;
            for(std::size_t i = 0; i < observed_idx.size(); ++i)
                sorted_obs.emplace_back(observed_y[i], observed_idx[i]);
            std::sort(sorted_obs.begin(), sorted_obs.end());

            // Derive min separation from median pairwise distance of observations
            std::vector<double> obs_dists;
            for(std::size_t a = 0; a < sorted_obs.size(); ++a)
                for(std::size_t b = a + 1; b < sorted_obs.size(); ++b)
                {
                    double d2 = 0.0;
                    for(std::size_t f = 0; f < n_features; ++f)
                    {
                        double diff =
                            features[sorted_obs[a].second][f] - features[sorted_obs[b].second][f];
                        d2 += diff * diff;
                    }
                    obs_dists.push_back(d2);
                }
            double min_sep2 = 0.5;
            if(!obs_dists.empty())
            {
                std::sort(obs_dists.begin(), obs_dists.end());
                min_sep2 = obs_dists[obs_dists.size() / 4]; // 25th percentile
            }

            const std::size_t max_seeds = std::max<std::size_t>(2, n_variants);
            std::vector<std::size_t> seeds;
            for(const auto& [ytime, yidx] : sorted_obs)
            {
                if(seeds.size() >= max_seeds)
                    break;
                bool far_enough = true;
                for(auto s : seeds)
                {
                    double d2 = 0.0;
                    for(std::size_t f = 0; f < n_features; ++f)
                    {
                        double diff = features[yidx][f] - features[s][f];
                        d2 += diff * diff;
                    }
                    if(d2 < min_sep2)
                    {
                        far_enough = false;
                        break;
                    }
                }
                if(far_enough)
                    seeds.push_back(yidx);
            }
            if(seeds.empty())
                seeds.push_back(n_best);

            const std::size_t per_seed = std::max<std::size_t>(1, nb_budget / seeds.size());
            std::size_t nb_improved    = 0;
            std::size_t nb_total       = 0;

            for(auto seed : seeds)
            {
                std::vector<std::pair<double, std::size_t>> dist_idx;
                for(std::size_t i = 0; i < n_configs; ++i)
                {
                    if(evaluated[i])
                        continue;
                    double d2 = 0.0;
                    for(std::size_t f = 0; f < n_features; ++f)
                    {
                        double diff = features[i][f] - features[seed][f];
                        d2 += diff * diff;
                    }
                    dist_idx.emplace_back(d2, i);
                }
                std::sort(dist_idx.begin(), dist_idx.end());

                const std::size_t k = std::min(per_seed, dist_idx.size());
                for(std::size_t ni = 0; ni < k; ++ni)
                {
                    std::size_t idx = dist_idx[ni].second;
                    float t         = evaluate(idx);
                    evaluated[idx]  = true;
                    nb_total++;
                    if(t < 0)
                        continue;
                    observed_y.push_back(static_cast<double>(t));
                    observed_idx.push_back(idx);
                    if(t < best_time)
                    {
                        best_time   = t;
                        best_config = all_configs[idx];
                        n_best      = idx;
                        nb_improved++;
                    }
                }
            }
            MIOPEN_LOG_I("BayesianSearch: phase5 neighborhood seeds="
                         << seeds.size() << " searched=" << nb_total
                         << " improved=" << nb_improved);
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

    MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning,
                         false,
                         "BayesOpt",
                         MIOPEN_GET_FN_NAME,
                         "     evaluated: " << std::to_string(observed_idx.size()) << "/"
                                            << std::to_string(n_configs) << " | best: #"
                                            << std::to_string(n_best) << " | time: " << best_time
                                            << "ms"
                                            << " | score: " << score << "x");
    MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning,
                         false,
                         "BayesOpt",
                         MIOPEN_GET_FN_NAME,
                         "     config: " << best_config);

    if(best_time < bo_tracker.best_time)
    {
        bo_tracker.best_time       = best_time;
        bo_tracker.best_solver     = s.SolverDbId();
        bo_tracker.best_solver_num = solver_num;
    }
    MIOPEN_LOG_XQ_CUSTOM(miopen::LoggingLevel::Warning,
                         false,
                         "BayesOpt",
                         MIOPEN_GET_FN_NAME,
                         ">> [BEST] Solver #" << std::to_string(bo_tracker.best_solver_num) << " "
                                              << bo_tracker.best_solver
                                              << " | time: " << bo_tracker.best_time << "ms");

    return best_config;
}

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_BAYESIAN_SEARCH_HPP_
