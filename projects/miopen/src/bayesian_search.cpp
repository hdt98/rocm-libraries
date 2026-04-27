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

#include <miopen/bayesian_search.hpp>

#include <cmath>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>
#include <string>
#include <cctype>

namespace miopen {
namespace solver {
namespace bayesian {

// ===================================================================
// PCA feature reduction (v25)
// Reduce high-dimensional feature spaces to top-k principal components
// explaining >= 95% variance. Uses Jacobi eigendecomposition (no
// external dependency). d×d covariance matrix is at most ~20×20.
// ===================================================================

PCAState PCAFit(const std::vector<double>& X, std::size_t n, std::size_t d, double variance_ratio)
{
    PCAState pca;
    pca.d_orig = d;

    if(n < 2 || d < 2)
    {
        pca.k = d;
        return pca;
    }

    // Center: compute column means
    pca.mean.assign(d, 0.0);
    for(std::size_t i = 0; i < n; ++i)
        for(std::size_t j = 0; j < d; ++j)
            pca.mean[j] += X[i * d + j];
    for(std::size_t j = 0; j < d; ++j)
        pca.mean[j] /= static_cast<double>(n);

    // Centered data
    std::vector<double> Xc(n * d);
    for(std::size_t i = 0; i < n; ++i)
        for(std::size_t j = 0; j < d; ++j)
            Xc[i * d + j] = X[i * d + j] - pca.mean[j];

    // Covariance matrix C = X^T X / (n-1), stored row-major d×d
    std::vector<double> C(d * d, 0.0);
    const double norm = 1.0 / static_cast<double>(n - 1);
    for(std::size_t i = 0; i < n; ++i)
        for(std::size_t p = 0; p < d; ++p)
            for(std::size_t q = p; q < d; ++q)
                C[p * d + q] += Xc[i * d + p] * Xc[i * d + q];
    for(std::size_t p = 0; p < d; ++p)
        for(std::size_t q = p; q < d; ++q)
        {
            C[p * d + q] *= norm;
            C[q * d + p] = C[p * d + q];
        }

    // Jacobi eigendecomposition of symmetric C
    std::vector<double> V(d * d, 0.0);
    for(std::size_t i = 0; i < d; ++i)
        V[i * d + i] = 1.0;

    constexpr int MAX_SWEEPS = 100;
    for(int sweep = 0; sweep < MAX_SWEEPS; ++sweep)
    {
        double off_diag = 0.0;
        for(std::size_t p = 0; p < d; ++p)
            for(std::size_t q = p + 1; q < d; ++q)
                off_diag += C[p * d + q] * C[p * d + q];
        if(off_diag < 1e-20)
            break;

        for(std::size_t p = 0; p < d; ++p)
        {
            for(std::size_t q = p + 1; q < d; ++q)
            {
                double apq = C[p * d + q];
                if(std::abs(apq) < 1e-15)
                    continue;

                double tau = (C[q * d + q] - C[p * d + p]) / (2.0 * apq);
                double t   = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                double c   = 1.0 / std::sqrt(1.0 + t * t);
                double s   = t * c;

                // Rotate C
                double app   = C[p * d + p] - t * apq;
                double aqq   = C[q * d + q] + t * apq;
                C[p * d + p] = app;
                C[q * d + q] = aqq;
                C[p * d + q] = 0.0;
                C[q * d + p] = 0.0;
                for(std::size_t r = 0; r < d; ++r)
                {
                    if(r == p || r == q)
                        continue;
                    double crp = C[r * d + p], crq = C[r * d + q];
                    C[r * d + p] = c * crp - s * crq;
                    C[p * d + r] = C[r * d + p];
                    C[r * d + q] = s * crp + c * crq;
                    C[q * d + r] = C[r * d + q];
                }
                // Rotate V
                for(std::size_t r = 0; r < d; ++r)
                {
                    double vrp = V[r * d + p], vrq = V[r * d + q];
                    V[r * d + p] = c * vrp - s * vrq;
                    V[r * d + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Extract eigenvalues and sort descending
    std::vector<double> eigenvalues(d);
    std::vector<std::size_t> order(d);
    for(std::size_t i = 0; i < d; ++i)
        eigenvalues[i] = std::max(C[i * d + i], 0.0);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return eigenvalues[a] > eigenvalues[b];
    });

    double total_var = 0.0;
    for(auto ev : eigenvalues)
        total_var += ev;
    if(total_var < 1e-15)
    {
        pca.k = d;
        return pca;
    }

    // Determine k: smallest number of components capturing >= variance_ratio
    double cum = 0.0;
    pca.k      = d;
    for(std::size_t i = 0; i < d; ++i)
    {
        cum += eigenvalues[order[i]];
        if(cum / total_var >= variance_ratio)
        {
            pca.k = i + 1;
            break;
        }
    }
    pca.k = std::max(pca.k, std::size_t(2)); // at least 2 components

    // Store top-k eigenvectors as rows of components (k × d)
    pca.components.resize(pca.k * d);
    for(std::size_t i = 0; i < pca.k; ++i)
    {
        std::size_t col = order[i];
        for(std::size_t j = 0; j < d; ++j)
            pca.components[i * d + j] = V[j * d + col];
    }

    return pca;
}

void PCATransform(const PCAState& pca,
                  const std::vector<double>& X_in,
                  std::size_t n,
                  std::vector<double>& X_out)
{
    if(pca.k == 0 || pca.k == pca.d_orig || pca.components.empty())
    {
        X_out = X_in;
        return;
    }
    const std::size_t d = pca.d_orig;
    const std::size_t k = pca.k;
    X_out.resize(n * k);
    for(std::size_t i = 0; i < n; ++i)
    {
        for(std::size_t c = 0; c < k; ++c)
        {
            double val = 0.0;
            for(std::size_t j = 0; j < d; ++j)
                val += (X_in[i * d + j] - pca.mean[j]) * pca.components[c * d + j];
            X_out[i * k + c] = val;
        }
    }
}

// ===================================================================
// GaussianProcess implementation
// ===================================================================

// Matern 5/2 kernel (isotropic).
// r^2 = sum_i ((x1[i] - x2[i]) / ls)^2
double GaussianProcess::Matern52(const double* x1, const double* x2, std::size_t d) const
{
    double r2 = 0.0;
    for(std::size_t i = 0; i < d; ++i)
    {
        double diff = (x1[i] - x2[i]) / length_scale_;
        r2 += diff * diff;
    }
    double r  = std::sqrt(r2);
    double s5 = std::sqrt(5.0);
    return (1.0 + s5 * r + (5.0 / 3.0) * r2) * std::exp(-s5 * r);
}

// Solve L * L^T * x = b via forward + backward substitution
void GaussianProcess::CholeskySolve(const std::vector<double>& L,
                                    const std::vector<double>& b,
                                    std::vector<double>& x,
                                    std::size_t n) const
{
    // Forward: L * z = b
    std::vector<double> z(n, 0.0);
    for(std::size_t i = 0; i < n; ++i)
    {
        double s = b[i];
        for(std::size_t j = 0; j < i; ++j)
            s -= L[i * n + j] * z[j];
        z[i] = s / L[i * n + i];
    }
    // Backward: L^T * x = z
    x.resize(n);
    for(std::size_t i = n; i-- > 0;)
    {
        double s = z[i];
        for(std::size_t j = i + 1; j < n; ++j)
            s -= L[j * n + i] * x[j]; // L^T[i][j] = L[j][i]
        x[i] = s / L[i * n + i];
    }
}

// Adaptive GP noise from measurement-to-signal variance ratio.
//
// After y-normalization to unit variance, the noise parameter represents the
// fraction of total variance attributable to measurement error:
//
//   noise = (measurement_std / signal_std)^2
//         = (MEASUREMENT_CV / data_cv)^2
//
// MEASUREMENT_CV is the coefficient of variation of repeated GPU timing
// measurements for the *same* configuration (~0.5% for modern GPUs).
//
// At data_cv ≈ 0.50 (typical for diverse solver configs), this gives
// noise = 1e-4 — identical to the previously hardcoded value.
// As data_cv decreases (tightly clustered timings where measurement jitter
// dominates), noise increases automatically.
//
// Examples:
//   data_cv = 0.50 → noise = 1.0e-4  (same as old default)
//   data_cv = 0.10 → noise = 2.5e-3
//   data_cv = 0.05 → noise = 1.0e-2
//   data_cv = 0.01 → noise = 5.0e-2  (capped)
double GaussianProcess::EstimateNoise(const std::vector<double>& y_obs, std::size_t n_obs)
{
    constexpr double MEASUREMENT_CV = 0.005;
    constexpr double NOISE_FLOOR    = 1e-4;
    constexpr double NOISE_CAP      = 0.05;

    if(n_obs < 4)
        return NOISE_FLOOR;

    double ymean = 0.0;
    for(std::size_t i = 0; i < n_obs; ++i)
        ymean += y_obs[i];
    ymean /= static_cast<double>(n_obs);

    if(ymean < 1e-10)
        return NOISE_FLOOR;

    double sum_sq = 0.0;
    for(std::size_t i = 0; i < n_obs; ++i)
        sum_sq += (y_obs[i] - ymean) * (y_obs[i] - ymean);
    double ystd = std::sqrt(sum_sq / static_cast<double>(n_obs));

    double cv = ystd / ymean;
    if(cv < 1e-10)
        return NOISE_CAP;

    double ratio     = MEASUREMENT_CV / cv;
    double noise_est = ratio * ratio;

    return std::max(NOISE_FLOOR, std::min(NOISE_CAP, noise_est));
}

void GaussianProcess::Fit(const std::vector<double>& X_obs,
                          const std::vector<double>& y_obs,
                          std::size_t n_obs,
                          std::size_t n_features)
{
    n_obs_      = n_obs;
    n_features_ = n_features;
    X_obs_      = X_obs;

    // Estimate noise from data before normalization
    noise_ = EstimateNoise(y_obs, n_obs);

    // Normalize y to zero mean, unit variance
    y_mean_ = 0.0;
    for(auto v : y_obs)
        y_mean_ += v;
    y_mean_ /= static_cast<double>(n_obs);

    double sum_sq = 0.0;
    for(auto v : y_obs)
        sum_sq += (v - y_mean_) * (v - y_mean_);
    y_std_ = std::sqrt(sum_sq / static_cast<double>(n_obs));
    if(y_std_ < 1e-10)
        y_std_ = 1.0;

    y_norm_.resize(n_obs);
    for(std::size_t i = 0; i < n_obs; ++i)
        y_norm_[i] = (y_obs[i] - y_mean_) / y_std_;

    // Isotropic length-scale: ls = median(pairwise euclidean distance) / sqrt(d).
    // Clamped to [1e-3, 10.0] for numerical stability.
    constexpr double LS_FLOOR = 1e-3;
    constexpr double LS_CAP   = 10.0;

    length_scale_ = 1.0;

    if(n_obs > 1 && n_features > 0)
    {
        const double sqrt_d = std::sqrt(static_cast<double>(n_features));
        std::vector<double> pair_dists;
        pair_dists.reserve(n_obs * (n_obs - 1) / 2);
        for(std::size_t i = 0; i < n_obs; ++i)
            for(std::size_t j = i + 1; j < n_obs; ++j)
            {
                double d2 = 0.0;
                for(std::size_t k = 0; k < n_features; ++k)
                {
                    double diff = X_obs[i * n_features + k] - X_obs[j * n_features + k];
                    d2 += diff * diff;
                }
                pair_dists.push_back(std::sqrt(d2));
            }
        std::sort(pair_dists.begin(), pair_dists.end());
        double med    = pair_dists[pair_dists.size() / 2];
        double ls     = med / sqrt_d;
        length_scale_ = std::max(LS_FLOOR, std::min(LS_CAP, ls));
    }

    // Build kernel matrix K + noise * I
    std::vector<double> K(n_obs * n_obs);
    for(std::size_t i = 0; i < n_obs; ++i)
    {
        for(std::size_t j = 0; j <= i; ++j)
        {
            double val       = Matern52(&X_obs[i * n_features], &X_obs[j * n_features], n_features);
            K[i * n_obs + j] = val;
            K[j * n_obs + i] = val;
        }
        K[i * n_obs + i] += noise_;
    }

    // Cholesky decomposition: K = L * L^T
    L_.assign(n_obs * n_obs, 0.0);
    for(std::size_t i = 0; i < n_obs; ++i)
    {
        for(std::size_t j = 0; j <= i; ++j)
        {
            double s = K[i * n_obs + j];
            for(std::size_t k = 0; k < j; ++k)
                s -= L_[i * n_obs + k] * L_[j * n_obs + k];

            if(i == j)
            {
                if(s <= 0.0)
                    s = noise_; // numerical safeguard
                L_[i * n_obs + j] = std::sqrt(s);
            }
            else
            {
                L_[i * n_obs + j] = s / L_[j * n_obs + j];
            }
        }
    }

    // alpha = K^{-1} * y_norm
    CholeskySolve(L_, y_norm_, alpha_, n_obs);
}

void GaussianProcess::Predict(const std::vector<double>& X_cand,
                              std::size_t n_cand,
                              std::vector<double>& out_mu,
                              std::vector<double>& out_sigma) const
{
    out_mu.resize(n_cand);
    out_sigma.resize(n_cand);

    for(std::size_t i = 0; i < n_cand; ++i)
    {
        // k_star[j] = kernel(X_cand[i], X_obs[j])
        std::vector<double> k_star(n_obs_);
        for(std::size_t j = 0; j < n_obs_; ++j)
            k_star[j] = Matern52(&X_cand[i * n_features_], &X_obs_[j * n_features_], n_features_);

        // mu_norm = k_star^T * alpha
        double mu_norm = 0.0;
        for(std::size_t j = 0; j < n_obs_; ++j)
            mu_norm += k_star[j] * alpha_[j];

        // v = L^{-1} * k_star  (forward substitution)
        std::vector<double> v(n_obs_);
        for(std::size_t j = 0; j < n_obs_; ++j)
        {
            double s = k_star[j];
            for(std::size_t k = 0; k < j; ++k)
                s -= L_[j * n_obs_ + k] * v[k];
            v[j] = s / L_[j * n_obs_ + j];
        }

        // var_norm = k(x*, x*) - v^T * v
        double k_self   = 1.0; // Matern52(x, x) = 1.0
        double var_norm = k_self;
        for(std::size_t j = 0; j < n_obs_; ++j)
            var_norm -= v[j] * v[j];
        if(var_norm < 0.0)
            var_norm = 0.0;

        // Un-normalize to original y-space
        out_mu[i]    = mu_norm * y_std_ + y_mean_;
        out_sigma[i] = std::sqrt(var_norm) * y_std_;
    }
}

// ===================================================================
// Expected Improvement
// ===================================================================

static double NormalCDF(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }

static double NormalPDF(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI); }

std::size_t SelectNextByEI(const std::vector<double>& mu,
                           const std::vector<double>& sigma,
                           double best_observed,
                           std::size_t n_cand,
                           double* out_max_ei)
{
    double best_ei       = -1.0;
    std::size_t best_idx = 0;

    for(std::size_t i = 0; i < n_cand; ++i)
    {
        double ei;
        if(sigma[i] < 1e-10)
        {
            ei = std::max(0.0, best_observed - mu[i]);
        }
        else
        {
            double z = (best_observed - mu[i]) / sigma[i];
            ei       = (best_observed - mu[i]) * NormalCDF(z) + sigma[i] * NormalPDF(z);
        }
        if(ei > best_ei)
        {
            best_ei  = ei;
            best_idx = i;
        }
    }

    if(out_max_ei != nullptr)
        *out_max_ei = best_ei;

    return best_idx;
}

// ===================================================================
// Thompson Sampling
// ===================================================================

std::size_t SelectNextByTS(const std::vector<double>& mu,
                           const std::vector<double>& sigma,
                           std::size_t n_cand,
                           std::mt19937& rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    double best_sample   = std::numeric_limits<double>::max();
    std::size_t best_idx = 0;

    for(std::size_t i = 0; i < n_cand; ++i)
    {
        double s = (sigma[i] < 1e-10) ? mu[i] : mu[i] + sigma[i] * normal(rng);
        if(s < best_sample)
        {
            best_sample = s;
            best_idx    = i;
        }
    }

    return best_idx;
}

// ===================================================================
// Config string → feature vector (two-pass, dynamic schema)
// ===================================================================

static std::string TrimWhitespace(const std::string& s)
{
    auto start = s.find_first_not_of(" \t\r\n");
    if(start == std::string::npos)
        return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string PrepareToken(const std::string& raw)
{
    std::string tok = TrimWhitespace(raw);
    // Strip CK V3 named-field prefixes like "BlkGemmPipelineScheduler: Intrawave"
    auto colon = tok.find(':');
    if(colon != std::string::npos)
        tok = TrimWhitespace(tok.substr(colon + 1));
    return tok;
}

static bool TryParseNumber(const std::string& tok, double& out)
{
    if(tok.empty())
    {
        out = 0.0;
        return true;
    }
    bool is_number = true;
    bool has_dot   = false;
    for(std::size_t i = 0; i < tok.size(); ++i)
    {
        char c = tok[i];
        if(c == '-' && i == 0)
            continue;
        if(c == '.' && !has_dot)
        {
            has_dot = true;
            continue;
        }
        if(!std::isdigit(static_cast<unsigned char>(c)))
        {
            is_number = false;
            break;
        }
    }
    if(is_number)
    {
        out = std::stod(tok);
        return true;
    }
    return false;
}

// Split a CK config string into (variant_name, param_tokens, split_k).
// CK format: "VariantName<p0, p1, ..., pN>"  or  "VariantName<...>+K"
// Returns false if not a CK-style string (no '<' found).
static bool ParseCKConfigString(const std::string& config_str,
                                std::string& out_variant,
                                std::vector<std::string>& out_params,
                                double& out_split_k)
{
    out_params.clear();
    out_split_k = -1.0;

    auto angle_open = config_str.find('<');
    if(angle_open == std::string::npos)
        return false;

    out_variant = config_str.substr(0, angle_open);

    // Find closing '>' and optional "+split_k"
    auto angle_close = config_str.rfind('>');
    if(angle_close == std::string::npos || angle_close <= angle_open)
        return false;

    std::string params_str = config_str.substr(angle_open + 1, angle_close - angle_open - 1);

    // Check for "+split_k" after '>'
    if(angle_close + 1 < config_str.size() && config_str[angle_close + 1] == '+')
    {
        std::string sk_str = config_str.substr(angle_close + 2);
        sk_str             = TrimWhitespace(sk_str);
        if(!sk_str.empty())
            out_split_k = std::stod(sk_str);
    }

    // Split params on ','
    std::istringstream ss(params_str);
    std::string token;
    while(std::getline(ss, token, ','))
        out_params.push_back(token);

    return true;
}

// Split an ASM/legacy config string into tokens.
// ASM format: "val0,val1,...,valN" or "val0,val1,...,valN[gks]"
static void ParseCSVConfigString(const std::string& config_str,
                                 std::vector<std::string>& out_params,
                                 double& out_gks)
{
    out_params.clear();
    out_gks = -1.0;

    std::string str = config_str;

    // Check for "[gemm_k_global_split]" suffix
    auto bracket_open = str.rfind('[');
    if(bracket_open != std::string::npos)
    {
        auto bracket_close = str.rfind(']');
        if(bracket_close != std::string::npos && bracket_close > bracket_open)
        {
            std::string gks_str = str.substr(bracket_open + 1, bracket_close - bracket_open - 1);
            gks_str             = TrimWhitespace(gks_str);
            if(!gks_str.empty())
                out_gks = std::stod(gks_str);
            str = str.substr(0, bracket_open);
        }
    }

    std::istringstream ss(str);
    std::string token;
    while(std::getline(ss, token, ','))
        out_params.push_back(token);
}

std::size_t BuildFeatureMatrix(const std::vector<std::string>& config_strings,
                               std::vector<std::vector<double>>& out_features,
                               std::vector<int>& out_variant_ids)
{
    const std::size_t n = config_strings.size();
    out_features.clear();
    out_variant_ids.assign(n, -1);
    if(n == 0)
        return 0;

    // --- Pass 1: discover schema ---
    // For each config, parse into (variant_or_empty, param_tokens, extra_suffix).
    // Track variant names and max param count.

    struct ParsedConfig
    {
        bool is_ck;
        std::string variant;
        std::vector<std::string> params;
        double suffix_value; // split_k (CK WrW) or gks (ASM), -1 if absent
    };

    std::vector<ParsedConfig> parsed(n);
    std::map<std::string, int> variant_map;
    std::size_t max_params = 0;
    bool has_any_ck        = false;
    bool has_any_suffix    = false;

    for(std::size_t i = 0; i < n; ++i)
    {
        std::string variant;
        std::vector<std::string> params;
        double suffix = -1.0;

        if(ParseCKConfigString(config_strings[i], variant, params, suffix))
        {
            parsed[i]  = {true, variant, std::move(params), suffix};
            has_any_ck = true;
        }
        else
        {
            ParseCSVConfigString(config_strings[i], params, suffix);
            parsed[i] = {false, "", std::move(params), suffix};
        }

        if(suffix >= 0.0)
            has_any_suffix = true;

        if(parsed[i].is_ck)
        {
            if(variant_map.find(parsed[i].variant) == variant_map.end())
                variant_map[parsed[i].variant] = static_cast<int>(variant_map.size());
            out_variant_ids[i] = variant_map[parsed[i].variant];
        }

        max_params = std::max(max_params, parsed[i].params.size());
    }

    const std::size_t has_suffix_col = has_any_suffix ? 1 : 0;

    // --- Pass 1.5: build per-column maps for non-numeric tokens ---
    std::vector<std::map<std::string, int>> col_ordinal(max_params);
    for(std::size_t i = 0; i < n; ++i)
    {
        for(std::size_t j = 0; j < parsed[i].params.size(); ++j)
        {
            std::string tok = PrepareToken(parsed[i].params[j]);
            double dummy;
            if(!TryParseNumber(tok, dummy))
            {
                auto& m = col_ordinal[j];
                if(m.find(tok) == m.end())
                    m[tok] = static_cast<int>(m.size());
            }
        }
    }

    // Compute expanded feature dimension with one-hot encoding:
    //   Variant: one-hot → variant_map.size() columns (0 if no CK)
    //   Each param column j: numeric → 1 col, categorical → col_ordinal[j].size() cols
    //   Suffix: 1 col (numeric)
    const std::size_t variant_width = has_any_ck ? variant_map.size() : 0;

    std::vector<std::size_t> col_start(max_params);
    std::vector<std::size_t> col_width(max_params);
    std::size_t param_total = 0;
    for(std::size_t j = 0; j < max_params; ++j)
    {
        col_start[j] = param_total;
        col_width[j] = col_ordinal[j].empty() ? 1 : col_ordinal[j].size();
        param_total += col_width[j];
    }

    const std::size_t n_features = variant_width + param_total + has_suffix_col;
    if(n_features == 0)
        return 0;

    // --- Pass 2: build feature vectors with one-hot encoding ---
    out_features.resize(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        out_features[i].assign(n_features, 0.0);

        // Variant: one-hot (CK only)
        if(variant_width > 0 && parsed[i].is_ck)
        {
            int vid                                        = variant_map[parsed[i].variant];
            out_features[i][static_cast<std::size_t>(vid)] = 1.0;
        }

        // Param columns
        const std::size_t base = variant_width;
        for(std::size_t j = 0; j < parsed[i].params.size(); ++j)
        {
            std::string tok = PrepareToken(parsed[i].params[j]);
            double num_val;
            if(TryParseNumber(tok, num_val))
            {
                out_features[i][base + col_start[j]] = num_val;
            }
            else
            {
                int oid = col_ordinal[j][tok];
                out_features[i][base + col_start[j] + static_cast<std::size_t>(oid)] = 1.0;
            }
        }

        // Suffix column (split_k or gks) — log₂ encoding for uniform spacing
        if(has_suffix_col)
        {
            std::size_t suffix_col      = variant_width + param_total;
            double sv                   = parsed[i].suffix_value;
            out_features[i][suffix_col] = (sv > 0.0) ? std::log2(sv) + 1.0 : 0.0;
        }
    }

    // --- Pass 3: per-feature min-max normalization to [0, 1], then drop constants ---
    std::vector<bool> col_active(n_features, false);
    for(std::size_t j = 0; j < n_features; ++j)
    {
        double fmin = out_features[0][j];
        double fmax = fmin;
        for(std::size_t i = 1; i < n; ++i)
        {
            fmin = std::min(fmin, out_features[i][j]);
            fmax = std::max(fmax, out_features[i][j]);
        }
        double range = fmax - fmin;
        if(range > 1e-12)
        {
            col_active[j] = true;
            for(std::size_t i = 0; i < n; ++i)
                out_features[i][j] = (out_features[i][j] - fmin) / range;
        }
        else
        {
            for(std::size_t i = 0; i < n; ++i)
                out_features[i][j] = 0.0;
        }
    }

    // Remove constant columns to reduce GP noise
    std::size_t n_active = 0;
    for(std::size_t j = 0; j < n_features; ++j)
        if(col_active[j])
            ++n_active;

    if(n_active > 0 && n_active < n_features)
    {
        for(std::size_t i = 0; i < n; ++i)
        {
            std::vector<double> compact(n_active);
            std::size_t k = 0;
            for(std::size_t j = 0; j < n_features; ++j)
                if(col_active[j])
                    compact[k++] = out_features[i][j];
            out_features[i] = std::move(compact);
        }
        return n_active;
    }

    return (n_active > 0) ? n_features : 0;
}

BayesOptTracker& GetBayesOptTracker()
{
    thread_local BayesOptTracker tracker;
    return tracker;
}

} // namespace bayesian
} // namespace solver
} // namespace miopen
