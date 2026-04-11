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
// GaussianProcess implementation
// ===================================================================

double GaussianProcess::Matern52(const double* x1, const double* x2, std::size_t d) const
{
    double r2 = 0.0;
    for(std::size_t i = 0; i < d; ++i)
    {
        double diff = (x1[i] - x2[i]) / length_scale_;
        r2 += diff * diff;
    }
    double r = std::sqrt(r2);
    double s5 = std::sqrt(5.0);
    // k(r) = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
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

void GaussianProcess::Fit(const std::vector<double>& X_obs,
                          const std::vector<double>& y_obs,
                          std::size_t n_obs,
                          std::size_t n_features)
{
    n_obs_      = n_obs;
    n_features_ = n_features;
    X_obs_      = X_obs;

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

    // Auto-tune length_scale: use median pairwise distance / sqrt(n_features)
    if(n_obs > 1)
    {
        std::vector<double> dists;
        dists.reserve(n_obs * (n_obs - 1) / 2);
        for(std::size_t i = 0; i < n_obs; ++i)
        {
            for(std::size_t j = i + 1; j < n_obs; ++j)
            {
                double d2 = 0.0;
                for(std::size_t k = 0; k < n_features; ++k)
                {
                    double diff = X_obs[i * n_features + k] - X_obs[j * n_features + k];
                    d2 += diff * diff;
                }
                dists.push_back(std::sqrt(d2));
            }
        }
        std::sort(dists.begin(), dists.end());
        length_scale_ = dists[dists.size() / 2];
        if(length_scale_ < 1e-10)
            length_scale_ = 1.0;
    }

    // Build kernel matrix K + noise * I
    std::vector<double> K(n_obs * n_obs);
    for(std::size_t i = 0; i < n_obs; ++i)
    {
        for(std::size_t j = 0; j <= i; ++j)
        {
            double val = Matern52(&X_obs[i * n_features], &X_obs[j * n_features], n_features);
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
            k_star[j] = Matern52(&X_cand[i * n_features_],
                                 &X_obs_[j * n_features_],
                                 n_features_);

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
        double k_self = 1.0; // Matern52(x, x) = 1.0
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

static double NormalCDF(double x)
{
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

static double NormalPDF(double x)
{
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

std::size_t SelectNextByEI(const std::vector<double>& mu,
                           const std::vector<double>& sigma,
                           double best_observed,
                           std::size_t n_cand)
{
    // EI(x) = (best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    // where z = (best - mu(x)) / sigma(x)
    // We MINIMIZE time, so "improvement" = best_observed - mu(x)
    double best_ei    = -1.0;
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
            ei = (best_observed - mu[i]) * NormalCDF(z) + sigma[i] * NormalPDF(z);
        }
        if(ei > best_ei)
        {
            best_ei  = ei;
            best_idx = i;
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

static double ParseToken(const std::string& raw)
{
    std::string tok = TrimWhitespace(raw);
    if(tok.empty())
        return 0.0;

    // Strip CK V3 named-field prefixes like "BlkGemmPipelineScheduler: Intrawave"
    auto colon = tok.find(':');
    if(colon != std::string::npos)
        tok = TrimWhitespace(tok.substr(colon + 1));

    // Try parsing as number
    bool is_number = !tok.empty();
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
        return std::stod(tok);

    // Known categorical mappings (ordinal encoding)
    // ConvSpecialization
    if(tok == "Default")       return 0.0;
    if(tok == "OddC")          return 1.0;
    if(tok == "Filter1x1Pad0") return 2.0;
    if(tok == "Filter1x1Stride1Pad0") return 3.0;
    if(tok == "Filter3x3")     return 4.0;
    // PipelineScheduler
    if(tok == "Intrawave")     return 0.0;
    if(tok == "Interwave")     return 1.0;
    // PipelineVersion
    if(tok == "v1")            return 1.0;
    if(tok == "v2")            return 2.0;
    if(tok == "v3")            return 3.0;
    if(tok == "v4")            return 4.0;
    if(tok == "v5")            return 5.0;

    // Fallback: hash for unknown strings (ASM direction, layout, precision, etc.)
    std::hash<std::string> hasher;
    return static_cast<double>(hasher(tok) % 10000) / 10000.0;
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
        sk_str = TrimWhitespace(sk_str);
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
            gks_str = TrimWhitespace(gks_str);
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
                               std::vector<std::vector<double>>& out_features)
{
    const std::size_t n = config_strings.size();
    out_features.clear();
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
    std::size_t max_params   = 0;
    bool has_any_ck          = false;
    bool has_any_suffix      = false;

    for(std::size_t i = 0; i < n; ++i)
    {
        std::string variant;
        std::vector<std::string> params;
        double suffix = -1.0;

        if(ParseCKConfigString(config_strings[i], variant, params, suffix))
        {
            parsed[i] = {true, variant, std::move(params), suffix};
            has_any_ck = true;
        }
        else
        {
            ParseCSVConfigString(config_strings[i], params, suffix);
            parsed[i] = {false, "", std::move(params), suffix};
        }

        if(suffix >= 0.0)
            has_any_suffix = true;

        if(parsed[i].is_ck &&
           variant_map.find(parsed[i].variant) == variant_map.end())
        {
            variant_map[parsed[i].variant] = static_cast<int>(variant_map.size());
        }

        max_params = std::max(max_params, parsed[i].params.size());
    }

    // Feature dimension:
    //   CK:  1 (variant) + max_params + (1 if any config has split_k suffix)
    //   CSV: max_params + (1 if any config has [gks] suffix)
    const std::size_t has_variant_col = has_any_ck ? 1 : 0;
    const std::size_t has_suffix_col  = has_any_suffix ? 1 : 0;
    const std::size_t n_features      = has_variant_col + max_params + has_suffix_col;

    if(n_features == 0)
        return 0;

    // --- Pass 2: build uniform feature vectors ---
    out_features.resize(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        out_features[i].assign(n_features, 0.0);
        std::size_t col = 0;

        // Variant column (CK only)
        if(has_variant_col)
        {
            if(parsed[i].is_ck)
                out_features[i][col] = static_cast<double>(variant_map[parsed[i].variant]);
            col++;
        }

        // Param columns (parsed tokens → numbers, padded with 0)
        for(std::size_t j = 0; j < parsed[i].params.size(); ++j)
            out_features[i][col + j] = ParseToken(parsed[i].params[j]);
        col += max_params;

        // Suffix column (split_k or gks)
        if(has_suffix_col)
        {
            out_features[i][col] = (parsed[i].suffix_value >= 0.0) ? parsed[i].suffix_value : 0.0;
        }
    }

    return n_features;
}

BayesOptTracker& GetBayesOptTracker()
{
    thread_local BayesOptTracker tracker;
    return tracker;
}

} // namespace bayesian
} // namespace solver
} // namespace miopen
