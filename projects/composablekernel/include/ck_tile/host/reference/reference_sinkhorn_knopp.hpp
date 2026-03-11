// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// CPU reference implementation of the naive Sinkhorn-Knopp algorithm:
//
// Given a square matrix X as input and an output matrix Y,
//
// C_ij = exp(X_ij)
// for n_iter iterations,
//   C_ij = C_ij / sum_i(C_ij)
//   C_ij = C_ij / sum_j(C_ij)
//
// Y_ij = C_ij
//
template <typename InDataType, typename ComputeDataType, typename OutDataType>
void sinkhorn_knopp_naive_ref(const HostTensor<InDataType>& x_n_n,
                              HostTensor<OutDataType>& y_n_n,
                              const int n_iter)
{
    // We assume input is an n x n matrix
    const index_t input_n = x_n_n.get_length(0);

    HostTensor<ComputeDataType> c_n_n({input_n, input_n}, {1, input_n});

    // Find max value for numerical stability
    ComputeDataType max_val = type_convert<ComputeDataType>(x_n_n(0, 0));
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            max_val = std::max(max_val, type_convert<ComputeDataType>(x_n_n(i, j)));
        }
    }

    // Exponentiate with max subtracted for numerical stability
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            c_n_n(i, j) = exp(type_convert<ComputeDataType>(x_n_n(i, j)) - max_val);
        }
    }

    // Iterate normalization on rows and columns
    for(auto it = 0; it < n_iter; ++it)
    {
        // Sum and scale each row by the total
        for(index_t i = 0; i < input_n; ++i)
        {
            ComputeDataType sum = 0.0;
            for(index_t j = 0; j < input_n; ++j)
            {
                sum += c_n_n(i, j);
            }
            // Avoid division by zero - if sum is too small, set uniform distribution
            if(sum > 1e-12)
            {
                for(index_t j = 0; j < input_n; ++j)
                {
                    c_n_n(i, j) /= sum;
                }
            }
            else
            {
                // Set to uniform distribution if row sum is too small
                ComputeDataType uniform_val = 1.0 / static_cast<ComputeDataType>(input_n);
                for(index_t j = 0; j < input_n; ++j)
                {
                    c_n_n(i, j) = uniform_val;
                }
            }
        }

        // Repeat columnwise
        for(index_t i = 0; i < input_n; ++i)
        {
            ComputeDataType sum = 0.0;
            for(index_t j = 0; j < input_n; ++j)
            {
                sum += c_n_n(j, i);
            }
            // Avoid division by zero - if sum is too small, set uniform distribution
            if(sum > 1e-12)
            {
                for(index_t j = 0; j < input_n; ++j)
                {
                    c_n_n(j, i) /= sum;
                }
            }
            else
            {
                // Set to uniform distribution if column sum is too small
                ComputeDataType uniform_val = 1.0 / static_cast<ComputeDataType>(input_n);
                for(index_t j = 0; j < input_n; ++j)
                {
                    c_n_n(j, i) = uniform_val;
                }
            }
        }
    }

    // Copy and cast to output type
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            y_n_n(i, j) = type_convert<OutDataType>(c_n_n(i, j));
        }
    }
}

// Log-domain implementation of Sinkhorn-Knopp algorithm
// More numerically stable for extreme values
template <typename InDataType, typename ComputeDataType, typename OutDataType>
void sinkhorn_knopp_log_domain_ref(const HostTensor<InDataType>& x_n_n,
                                   HostTensor<OutDataType>& y_n_n,
                                   const int n_iter)
{
    const index_t input_n = x_n_n.get_length(0);

    // Work in log domain
    HostTensor<ComputeDataType> log_c_n_n({input_n, input_n}, {1, input_n});

    // Find max value for numerical stability
    ComputeDataType max_val = type_convert<ComputeDataType>(x_n_n(0, 0));
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            max_val = std::max(max_val, type_convert<ComputeDataType>(x_n_n(i, j)));
        }
    }

    // Convert to log domain (log(exp(x - max)) = x - max)
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            log_c_n_n(i, j) = type_convert<ComputeDataType>(x_n_n(i, j)) - max_val;
        }
    }

    // Check if matrix is degenerate (all values too negative)
    bool is_degenerate                    = true;
    constexpr ComputeDataType min_log_val = -80.0f; // exp(-80) ≈ 1e-35
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            if(log_c_n_n(i, j) > min_log_val)
            {
                is_degenerate = false;
                break;
            }
        }
        if(!is_degenerate)
            break;
    }

    if(is_degenerate)
    {
        // Set to uniform distribution
        ComputeDataType uniform_val = 1.0f / static_cast<ComputeDataType>(input_n * input_n);
        for(index_t i = 0; i < input_n; ++i)
        {
            for(index_t j = 0; j < input_n; ++j)
            {
                y_n_n(i, j) = type_convert<OutDataType>(uniform_val);
            }
        }
        return;
    }

    // Helper function for log-sum-exp
    auto log_sum_exp = [](const ComputeDataType* log_values, index_t n) -> ComputeDataType {
        ComputeDataType max_log_val = log_values[0];
        for(index_t i = 1; i < n; ++i)
        {
            max_log_val = std::max(max_log_val, log_values[i]);
        }

        if(std::isinf(max_log_val) && max_log_val < 0)
        {
            return -INFINITY;
        }

        ComputeDataType sum = 0.0f;
        for(index_t i = 0; i < n; ++i)
        {
            sum += std::exp(log_values[i] - max_log_val);
        }

        return std::log(sum) + max_log_val;
    };

    // Sinkhorn iterations in log domain
    for(int it = 0; it < n_iter; ++it)
    {
        // Normalize rows in log domain
        for(index_t i = 0; i < input_n; ++i)
        {
            ComputeDataType row_values[16]; // Max size for typical use
            for(index_t j = 0; j < input_n; ++j)
            {
                row_values[j] = log_c_n_n(i, j);
            }

            ComputeDataType log_row_sum = log_sum_exp(row_values, input_n);

            if(std::isinf(log_row_sum) && log_row_sum < 0)
            {
                // Set row to uniform in log domain
                ComputeDataType log_uniform = -std::log(static_cast<ComputeDataType>(input_n));
                for(index_t j = 0; j < input_n; ++j)
                {
                    log_c_n_n(i, j) = log_uniform;
                }
            }
            else
            {
                // Normalize: log(val/sum) = log(val) - log(sum)
                for(index_t j = 0; j < input_n; ++j)
                {
                    log_c_n_n(i, j) -= log_row_sum;
                }
            }
        }

        // Normalize columns in log domain
        for(index_t j = 0; j < input_n; ++j)
        {
            ComputeDataType col_values[16]; // Max size for typical use
            for(index_t i = 0; i < input_n; ++i)
            {
                col_values[i] = log_c_n_n(i, j);
            }

            ComputeDataType log_col_sum = log_sum_exp(col_values, input_n);

            if(std::isinf(log_col_sum) && log_col_sum < 0)
            {
                // Set column to uniform in log domain
                ComputeDataType log_uniform = -std::log(static_cast<ComputeDataType>(input_n));
                for(index_t i = 0; i < input_n; ++i)
                {
                    log_c_n_n(i, j) = log_uniform;
                }
            }
            else
            {
                // Normalize: log(val/sum) = log(val) - log(sum)
                for(index_t i = 0; i < input_n; ++i)
                {
                    log_c_n_n(i, j) -= log_col_sum;
                }
            }
        }
    }

    // Convert back from log domain and write to output
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            ComputeDataType val = std::exp(log_c_n_n(i, j));
            // Clamp to reasonable range
            val         = std::max(val, static_cast<ComputeDataType>(1e-35f));
            val         = std::min(val, static_cast<ComputeDataType>(1.0f));
            y_n_n(i, j) = type_convert<OutDataType>(val);
        }
    }
}

} // namespace ck_tile
