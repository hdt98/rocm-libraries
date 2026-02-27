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

} // namespace ck_tile
