// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename ComputeDataType>
auto row_sum_ref(const HostTensor<ComputeDataType>& x_b_n_n)
{
    const index_t batchsize = x_b_n_n.get_length(0);
    const index_t input_n   = x_b_n_n.get_length(1);

    HostTensor<ComputeDataType> acc_n({batchsize, input_n}, {input_n, 1});

    auto f = [&](const index_t b) {
        for(index_t i = 0; i < input_n; ++i)
        {
            acc_n(b, i) = 0;
            for(index_t j = 0; j < input_n; ++j)
            {
                acc_n(b, i) += x_b_n_n(b, i, j);
            }
        }
    };

    make_ParallelTensorFunctor(f,
                               x_b_n_n.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
    return acc_n;
}

template <typename ComputeDataType>
auto col_sum_ref(const HostTensor<ComputeDataType>& x_b_n_n)
{

    const index_t batchsize = x_b_n_n.get_length(0);
    const index_t input_n   = x_b_n_n.get_length(1);
    HostTensor<ComputeDataType> acc_n({batchsize, input_n}, {input_n, 1});

    auto f = [&](const index_t b) {
        for(index_t i = 0; i < input_n; ++i)
        {
            acc_n(b, i) = 0;
            for(index_t j = 0; j < input_n; ++j)
            {
                acc_n(b, i) += x_b_n_n(b, j, i);
            }
        }
    };

    make_ParallelTensorFunctor(f,
                               x_b_n_n.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
    return acc_n;
}

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
void sinkhorn_knopp_naive_ref(const HostTensor<InDataType>& x_b_n_n,
                              HostTensor<OutDataType>& y_b_n_n,
                              const int n_iter)
{
    // We assume input is an b x n x n matrix
    const index_t input_n = x_b_n_n.get_length(1);

    auto f = [&](const index_t b) {
        HostTensor<ComputeDataType> c_n_n({input_n, input_n}, {input_n, 1});

        // Exponentiate to enforce nonnegativity; required for algorithm to converge
        for(index_t i = 0; i < input_n; ++i)
        {
            for(index_t j = 0; j < input_n; ++j)
            {
                c_n_n(i, j) = exp(type_convert<ComputeDataType>(x_b_n_n(b, i, j)));
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
                for(index_t j = 0; j < input_n; ++j)
                {
                    c_n_n(i, j) /= sum;
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
                for(index_t j = 0; j < input_n; ++j)
                {
                    c_n_n(j, i) /= sum;
                }
            }
        }

        // Copy and cast to output type
        for(index_t i = 0; i < input_n; ++i)
        {
            for(index_t j = 0; j < input_n; ++j)
            {
                y_b_n_n(b, i, j) = type_convert<OutDataType>(c_n_n(i, j));
            }
        }
    };

    make_ParallelTensorFunctor(f,
                               x_b_n_n.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
}

} // namespace ck_tile
