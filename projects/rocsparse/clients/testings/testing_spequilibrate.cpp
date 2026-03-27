/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "testing_spequilibrate.hpp"
#include "testing.hpp"

#include <set>
#include <vector>

// symbolic_csr: compute the symbolic LU factorization with fill-in for an unsymmetric CSR matrix.
//
// Given a square m x m matrix A in CSR format, this function computes the sparsity patterns of
// the strict lower triangular factor L and the upper triangular factor U (including diagonal)
// such that A ~ L * U (symbolically).  Fill-in entries are positions in L+U that are not
// present in A.
//
// Algorithm:
//   For each row i (0-based):
//     1. Initialise the working row set with the original entries of row i.
//     2. Walk through the lower-triangle entries (col < i) of the working set in increasing
//        column order.  For each such entry at column j, add all entries from the already-
//        computed U row j to the working set.  Because U_pattern[j] contains only entries
//        >= j, any new entries inserted are >= j, and entries < i that are inserted after
//        the current iterator position will be visited in subsequent loop iterations.
//        This correctly propagates fill-in through the entire row.
//     3. Split the working set: entries with col < i go to strict-L for row i; entries
//        with col >= i go to U for row i (the col == i entry is the diagonal).
//
// The resulting L_col_ind / U_col_ind arrays use the same index base as the input.
template <typename I, typename J>
static void symbolic_csr(J                    m,
                         const I*             csr_row_ptr,
                         const J*             csr_col_ind,
                         rocsparse_index_base base,
                         std::vector<I>&      L_row_ptr,
                         std::vector<J>&      L_col_ind,
                         std::vector<I>&      U_row_ptr,
                         std::vector<J>&      U_col_ind)
{
    L_row_ptr.assign(m + 1, static_cast<I>(0));
    U_row_ptr.assign(m + 1, static_cast<I>(0));

    // U_pattern[k] holds the sorted column indices (0-based) of U row k (including diagonal).
    // L_pattern[k] holds the sorted column indices (0-based) of strict-L row k.
    std::vector<std::vector<J>> U_pattern(m);
    std::vector<std::vector<J>> L_pattern(m);

    for(J i = 0; i < m; ++i)
    {
        // Collect the original entries of row i into a sorted working set (0-based indices).
        std::set<J> row_set;
        for(I p = csr_row_ptr[i] - base; p < csr_row_ptr[i + 1] - base; ++p)
        {
            row_set.insert(csr_col_ind[p] - base);
        }

        // Walk the lower-triangle entries in increasing column order.
        // std::set iterators are not invalidated by insert, and new elements inserted
        // at positions > *it will be visited by subsequent iterator increments.
        // This guarantees that fill-in entries in the strict lower triangle are also
        // processed, correctly propagating fill-in from earlier rows.
        for(auto it = row_set.begin(); it != row_set.end() && *it < i; ++it)
        {
            J j = *it;
            // Add all upper entries of the already-computed U row j to the working set.
            // These entries are all >= j; those that are < i become new lower-triangle
            // entries that will be visited later in this loop.
            for(J col : U_pattern[j])
            {
                row_set.insert(col);
            }
        }

        // Split the completed working set into strict-L and U (including diagonal).
        for(J col : row_set)
        {
            if(col < i)
                L_pattern[i].push_back(col); // L_pattern is built in sorted order
            else
                U_pattern[i].push_back(col); // U_pattern is built in sorted order
        }
    }

    // Build CSR row pointer arrays.
    for(J i = 0; i < m; ++i)
    {
        L_row_ptr[i + 1] = L_row_ptr[i] + static_cast<I>(L_pattern[i].size());
        U_row_ptr[i + 1] = U_row_ptr[i] + static_cast<I>(U_pattern[i].size());
    }

    // Fill column-index arrays, converting back to the requested index base.
    L_col_ind.resize(L_row_ptr[m]);
    U_col_ind.resize(U_row_ptr[m]);

    I lpos = 0;
    I upos = 0;
    for(J i = 0; i < m; ++i)
    {
        for(J col : L_pattern[i])
            L_col_ind[lpos++] = col + static_cast<J>(base);
        for(J col : U_pattern[i])
            U_col_ind[upos++] = col + static_cast<J>(base);
    }
}

void testing_spequilibrate_extra(const Arguments& arg) {}

template <typename I, typename J, typename T>
void testing_spequilibrate_bad_arg(const Arguments& arg)
{
    // symbolic_csr is a host-only utility; no GPU bad-arg tests required.
}

template <typename I, typename J, typename T>
void testing_spequilibrate(const Arguments& arg)
{
    // symbolic_csr requires a square matrix.
    if(arg.M != arg.N)
    {
        return;
    }

    static constexpr bool       full_rank      = true;
    static constexpr bool       to_int         = false;
    rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);

    host_csr_matrix<T, I, J> hA;
    matrix_factory.init_csr(hA);

    const J m = static_cast<J>(hA.m);
    if(m == 0)
    {
        return;
    }

    std::vector<I> L_row_ptr;
    std::vector<J> L_col_ind;
    std::vector<I> U_row_ptr;
    std::vector<J> U_col_ind;

    if(arg.unit_check || arg.timing)
    {
        symbolic_csr<I, J>(m,
                           hA.ptr,
                           hA.ind,
                           hA.base,
                           L_row_ptr,
                           L_col_ind,
                           U_row_ptr,
                           U_col_ind);
    }

    if(arg.timing)
    {
        const int number_cold_calls = 2;
        const int number_hot_calls  = arg.iters;

        // Warm-up runs.
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            symbolic_csr<I, J>(m,
                               hA.ptr,
                               hA.ind,
                               hA.base,
                               L_row_ptr,
                               L_col_ind,
                               U_row_ptr,
                               U_col_ind);
        }

        // Timed runs.
        double cpu_time_used = get_time_us();
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            symbolic_csr<I, J>(m,
                               hA.ptr,
                               hA.ind,
                               hA.base,
                               L_row_ptr,
                               L_col_ind,
                               U_row_ptr,
                               U_col_ind);
        }
        cpu_time_used = (get_time_us() - cpu_time_used) / number_hot_calls;

        const I A_nnz   = hA.nnz;
        const I L_nnz   = L_row_ptr[m];
        const I U_nnz   = U_row_ptr[m];
        const I fill_in = (L_nnz + U_nnz) - A_nnz;

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::nnz,
                            A_nnz,
                            "L_nnz",
                            L_nnz,
                            "U_nnz",
                            U_nnz,
                            "fill_in",
                            fill_in,
                            display_key_t::time_ms,
                            get_gpu_time_msec(cpu_time_used));
    }
}

#define INSTANTIATE(I, J, T)                                                     \
    template void testing_spequilibrate_bad_arg<I, J, T>(const Arguments& arg); \
    template void testing_spequilibrate<I, J, T>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

#undef INSTANTIATE
