/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "testing_getrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> getrf_tuple;

// each matrix_size_range vector is a {m, lda}

// case when m = -1 and n = -1 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {32, 32},
    {50, 50},
    {70, 100},
};

const vector<int> n_size_range = {
    // invalid
    -1,
    // normal (valid) samples
    16,
    20,
    40,
    100,
};

// cuBLAS getrfBatched only supports square matrices (m == n).
// Each entry is {{m, lda}, n} with n == m to ensure square inputs.
const vector<getrf_tuple> batched_square_range = {
    // invalid (triggers bad-args test when m == -1 and n == -1)
    {{-1, 1}, -1},
    // normal (valid) square samples
    {{32, 32}, 32},
    {{50, 50}, 50},
    {{70, 100}, 70},
};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {
//     {192, 192},
//     {640, 640},
//     {1000, 1024},
// };

// const vector<int> large_n_size_range = {
//     45,
//     64,
//     520,
//     1024,
//     2000,
// };

Arguments getrf_setup_arguments(getrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int         n_size      = std::get<1>(tup);

    Arguments arg;

    arg.set<int>("m", matrix_size[0]);
    arg.set<int>("lda", matrix_size[1]);

    arg.set<int>("n", n_size);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, bool NPVT, typename I, typename SIZE>
class GETRF_BASE : public ::TestWithParam<getrf_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == -1 && arg.peek<rocblas_int>("n") == -1)
            testing_getrf_bad_arg<API, BATCHED, STRIDED, T, I, SIZE>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_getrf<API, BATCHED, STRIDED, NPVT, T, I, SIZE>(arg);
    }
};

class Getrf : public GETRF_BASE<API_NORMAL, false, int, int>
{
};

class GetrfNpvt : public GETRF_BASE<API_NORMAL, true, int, int>
{
};

class GetrfFortran : public GETRF_BASE<API_FORTRAN, false, int, int>
{
};

class GetrfCompat : public GETRF_BASE<API_COMPAT, false, int, int>
{
};

class GetrfCompat64 : public GETRF_BASE<API_COMPAT, false, int64_t, size_t>
{
};

class GetrfCompatNpvt64 : public GETRF_BASE<API_COMPAT, true, int64_t, size_t>
{
};

class GetrfBatched : public GETRF_BASE<API_NORMAL, false, int, int>
{
};

class GetrfNpvtBatched : public GETRF_BASE<API_NORMAL, true, int, int>
{
};

// non-batch tests

TEST_P(Getrf, Float)
{
    run_tests<false, false, float>();
}

TEST_P(Getrf, Double)
{
    run_tests<false, false, double>();
}

TEST_P(Getrf, FloatComplex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(Getrf, DoubleComplex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}

TEST_P(GetrfNpvt, Float)
{
    run_tests<false, false, float>();
}

TEST_P(GetrfNpvt, Double)
{
    run_tests<false, false, double>();
}

TEST_P(GetrfNpvt, FloatComplex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(GetrfNpvt, DoubleComplex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}

TEST_P(GetrfFortran, Float)
{
    run_tests<false, false, float>();
}

TEST_P(GetrfFortran, Double)
{
    run_tests<false, false, double>();
}

TEST_P(GetrfFortran, FloatComplex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(GetrfFortran, DoubleComplex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}

TEST_P(GetrfCompat, Float)
{
    run_tests<false, false, float>();
}

TEST_P(GetrfCompat, Double)
{
    run_tests<false, false, double>();
}

TEST_P(GetrfCompat, FloatComplex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(GetrfCompat, DoubleComplex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}

TEST_P(GetrfCompat64, Float)
{
    run_tests<false, false, float>();
}

TEST_P(GetrfCompat64, Double)
{
    run_tests<false, false, double>();
}

TEST_P(GetrfCompat64, FloatComplex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(GetrfCompat64, DoubleComplex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}

TEST_P(GetrfCompatNpvt64, Float)
{
    run_tests<false, false, float>();
}

TEST_P(GetrfCompatNpvt64, Double)
{
    run_tests<false, false, double>();
}

TEST_P(GetrfCompatNpvt64, FloatComplex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(GetrfCompatNpvt64, DoubleComplex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}

// batched tests

TEST_P(GetrfBatched, BatchedFloat)
{
    run_tests<true, false, float>();
}

TEST_P(GetrfBatched, BatchedDouble)
{
    run_tests<true, false, double>();
}

TEST_P(GetrfBatched, BatchedFloatComplex)
{
    run_tests<true, false, hipsolverComplex>();
}

TEST_P(GetrfBatched, BatchedDoubleComplex)
{
    run_tests<true, false, hipsolverDoubleComplex>();
}

TEST_P(GetrfNpvtBatched, BatchedFloat)
{
    run_tests<true, false, float>();
}

TEST_P(GetrfNpvtBatched, BatchedDouble)
{
    run_tests<true, false, double>();
}

TEST_P(GetrfNpvtBatched, BatchedFloatComplex)
{
    run_tests<true, false, hipsolverComplex>();
}

TEST_P(GetrfNpvtBatched, BatchedDoubleComplex)
{
    run_tests<true, false, hipsolverDoubleComplex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          Getrf,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         Getrf,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GetrfNpvt,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GetrfNpvt,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GetrfFortran,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GetrfFortran,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GetrfCompat,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GetrfCompat,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GetrfCompat64,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GetrfCompat64,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GetrfCompatNpvt64,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GetrfCompatNpvt64,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GetrfBatched, ValuesIn(batched_square_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GetrfNpvtBatched, ValuesIn(batched_square_range));
