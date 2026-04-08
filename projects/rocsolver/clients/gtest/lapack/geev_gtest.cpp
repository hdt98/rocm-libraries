/* **************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "common/lapack/testing_geev.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> geev_tuple;

// each size_range vector is a {n, lda, ldvl, ldvr}

// each op_range vector is a {jobvl, jobvr}

// case when n == 0 and jobvl == N and jobvr == N will also execute the bad arguments test
// (null handle, null pointers, invalid sizes)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 1, 1},
    // invalid
    {-1, 1, 1, 1},
    {10, 5, 10, 10},
    // normal (valid) samples
    {1, 1, 1, 1},
    {4, 4, 4, 4},
    {8, 8, 8, 8},
    {16, 16, 16, 16},
    {32, 32, 32, 32},
    {50, 60, 50, 50}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{64, 64, 64, 64}, {128, 128, 128, 128}, {256, 256, 256, 256}, {512, 512, 512, 512}};

// {jobvl, jobvr}
const vector<vector<printable_char>> op_range = {{'N', 'N'}, {'N', 'V'}, {'V', 'N'}, {'V', 'V'}};

Arguments geev_setup_arguments(geev_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<printable_char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);
    arg.set<rocblas_int>("ldvl", size[2]);
    arg.set<rocblas_int>("ldvr", size[3]);

    arg.set<char>("jobvl", op[0]);
    arg.set<char>("jobvr", op[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class GEEV : public ::TestWithParam<geev_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = geev_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("jobvl") == 'N'
           && arg.peek<char>("jobvr") == 'N')
            testing_geev_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_geev<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GEEV, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEEV, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEEV, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEEV, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEEV, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEEV, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEEV, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEEV, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GEEV, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEEV, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEEV, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEEV, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEEV,
                         Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEEV,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));
