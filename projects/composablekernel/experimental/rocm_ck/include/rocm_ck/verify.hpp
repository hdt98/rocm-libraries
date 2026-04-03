// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host — result verification for examples and tests.
//
// Compares GPU output against CPU reference using combined relative+absolute
// tolerance. Provides structured results and formatted reporting.

#pragma once

#include <rocm_ck/datatype_convert.hpp>

#include <cmath>
#include <cstdio>

namespace rocm_ck {

/// Result of element-wise verification.
struct VerifyResult
{
    bool passed;
    int first_mismatch; // index of first failing element, -1 if passed
    int count;          // total elements compared
    float got;          // value at first mismatch
    float expected;     // reference value at first mismatch
};

/// Compare result against reference using combined relative+absolute tolerance.
///
/// Formula: |result[i] - ref[i]| <= tol * |ref[i]| + tol
///
/// This handles both large values (relative error dominates) and values
/// near zero (absolute error dominates).
inline VerifyResult verify(const float* result, const float* ref, int count, float tolerance)
{
    for(int i = 0; i < count; ++i)
    {
        float diff = std::fabs(result[i] - ref[i]);
        if(diff > tolerance * std::fabs(ref[i]) + tolerance)
            return {false, i, count, result[i], ref[i]};
    }
    return {true, -1, count, 0.0f, 0.0f};
}

/// Convenience overload: look up tolerance from the output DataType.
inline VerifyResult verify(const float* result, const float* ref, int count, DataType output_dtype)
{
    return verify(result, ref, count, toleranceFor(output_dtype));
}

/// Print verification result. Returns whether the test passed.
inline bool reportVerify(const char* name, const VerifyResult& r)
{
    if(r.passed)
    {
        std::printf("%s: PASSED\n", name);
    }
    else
    {
        std::printf("%s: FAILED\n", name);
        std::fprintf(stderr,
                     "  MISMATCH at [%d]: got %f, expected %f (diff=%e)\n",
                     r.first_mismatch,
                     r.got,
                     r.expected,
                     r.got - r.expected);
    }
    return r.passed;
}

} // namespace rocm_ck
