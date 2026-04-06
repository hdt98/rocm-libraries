// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/verify.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using ::rocm_ck::DataType;
using ::rocm_ck::reportVerify;
using ::rocm_ck::toleranceFor;
using ::rocm_ck::verify;
using ::rocm_ck::VerifyResult;

// ============================================================================
// VerifyResult structure
// ============================================================================

TEST(VerifyResult, InitializesWithPassedState)
{
    VerifyResult r{true, -1, 100, 0.0f, 0.0f};
    EXPECT_TRUE(r.passed);
    EXPECT_EQ(r.first_mismatch, -1);
    EXPECT_EQ(r.count, 100);
}

TEST(VerifyResult, InitializesWithFailedState)
{
    VerifyResult r{false, 42, 100, 1.5f, 2.0f};
    EXPECT_FALSE(r.passed);
    EXPECT_EQ(r.first_mismatch, 42);
    EXPECT_EQ(r.count, 100);
    EXPECT_FLOAT_EQ(r.got, 1.5f);
    EXPECT_FLOAT_EQ(r.expected, 2.0f);
}

// ============================================================================
// verify() with explicit tolerance
// ============================================================================

TEST(Verify, PassesWhenArraysMatch)
{
    float result[] = {1.0f, 2.0f, 3.0f};
    float ref[]    = {1.0f, 2.0f, 3.0f};
    VerifyResult r = verify(result, ref, 3, 1e-5f);
    EXPECT_TRUE(r.passed);
    EXPECT_EQ(r.first_mismatch, -1);
}

TEST(Verify, PassesWhenDifferenceWithinTolerance)
{
    float result[] = {1.0f, 2.00001f, 3.0f};
    float ref[]    = {1.0f, 2.0f, 3.0f};
    VerifyResult r = verify(result, ref, 3, 1e-3f);
    EXPECT_TRUE(r.passed);
}

TEST(Verify, FailsWhenDifferenceExceedsTolerance)
{
    float result[] = {1.0f, 2.1f, 3.0f};
    float ref[]    = {1.0f, 2.0f, 3.0f};
    VerifyResult r = verify(result, ref, 3, 1e-5f);
    EXPECT_FALSE(r.passed);
    EXPECT_EQ(r.first_mismatch, 1);
    EXPECT_FLOAT_EQ(r.got, 2.1f);
    EXPECT_FLOAT_EQ(r.expected, 2.0f);
}

TEST(Verify, ReportsFirstMismatchIndex)
{
    float result[] = {1.0f, 2.0f, 3.5f, 4.0f};
    float ref[]    = {1.0f, 2.0f, 3.0f, 4.0f};
    VerifyResult r = verify(result, ref, 4, 1e-5f);
    EXPECT_FALSE(r.passed);
    EXPECT_EQ(r.first_mismatch, 2);
}

TEST(Verify, UsesCombinedRelativeAndAbsoluteTolerance)
{
    // Formula: |result - ref| <= tol * |ref| + tol
    // Test with large reference (relative dominates)
    float result_large[] = {100.0f};
    float ref_large[]    = {100.1f};
    float tol            = 1e-2f; // 1% relative + 0.01 absolute
    // diff = 0.1, threshold = 0.01 * 100.1 + 0.01 = 1.011
    VerifyResult r_large = verify(result_large, ref_large, 1, tol);
    EXPECT_TRUE(r_large.passed);

    // Test with small reference (absolute dominates)
    float result_small[] = {0.0f};
    float ref_small[]    = {0.009f};
    // diff = 0.009, threshold = 0.01 * 0.009 + 0.01 = 0.01009
    VerifyResult r_small = verify(result_small, ref_small, 1, tol);
    EXPECT_TRUE(r_small.passed);
}

TEST(Verify, HandlesZeroReference)
{
    float result[] = {0.001f};
    float ref[]    = {0.0f};
    float tol      = 1e-3f;
    // diff = 0.001, threshold = tol * 0 + tol = 0.001
    VerifyResult r = verify(result, ref, 1, tol);
    EXPECT_TRUE(r.passed); // exactly at threshold
}

TEST(Verify, HandlesNegativeValues)
{
    float result[] = {-1.0f, -2.0f};
    float ref[]    = {-1.0f, -2.0f};
    VerifyResult r = verify(result, ref, 2, 1e-5f);
    EXPECT_TRUE(r.passed);
}

TEST(Verify, DetectsMismatchInNegativeValues)
{
    float result[] = {-1.5f};
    float ref[]    = {-1.0f};
    VerifyResult r = verify(result, ref, 1, 1e-5f);
    EXPECT_FALSE(r.passed);
    EXPECT_FLOAT_EQ(r.got, -1.5f);
    EXPECT_FLOAT_EQ(r.expected, -1.0f);
}

TEST(Verify, PassesEmptyArray)
{
    float dummy[]  = {0.0f};
    VerifyResult r = verify(dummy, dummy, 0, 1e-5f);
    EXPECT_TRUE(r.passed);
    EXPECT_EQ(r.count, 0);
}

// ============================================================================
// verify() with DataType dispatch
// ============================================================================

TEST(Verify, DispatchesToleranceFromDataType)
{
    float result[] = {1.0f, 2.0f};
    float ref[]    = {1.0f, 2.0f};
    VerifyResult r = verify(result, ref, 2, DataType::FP32);
    EXPECT_TRUE(r.passed);
}

TEST(Verify, UsesToleranceForFP8)
{
    // FP8 tolerance = 0.125 (see toleranceFor)
    float result[] = {1.1f};
    float ref[]    = {1.0f};
    // diff = 0.1, threshold = 0.125 * 1.0 + 0.125 = 0.25
    VerifyResult r = verify(result, ref, 1, DataType::FP8_FNUZ);
    EXPECT_TRUE(r.passed); // within FP8 tolerance
}

TEST(Verify, UsesToleranceForBF8)
{
    // BF8 tolerance = 0.25
    float result[] = {1.2f};
    float ref[]    = {1.0f};
    // diff = 0.2, threshold = 0.25 * 1.0 + 0.25 = 0.5
    VerifyResult r = verify(result, ref, 1, DataType::BF8_FNUZ);
    EXPECT_TRUE(r.passed); // within BF8 tolerance
}

TEST(Verify, UsesToleranceForBF16)
{
    // BF16 tolerance = 0.1
    float result[] = {1.05f};
    float ref[]    = {1.0f};
    // diff = 0.05, threshold = 0.1 * 1.0 + 0.1 = 0.2
    VerifyResult r = verify(result, ref, 1, DataType::BF16);
    EXPECT_TRUE(r.passed);
}

TEST(Verify, FailsWhenExceedingDataTypeTolerance)
{
    // FP32 tolerance = 1e-5
    float result[] = {1.0001f};
    float ref[]    = {1.0f};
    // diff = 0.0001, threshold ≈ 1e-5 * 1.0 + 1e-5 ≈ 2e-5
    VerifyResult r = verify(result, ref, 1, DataType::FP32);
    EXPECT_FALSE(r.passed);
}

// ============================================================================
// reportVerify()
// ============================================================================

TEST(ReportVerify, ReturnsTrueOnPass)
{
    VerifyResult r{true, -1, 10, 0.0f, 0.0f};
    testing::internal::CaptureStdout();
    bool result        = reportVerify("TestPass", r);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_TRUE(result);
    EXPECT_NE(output.find("PASSED"), std::string::npos);
}

TEST(ReportVerify, ReturnsFalseOnFail)
{
    VerifyResult r{false, 5, 10, 1.5f, 2.0f};
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    bool result     = reportVerify("TestFail", r);
    std::string out = testing::internal::GetCapturedStdout();
    std::string err = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(result);
    EXPECT_NE(out.find("FAILED"), std::string::npos);
    EXPECT_NE(err.find("MISMATCH at [5]"), std::string::npos);
}

TEST(ReportVerify, IncludesTestNameInOutput)
{
    VerifyResult r{true, -1, 10, 0.0f, 0.0f};
    testing::internal::CaptureStdout();
    reportVerify("MyCustomTest", r);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("MyCustomTest"), std::string::npos);
}

TEST(ReportVerify, ShowsMismatchDetails)
{
    VerifyResult r{false, 42, 100, 3.14f, 2.71f};
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    reportVerify("MismatchTest", r);
    std::string err = testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("[42]"), std::string::npos);
    EXPECT_NE(err.find("3.14"), std::string::npos);
    EXPECT_NE(err.find("2.71"), std::string::npos);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(Verify, HandlesInfinity)
{
    float result[] = {std::numeric_limits<float>::infinity()};
    float ref[]    = {std::numeric_limits<float>::infinity()};
    VerifyResult r = verify(result, ref, 1, 1e-5f);
    EXPECT_TRUE(r.passed); // inf == inf
}

TEST(Verify, DetectsInfinityMismatch)
{
    float result[] = {std::numeric_limits<float>::infinity()};
    float ref[]    = {1.0f};
    VerifyResult r = verify(result, ref, 1, 1e-5f);
    EXPECT_FALSE(r.passed);
}

TEST(Verify, NaNSlipsThroughComparison)
{
    // NaN - x = NaN, fabs(NaN) = NaN, and NaN > threshold is false,
    // so NaN does NOT trigger a mismatch. This is a known limitation.
    float result[] = {std::numeric_limits<float>::quiet_NaN()};
    float ref[]    = {1.0f};
    VerifyResult r = verify(result, ref, 1, 1e-5f);
    EXPECT_TRUE(r.passed);
}

TEST(Verify, NaNInReferenceSlipsThrough)
{
    float result[] = {1.0f};
    float ref[]    = {std::numeric_limits<float>::quiet_NaN()};
    VerifyResult r = verify(result, ref, 1, 1e-5f);
    EXPECT_TRUE(r.passed);
}
