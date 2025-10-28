// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <random>
#include <vector>

#include <hipdnn_sdk/test_utilities/Seeds.hpp>

namespace test_bn_common
{

struct Batchnorm2dTestCase
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm2dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w
                  << " seed:" << tc.seed << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, h, w};
    }
};

struct Batchnorm3dTestCase
{
    int64_t n;
    int64_t c;
    int64_t d;
    int64_t h;
    int64_t w;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm3dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " d:" << tc.d << " h:" << tc.h
                  << " w:" << tc.w << " seed:" << tc.seed << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, d, h, w};
    }
};

inline std::vector<Batchnorm2dTestCase> getBatchnorm2dTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {1, 3, 14, 14, seed},
        {2, 3, 14, 14, seed},
    };
}

inline std::vector<Batchnorm2dTestCase> getBnFwdInferenceTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {1, 3, 14, 14, seed},
        {1, 256, 1, 1, seed},
        {2, 3, 1, 1, seed},
        {32, 1, 14, 14, seed},
        {32, 3, 1, 14, seed},
        {32, 3, 14, 1, seed},
    };
}

inline std::vector<Batchnorm2dTestCase> getBnFwdInferenceFullTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {64, 64, 112, 112, seed},
        {64, 512, 14, 14, seed},
    };
}

inline std::vector<Batchnorm3dTestCase> getBnFwdInference3dTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {2, 3, 3, 1, 1, seed},
        {16, 3, 8, 14, 14, seed},
    };
}

inline std::vector<Batchnorm2dTestCase> getBnBwdTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return std::vector<Batchnorm2dTestCase>{
        {1, 3, 14, 14, seed},
        // MIOpen segfaults for this case, re-enable when fix is released:
        // https://github.com/ROCm/rocm-libraries/pull/1197
        // {1, 256, 1, 1, seed}, // Would produce near-zero variance in theory
        {2, 3, 1, 1, seed},
        {32, 1, 14, 14, seed},
        {32, 3, 1, 14, seed},
        {32, 3, 14, 1, seed},
    };
}

inline std::vector<Batchnorm2dTestCase> getBnBwdFullTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return std::vector<Batchnorm2dTestCase>{
        {64, 64, 112, 112, seed},
        {64, 512, 14, 14, seed},
    };
}

inline std::vector<Batchnorm3dTestCase> getBnBwd3dTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return std::vector<Batchnorm3dTestCase>{
        {2, 3, 3, 1, 1, seed},
        {16, 3, 8, 14, 14, seed},
    };
}

inline std::vector<Batchnorm2dTestCase> getBnFwdTrainingSmoke2dTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        // {1, 256, 1, 1, seed}, // miopen's driver command for this shape fails. There is a PR in miopen that fixes this issue.
        {2, 3, 1, 1, seed}, // Minimal case
        {32, 3, 1, 14, seed}, // Typical small training case
    };
}

inline std::vector<Batchnorm2dTestCase> getBnFwdTrainingFull2dTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {1, 3, 14, 14, seed},
        {2, 3, 1, 1, seed},
        {32, 1, 14, 14, seed},
        {32, 3, 1, 14, seed},
        {32, 3, 14, 1, seed},
        {64, 64, 112, 112, seed}, // Large regression case
        {64, 512, 14, 14, seed}, // Many channels
    };
}

inline std::vector<Batchnorm3dTestCase> getBnFwdTrainingSmoke3dTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {2, 3, 3, 1, 1, seed}, // Minimal 3D case
        {2, 3, 2, 4, 4, seed}, // Small case with non-1 spatial dims
    };
}

inline std::vector<Batchnorm3dTestCase> getBnFwdTrainingFull3dTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {2, 3, 3, 1, 1, seed}, // Minimal case
        {2, 3, 2, 4, 4, seed}, // Small case
        {16, 3, 8, 14, 14, seed}, // Larger regression case
    };
}

} // namespace test_bn_common
