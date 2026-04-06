// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "miopendriver_common.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

#include <miopen/miopen.h>
#include <miopen/process.hpp>

namespace miopen_conv_deterministic {

// clang-format off
static const std::string shape_3d =
    "-n 1 -c 1 --in_d 10 -H 11 -W 12 -k 1 --fil_d 1 -y 2 -x 5 --pad_d 0 -p 1 -q 4 --conv_stride_d 1 -u 1 -v 1 --dilation_d 2 -l 2 -j 2 --spatial_dim 3 "
    "-m conv -g 1 --iter 1 -V 0";
// clang-format on

// Returns a pair of {test args without deterministic, test args with deterministic}
std::vector<std::pair<std::string, std::string>>
GetTestCasePairs(const std::string& mode, const std::string& shape, int forw_flag)
{
    const std::string dir  = " -F " + std::to_string(forw_flag);
    const std::string base = mode + " " + shape + dir;
    return {{base, base + " --deterministic 1"}};
}

miopen::ProcessEnvironmentMap MakeEnv(const std::string& tmp_dir)
{
    miopen::ProcessEnvironmentMap envs;
    envs["MIOPEN_FIND_MODE"]    = "2"; // immediate mode - uses AI heuristics
    envs["MIOPEN_LOG_LEVEL"]    = "5"; // Info level to capture messages
    envs["MIOPEN_USER_DB_PATH"] = tmp_dir;
    return envs;
}

class GPU_MIOpenDriverConvDeterministicTest_NoFlag_FP32
    : public testing::TestWithParam<std::pair<std::string, std::string>>
{
};

class GPU_MIOpenDriverConvDeterministicTest_Enabled_FP32
    : public testing::TestWithParam<std::pair<std::string, std::string>>
{
};

class GPU_MIOpenDriverConvDeterministicTest_Reproducible_FP32
    : public testing::TestWithParam<std::pair<std::string, std::string>>
{
};

// ----------------------------------------------------------------------------
// Test 1: Without --deterministic being set, log must not mention
// "Restricting convolution to deterministic kernels."
// ----------------------------------------------------------------------------
TEST_P(GPU_MIOpenDriverConvDeterministicTest_NoFlag_FP32, NoDeterministicLog)
{
    const auto tmp_dir = std::string{"/tmp/miopen_det_test_noflag"};
    miopen::fs::remove_all(tmp_dir);

    testing::internal::CaptureStderr();
    RunMIOpenDriverTestCommand({GetParam().first}, MakeEnv(tmp_dir));
    const auto output = testing::internal::GetCapturedStderr();

    EXPECT_THAT(output,
                Not(testing::HasSubstr("Restricting convolution to deterministic kernels.")));

    miopen::fs::remove_all(tmp_dir);
}

// ----------------------------------------------------------------------------
// Test 2: With --deterministic 1 the log mentions
// "Restricting convolution to deterministic kernels."
// ----------------------------------------------------------------------------
TEST_P(GPU_MIOpenDriverConvDeterministicTest_Enabled_FP32, RunsSuccessfullyAndLogsOverride)
{
    const auto tmp_dir = std::string{"/tmp/miopen_det_test_enabled"};
    miopen::fs::remove_all(tmp_dir);

    testing::internal::CaptureStderr();
    RunMIOpenDriverTestCommand({GetParam().second}, MakeEnv(tmp_dir));
    const auto output = testing::internal::GetCapturedStderr();

    EXPECT_THAT(output, testing::HasSubstr("Restricting convolution to deterministic kernels."));

    miopen::fs::remove_all(tmp_dir);
}

// ----------------------------------------------------------------------------
// Test 3: Invalid --deterministic value causes non-zero exit
// ----------------------------------------------------------------------------
TEST(GPU_MIOpenDriverConvDeterministicTest_InvalidValue_FP32, ExitsOnInvalidValue)
{
    const auto tmp_dir = std::string{"/tmp/miopen_det_test_invalid"};
    miopen::fs::remove_all(tmp_dir);

    const std::string bad_args =
        miopendriver::basearg::conv::Half + " " + shape_3d + " -F 4 --deterministic 2";

    int result = 0;
    miopen::Process p{MIOpenDriverExePath().string()};
    std::stringstream ss;
    EXPECT_NO_THROW(result = p(bad_args, "", &ss, MakeEnv(tmp_dir)));
    EXPECT_NE(result, 0) << "Should exit with a non-zero code on invalid deterministic value";
    EXPECT_THAT(ss.str(), testing::HasSubstr("Invalid deterministic value"));

    miopen::fs::remove_all(tmp_dir);
}

// ----------------------------------------------------------------------------
// Test 4: Reproducibility - two runs with --deterministic 1 produce
//         identical output files (--dump_output)
// ----------------------------------------------------------------------------
TEST_P(GPU_MIOpenDriverConvDeterministicTest_Reproducible_FP32, BitExactAcrossRuns)
{
    const auto run1_dir = std::string{"/tmp/miopen_det_repro_run1"};
    const auto run2_dir = std::string{"/tmp/miopen_det_repro_run2"};
    miopen::fs::remove_all(run1_dir);
    miopen::fs::remove_all(run2_dir);
    miopen::fs::create_directories(run1_dir);
    miopen::fs::create_directories(run2_dir);

    auto make_env = [](const std::string& tmp) {
        miopen::ProcessEnvironmentMap envs;
        envs["MIOPEN_DEBUG_FIND_ONLY_SOLVER"] = "ConvHipImplicitGemm3DGroupWrwXdlops";
        envs["MIOPEN_LOG_LEVEL"]              = "4";
        envs["MIOPEN_USER_DB_PATH"]           = tmp;
        return envs;
    };

    // --dump_output writes binary output files to the working directory.
    // Run the driver twice from separate directories to get separate output files.
    const std::string det_args = GetParam().second + " -o 1";

    {
        auto envs = make_env(run1_dir + "/db");
        miopen::fs::create_directories(run1_dir + "/db");
        miopen::Process p{MIOpenDriverExePath().string()};
        std::stringstream ss;
        int rc = 0;
        // Change CWD so dump files land in run1_dir
        EXPECT_NO_THROW(rc = p(det_args, run1_dir, &ss, envs));
        EXPECT_EQ(rc, 0);
    }
    {
        auto envs = make_env(run2_dir + "/db");
        miopen::fs::create_directories(run2_dir + "/db");
        miopen::Process p{MIOpenDriverExePath().string()};
        std::stringstream ss;
        int rc = 0;
        EXPECT_NO_THROW(rc = p(det_args, run2_dir, &ss, envs));
        EXPECT_EQ(rc, 0);
    }

    // Compare all dump_*.bin files between the two runs
    for(const auto& entry : miopen::fs::directory_iterator(run1_dir))
    {
        const auto& p1 = entry.path();
        if(p1.extension() != ".bin")
            continue;

        const auto p2 = miopen::fs::path{run2_dir} / p1.filename();
        ASSERT_TRUE(miopen::fs::exists(p2))
            << "Output file missing from second run: " << p2.string();

        const auto sz1 = miopen::fs::file_size(p1);
        const auto sz2 = miopen::fs::file_size(p2);
        ASSERT_EQ(sz1, sz2) << "Output file size mismatch: " << p1.filename().string();

        std::ifstream f1(p1, std::ios::binary);
        std::ifstream f2(p2, std::ios::binary);
        ASSERT_TRUE(f1.is_open() && f2.is_open());

        std::vector<char> buf1(sz1), buf2(sz2);
        f1.read(buf1.data(), static_cast<std::streamsize>(sz1));
        f2.read(buf2.data(), static_cast<std::streamsize>(sz2));
        EXPECT_EQ(buf1, buf2) << "Bit-exact mismatch in " << p1.filename().string();
    }

    miopen::fs::remove_all(run1_dir);
    miopen::fs::remove_all(run2_dir);
}

// 3D WRW
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_MIOpenDriverConvDeterministicTest_NoFlag_FP32,
    testing::ValuesIn(GetTestCasePairs(miopendriver::basearg::conv::Float, shape_3d, 4)));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_MIOpenDriverConvDeterministicTest_Enabled_FP32,
    testing::ValuesIn(GetTestCasePairs(miopendriver::basearg::conv::Float, shape_3d, 4)));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_MIOpenDriverConvDeterministicTest_Reproducible_FP32,
    testing::ValuesIn(GetTestCasePairs(miopendriver::basearg::conv::Float, shape_3d, 4)));

} // namespace miopen_conv_deterministic
