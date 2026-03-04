// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include "ck/ck.hpp"
#include "ck/utility/device_arch.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/hip_check_error.hpp"

using namespace ck;

__global__ void kernel_matches_all(bool* result)
{
    *result = matches_with_compilation_target(DeviceArch::All);
}

__global__ void kernel_matches_gfx950(bool* result)
{
    *result = matches_with_compilation_target(DeviceArch::Gfx950);
}

class MatchesWithCompilationTargetTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        auto err = hipMalloc(&d_result, sizeof(bool));
        ASSERT_EQ(err, hipSuccess);
    }

    void TearDown() override { hip_check_error(hipFree(d_result)); }

    bool launch_and_get(void (*kernel)(bool*))
    {
        bool h_result = false;
        kernel<<<1, 1>>>(d_result);
        auto err = hipDeviceSynchronize();
        EXPECT_EQ(err, hipSuccess);
        err = hipMemcpy(&h_result, d_result, sizeof(bool), hipMemcpyDeviceToHost);
        EXPECT_EQ(err, hipSuccess);
        return h_result;
    }

    bool* d_result = nullptr;
};

class IsSupportedTest : public ::testing::Test
{
    protected:
    void SetUp() override { device_name = ck::get_device_name(); }
    std::string device_name;
};

TEST_F(MatchesWithCompilationTargetTest, DeviceArch_All_always_returns_true)
{
    // DeviceArch::All should match regardless of the compilation target.
    EXPECT_TRUE(launch_and_get(kernel_matches_all));
}

TEST_F(MatchesWithCompilationTargetTest, DeviceArch_Gfx950_matches_compilation_target)
{
    const bool result = launch_and_get(kernel_matches_gfx950);

    const auto& device_name = ck::get_device_name();
    if(device_name == "gfx950")
    {
        EXPECT_TRUE(result);
    }
    else
    {
        EXPECT_FALSE(result);
    }
}

TEST_F(IsSupportedTest, DeviceArch_All)
{
    // DeviceArch::All must be supported on every device.
    EXPECT_TRUE(is_supported(DeviceArch::All));
}

TEST_F(IsSupportedTest, DeviceArch_Gfx950)
{
    if(device_name == "gfx950")
    {
        EXPECT_TRUE(is_supported(DeviceArch::Gfx950));
    }
    else
    {
        EXPECT_FALSE(is_supported(DeviceArch::Gfx950));
    }
}

TEST(Printing, DeviceArch)
{
    std::stringstream ss;
    ss << DeviceArch::All;
    EXPECT_EQ(ss.str(), "All");

    ss.str("");
    ss << DeviceArch::Gfx950;
    EXPECT_EQ(ss.str(), "gfx950");
}
