// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <sstream>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "hip_helper.hpp"

// This test event listener ensures that HIP errors are cleaned up after every test, and will flag
// tests that don't clean up their own errors
class HIPErrorHandler : public testing::EmptyTestEventListener
{
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        std::ostringstream oss;
        oss << " after test " << test_info.test_suite_name() << "." << test_info.name() << ".";

        HIPGtest::ExpectHipSuccess(oss.str());
    }
};

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new HIPErrorHandler);

    return RUN_ALL_TESTS();
}
