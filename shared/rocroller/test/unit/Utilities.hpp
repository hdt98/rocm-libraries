// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <common/Utilities.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef TEST
#undef TEST // Rely on TEST_F and TEST_P instead
#endif

// PrintTo for std::pair<std::string, std::string> to avoid parentheses in gtest-generated test
// names. CMake cannot parse add_test() calls where the test name contains a ')' character,
// which is what the default pair printer produces (e.g. ("N", "N")).
inline void PrintTo(std::pair<std::string, std::string> const& p, std::ostream* os)
{
    *os << p.first << p.second;
}

MATCHER_P(HasHipSuccess, p, "")
{
    auto result = arg;
    if(result != hipSuccess)
    {
        std::string msg = hipGetErrorString(result);
        *result_listener << msg;
    }
    return result == hipSuccess;
}
