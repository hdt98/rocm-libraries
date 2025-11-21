/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once

#include <string>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

struct HIPGtest
{
    // This helper ensures that HIP errors are cleaned up after every test, and will flag
    // tests that don't clean up their own errors
    static void ExpectHipSuccess(std::string postfix = "")
    {
        auto hipError    = hipGetLastError();
        auto hipExtError = hipExtGetLastError();

        EXPECT_EQ(hipError, hipSuccess)
            << " hipGetLastError returned error code " << hipError << " ("
            << hipGetErrorString(hipExtError) << ")" << postfix.c_str();
        // " after test "
        //            << test_info.test_suite_name() << "." << test_info.name()
        //            << ". Error string: " << hipGetErrorString(hipError);
        EXPECT_EQ(hipExtError, hipSuccess)
            << " hipExtGetLastError returned error code " << hipExtError << " ("
            << hipGetErrorString(hipExtError) << ")" << postfix.c_str();
        //          << " after test "
        //            << test_info.test_suite_name() << "." << test_info.name()
        //            << ". Error string: " << hipGetErrorString(hipExtError);
    }

    // This helper checks that an exact hip runtime API error has occurred, and if so, clears the
    // error.
    static void ExpectHipError(const hipError_t error)
    {
        auto hipError = hipPeekAtLastError(); // peek so that error persists if it does not match

        if(hipError != error)
        {
            EXPECT_EQ(hipError, error)
                << " expected hip error " << error << " (" << hipGetErrorString(error)
                << ") but hipPeekAtLastError returned " << hipError << " ("
                << hipGetErrorString(hipError) << ")";
        }
        else
        {
            (void)hipGetLastError(); // clear the error
        }
    }
};
