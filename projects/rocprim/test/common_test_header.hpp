// MIT License
//
// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef ROCPRIM_COMMON_TEST_HEADER_HPP_
#define ROCPRIM_COMMON_TEST_HEADER_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/utils.hpp"

// Google Test
#include <gtest/gtest.h>

// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

#if __has_include(<valgrind/valgrind.h>)
    #include <valgrind/valgrind.h>
    #define HAS_VALGRIND_H 1
#else
    #define HAS_VALGRIND_H 0
#endif

// OS-specific includes for detecting available host memory
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <iostream>

#define HIP_CHECK_MEMORY(condition)                                                         \
    {                                                                                       \
        hipError_t error = condition;                                                       \
        if(error == hipErrorOutOfMemory)                                                    \
        {                                                                                   \
            (void) hipGetLastError();                                                       \
            std::cout << "Out of memory. Skipping size = " << size << std::endl;            \
            break;                                                                          \
        }                                                                                   \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cout << "HIP error: " << hipGetErrorString(error) << " line: " << __LINE__ \
                      << std::endl;                                                         \
            exit(error);                                                                    \
        }                                                                                   \
    }

#ifndef ROCPRIM_HAS_INT128_SUPPORT
    #define ROCPRIM_HAS_INT128_SUPPORT 1
#endif

#define INSTANTIATE_TYPED_TEST_EXPANDED_1(line, test_suite_name, ...)         \
    namespace Id##line                                                        \
    {                                                                         \
        using test_type = __VA_ARGS__;                                        \
        INSTANTIATE_TYPED_TEST_SUITE_P(Id##line, test_suite_name, test_type); \
    }

#define INSTANTIATE_TYPED_TEST_EXPANDED(line, test_suite_name, ...) \
    INSTANTIATE_TYPED_TEST_EXPANDED_1(line, test_suite_name, __VA_ARGS__)

// Used in input file for rocprim_test_add_parallel.
// Instantiate a typed test suite with a unique name based on line number.
// Do not call this macro twice on the same line.
#define INSTANTIATE_TYPED_TEST(test_suite_name, ...) \
    INSTANTIATE_TYPED_TEST_EXPANDED(__LINE__, test_suite_name, __VA_ARGS__)

#include <cstdlib>
#include <string>
#include <cctype>

namespace test_common_utils
{

inline int obtain_device_from_ctest()
{
    static const std::string rg0    = "CTEST_RESOURCE_GROUP_0";
    char*                    env    = common::__get_env(rg0.c_str());
    int                      device = 0;
    if(env != nullptr)
    {
        std::string amdgpu_target(env);
        std::transform(
            amdgpu_target.cbegin(),
            amdgpu_target.cend(),
            amdgpu_target.begin(),
            // Feeding std::toupper plainly results in implicitly truncating conversions between int and char triggering warnings.
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        char*       env_reqs = common::__get_env((rg0 + "_" + amdgpu_target).c_str());
        std::string reqs(env_reqs);
        device = std::atoi(
            reqs.substr(reqs.find(':') + 1, reqs.find(',') - (reqs.find(':') + 1)).c_str());
        common::clean_env(env_reqs);
    }
    common::clean_env(env);
    return device;
}
}

template<bool continue_on_fail=true>
class TestControl
{
public:
	struct SysInfo
	{
		// Other members are valid only when this is true.
		bool is_initialized = false;
		bool is_apu = false;
		// Valid only when is_apu is true.
		size_t unified_mem_limit = 0;
		// Valid only when is_apu is false.
		size_t host_mem_limit = 0;
		size_t dev_mem_limit = 0;
	};

	TestControl(const float padding_factor=0.1) :
		host_usage(0), dev_usage(0), padding_factor(padding_factor)
    {
        if (!this->sys_info.is_initialized)
            this->init_info();
    }
    
	inline void log_host_usage(const size_t bytes)
    {
        std::cout << "\nLogging host usage: " << bytes << std::endl;
        this->host_usage += bytes;
        this->mem_check();
    }

    inline void log_dev_usage(const size_t bytes)
    {
        std::cout << "\nLogging dev usage: " << bytes << std::endl;
        this->dev_usage += bytes;
        this->mem_check();
    }

	bool continue_on_failure;
	
private:
	inline void mem_check()
        {
            std::cout << "--- mem_check ---" << std::endl;
            bool success = false;
            
            if (TestControl::sys_info.is_apu)
            {
                const size_t total_usage = this->host_usage + this->dev_usage;
                const size_t limit = static_cast<size_t>(TestControl::sys_info.unified_mem_limit * (1 - this->padding_factor));

                std::cout << "total_usage: " << total_usage << std::endl;
                std::cout << "limit: " << limit << std::endl;

                success = total_usage <= limit;
            }
            else
            {
                const size_t host_limit = static_cast<size_t>(TestControl::sys_info.host_mem_limit * (1 - this->padding_factor));
                const size_t dev_limit = static_cast<size_t>(TestControl::sys_info.dev_mem_limit * (1 - this->padding_factor));

                std::cout << "host usage: " << this->host_usage << std::endl;
                std::cout << "host limit: " << host_limit << std::endl;
                std::cout << "dev usage: " << this->dev_usage << std::endl;
                std::cout << "dev limit: " << dev_limit << std::endl;
                
                success = (this->host_usage <= host_limit &&
                           this->dev_usage <= dev_limit);
            }
            std::cout.flush();

            if (!success)
            {
#if continue_on_fail
                std::cout << "issuing continue" << std::endl;
                std::cout << "Skipping size - not enough memory.";
                continue;
#else
                std::cout << "Calling GTEST_SKIP" << std::endl;
                GTEST_SKIP() << "Skipping test - not enough memory.";
#endif
            }
    }
    
	static size_t get_host_memory()
    {
            size_t size = 0;
#ifdef _WIN32
            MEMORYSTATUSEX mem_status;
            mem_status.dsLength = sizeof(mem_status);
            GlobalMemoryStatusEx(&mem_status);
            size = status.ullTotalPhys;
#else
            size = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
#endif

            return static_cast<size_t>(size);
    }
    
	static void init_info()
    {
        std::cout << "*** init_info() ***" << std::endl;
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, 0));
        TestControl::sys_info.is_apu = static_cast<bool>(props.integrated);

        if (TestControl::sys_info.is_apu)
        {
            size_t free_dev_mem;
            size_t total_dev_mem;
            HIP_CHECK(hipMemGetInfo(&free_dev_mem, &total_dev_mem));

            size_t total_host_mem = TestControl::get_host_memory();
            size_t max_dev_shared_mem = total_dev_mem / 2;

            size_t dedicated_dev_mem = total_dev_mem - max_dev_shared_mem;
            size_t total_sys_mem = dedicated_dev_mem + total_host_mem;

            TestControl::sys_info.unified_mem_limit = total_sys_mem;
            std::cout << "free_dev_mem: " << free_dev_mem << std::endl;
            std::cout << "total_dev_mem: " << total_dev_mem << std::endl;
            std::cout << "total_host_mem: " << total_host_mem << std::endl;
            std::cout << "unified_mem_limit: " << TestControl::sys_info.unified_mem_limit << std::endl;
        }
        else
        {
            TestControl::sys_info.host_mem_limit = props.totalGlobalMem;
            TestControl::sys_info.dev_mem_limit = TestControl::get_host_memory();
            std::cout << "host_mem_limit: " << TestControl::sys_info.host_mem_limit << std::endl;
            std::cout << "dev_mem_limit: " << TestControl::sys_info.dev_mem_limit << std::endl;
        }
            
        TestControl::sys_info.is_initialized = true;
    }

    inline static SysInfo sys_info = {};
	float padding_factor;
	size_t host_usage;
	size_t dev_usage;
};

#endif // ROCPRIM_COMMON_TEST_HEADER_HPP_
