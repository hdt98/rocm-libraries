// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef TEST_TEST_UTILS_MEMORY_CHECK_HPP_
#define TEST_TEST_UTILS_MEMORY_CHECK_HPP_

#include <hip/hip_runtime.h>

#if defined(WIN32)
#include <windows.h>
#include <tchar.h>
#else
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#endif

#include "../../common/utils.hpp"

namespace test_utils
{
// 32GB
constexpr static size_t minimum_memory_required_bytes = 34359738368;

inline unsigned long long get_total_system_memory(bool is_apu)
{
    unsigned long long total_system_memory = 0;
#if defined(WIN32)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx (&statex);
    total_system_memory = statex.ullTotalPhys;
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string label;
    unsigned long long value;
    std::string unit;

    while (meminfo >> label >> value >> unit)
    {
        if (label == "MemTotal:")
        {
            total_system_memory = value;
        }

        // Stop once totalMem is found
        if (total_system_memory > 0) break;
    }
#endif
    if(is_apu)
    {
        size_t gpu_free_memory, gpu_total_memory;
        HIP_CHECK(hipMemGetInfo(&gpu_free_memory, &gpu_total_memory));

        // For APUs, OS will share up to half of visible system memory.
        // "Visible system memory" is total CPU RAM minus the carved-out
        // dedicated GPU memory.
        unsigned long long shared_gpu_memory = total_system_memory / 2;
        unsigned long long dedicated_gpu_memory = gpu_total_memory - shared_gpu_memory;
        total_system_memory = total_system_memory + dedicated_gpu_memory;
    }

    return total_system_memory;
}


// drop MemChk class in here
// We need to be able to skip tests in 2 ways:
// 1. Some tests loop though multiple input sizes.
//    When we hit a size that causes out of memory, skip the size and continue the next iteration.
// 2. Some tests only use a single input size. In this case, skip the entire test when we run out of memory.
//
// Idea:
// Class with a static member struct to store info about memory limits.
// Instantiate it at the top of every GTest function.
// Every time we want to allocate memory on host or device, first call
// functions to log the allocation with the MemCheck instance.
// The log functions check memory usage and, if it exceeds the threshold,
// issue a `continue` (goto next iteration of enclosing loop)
// or `GTEST_SKIP()` depending on the `continue_on_fail` template arg.
// Eg:
// MemCheck<continue_on_fail> checker;
//
// // log host allocation
// checker.log_host_usage(sizeof(type) * size);
// std::vector<type> host_vec(size);
// ...
// // log device allocation
// checker.log_dev_usage(sizeof(type) * size);
// int d_ptr;
// HIP_CHECK(hipMalloc(&d_ptr, sizeof(type) * size));
//
// Haven't tested this much yet, but so far the limits don't seem quite right.

// OS-specific includes for detecting available host memory
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// When continue_on_fail == true, issue a continue statement (skip to next
// iteration of enclosing loop, which runs the next input size) when mem check fails.
// Otherwise, issue a GTEST_SKIP() call to skip the test completely.
template<bool continue_on_fail=true>
class MemCheck
{
public:
	struct SysInfo
	{
		// Other members are valid only when this is true.
		bool is_initialized = false;

		// Is this an APU or discrete GPU?
		bool is_apu = false;

        // Valid when is_apu is true.
		size_t unified_mem_limit = 0;
		
		// Valid when is_apu is false.
		size_t host_mem_limit = 0;
		size_t dev_mem_limit = 0;
	};

	// padding_factor is a value in [0, 1] that indicates how much of a buffer we should leave below
	// the calculuated memory limits.
	// i.e when allocations are >= actual_limit * (1 - padding_factor), then assume we're out of memory.
	MemCheck(const float padding_factor=0.1) :
		host_usage(0), dev_usage(0), padding_factor(padding_factor)
    {
        std::cout << "*********** MemCheck() ***************" << std::endl;
        if (!this->sys_info.is_initialized)
            this->init_info();
    }

	// Call this before host allocs. Must be inline to allow mem_check to issue continue/GTEST_SKIP() statements
    template<typename T>
    inline void alloc_host(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
        std::cout << "\nAlloc host usage (sizeof(T): " << sizeof(T) << "): " << (bytes >> 20) << std::endl;
        this->host_usage += bytes;
        this->mem_check();
    }

    inline void alloc_host_bytes(const size_t bytes)
    {
        std::cout << "\nAlloc host usage: " << (bytes >> 20) << std::endl;
        this->host_usage += bytes;
        this->mem_check();
    }

    template<typename T>
    inline void free_host(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
        std::cout << "\nFree host usage: " << (bytes >> 20) << std::endl;
        this->host_usage -= bytes;
    }

    inline void free_host_bytes(const size_t bytes)
    {
        std::cout << "\nFree host usage: " << (bytes >> 20) << std::endl;
        this->host_usage -= bytes;
    }

	// Call this before dev allocs
    template<typename T>
    inline void alloc_device(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
        std::cout << "\nAlloc dev usage (sizeof(T): " << sizeof(T) << "): " << (bytes >> 20) << std::endl;
        this->dev_usage += bytes;
        this->mem_check();
    }

    inline void alloc_device_bytes(const size_t bytes)
    {
        std::cout << "\nAlloc dev usage: " << (bytes >> 20) << std::endl;
        this->dev_usage += bytes;
        this->mem_check();
    }

    template<typename T>
    inline void free_device(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
        std::cout << "\nFree dev usage: " << (bytes >> 20) << std::endl;
        this->dev_usage -= bytes;
    }

    inline void free_device_bytes(const size_t bytes)
    {
        std::cout << "\nFree dev usage: " << (bytes >> 20) << std::endl;
        this->dev_usage -= bytes;
    }

private:

	inline void mem_check()
        {
            std::cout << "--- mem_check ---" << std::endl;
            bool success = false;
            
            if (MemCheck::sys_info.is_apu)
            {
                const size_t total_usage = this->host_usage + this->dev_usage;
                const size_t limit = static_cast<size_t>(MemCheck::sys_info.unified_mem_limit * (1 - this->padding_factor));

                std::cout << "    host_usage: " << (host_usage >> 20) << std::endl;
                std::cout << "    device_usage: " << (dev_usage >> 20) << std::endl;
                std::cout << "    total_usage: " << (total_usage >> 20) << std::endl;
                std::cout << "    limit: " << (limit >> 20) << std::endl;

                success = total_usage <= limit;
            }
            else
            {
                const size_t host_limit = static_cast<size_t>(MemCheck::sys_info.host_mem_limit * (1 - this->padding_factor));
                const size_t dev_limit = static_cast<size_t>(MemCheck::sys_info.dev_mem_limit * (1 - this->padding_factor));

                std::cout << "    host usage: " << this->host_usage << std::endl;
                std::cout << "    host limit: " << host_limit << std::endl;
                std::cout << "    dev usage: " << this->dev_usage << std::endl;
                std::cout << "    dev limit: " << dev_limit << std::endl;
                
                success = (this->host_usage <= host_limit &&
                           this->dev_usage <= dev_limit);
            }
            std::cout.flush();

            if (!success)
            {
#if continue_on_fail
                std::cout << "Issuing continue." << std::endl;
                std::cout << "Skipping size - not enough memory.";
                continue;
#else
                std::cout << "Calling GTEST_SKIP" << std::endl;
                GTEST_SKIP() << "Skipping test - not enough memory.";
#endif
            }
    }

	// Returns total system memory minus the amount that's been carved out for the GPU.
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

	// Do some runtime queries to see if we're on an APU, and try to calculate the
	// amount of available GPU and host memory.
	static void init_info()
    {
        std::cout << "*** init_info() ***" << std::endl;
		
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, 0));
        MemCheck::sys_info.is_apu = static_cast<bool>(props.integrated);

        if (MemCheck::sys_info.is_apu)
        {
            size_t free_dev_mem;
            size_t total_dev_mem;
			// hipMemGetInfo fetches total free GPU memory, but currently
			// on APUs this does not get updated when host allocations are made.
            HIP_CHECK(hipMemGetInfo(&free_dev_mem, &total_dev_mem));

            size_t total_host_mem = MemCheck::get_host_memory();
            size_t max_dev_shared_mem = total_dev_mem / 2;

            size_t dedicated_dev_mem = total_dev_mem - max_dev_shared_mem;
            size_t total_sys_mem = dedicated_dev_mem + total_host_mem;

            MemCheck::sys_info.unified_mem_limit = total_sys_mem;
            std::cout << "free_dev_mem: " << free_dev_mem << std::endl;
            std::cout << "total_dev_mem: " << total_dev_mem << std::endl;
            std::cout << "total_host_mem: " << total_host_mem << std::endl;
            std::cout << "unified_mem_limit: " << MemCheck::sys_info.unified_mem_limit << std::endl;
        }
        else
        {
			// Here all GPU memory is dedicated and there's no shared memory.
            MemCheck::sys_info.host_mem_limit = props.totalGlobalMem;
            MemCheck::sys_info.dev_mem_limit = MemCheck::get_host_memory();
            std::cout << "host_mem_limit: " << MemCheck::sys_info.host_mem_limit << std::endl;
            std::cout << "dev_mem_limit: " << MemCheck::sys_info.dev_mem_limit << std::endl;
        }
            
        MemCheck::sys_info.is_initialized = true;
    }

	// This must be inline to prevent multiple definition errors
    inline static SysInfo sys_info = {};
	
	float padding_factor;
	size_t host_usage;
	size_t dev_usage;
};


}
#endif // TEST_TEST_UTILS_MEMORY_CHECK_HPP_