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

// todo: remove templating
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
    inline bool alloc_host(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
        std::cout << "\nAlloc host usage (sizeof(T): " << sizeof(T) << "): " << (bytes >> 20) << std::endl;
        this->host_usage += bytes;
        return this->mem_check_host();
    }

    inline bool alloc_host_bytes(const size_t bytes)
    {
        std::cout << "\nAlloc host usage: " << (bytes >> 20) << std::endl;
        this->host_usage += bytes;
        return this->mem_check_host();
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
    inline bool alloc_device(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
        std::cout << "\nAlloc dev usage (sizeof(T): " << sizeof(T) << "): " << (bytes >> 20) << std::endl;
        this->dev_usage += bytes;
        return this->mem_check_device();
    }

    inline bool alloc_device_bytes(const size_t bytes)
    {
        std::cout << "\nAlloc dev usage: " << (bytes >> 20) << std::endl;
        this->dev_usage += bytes;
        return this->mem_check_device();
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

    bool mem_check_host()
    {
        const size_t host_limit = static_cast<size_t>(MemCheck::sys_info.host_mem_limit * (1 - padding_factor));

        std::cout << "mem_check_host()" << std::endl;
        std::cout << "    host usage: " << (host_usage >> 20) << std::endl;
        std::cout << "    host limit: " << (host_limit >> 20) << std::endl;

        bool success = false;
        if (MemCheck::sys_info.is_apu)
        {
            // assume all device usage is shared and subtracts from host memory
            std::cout << "    device usage: " << (dev_usage >> 20) << std::endl;
            success = host_usage <= (host_limit - dev_usage);
        }
        else
        {
            success = host_usage <= host_limit;
        }
        std::cout.flush();
        return success;
    }

    bool mem_check_device()
    {
        const size_t dev_limit = static_cast<size_t>(MemCheck::sys_info.dev_mem_limit * (1 - padding_factor));

        std::cout << "mem_check_device()" << std::endl;
        std::cout << "    device usage: " << (dev_usage >> 20) << std::endl;
        std::cout << "    device limit: " << (dev_limit >> 20) << std::endl;

        bool success = false;
        if (MemCheck::sys_info.is_apu)
        {
            // Any memory used in excess of host_limit - dev_limit will spill
            // into the device's shared memory, reducing the device limit
            const size_t host_limit = static_cast<size_t>(MemCheck::sys_info.host_mem_limit * (1 - padding_factor));
            size_t host_unshared_limit = host_limit - dev_limit;
            size_t spill = host_usage > host_unshared_limit ? 
                           host_usage - host_unshared_limit : 0UL;

            std::cout << "    host usage: " << (host_usage >> 20) << std::endl;
            std::cout << "    spill into device: " << (spill >> 20) << std::endl;

            success = dev_usage <= dev_limit - spill;
        }
        else
        {
            success = dev_usage <= dev_limit;
        }
        std::cout.flush();
        return success;
    }

	// Returns total system memory.
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

	// Do some runtime queries to see if we're on an APU, and retrieve the
	// amount of available GPU and host memory.
	static void init_info()
    {
        std::cout << "*** init_info() ***" << std::endl;

        size_t free_dev_mem;
        size_t total_dev_mem;
        HIP_CHECK(hipMemGetInfo(&free_dev_mem, &total_dev_mem));
        std::cout << "total device memory: " << (total_dev_mem >> 20) << std::endl;
        MemCheck::sys_info.dev_mem_limit = total_dev_mem;

        size_t total_host_mem = MemCheck::get_host_memory();
        std::cout << "total host memory: " << (total_host_mem >> 20) << std::endl;
        MemCheck::sys_info.host_mem_limit = total_host_mem;

        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, 0));
        MemCheck::sys_info.is_apu = static_cast<bool>(props.integrated);
        std::cout << "is_apu: " << MemCheck::sys_info.is_apu << std::endl;
 
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