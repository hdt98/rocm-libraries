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

#ifdef _WIN32
#include <windows.h>
#else
#include <fstream>
#include <string>
#include <unistd.h>
#endif

#ifdef MEMCHECK_LOGGING
#include <iostream>
#endif

#include "../../common/utils.hpp"

namespace test_utils
{
// 32GB
constexpr static size_t minimum_memory_required_bytes = 34359738368;

inline unsigned long long get_total_system_memory(bool is_apu)
{
    unsigned long long total_system_memory = 0;
#ifdef _WIN32
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


// MemCheck tracks host and device memory usage to skip test sizes that would
// exceed available memory. This is needed on APU systems where CPU and GPU share
// a single memory pool, making it easy to exhaust memory with large test inputs.
//
// The MemCheck alloc() functions are called before the actual memory allocation
// so the code can gracefully handle an out-of-memory situation.
//
// The MemCheck free() functions must be called when memory is freed before the
// end of the MemCheck object's scope.
// 
// For tests with a single input size (skip the whole test):
//
//   MemCheck mem_check;
//   if(!mem_check.alloc_device_bytes(sizeof(type) * size)) GTEST_SKIP();
//
// For tests with multiple input sizes (skip the size or break out):
//
//   for(auto size : sizes)
//   {
//       MemCheck mem_check;
//       if(!mem_check.alloc_host_bytes(sizeof(type) * size)) continue;
//       std::vector<type> host_vec(size);
//       if(!mem_check.alloc_device_bytes(sizeof(type) * size)) continue;
//       type* d_ptr;
//       HIP_CHECK(hipMalloc(&d_ptr, sizeof(type) * size));
//       // ... run test ...
//       HIP_CHECK(hipFree(d_ptr));
//   }
//
//  It is possible to create one MemCheck object outside of the size loop, but
//  that introduces the possibility of forgetting to call free() on something
//  allocated inside the loop, and thus incorrect memory tracking for subsequent
//  loop iterations.  Creating a new MemCheck object each time ensures correct
//  tracking for the current size.
//
// Define MEMCHECK_LOGGING at compile time to enable diagnostic output.

// #define MEMCHECK_LOGGING

class MemCheck
{
public:
	// padding_factor is a value in [0, 1] that indicates how much of a buffer we should leave below
	// the calculated memory limits.
	// i.e when allocations are >= actual_limit * (1 - padding_factor), then assume we're out of memory.
	MemCheck(const hipStream_t stream = 0, const float padding_factor = 0.1f) :
		padding_factor(padding_factor)
    {
        // Some of this information could be queried once and stored as static data
        // to be shared across multiple instances of MemCheck.
        // This constructor is not meant to be in performance critical code so it's OK
        // if we end up repeatedly querying the same data to keep things simpler.

        size_t free_dev_mem;
        HIP_CHECK(hipMemGetInfo(&free_dev_mem, &dev_limit));
        dev_usage = dev_limit - free_dev_mem;

#ifdef _WIN32
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        GlobalMemoryStatusEx(&mem_status);
        host_limit = mem_status.ullTotalPhys;
        host_usage = mem_status.ullTotalPhys - mem_status.ullAvailPhys;
#else
        host_limit = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);

        // MemAvailable accounts for reclaimable page cache and slab memory,
        // so it more accurately reflects what the OS can give to a new allocation
        // than MemFree alone.
        size_t mem_available = 0;
        {
            std::ifstream meminfo("/proc/meminfo");
            std::string label;
            size_t value;
            std::string unit;
            while(meminfo >> label >> value >> unit)
            {
                if(label == "MemAvailable:")
                {
                    mem_available = value * 1024; // kB → bytes
                    break;
                }
            }
        }
        host_usage = host_limit - mem_available;
#endif

        rocprim::detail::target_arch arch;
        HIP_CHECK(rocprim::detail::host_target_arch(stream, arch));
        is_apu = test_utils::is_apu(arch);

        if (is_apu) {
            // For APUs, OS will share up to half of system memory, not exceeding the
            // amount of reported device memory.
            dev_shared = std::min(dev_limit, host_limit / 2);
            // The carved memory is the difference between the total memory and shared.
            dev_carved = dev_limit - dev_shared;
        }

#ifdef MEMCHECK_LOGGING
        std::cout << "MemCheck: device " << toMB(dev_usage) << "/" << toMB(dev_limit)
                  << " MiB, host " << toMB(host_usage) << "/" << toMB(host_limit)
                  << " MiB, dev_shared: " << toMB(dev_shared) << " MiB, dev_carved: " << toMB(dev_carved)
                  << " MiB, is_apu=" << is_apu << std::endl;
#endif
    }

	// Call this before host allocs
    template<typename T>
    inline bool alloc_host(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
#ifdef MEMCHECK_LOGGING
        std::cout << "alloc host: " << toMB(bytes) << " MiB" << std::endl;
#endif
        this->host_usage += bytes;
        return this->mem_check_host();
    }

    inline bool alloc_host_bytes(const size_t bytes)
    {
#ifdef MEMCHECK_LOGGING
        std::cout << "alloc host: " << toMB(bytes) << " MiB" << std::endl;
#endif
        this->host_usage += bytes;
        return this->mem_check_host();
    }

    // Call this before host frees
    template<typename T>
    inline void free_host(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
#ifdef MEMCHECK_LOGGING
        std::cout << "free host: " << toMB(bytes) << " MiB" << std::endl;
#endif
        this->host_usage -= bytes;
    }

    inline void free_host_bytes(const size_t bytes)
    {
#ifdef MEMCHECK_LOGGING
        std::cout << "free host: " << toMB(bytes) << " MiB" << std::endl;
#endif
        this->host_usage -= bytes;
    }

	// Call this before dev allocs
    template<typename T>
    inline bool alloc_device(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
#ifdef MEMCHECK_LOGGING
        std::cout << "alloc device: " << toMB(bytes) << " MiB" << std::endl;
#endif
        this->dev_usage += bytes;
        return this->mem_check_device();
    }

    inline bool alloc_device_bytes(const size_t bytes)
    {
#ifdef MEMCHECK_LOGGING
        std::cout << "alloc device: " << toMB(bytes) << " MiB" << std::endl;
#endif
        this->dev_usage += bytes;
        return this->mem_check_device();
    }

    // Call this before dev frees
    template<typename T>
    inline void free_device(const size_t size)
    {
        size_t bytes = sizeof(T) * size;
#ifdef MEMCHECK_LOGGING
        std::cout << "free device: " << toMB(bytes) << " MiB" << std::endl;
#endif
        this->dev_usage -= bytes;
    }

    inline void free_device_bytes(const size_t bytes)
    {
#ifdef MEMCHECK_LOGGING
        std::cout << "free device: " << toMB(bytes) << " MiB" << std::endl;
#endif
        this->dev_usage -= bytes;
    }

private:

    static size_t toMB(size_t bytes) { return bytes >> 20; }

    // Returns true if there is enough memory, false if the caller should skip.
    // GTEST_SKIP() cannot be called here because it only exits the immediate
    // function, not the enclosing test body.  The caller must act on the
    // return value directly from within the test.
    // Making the function inline does not enable GTEST_SKIP() to work, because
    // inline functions are still functions and not macros.
    bool mem_check_host()
    {
        // Reduce the host_limit by a padding factor as a safety margin.
        size_t host_limit_padded = static_cast<size_t>(host_limit * (1 - padding_factor));

        bool success;
        if (is_apu)
        {
            // The spill into shared memory is the device memory used that exceeds
            // the device's carved memory.
            size_t spill = 0;
            if (dev_usage > dev_carved) spill = dev_usage - dev_carved;

            if (host_limit_padded > spill)
            {
                // reduce the host limit by the amount of spill into shared memory
                host_limit_padded -= spill;
            }
            else
            {
                // The amount we're spilling exceeds the total padded host memory,
                //  something is likely wrong but we need to handle it cleanly.
                host_limit_padded = 0;
            }

            success = host_usage <= host_limit_padded;

#ifdef MEMCHECK_LOGGING
            std::cout << "mem_check_host: host=" << toMB(host_usage) << "/" << toMB(host_limit_padded)
                      << " MiB, device=" << toMB(dev_usage) << " MiB, spill=" << toMB(spill)
                      << " MiB" << std::endl;
#endif
        }
        else
        {
            success = host_usage <= host_limit_padded;
#ifdef MEMCHECK_LOGGING
            std::cout << "mem_check_host: host=" << toMB(host_usage) << "/" << toMB(host_limit_padded)
                      << " MiB" << std::endl;
#endif
        }
#ifdef MEMCHECK_LOGGING
        if (!success)
            std::cout << "mem_check_host: out of memory" << std::endl;
#endif
        return success;
    }

    bool mem_check_device()
    {
        // Reduce the dev_limit by a padding factor as a safety margin.
        size_t dev_limit_padded = static_cast<size_t>(dev_limit * (1 - padding_factor));

        bool success;
        if (is_apu)
        {
            size_t host_unshared = host_limit - dev_shared;

            // The spill into shared memory is the amount of memory used that exceeds the host's
            //  unshared memory.
            size_t spill = 0;
            if (host_usage > host_unshared) spill = host_usage - host_unshared; 

            if (dev_limit_padded > spill)
            {
                // reduce the device limit by the amount of spill into shared memory
                dev_limit_padded -= spill;
            }
            else
            {
                // The amount we're spilling exceeds the total device padded memory,
                //  something is likely wrong but we need to handle it cleanly.
                dev_limit_padded = 0;
            }

            success = dev_usage <= dev_limit_padded;

#ifdef MEMCHECK_LOGGING
            std::cout << "mem_check_device: device=" << toMB(dev_usage) << "/" << toMB(dev_limit_padded)
                      << " MiB, host=" << toMB(host_usage) << " MiB, spill=" << toMB(spill)
                      << " MiB" << std::endl;
#endif
        }
        else
        {
            success = dev_usage <= dev_limit_padded;
#ifdef MEMCHECK_LOGGING
            std::cout << "mem_check_device: device=" << toMB(dev_usage) << "/" << toMB(dev_limit_padded)
                      << " MiB" << std::endl;
#endif
        }
#ifdef MEMCHECK_LOGGING
        if (!success)
            std::cout << "mem_check_device: out of memory" << std::endl;
#endif
        return success;
    }

	bool is_apu = false;
	size_t host_limit = 0;
	size_t dev_limit = 0;
    size_t dev_shared = 0;
    size_t dev_carved = 0;
	float padding_factor;
	size_t host_usage = 0;
	size_t dev_usage = 0;
};

}
#endif // TEST_TEST_UTILS_MEMORY_CHECK_HPP_