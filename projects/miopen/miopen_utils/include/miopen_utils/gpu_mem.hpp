/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_UTILS_GPU_MEM_HPP
#define GUARD_MIOPEN_UTILS_GPU_MEM_HPP

// GPU memory allocation wrapper used by both MIOpenDriver and tests.
// Extracted from driver/driver.hpp to break the miopen_utils → driver dependency.

#include <miopen/miopen.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#if MIOPEN_BACKEND_OPENCL
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#elif MIOPEN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#include <common_utils/errors.hpp>
#endif

// Minimal status type for FillBufferWithNans return value.
#ifndef STATUS_SUCCESS
#define STATUS_SUCCESS 0
typedef int status_t;
#endif

struct GPUMem
{
    enum class Check
    {
        None,
        Front,
        Back,
    };

#if MIOPEN_BACKEND_OPENCL
    GPUMem(){};
    GPUMem(cl_context& ctx, size_t psz, size_t pdata_sz, Check ch = Check::None)
        : sz(psz), data_sz(pdata_sz)
    {
        buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, data_sz * sz, nullptr, nullptr);
    }

    int ToGPU(cl_command_queue& q, void* p) const
    {
        return clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, data_sz * sz, p, 0, nullptr, nullptr);
    }
    int FromGPU(cl_command_queue& q, void* p) const
    {
        return clEnqueueReadBuffer(q, buf, CL_TRUE, 0, data_sz * sz, p, 0, nullptr, nullptr);
    }

    cl_mem GetMem() const { return buf; }
    size_t GetSize() const { return sz * data_sz; }

    ~GPUMem() { clReleaseMemObject(buf); }

    cl_mem buf;
    size_t sz;
    size_t data_sz;

#elif MIOPEN_BACKEND_HIP

    GPUMem(){};
    GPUMem(uint32_t ctx, size_t psz, size_t pdata_sz, Check ch = Check::None)
        : _ctx(ctx), sz(psz), data_sz(pdata_sz), check(ch)
    {
        auto status = hipMalloc(static_cast<void**>(&buf), GetTotalSize(GetSize()));
        if(status != hipSuccess)
            COMMON_THROW_HIP_STATUS(status,
                                    "[MIOpenDriver] hipMalloc " + std::to_string(GetSize()));
        buf = static_cast<char*>(buf) + GetOffsetToUserBuffer();
    }

    int ToGPU(hipStream_t q, void* p)
    {
        _q = q;
        return static_cast<int>(hipMemcpy(buf, p, GetSize(), hipMemcpyHostToDevice));
    }
    int FromGPU(hipStream_t q, void* p)
    {
        (void)hipDeviceSynchronize();
        _q = q;
        return static_cast<int>(hipMemcpy(p, buf, GetSize(), hipMemcpyDeviceToHost));
    }

    template <typename Tgpu>
    status_t FillBufferWithNans(miopenHandle_t handle, const miopenTensorDescriptor_t tensorDesc)
    {
        if(std::is_same<Tgpu, int8_t>::value)
        {
            Tgpu max = std::numeric_limits<Tgpu>::max();
            miopenSetTensor(handle, tensorDesc, GetMem(), &max);
        }
        else
        {
            Tgpu nan = std::numeric_limits<Tgpu>::quiet_NaN();
            miopenSetTensor(handle, tensorDesc, GetMem(), &nan);
        }

        return STATUS_SUCCESS;
    }

    void* GetMem() { return buf; }
    size_t GetSize() { return sz * data_sz; }

    size_t GetTotalSize(size_t userSize)
    {
        if(check == Check::None)
            return userSize;

        constexpr size_t maxPadding = 2ULL * 1024 * 1024 - 1;

        auto roundUpToPageAlignment = [&](size_t bytes) {
            return (bytes + maxPadding) & ~maxPadding;
        };

        return roundUpToPageAlignment(userSize);
    }

    size_t GetOffsetToUserBuffer()
    {
        if(check == Check::Back)
        {
            auto userSize = GetSize();
            return GetTotalSize(userSize) - userSize;
        }
        return 0;
    }

    ~GPUMem()
    {
        buf = static_cast<char*>(buf) - GetOffsetToUserBuffer();
        (void)hipFree(buf);
    }

    hipStream_t _q; // Place holder for opencl context
    uint32_t _ctx;
    void* buf;
    size_t sz;
    size_t data_sz;
    Check check;
#endif
};

#endif // GUARD_MIOPEN_UTILS_GPU_MEM_HPP
