/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/env.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/logger.hpp>
#include <miopen/oclkernel.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_LOG_KERNEL_NAMES)

static std::string DimToFormattedString(const size_t* dims, size_t count)
{
    std::stringstream ss;
    ss << '{';
    for(size_t i = 0; i < count; ++i)
    {
        if(i > 0)
            ss << ", ";
        else
            ss << ' ';
        ss << dims[i];
    }
    ss << " }";
    return ss.str();
}

void OCLKernelInvoke::run() const
{
    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", work_dim = " << work_dim << ", global_work_offset = "
                  << DimToFormattedString(global_work_offset.data(), work_dim)
                  << ", global_work_dim = " << DimToFormattedString(gdims.data(), work_dim)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), work_dim));

    const auto log_level = env::value(MIOPEN_LOG_KERNEL_NAMES);
    const bool is_tuning_mode = GetKernelTuningMode();
    const bool is_transpose = IsTransposeOrTransformKernel(GetName());
    const auto exec_id = GetKernelExecutionCounter();
    
    // Enhanced logging levels:
    // Level 0: No logging
    // Level 1: Only main conv kernel per executed solution (not find/search)
    // Level 2: All kernels for executed solution (not find/search)
    // Level 3: Only main conv kernels for all solutions (including find/search)
    // Level 4: All kernels for all solutions (including find/search)
    
    bool should_log_individual = false;
    bool should_accumulate = false;
    
    if(log_level == 0)
    {
        // No logging
    }
    else if(log_level == 1)
    {
        should_accumulate = !is_tuning_mode;
    }
    else if(log_level == 2)
    {
        should_log_individual = !is_tuning_mode;
    }
    else if(log_level == 3)
    {
        should_accumulate = true;
    }
    else // log_level >= 4
    {
        should_log_individual = true;
    }

    MIOPEN_HANDLE_LOCK

    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
    {
        MIOPEN_THROW("MIOPEN_DEVICE_ARCH used, escaping launching kernel");
    }

    cl_event ev;
    /* way to run OCL group larger than 256
     * hack to ensure local_size == 0, just checking that the 1st dim is 0
     * may want to use a better solution*/
    cl_int status = clEnqueueNDRangeKernel(queue,
                                           kernel.get(),
                                           work_dim,
                                           ((work_dim == 0) ? nullptr : global_work_offset.data()),
                                           gdims.data(),
                                           ((ldims[0] == 0) ? nullptr : ldims.data()),
                                           0,
                                           nullptr,
                                           (callback || should_log_individual || should_accumulate) ? &ev : nullptr);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Running kernel failed: ");
    }
    else if(should_log_individual || should_accumulate)
    {
        clWaitForEvents(1, &ev);
        
        cl_ulong start_time = 0;
        cl_ulong end_time = 0;
        
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
        
        float elapsed_time = (end_time - start_time) / 1000000.0f; // Convert nanoseconds to milliseconds
        
        if(should_log_individual)
        {
            // Log each kernel individually
            std::cerr << "[KERNEL:" << exec_id << "] " << GetName() << " : " << elapsed_time << " ms" << std::endl;
        }
        else if(should_accumulate)
        {
            // Accumulate for solution-level reporting
            auto& accum = GetSolutionTimingAccumulator();
            
            if(accum.exec_id != exec_id)
            {
                if(accum.exec_id > 0 && !accum.main_kernel_name.empty())
                {
                    std::cerr << "[KERNEL:" << accum.exec_id << "] " << accum.main_kernel_name 
                              << " : " << accum.total_time << " ms";
                    if(accum.skipped_count > 0)
                    {
                        std::cerr << " (+" << accum.skipped_count << " transpose/transform kernels)";
                    }
                    std::cerr << std::endl;
                }
                accum.Reset(exec_id);
            }
            
            accum.total_time += elapsed_time;
            accum.kernel_count++;
            
            if(is_transpose)
            {
                accum.skipped_count++;
            }
            else if(accum.main_kernel_name.empty())
            {
                accum.main_kernel_name = GetName();
            }
        }
        
        if(callback)
        {
            callback(ev);
        }
        clReleaseEvent(ev);
    }
    else if(callback)
    {
        clWaitForEvents(1, &ev);
        callback(ev);
    }
}

std::string OCLKernelInvoke::GetName() const
{
    std::array<char, 200> buffer{};

    cl_int status =
        clGetKernelInfo(kernel.get(), CL_KERNEL_FUNCTION_NAME, 200, buffer.data(), nullptr);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Error getting kernel name");
    }
    return buffer.data();
}

OCLKernelInvoke OCLKernel::Invoke(cl_command_queue q, std::function<void(cl_event&)> callback) const
{
#ifndef NDEBUG
    MIOPEN_LOG_I(GetName());
#endif
    OCLKernelInvoke result{q, kernel, gdims.size(), {}, {}, {}, callback};
    std::copy(gdims.begin(), gdims.end(), result.gdims.begin());
    std::copy(ldims.begin(), ldims.end(), result.ldims.begin());
    return result;
}

std::string OCLKernel::GetName() const
{
    std::array<char, 200> buffer{};

    cl_int status =
        clGetKernelInfo(kernel.get(), CL_KERNEL_FUNCTION_NAME, 200, buffer.data(), nullptr);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Error getting kernel name");
    }
    return buffer.data();
}

} // namespace miopen
