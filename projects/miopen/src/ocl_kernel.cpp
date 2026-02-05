// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <miopen/env.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/logger.hpp>
#include <miopen/oclkernel.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_PERFORMANCE_LOGS)

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

    const auto base_level = env::value(MIOPEN_PERFORMANCE_LOGS);
    const bool is_tuning_mode = GetKernelTuningMode();
    const bool is_transpose = IsTransposeOrTransformKernel(GetName());
    const auto exec_id = GetKernelExecutionCounter();
    
    // Enhanced logging levels:
    // Level 0: No logging
    // Level 1: Only main conv kernel per executed solution (not find/search)
    // Level 2: All kernels for executed solution (not find/search)
    // Level 3: Only main conv kernels for all solutions (including find/search)
    // Level 4: All kernels for all solutions (including find/search)
    // Add 256 to enable JSON mode
    
    bool should_log_individual = false;
    bool should_accumulate = false;
    
    if(base_level == 0)
    {
        // No logging
    }
    else if(base_level == 1)
    {
        should_accumulate = !is_tuning_mode;
    }
    else if(base_level == 2)
    {
        should_log_individual = !is_tuning_mode;
    }
    else if(base_level == 3)
    {
        should_accumulate = true;
    }
    else // base_level >= 4
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
        
        if(should_log_individual || should_accumulate)
        {
            // Log to JSON accumulator
            if(IsJsonModeEnabled(log_level))
            {
                AddKernelToJsonAccumulator(exec_id, GetName(), elapsed_time, is_transpose);
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
