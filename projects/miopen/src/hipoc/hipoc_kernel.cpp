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
#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/handle.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/logger.hpp>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#define WORKAROUND_SWDEV_448157 1

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_LOG_KERNEL_NAMES)

namespace miopen {

HipEventProfiler::HipEventProfiler(const Handle& handle_)
    : handle(handle_), start(nullptr), stop(nullptr)
{
    if(handle.IsProfilingEnabled())
    {
        start = make_hip_event();
        stop  = make_hip_event();
        hipEventRecord(start.get(), handle.GetStream());
    }
}

HipEventProfiler::~HipEventProfiler()
{
    if(start)
    {
        hipEventRecord(stop.get(), handle.GetStream());
        hipEventSynchronize(stop.get());
        float event_time = 0.0f;
        hipEventElapsedTime(&event_time, start.get(), stop.get());
        handle.ResetKernelTime();
        handle.AccumKernelTime(event_time);
    }
}

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

void HIPOCKernelInvoke::run(void* args, std::size_t size) const
{
    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", global_work_dim = " << DimToFormattedString(gdims.data(), 3)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), 3));

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
        // Only main kernels, only executed (not tuning)
        should_accumulate = !is_tuning_mode;
    }
    else if(log_level == 2)
    {
        // All kernels, only executed (not tuning)
        should_log_individual = !is_tuning_mode;
    }
    else if(log_level == 3)
    {
        // Only main kernels, all solutions (including tuning)
        should_accumulate = true;
    }
    else // log_level >= 4
    {
        // All kernels, all solutions (including tuning)
        should_log_individual = true;
    }
    
    HipEventPtr log_start = nullptr;
    HipEventPtr log_stop  = nullptr;

    if(should_log_individual || should_accumulate)
    {
        log_start = make_hip_event();
        log_stop  = make_hip_event();
        hipEventRecord(log_start.get(), stream);
    }

    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;
    void* config[]    = {// HIP_LAUNCH_PARAM_* are macros that do horrible things
                      // NOLINTNEXTLINE cppcoreguidelines-pro-type-cstyle-cast
                      HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      args,
                      // NOLINTNEXTLINE cppcoreguidelines-pro-type-cstyle-cast
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &size,
                      // NOLINTNEXTLINE cppcoreguidelines-pro-type-cstyle-cast
                      HIP_LAUNCH_PARAM_END};
    if(callback)
    {
        start = make_hip_event();
        stop  = make_hip_event();
    }

    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
    {
        MIOPEN_THROW("MIOPEN_DEVICE_ARCH used, escaping launching kernel");
    }

    MIOPEN_HANDLE_LOCK

    auto status = hipExtModuleLaunchKernel(fun,
                                           gdims[0],
                                           gdims[1],
                                           gdims[2],
                                           ldims[0],
                                           ldims[1],
                                           ldims[2],
                                           0,
                                           stream,
                                           nullptr,
                                           reinterpret_cast<void**>(&config),
                                           start.get(),
                                           stop.get());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed to launch kernel");

    if(should_log_individual || should_accumulate)
    {
        hipEventRecord(log_stop.get(), stream);
        hipEventSynchronize(log_stop.get());
        float elapsed_time = 0.0f;
        hipEventElapsedTime(&elapsed_time, log_start.get(), log_stop.get());
        
        if(should_log_individual)
        {
            // Log each kernel individually
            std::cerr << "[KERNEL:" << exec_id << "] " << GetName() << " : " << elapsed_time << " ms" << std::endl;
        }
        else if(should_accumulate)
        {
            // Accumulate for solution-level reporting
            auto& accum = GetSolutionTimingAccumulator();
            
            // Reset accumulator if we're starting a new execution
            if(accum.exec_id != exec_id)
            {
                // Print previous solution if we have data
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
                // Store the first non-transpose kernel as the main kernel
                accum.main_kernel_name = GetName();
            }
        }
    }

    if(callback)
    {
#if 0
        auto start_time = std::chrono::system_clock::now();
        while(hipEventQuery(stop.get()) == hipErrorNotReady)
        {
            std::this_thread::yield();
            if((std::chrono::system_clock::now() - start_time) > std::chrono::seconds(60))
            {
                std::cerr << "Timeout: HIPOCKernelInvoke::run" << std::endl;
                std::abort();
            }
        }
#else
        hipEventSynchronize(stop.get());
#endif
        callback(start.get(), stop.get());
    }
}

void HIPOCKernelInvoke::run_cooperative(void** kern_args) const
{
    hipError_t status;

    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", global_work_dim = " << DimToFormattedString(gdims.data(), 3)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), 3));

    const auto log_level = env::value(MIOPEN_LOG_KERNEL_NAMES);
    const bool is_tuning_mode = GetKernelTuningMode();
    const bool is_transpose = IsTransposeOrTransformKernel(GetName());
    const auto exec_id = GetKernelExecutionCounter();
    
    // Enhanced logging levels (same as run())
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
    
    HipEventPtr log_start = nullptr;
    HipEventPtr log_stop  = nullptr;

    if(should_log_individual || should_accumulate)
    {
        log_start = make_hip_event();
        log_stop  = make_hip_event();
        hipEventRecord(log_start.get(), stream);
    }

    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
    {
        MIOPEN_THROW("MIOPEN_DEVICE_ARCH used, escaping launching kernel");
    }

    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;

    if(callback)
    {
        start = make_hip_event();
        stop  = make_hip_event();
    }

#if WORKAROUND_SWDEV_448157
    if(gdims[0] >= (1ULL << 32) || gdims[1] >= (1ULL << 32) || gdims[2] >= (1ULL << 32))
        MIOPEN_THROW("gridDim x blockDim >= 2^32");

    if(gdims[0] % ldims[0] != 0 || gdims[1] % ldims[1] != 0 || gdims[2] % ldims[2] != 0)
        MIOPEN_THROW(miopenStatusInternalError);

    unsigned grid_dim_x = gdims[0] / ldims[0];
    unsigned grid_dim_y = gdims[1] / ldims[1];
    unsigned grid_dim_z = gdims[2] / ldims[2];

    MIOPEN_HANDLE_LOCK

    if(callback)
    {
        status = hipEventRecord(start.get(), stream);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "hipEventRecord() failed");
    }

    status = hipModuleLaunchCooperativeKernel(fun,
                                              grid_dim_x,
                                              grid_dim_y,
                                              grid_dim_z,
                                              ldims[0],
                                              ldims[1],
                                              ldims[2],
                                              0,
                                              stream,
                                              kern_args);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed to launch kernel");

    if(should_log_individual || should_accumulate)
    {
        hipEventRecord(log_stop.get(), stream);
        hipEventSynchronize(log_stop.get());
        float elapsed_time = 0.0f;
        hipEventElapsedTime(&elapsed_time, log_start.get(), log_stop.get());
        
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
    }

    if(callback)
    {
        status = hipEventRecord(stop.get(), stream);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "hipEventRecord() failed");
    }
#else
#error "Doesn't work without workaround"
#endif // WORKAROUND_SWDEV_448157

    if(callback)
    {
        hipEventSynchronize(stop.get());
        callback(start.get(), stop.get());
    }
}

HIPOCKernelInvoke HIPOCKernel::Invoke(hipStream_t stream,
                                      std::function<void(hipEvent_t, hipEvent_t)> callback,
                                      bool coop_launch) const
{
    return HIPOCKernelInvoke{stream, fun, ldims, gdims, name, callback, coop_launch};
}
} // namespace miopen
