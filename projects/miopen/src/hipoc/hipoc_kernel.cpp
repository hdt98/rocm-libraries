// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_PERFORMANCE_LOGS)

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

    const auto log_level = env::value(MIOPEN_PERFORMANCE_LOGS);
    const uint64_t base_level = log_level & 0xFF; // Extract base level (0-255)
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
        // Only main kernels, only executed (not tuning)
        should_accumulate = !is_tuning_mode;
    }
    else if(base_level == 2)
    {
        // All kernels, only executed (not tuning)
        should_log_individual = !is_tuning_mode;
    }
    else if(base_level == 3)
    {
        // Only main kernels, all solutions (including tuning)
        should_accumulate = true;
    }
    else // base_level >= 4
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
        
        if(should_log_individual || should_accumulate)
        {
            // Log to JSON accumulator
            if(IsJsonModeEnabled(log_level))
            {
                AddKernelToJsonAccumulator(exec_id, GetName(), elapsed_time, is_transpose);
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

    const auto log_level = env::value(MIOPEN_PERFORMANCE_LOGS);
    const uint64_t base_level = log_level & 0xFF; // Extract base level (0-255)
    const bool is_tuning_mode = GetKernelTuningMode();
    const bool is_transpose = IsTransposeOrTransformKernel(GetName());
    const auto exec_id = GetKernelExecutionCounter();
    
    // Enhanced logging levels (same as run())
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
        
        if(should_log_individual || should_accumulate)
        {
            // Log to JSON accumulator
            if(IsJsonModeEnabled(log_level))
            {
                AddKernelToJsonAccumulator(exec_id, GetName(), elapsed_time, is_transpose);
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
