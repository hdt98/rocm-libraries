/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP
#define GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP

#include <iostream>
#include <string>

namespace miopen {

/// Thread-local flag to indicate if we're in kernel tuning/search mode
/// This is used by kernel logging to distinguish between:
/// - Kernels executed during find/search/benchmark (tuning mode)
/// - Kernels executed for actual computation (execution mode)
inline bool& GetKernelTuningMode()
{
    thread_local bool is_tuning = false;
    return is_tuning;
}

/// Thread-local counter for execution/iteration tracking
/// Used to group related kernels that belong to the same logical execution
inline size_t& GetKernelExecutionCounter()
{
    thread_local size_t counter = 0;
    return counter;
}

/// Increment and return the execution counter
inline size_t IncrementKernelExecutionCounter()
{
    return ++GetKernelExecutionCounter();
}

/// Get the last printed solution name
inline std::string& GetLastPrintedSolutionName()
{
    thread_local std::string last_solution;
    return last_solution;
}

/// Log solution name if appropriate for the current log level
/// Only prints if the solution name has changed since the last call
inline void LogSolutionName(const std::string& solution_name, uint64_t log_level)
{
    const bool is_tuning_mode = GetKernelTuningMode();
    
    // Determine if we should log based on level and mode
    bool should_log = false;
    
    if(log_level == 0)
    {
        // No logging
    }
    else if(log_level == 1 || log_level == 2)
    {
        // Only executed solutions (not tuning)
        should_log = !is_tuning_mode;
    }
    else // log_level >= 3
    {
        // All solutions including tuning
        should_log = true;
    }
    
    if(should_log && !solution_name.empty())
    {
        auto& last_solution = GetLastPrintedSolutionName();
        // Only print if solution name has changed
        if(solution_name != last_solution)
        {
            std::cerr << "[SOLUTION:" << solution_name << "]" << std::endl;
            last_solution = solution_name;
        }
    }
}

/// Check if a kernel name indicates it's a transpose or transformation kernel
inline bool IsTransposeOrTransformKernel(const std::string& kernel_name)
{
    // Check for common transpose/transformation patterns
    return kernel_name.find("transpose") != std::string::npos ||
           kernel_name.find("Transpose") != std::string::npos ||
           kernel_name.find("transform") != std::string::npos ||
           kernel_name.find("Transform") != std::string::npos ||
           kernel_name.find("SubTensor") != std::string::npos ||
           kernel_name.find("reorder") != std::string::npos ||
           kernel_name.find("Reorder") != std::string::npos;
}

/// Thread-local storage for solution-level timing accumulation
struct SolutionTimingAccumulator
{
    size_t exec_id = 0;
    float total_time = 0.0f;
    int kernel_count = 0;
    int skipped_count = 0;
    std::string main_kernel_name;
    
    void Reset(size_t new_exec_id)
    {
        exec_id = new_exec_id;
        total_time = 0.0f;
        kernel_count = 0;
        skipped_count = 0;
        main_kernel_name.clear();
    }
};

inline SolutionTimingAccumulator& GetSolutionTimingAccumulator()
{
    thread_local SolutionTimingAccumulator accumulator;
    return accumulator;
}

/// RAII helper to set tuning mode for a scope
class ScopedKernelTuningMode
{
public:
    ScopedKernelTuningMode() : prev_value(GetKernelTuningMode())
    {
        GetKernelTuningMode() = true;
    }

    ~ScopedKernelTuningMode()
    {
        GetKernelTuningMode() = prev_value;
    }

    ScopedKernelTuningMode(const ScopedKernelTuningMode&) = delete;
    ScopedKernelTuningMode& operator=(const ScopedKernelTuningMode&) = delete;

private:
    bool prev_value;
};

} // namespace miopen

#endif // GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP
