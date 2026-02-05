// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP
#define GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>

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

/// Structure to hold kernel execution data for JSON output
struct KernelExecutionData
{
    size_t exec_id;
    std::string kernel_name;
    float time_ms;
    std::string timestamp;
    bool is_transform;
};

/// Structure to hold solution data for JSON output
struct SolutionExecutionData
{
    std::string solution_name;
    std::string phase;
    std::vector<KernelExecutionData> kernels;
    
    void Clear()
    {
        solution_name.clear();
        phase.clear();
        kernels.clear();
    }
};

/// Get current timestamp in ISO 8601 format
inline std::string GetCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()) % 1000000;
    
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(6) << microseconds.count();
    
    return oss.str();
}

/// Escape string for JSON
inline std::string JsonEscape(const std::string& str)
{
    std::ostringstream oss;
    for (char c : str)
    {
        switch (c)
        {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 32 || c > 126)
                {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') 
                        << static_cast<int>(c);
                }
                else
                {
                    oss << c;
                }
        }
    }
    return oss.str();
}

/// Thread-local JSON accumulator
inline SolutionExecutionData& GetJsonAccumulator()
{
    thread_local SolutionExecutionData accumulator;
    return accumulator;
}

/// Output accumulated JSON data
inline void FlushJsonAccumulator()
{
    auto& data = GetJsonAccumulator();
    
    if(data.kernels.empty())
        return;
    
    std::cerr << "{\"solution\":\"" << JsonEscape(data.solution_name) << "\","
              << "\"phase\":\"" << data.phase << "\","
              << "\"kernels\":[";
    
    for(size_t i = 0; i < data.kernels.size(); ++i)
    {
        const auto& k = data.kernels[i];
        if(i > 0) std::cerr << ",";
        std::cerr << "{\"exec_id\":" << k.exec_id << ","
                  << "\"kernel_name\":\"" << JsonEscape(k.kernel_name) << "\","
                  << "\"time_ms\":" << k.time_ms << ","
                  << "\"timestamp\":\"" << k.timestamp << "\","
                  << "\"is_transform\":" << (k.is_transform ? "true" : "false") << "}";
    }
    
    std::cerr << "]}" << std::endl;
    data.Clear();
}

/// Manually flush any pending JSON data (call at program end or context switch)
inline void FinalizeJsonLogging()
{
    FlushJsonAccumulator();
}

/// Add kernel to JSON accumulator
inline void AddKernelToJsonAccumulator(size_t exec_id,
                                       const std::string& kernel_name,
                                       float time_ms,
                                       bool is_transform)
{
    auto& data = GetJsonAccumulator();
    data.kernels.push_back({exec_id, kernel_name, time_ms, GetCurrentTimestamp(), is_transform});
}

/// Check if JSON mode is enabled (bit flag in MIOPEN_PERFORMANCE_LOGS)
/// JSON mode is enabled when bit 8 is set (value >= 256)
inline bool IsJsonModeEnabled(uint64_t log_level)
{
    return log_level > 0;
}

/// Log solution name if appropriate for the current log level
/// Only prints if the solution name has changed since the last call
inline void LogSolutionName(const std::string& solution_name, uint64_t log_level)
{
    const bool is_tuning_mode = GetKernelTuningMode();
    const bool json_mode = IsJsonModeEnabled(log_level);
    const uint64_t base_level = log_level & 0xFF; // Get base level without JSON flag
    
    // Determine if we should log based on level and mode
    bool should_log = false;
    
    if(base_level == 0)
    {
        // No logging
    }
    else if(base_level == 1 || base_level == 2)
    {
        // Only executed solutions (not tuning)
        should_log = !is_tuning_mode;
    }
    else // base_level >= 3
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
            if(json_mode)
            {
                // Flush previous solution's JSON data
                FlushJsonAccumulator();
                
                // Set up new solution in accumulator
                auto& data = GetJsonAccumulator();
                data.solution_name = solution_name;
                data.phase = is_tuning_mode ? "tuning" : "execution";
            }
            else
            {
                // Traditional text output
                std::cerr << "[SOLUTION:" << solution_name << "]" << std::endl;
            }
            last_solution = solution_name;
        }
    }
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
