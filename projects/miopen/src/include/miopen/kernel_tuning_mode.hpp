// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP
#define GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

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

/// Get the last printed solver ID
inline uint64_t& GetLastPrintedSolverId()
{
    thread_local uint64_t last_solver_id = 0;
    return last_solver_id;
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
    std::string main_kernel_name;
    
    void Reset(size_t new_exec_id)
    {
        exec_id = new_exec_id;
        total_time = 0.0f;
        kernel_count = 0;
        main_kernel_name.clear();
    }
};

inline SolutionTimingAccumulator& GetSolutionTimingAccumulator()
{
    thread_local SolutionTimingAccumulator accumulator;
    return accumulator;
}

/// Structure to hold kernel execution data for JSON output (single execution)
struct KernelExecutionData
{
    size_t exec_id;
    std::string kernel_name;
    float time_ms;
    int transformation_count;
    std::vector<std::string> transformation_kernel_names;  // Collected transformation kernel names
    bool is_transformation;
};

/// Structure to hold grouped kernel data by kernel_name (multiple executions)
struct GroupedKernelData
{
    std::string kernel_name;
    std::vector<float> time_executions_ms;  // Renamed from time_ms_array
    float time_ms;                          // Average time (excluding first)
    float time_std_ms;                      // Standard deviation (excluding first)
    float time_min_ms;                      // Minimum time (excluding first)
    float time_max_ms;                      // Maximum time (excluding first)
    int number_of_transformations;
    std::vector<std::string> transformation_kernels;  // Names of transformation kernels
    size_t exec_number;  // First/lowest exec_id
    bool is_transformation;
};

/// Structure to accumulate kernels by exec_id
struct ExecIdAccumulator
{
    size_t exec_id = 0;
    std::string main_kernel_name;
    float total_time_ms = 0.0f;
    int transformation_count = 0;
    std::vector<std::string> transformation_kernel_names;  // Track transformation kernel names
    bool has_data = false;
    
    void Reset()
    {
        exec_id = 0;
        main_kernel_name.clear();
        total_time_ms = 0.0f;
        transformation_count = 0;
        transformation_kernel_names.clear();
        has_data = false;
    }
};

/// Structure to hold solution data for JSON output
struct SolutionExecutionData
{
    std::string solution_name;
    uint64_t solver_id;
    std::string phase;
    std::vector<KernelExecutionData> kernels;
    
    void Clear()
    {
        solution_name.clear();
        solver_id = 0;
        phase.clear();
        kernels.clear();
    }
};

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

/// Thread-local exec_id accumulator for kernel grouping
inline ExecIdAccumulator& GetCurrentExecIdAccumulator()
{
    thread_local ExecIdAccumulator accumulator;
    return accumulator;
}

/// Calculate statistics for grouped kernel data (excluding first execution)
inline void CalculateStatistics(GroupedKernelData& data)
{
    const auto& times = data.time_executions_ms;
    if(times.size() <= 1)
    {
        // If only one execution, use that value for all stats
        data.time_ms = times.empty() ? 0.0f : times[0];
        data.time_std_ms = 0.0f;
        data.time_min_ms = data.time_ms;
        data.time_max_ms = data.time_ms;
        return;
    }
    
    // Calculate stats excluding first execution
    float sum = 0.0f;
    float min_val = times[1];
    float max_val = times[1];
    
    for(size_t i = 1; i < times.size(); ++i)
    {
        sum += times[i];
        min_val = std::min(min_val, times[i]);
        max_val = std::max(max_val, times[i]);
    }
    
    data.time_ms = sum / (times.size() - 1);
    data.time_min_ms = min_val;
    data.time_max_ms = max_val;
    
    // Calculate standard deviation
    float sq_sum = 0.0f;
    for(size_t i = 1; i < times.size(); ++i)
    {
        float diff = times[i] - data.time_ms;
        sq_sum += diff * diff;
    }
    data.time_std_ms = std::sqrt(sq_sum / (times.size() - 1));
}

/// Finalize the current exec_id accumulator and add it to the solution kernels
inline void FinalizeCurrentExecId()
{
    auto& exec_accum = GetCurrentExecIdAccumulator();
    
    if(!exec_accum.has_data)
        return;
    
    auto& data = GetJsonAccumulator();
    data.kernels.push_back({
        exec_accum.exec_id,
        exec_accum.main_kernel_name,
        exec_accum.total_time_ms,
        exec_accum.transformation_count,
        exec_accum.transformation_kernel_names,  // Pass transformation kernel names
        false  // is_transformation - this is the accumulated main kernel
    });
    
    exec_accum.Reset();
}

/// Output accumulated JSON data
inline void FlushJsonAccumulator()
{
    // Finalize any pending exec_id before flushing
    FinalizeCurrentExecId();
    
    auto& data = GetJsonAccumulator();
    
    if(data.kernels.empty())
        return;
    
    // Group kernels by kernel_name and accumulate transformation kernels
    std::map<std::string, GroupedKernelData> grouped_kernels;
    
    for(const auto& k : data.kernels)
    {
        auto& grouped = grouped_kernels[k.kernel_name];
        
        // Initialize on first occurrence
        if(grouped.time_executions_ms.empty())
        {
            grouped.kernel_name = k.kernel_name;
            grouped.number_of_transformations = k.transformation_count;
            grouped.exec_number = k.exec_id;  // First/lowest exec_id
            grouped.is_transformation = k.is_transformation;
            
            // Use the transformation_kernels from the first exec_id amalgamation
            // This avoids duplicates when grouping by kernel_name
            grouped.transformation_kernels = k.transformation_kernel_names;
        }
        
        // Append timing data
        grouped.time_executions_ms.push_back(k.time_ms);
        
        // No need to accumulate transformation kernels from subsequent executions
        // as they were already collected in the first amalgamation (by exec_id)
    }
    
    // Calculate statistics for each grouped kernel
    for(auto& entry : grouped_kernels)
    {
        CalculateStatistics(entry.second);
    }
    
    // Output JSON
    std::cerr << "{\"solution\":\"" << JsonEscape(data.solution_name) << "\","
              << "\"solver_id\":" << data.solver_id << ","
              << "\"phase\":\"" << data.phase << "\","
              << "\"kernels\":[";
    
    bool first = true;
    for(const auto& entry : grouped_kernels)
    {
        const auto& g = entry.second;
        if(!first) std::cerr << ",";
        first = false;
        
        std::cerr << "{\"kernel_name\":\"" << JsonEscape(g.kernel_name) << "\","
                  << "\"time_executions_ms\":[";
        
        for(size_t i = 0; i < g.time_executions_ms.size(); ++i)
        {
            if(i > 0) std::cerr << ",";
            std::cerr << g.time_executions_ms[i];
        }
        
        std::cerr << "],"
                  << "\"time_ms\":" << g.time_ms << ","
                  << "\"time_std_ms\":" << g.time_std_ms << ","
                  << "\"time_min_ms\":" << g.time_min_ms << ","
                  << "\"time_max_ms\":" << g.time_max_ms << ","
                  << "\"number_of_transformations\":" << g.number_of_transformations << ",";
        
        // Output transformation_kernels array
        std::cerr << "\"transformation_kernels\":[";
        for(size_t i = 0; i < g.transformation_kernels.size(); ++i)
        {
            if(i > 0) std::cerr << ",";
            std::cerr << "\"" << JsonEscape(g.transformation_kernels[i]) << "\"";
        }
        std::cerr << "],";
        
        std::cerr << "\"exec_number\":" << g.exec_number << ","
                  << "\"is_transformation\":" << (g.is_transformation ? "true" : "false") << "}";
    }
    
    std::cerr << "]}" << std::endl;
    data.Clear();
}

/// Manually flush any pending JSON data (call at program end or context switch)
inline void FinalizeJsonLogging()
{
    FlushJsonAccumulator();
}

/// Check if we should list kernels individually without exec_id accumulation
/// This is enabled when MIOPEN_PERFORMANCE_LOGS is 2 or 4
inline bool IsIndividualKernelListingMode(uint64_t log_level)
{
    return log_level == 2 || log_level == 4;
}

/// Add kernel to JSON accumulator with optional exec_id grouping
/// If in individual kernel listing mode (log_level 2 or 4), kernels are added individually
/// Otherwise, kernels are accumulated by exec_id
inline void AddKernelToJsonAccumulator(size_t exec_id,
                                       const std::string& kernel_name,
                                       float time_ms,
                                       bool is_transform,
                                       uint64_t log_level)
{
    // Check if we should list kernels individually
    if(IsIndividualKernelListingMode(log_level))
    {
        // Individual kernel listing mode - add directly without exec_id accumulation
        auto& data = GetJsonAccumulator();
        data.kernels.push_back({
            exec_id,
            kernel_name,
            time_ms,
            0,  // transformation_count is 0 for individual kernels
            {},  // transformation_kernel_names - empty for individual kernels
            is_transform
        });
    }
    else
    {
        // Standard mode - accumulate by exec_id
        auto& exec_accum = GetCurrentExecIdAccumulator();
        
        // Check if we've moved to a new exec_id
        if(exec_accum.has_data && exec_accum.exec_id != exec_id)
        {
            // Finalize the previous exec_id before starting a new one
            FinalizeCurrentExecId();
        }
        
        // Initialize or update the current exec_id accumulator
        if(!exec_accum.has_data)
        {
            exec_accum.exec_id = exec_id;
            exec_accum.has_data = true;
        }
        
        // Add timing to total
        exec_accum.total_time_ms += time_ms;
        
        // Track transformation kernels or set main kernel name
        if(is_transform)
        {
            exec_accum.transformation_count++;
            exec_accum.transformation_kernel_names.push_back(kernel_name);
        }
        else
        {
            // This is a main kernel - use it as the representative kernel name
            // If there are multiple non-transform kernels, the last one wins
            exec_accum.main_kernel_name = kernel_name;
            exec_accum.transformation_kernel_names.push_back(kernel_name);
        }
    }
}

/// Check if JSON mode is enabled (bit flag in MIOPEN_PERFORMANCE_LOGS)
/// JSON mode is enabled when bit 8 is set (value >= 256)
inline bool IsPerformanceLoggingEnabled(uint64_t log_level)
{
    return log_level > 0;
}

/// Check if this kernel should be logged
/// - When log_level <= 2: Only log execution phase kernels (not tuning)
/// - When log_level > 2: Log both tuning and execution phase kernels
inline bool IsLoggingKernel(uint64_t log_level, bool is_tuning)
{
    if (log_level > 2)
    {
        // Log all kernels (both tuning and execution) when log_level > 2
        return true;
    }
    else if (log_level > 0)
    {
        // Only log execution phase kernels when 0 < log_level <= 2
        return !is_tuning;
    }
    return false;
}

/// Log solution name if appropriate for the current log level
/// Only prints if the solution name or solver_id has changed since the last call
inline void LogSolutionName(const std::string& solution_name, uint64_t solver_id, uint64_t log_level)
{
    const bool is_tuning_mode = GetKernelTuningMode();
    const bool logging_enabled = IsLoggingKernel(log_level, is_tuning_mode);
    
    // Always update tracking variables when solution/solver changes, regardless of logging state
    // This prevents stale data from appearing when logging state changes
    if(!solution_name.empty() && logging_enabled)
    {
        auto& last_solution = GetLastPrintedSolutionName();
        auto& last_solver_id = GetLastPrintedSolverId();
        
        // Check if solution name or solver_id has changed
        if(solution_name != last_solution || solver_id != last_solver_id)
        {
            // Flush previous solution's JSON data
            FlushJsonAccumulator();
            
            // Set up new solution in accumulator
            auto& data = GetJsonAccumulator();
            data.solution_name = solution_name;
            data.solver_id = solver_id;
            data.phase = is_tuning_mode ? "tuning" : "execution";
            
            // Always update tracking to prevent stale state
            last_solution = solution_name;
            last_solver_id = solver_id;
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
