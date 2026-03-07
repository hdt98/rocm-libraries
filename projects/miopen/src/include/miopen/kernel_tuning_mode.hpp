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
#include <miopen/env.hpp>
#include <miopen/utility/modified_z.hpp>

// Declare the performance logs environment variable
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_PERFORMANCE_LOGS)

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
inline size_t IncrementKernelExecutionCounter() { return ++GetKernelExecutionCounter(); }

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
           kernel_name.find("Reorder") != std::string::npos ||
           // Im2Col and Col2Im transformations (2D and 3D)
           kernel_name.find("Im2d2Col") != std::string::npos ||
           kernel_name.find("Im3d2Col") != std::string::npos ||
           kernel_name.find("Col2Im2d") != std::string::npos ||
           kernel_name.find("Col2Im3d") != std::string::npos ||
           kernel_name.find("Im2Col") != std::string::npos ||
           kernel_name.find("Col2Im") != std::string::npos ||
           // NCHW to CNHW and vice versa transformations
           kernel_name.find("NCHW2CNHW") != std::string::npos ||
           kernel_name.find("CNHW2NCHW") != std::string::npos ||
           kernel_name.find("NCHW2Vec") != std::string::npos ||
           kernel_name.find("NCHWVec") != std::string::npos ||
           // Packed matrix transpose
           kernel_name.find("MN2NM") != std::string::npos;
}

/// Thread-local storage for solution-level timing accumulation
struct SolutionTimingAccumulator
{
    size_t exec_id   = 0;
    float total_time = 0.0f;
    int kernel_count = 0;
    std::string main_kernel_name;

    void Reset(size_t new_exec_id)
    {
        exec_id      = new_exec_id;
        total_time   = 0.0f;
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
    std::vector<std::string> transformation_kernel_names; // Collected transformation kernel names
    bool is_transformation;
};

/// Structure to hold performance config data with its kernels
struct PerformanceConfigData
{
    std::string config_name;        // Kernel name (if available) or solution name
    std::string config_descriptor;  // Performance config parameters string
    std::vector<KernelExecutionData> kernels;
    std::vector<float> invoker_times_ms;  // Direct timing from invoker (if available)
    
    void Clear()
    {
        config_name.clear();
        config_descriptor.clear();
        kernels.clear();
        invoker_times_ms.clear();
    }
};

/// Structure to hold grouped kernel data by kernel_name (multiple executions)
struct GroupedKernelData
{
    std::string kernel_name;
    std::vector<float> time_executions_ms;
    bool is_transformation;
};

/// Structure to accumulate kernels by exec_id
struct ExecIdAccumulator
{
    size_t exec_id = 0;
    std::string main_kernel_name;
    float total_time_ms      = 0.0f;
    int transformation_count = 0;
    std::vector<std::string> transformation_kernel_names; // Track transformation kernel names
    bool has_data = false;

    void Reset()
    {
        exec_id = 0;
        main_kernel_name.clear();
        total_time_ms        = 0.0f;
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
    std::vector<PerformanceConfigData> performance_configs; // Array of performance configs
    
    // Current config being accumulated (temporary)
    PerformanceConfigData current_config;
    
    void Clear()
    {
        solution_name.clear();
        solver_id = 0;
        phase.clear();
        performance_configs.clear();
        current_config.Clear();
    }
    
    void FinalizeCurrentConfig()
    {
        if(!current_config.config_name.empty() && !current_config.kernels.empty())
        {
            performance_configs.push_back(current_config);
            current_config.Clear();
        }
    }
};

/// Escape string for JSON
inline std::string JsonEscape(const std::string& str)
{
    std::ostringstream oss;
    for(char c : str)
    {
        switch(c)
        {
        case '"': oss << "\\\""; break;
        case '\\': oss << "\\\\"; break;
        case '\b': oss << "\\b"; break;
        case '\f': oss << "\\f"; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default:
            if(c < 32 || c > 126)
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


/// Finalize the current exec_id accumulator and add it to the current config's kernels
inline void FinalizeCurrentExecId()
{
    auto& exec_accum = GetCurrentExecIdAccumulator();

    if(!exec_accum.has_data)
        return;

    auto& data = GetJsonAccumulator();
    data.current_config.kernels.push_back({
        exec_accum.exec_id,
        exec_accum.main_kernel_name,
        exec_accum.total_time_ms,
        exec_accum.transformation_count,
        exec_accum.transformation_kernel_names, // Pass transformation kernel names
        false // is_transformation - this is the accumulated main kernel
    });

    exec_accum.Reset();
}

/// Add a new performance config with its name, descriptor and optional invoker timings
inline void AddPerformanceConfig(const std::string& config_name,
                                  const std::string& config_descriptor = "",
                                  const std::vector<float>& invoker_times_ms = {})
{
    auto& data = GetJsonAccumulator();
    
    // Finalize the previous config if it exists
    data.FinalizeCurrentConfig();
    
    // Start a new config
    data.current_config.config_name = config_name;
    data.current_config.config_descriptor = config_descriptor;
    data.current_config.invoker_times_ms = invoker_times_ms;
}

/// Output accumulated JSON data
inline void FlushJsonAccumulator()
{
    // Finalize any pending exec_id and config
    FinalizeCurrentExecId();
    
    auto& data = GetJsonAccumulator();
    data.FinalizeCurrentConfig();

    if(data.performance_configs.empty())
        return;

    // Output JSON with solution-level metrics
    std::cerr << "{\"solution\":\"" << JsonEscape(data.solution_name) << "\","
              << "\"solver_id\":" << data.solver_id << ","
              << "\"phase\":\"" << data.phase << "\","
              << "\"performance_configs\":[";

    // Output each performance config
    bool first_config = true;
    for(const auto& config : data.performance_configs)
    {
        if(!first_config)
            std::cerr << ",";
        first_config = false;
        
        // Determine config_name: prefer first kernel name, fallback to config.config_name
        std::string display_name = config.config_name;
        if(!config.kernels.empty())
        {
            // Use first non-transformation kernel name if available
            for(const auto& k : config.kernels)
            {
                if(!k.is_transformation && !k.kernel_name.empty())
                {
                    display_name = k.kernel_name;
                    break;
                }
            }
        }
        
        // If all kernels are transformations or there are no kernels, use the solution name
        if(display_name == config.config_name)
        {
            display_name = data.solution_name;
        }
        
        std::cerr << "{\"config_name\":\"" << JsonEscape(display_name) << "\",";
        
        // Output config_descriptor if present
        if(!config.config_descriptor.empty())
        {
            std::cerr << "\"config_descriptor\":\"" << JsonEscape(config.config_descriptor) << "\",";
        }
        
        // Group kernels by kernel_name for this config
        std::map<std::string, GroupedKernelData> grouped_kernels;
        
        // Variables for config-level aggregation
        std::vector<float> all_exec_times;
        size_t min_exec_number = std::numeric_limits<size_t>::max();
        int total_transformations = 0;
        
        // Check if we have invoker timings - if so, use those instead of kernel accumulation
        if(!config.invoker_times_ms.empty())
        {
            all_exec_times = config.invoker_times_ms;
            min_exec_number = 1;  // Invoker timings start from execution 1
            
            // Still count transformations from kernels if available
            for(const auto& k : config.kernels)
            {
                if(k.is_transformation || k.transformation_count > 0)
                {
                    total_transformations += k.transformation_count;
                }
            }
        }
        else
        {
            // Fallback to original kernel accumulation logic
            for(const auto& k : config.kernels)
            {
                auto& grouped = grouped_kernels[k.kernel_name];

                // Initialize on first occurrence
                if(grouped.time_executions_ms.empty())
                {
                    grouped.kernel_name = k.kernel_name;
                    grouped.is_transformation = k.is_transformation;
                }

                // Append timing data
                grouped.time_executions_ms.push_back(k.time_ms);
                
                // Accumulate for config-level stats
                all_exec_times.push_back(k.time_ms);
                min_exec_number = std::min(min_exec_number, k.exec_id);
                if(k.is_transformation || k.transformation_count > 0)
                {
                    total_transformations += k.transformation_count;
                }
            }
        }
        
        // Calculate config-level statistics
        float config_time_ms = 0.0f;
        float config_time_std = 0.0f;
        float config_time_min = 0.0f;
        float config_time_max = 0.0f;
        
        if(!all_exec_times.empty())
        {
            if(all_exec_times.size() <= 1)
            {
                config_time_ms = all_exec_times[0];
                config_time_std = 0.0f;
                config_time_min = config_time_ms;
                config_time_max = config_time_ms;
            }
            else
            {
                // Use removeHighOutliersAndGetMean for more accurate timing
                // (same as GenericSearch does)
                std::vector<float> samples_copy = all_exec_times;
                config_time_ms = removeHighOutliersAndGetMean(samples_copy, 2.0f);
                
                // samples_copy is now sorted by removeHighOutliersAndGetMean
                config_time_min = samples_copy.front();
                config_time_max = samples_copy.back();
                
                // Calculate standard deviation using the mean from outlier removal
                float sq_sum = 0.0f;
                int count = 0;
                for(size_t i = 0; i < all_exec_times.size(); ++i)
                {
                    float diff = all_exec_times[i] - config_time_ms;
                    sq_sum += diff * diff;
                    count++;
                }
                config_time_std = (count > 1) ? std::sqrt(sq_sum / count) : 0.0f;
            }
        }
        
        // Output config-level aggregated stats
        std::cerr << "\"exec_number\":" << min_exec_number << ",";
        
        std::cerr << "\"time_executions_ms\":[";
        for(size_t i = 0; i < all_exec_times.size(); ++i)
        {
            if(i > 0)
                std::cerr << ",";
            std::cerr << all_exec_times[i];
        }
        std::cerr << "],";
        
        std::cerr << "\"time_ms\":" << config_time_ms << ","
                  << "\"time_std_ms\":" << config_time_std << ","
                  << "\"time_min_ms\":" << config_time_min << ","
                  << "\"time_max_ms\":" << config_time_max << ","
                  << "\"number_of_transformations\":" << total_transformations << ",";
        
        // Output simplified kernels array
        std::cerr << "\"kernels\":[";
        bool first_kernel = true;
        for(const auto& entry : grouped_kernels)
        {
            const auto& g = entry.second;
            if(!first_kernel)
                std::cerr << ",";
            first_kernel = false;

            std::cerr << "{\"kernel_name\":\"" << JsonEscape(g.kernel_name) << "\","
                      << "\"time_executions_ms\":[";

            for(size_t i = 0; i < g.time_executions_ms.size(); ++i)
            {
                if(i > 0)
                    std::cerr << ",";
                std::cerr << g.time_executions_ms[i];
            }

            std::cerr << "],";
            std::cerr << "\"is_transformation\":" << (g.is_transformation ? "true" : "false") << "}";
        }
        
        std::cerr << "]}"; // Close kernels array and config object
    }
    
    std::cerr << "]}" << std::endl; // Close performance_configs array and root object
    data.Clear();
}

/// Manually flush any pending JSON data (call at program end or context switch)
inline void FinalizeJsonLogging() { FlushJsonAccumulator(); }

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
                                       bool is_transform)
{
    const auto log_level = env::value(MIOPEN_PERFORMANCE_LOGS);
    // Check if we should list kernels individually
    if(IsIndividualKernelListingMode(log_level))
    {
        // Individual kernel listing mode - add directly to current config without exec_id accumulation
        auto& data = GetJsonAccumulator();
        data.current_config.kernels.push_back({exec_id,
                                               kernel_name,
                                               time_ms,
                                               0,  // transformation_count is 0 for individual kernels
                                               {}, // transformation_kernel_names - empty for individual kernels
                                               is_transform});
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
            exec_accum.exec_id  = exec_id;
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
inline bool IsPerformanceLoggingEnabled() 
{ 
    const auto log_level = env::value(MIOPEN_PERFORMANCE_LOGS);
    return log_level > 0; 
}

/// Check if this kernel should be logged
/// - When log_level <= 2: Only log execution phase kernels (not tuning)
/// - When log_level > 2: Log both tuning and execution phase kernels
inline bool IsLoggingKernel(bool is_tuning)
{
    const auto log_level = env::value(MIOPEN_PERFORMANCE_LOGS);
    if(log_level > 2)
    {
        // Log all kernels (both tuning and execution) when log_level > 2
        return true;
    }
    else if(log_level > 0)
    {
        // Only log execution phase kernels when 0 < log_level <= 2
        return !is_tuning;
    }
    return false;
}

/// Log solution name if appropriate for the current log level
/// Only prints if the solution name has changed since the last call
inline void LogSolutionName(const std::string& solution_name, uint64_t solver_id)
{
    const bool is_tuning_mode  = GetKernelTuningMode();
    const bool logging_enabled = IsLoggingKernel(is_tuning_mode);
    if(logging_enabled && !solution_name.empty())
    {
        auto& last_solution = GetLastPrintedSolutionName();
        auto& last_solver_id = GetLastPrintedSolverId();
        
        // Check if solution name or solver_id has changed
        if(solution_name != last_solution || solver_id != last_solver_id)
        {
            // Flush previous solution's JSON data
            FlushJsonAccumulator();

            // Set up new solution in accumulator
            auto& data         = GetJsonAccumulator();
            data.solution_name = solution_name;
            data.solver_id     = solver_id;
            data.phase         = is_tuning_mode ? "tuning" : "execution";
            last_solution      = solution_name;
            last_solver_id     = solver_id;
        }
    }
}

/// RAII helper to set tuning mode for a scope
class ScopedKernelTuningMode
{
public:
    ScopedKernelTuningMode() : prev_value(GetKernelTuningMode()) { GetKernelTuningMode() = true; }

    ~ScopedKernelTuningMode() { GetKernelTuningMode() = prev_value; }

    ScopedKernelTuningMode(const ScopedKernelTuningMode&)            = delete;
    ScopedKernelTuningMode& operator=(const ScopedKernelTuningMode&) = delete;

private:
    bool prev_value;
};

} // namespace miopen

#endif // GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP
