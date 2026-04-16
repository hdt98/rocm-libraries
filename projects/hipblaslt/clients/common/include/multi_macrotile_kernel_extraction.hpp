/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 *******************************************************************************/

#ifndef MULTI_MACROTILE_KERNEL_EXTRACTION_HPP
#define MULTI_MACROTILE_KERNEL_EXTRACTION_HPP

#pragma once

#include "multi_macrotile_fused.hpp"
#include "multi_macrotile_fused_kernel.hpp"
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hip/hip_runtime.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <vector>

/**
 * @file multi_macrotile_kernel_extraction.hpp
 * @brief Extract Tensile kernel function pointers from hipBLASLt library
 *
 * This file implements the critical Phase 2 functionality:
 * extracting device function pointers for Tensile GEMM kernels
 * from hipBLASLt's code objects so they can be called from
 * the fused dispatch kernel.
 */

/**
 * @brief Kernel extraction context
 *
 * Manages loaded HIP modules and extracted kernel functions.
 * This is a singleton that caches loaded modules and function pointers
 * to avoid repeated expensive extractions.
 */
class KernelExtractionContext
{
private:
    // Map from code object path to loaded HIP module
    std::unordered_map<std::string, hipModule_t> loaded_modules_;

    // Map from kernel name to device function pointer
    std::unordered_map<std::string, hipFunction_t> kernel_functions_;

    // Singleton instance
    static std::shared_ptr<KernelExtractionContext> instance_;

    KernelExtractionContext() = default;

public:
    static std::shared_ptr<KernelExtractionContext> getInstance()
    {
        if (!instance_)
        {
            instance_ = std::shared_ptr<KernelExtractionContext>(new KernelExtractionContext());
        }
        return instance_;
    }

    ~KernelExtractionContext()
    {
        // Unload all modules
        for (auto& pair : loaded_modules_)
        {
            hipModuleUnload(pair.second);
        }
    }

    /**
     * @brief Load a code object file and cache the module handle
     *
     * @param co_path Path to .co or .hsaco file
     * @return HIP module handle, or nullptr on failure
     */
    hipModule_t loadCodeObject(const std::string& co_path)
    {
        // Check if already loaded
        auto it = loaded_modules_.find(co_path);
        if (it != loaded_modules_.end())
        {
            return it->second;
        }

        // Load the code object
        hipModule_t module;
        hipError_t err = hipModuleLoad(&module, co_path.c_str());

        if (err != hipSuccess)
        {
            std::cerr << "ERROR: hipModuleLoad failed for " << co_path << std::endl;
            std::cerr << "  Error code: " << err << std::endl;
            return nullptr;
        }

        std::cout << "Loaded code object: " << co_path << std::endl;

        // Cache the module
        loaded_modules_[co_path] = module;
        return module;
    }

    /**
     * @brief Extract a kernel function pointer from a loaded module
     *
     * @param module HIP module handle
     * @param kernel_name Name of the kernel to extract
     * @return Device function pointer, or nullptr on failure
     */
    hipFunction_t extractKernelFunction(hipModule_t module, const std::string& kernel_name)
    {
        // Check cache
        auto it = kernel_functions_.find(kernel_name);
        if (it != kernel_functions_.end())
        {
            return it->second;
        }

        // Extract the function
        hipFunction_t kernel_func;
        hipError_t err = hipModuleGetFunction(&kernel_func, module, kernel_name.c_str());

        if (err != hipSuccess)
        {
            std::cerr << "ERROR: hipModuleGetFunction failed for " << kernel_name << std::endl;
            std::cerr << "  Error code: " << err << std::endl;

            // Try demangled name (sometimes Tensile kernels are mangled)
            std::string demangled = tryDemangleName(kernel_name);
            if (!demangled.empty() && demangled != kernel_name)
            {
                err = hipModuleGetFunction(&kernel_func, module, demangled.c_str());
                if (err == hipSuccess)
                {
                    std::cout << "  Success with demangled name: " << demangled << std::endl;
                    kernel_functions_[kernel_name] = kernel_func;
                    return kernel_func;
                }
            }

            return nullptr;
        }

        std::cout << "Extracted kernel function: " << kernel_name << std::endl;

        // Cache the function
        kernel_functions_[kernel_name] = kernel_func;
        return kernel_func;
    }

    /**
     * @brief Try to demangle a kernel name
     *
     * Tensile kernels may be C++ mangled. This tries common patterns.
     */
    std::string tryDemangleName(const std::string& name)
    {
        // For now, just return the name as-is
        // In a full implementation, we'd use __cxa_demangle or similar
        return name;
    }

    /**
     * @brief Get cached kernel function
     */
    hipFunction_t getCachedFunction(const std::string& kernel_name)
    {
        auto it = kernel_functions_.find(kernel_name);
        return (it != kernel_functions_.end()) ? it->second : nullptr;
    }
};

// Initialize static instance
std::shared_ptr<KernelExtractionContext> KernelExtractionContext::instance_ = nullptr;

/**
 * @brief Get the Tensile library path for the current device
 *
 * @param handle hipBLASLt handle
 * @return Path to Tensile library directory
 */
inline std::string getTensileLibraryPath(hipblasLtHandle_t handle)
{
    // Get from environment variable if set
    const char* env_path = std::getenv("HIPBLASLT_TENSILE_LIBPATH");
    if (env_path != nullptr && env_path[0] != '\0')
    {
        return std::string(env_path);
    }

    // Default paths to try
    std::vector<std::string> search_paths = {
        "./Tensile/library",                                    // Build tree
        "../lib/hipblaslt/library",                            // Install tree (relative to bin)
        "/opt/rocm/lib/hipblaslt/library",                    // System install
        "../hipblaslt/lib/hipblaslt/library",                 // ROCm install
    };

    // Check which paths exist
    for (const auto& path : search_paths)
    {
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        {
            return path;
        }
    }

    // Fallback
    return "./Tensile/library";
}

/**
 * @brief Get the code object file path for the current GPU architecture
 *
 * @param library_path Tensile library base path
 * @param arch GPU architecture (e.g., "gfx950")
 * @return Path to .hsaco file
 */
inline std::string getCodeObjectPath(const std::string& library_path, const std::string& arch)
{
    // Try multiple naming patterns
    std::vector<std::string> patterns = {
        library_path + "/Kernels.so-000-" + arch + ".hsaco",
        library_path + "/Kernels.so-000-" + arch + "-xnack-.hsaco",
        library_path + "/Kernels.so-000-" + arch + "-xnack+.hsaco",
        library_path + "/" + arch + "/Kernels.so-000-" + arch + ".hsaco",
    };

    for (const auto& path : patterns)
    {
        struct stat st;
        if (stat(path.c_str(), &st) == 0)
        {
            return path;
        }
    }

    // Default to first pattern
    return patterns[0];
}

/**
 * @brief Get GPU architecture string for current device
 *
 * @param device_id HIP device ID
 * @return Architecture string (e.g., "gfx950")
 */
inline std::string getGpuArchitecture(int device_id)
{
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id);

    // The gcnArchName contains the architecture
    std::string arch_name = prop.gcnArchName;

    // Extract just the gfxXXX part
    size_t pos = arch_name.find("gfx");
    if (pos != std::string::npos)
    {
        size_t end = arch_name.find(':', pos);
        if (end != std::string::npos)
        {
            return arch_name.substr(pos, end - pos);
        }
        return arch_name.substr(pos);
    }

    return "gfx950"; // Fallback
}

/**
 * @brief Extract solution index from algorithm
 *
 * @param algo hipBLASLt algorithm
 * @return Solution index (first 4 bytes of algo.data)
 */
inline int getSolutionIndexFromAlgo(const hipblasLtMatmulAlgo_t& algo)
{
    // Solution index is stored in first 4 bytes of algo.data
    const int* index_ptr = reinterpret_cast<const int*>(algo.data);
    return *index_ptr;
}

/**
 * @brief Create a kernel dispatch table from sub-problems
 *
 * This is the main function that extracts all necessary kernel functions
 * for a multi-MacroTile dispatch.
 *
 * @param handle hipBLASLt handle
 * @param subProblems Vector of sub-problem descriptors
 * @param algorithms Vector of algorithms (one per sub-problem)
 * @param device_id HIP device ID
 * @return Populated kernel dispatch table
 */
inline KernelDispatchTable createKernelDispatchTable(
    hipblasLtHandle_t handle,
    const std::vector<GemmSubProblem>& subProblems,
    const std::vector<hipblasLtMatmulAlgo_t>& algorithms,
    int device_id)
{
    KernelDispatchTable table = {};

    // Get extraction context
    auto ctx = KernelExtractionContext::getInstance();

    // Get library path and architecture
    std::string library_path = getTensileLibraryPath(handle);
    std::string arch = getGpuArchitecture(device_id);
    std::string co_path = getCodeObjectPath(library_path, arch);

    std::cout << "Multi-MacroTile Fused Kernel Extraction:" << std::endl;
    std::cout << "  Library path: " << library_path << std::endl;
    std::cout << "  GPU arch: " << arch << std::endl;
    std::cout << "  Code object: " << co_path << std::endl;

    // Load the code object
    hipModule_t module = ctx->loadCodeObject(co_path);
    if (!module)
    {
        std::cerr << "ERROR: Failed to load code object" << std::endl;
        return table;
    }

    // Extract kernel for each sub-problem
    for (size_t i = 0; i < subProblems.size() && i < algorithms.size(); i++)
    {
        const auto& algo = algorithms[i];

        // Get solution index
        int solution_idx = getSolutionIndexFromAlgo(algo);

        // Get kernel name
        std::string kernel_name;
        try
        {
            kernel_name = hipblaslt_ext::getKernelNameFromAlgo(handle,
                                                               const_cast<hipblasLtMatmulAlgo_t&>(algo));
        }
        catch (...)
        {
            std::cerr << "ERROR: Failed to get kernel name for sub-problem " << i << std::endl;
            continue;
        }

        if (kernel_name.empty())
        {
            std::cerr << "ERROR: Empty kernel name for sub-problem " << i << std::endl;
            continue;
        }

        std::cout << "  Sub-problem " << i << ":" << std::endl;
        std::cout << "    Solution index: " << solution_idx << std::endl;
        std::cout << "    Kernel name: " << kernel_name << std::endl;

        // Check if we already have this kernel
        hipFunction_t kernel_func = ctx->getCachedFunction(kernel_name);

        if (!kernel_func)
        {
            // Extract the function
            kernel_func = ctx->extractKernelFunction(module, kernel_name);
        }

        if (!kernel_func)
        {
            std::cerr << "ERROR: Failed to extract kernel function" << std::endl;
            continue;
        }

        // Add to dispatch table (check for duplicates)
        bool found = false;
        for (int j = 0; j < table.num_kernels; j++)
        {
            if (table.solution_indices[j] == solution_idx)
            {
                found = true;
                break;
            }
        }

        if (!found && table.num_kernels < MAX_KERNEL_VARIANTS)
        {
            table.solution_indices[table.num_kernels] = solution_idx;
            table.kernel_funcs[table.num_kernels] = reinterpret_cast<GemmKernelFunc>(kernel_func);
            table.num_kernels++;

            std::cout << "    Added to dispatch table (entry " << table.num_kernels - 1 << ")" << std::endl;
        }
    }

    std::cout << "Kernel extraction complete: " << table.num_kernels << " unique kernels" << std::endl;

    return table;
}

#endif // MULTI_MACROTILE_KERNEL_EXTRACTION_HPP
