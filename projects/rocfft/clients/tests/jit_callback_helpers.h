// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef JIT_CALLBACK_HELPERS_H
#define JIT_CALLBACK_HELPERS_H

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// check if hipRTC is available in the environment
static bool rtc_available()
{
    static bool checked   = false;
    static bool available = false;

    if(!checked)
    {
        int          major, minor;
        hiprtcResult result = hiprtcVersion(&major, &minor);
        available           = (result == HIPRTC_SUCCESS);
        checked             = true;
    }
    return available;
}

// get the current device architecture
static std::string get_device_arch()
{
    static std::string arch;
    if(arch.empty())
    {
        hipDeviceProp_t prop;
        int             device;
        if(hipGetDevice(&device) == hipSuccess
           && hipGetDeviceProperties(&prop, device) == hipSuccess)
        {
            arch = std::string("--offload-arch=") + prop.gcnArchName;
        }
        else
        {
            throw std::runtime_error("Cannot determine GPU architecture - no device available");
        }
    }
    return arch;
}

// RAII wrapper for hiprtcProgram
struct HiprtcProgramGuard
{
    hiprtcProgram prog = nullptr;

    ~HiprtcProgramGuard()
    {
        if(prog != nullptr)
        {
            hiprtcDestroyProgram(&prog);
        }
    }
};

// compile callback source code to native GPU code
static std::vector<char> compile_callback_to_spirv(const std::string& source,
                                                   const std::string& name = "callback.hip")
{
    HiprtcProgramGuard guard;

    hiprtcResult result
        = hiprtcCreateProgram(&guard.prog, source.c_str(), name.c_str(), 0, nullptr, nullptr);
    if(result != HIPRTC_SUCCESS)
    {
        throw std::runtime_error("hiprtcCreateProgram failed");
    }

    // get device architecture dynamically
    std::string              arch_option = get_device_arch();
    std::vector<const char*> options;
    options.push_back(arch_option.c_str());

    hiprtcResult compile_result = hiprtcCompileProgram(guard.prog, options.size(), options.data());

    if(compile_result != HIPRTC_SUCCESS)
    {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(guard.prog, &log_size);
        if(log_size > 0)
        {
            std::vector<char> log(log_size);
            hiprtcGetProgramLog(guard.prog, log.data());
            throw std::runtime_error(std::string("Compilation failed:\n") + log.data());
        }
        throw std::runtime_error("Compilation failed with no log");
    }

    // try to get bitcode (linkable format) instead of code
    size_t       code_size      = 0;
    hiprtcResult bitcode_result = hiprtcGetBitcodeSize(guard.prog, &code_size);
    if(bitcode_result != HIPRTC_SUCCESS)
    {
        // fallback to regular code if bitcode not available
        if(hiprtcGetCodeSize(guard.prog, &code_size) != HIPRTC_SUCCESS)
        {
            throw std::runtime_error("Both hiprtcGetBitcodeSize and hiprtcGetCodeSize failed");
        }

        std::vector<char> code(code_size);
        if(hiprtcGetCode(guard.prog, code.data()) != HIPRTC_SUCCESS)
        {
            throw std::runtime_error("hiprtcGetCode failed");
        }
        return code;
    }

    std::vector<char> code(code_size);
    if(hiprtcGetBitcode(guard.prog, code.data()) != HIPRTC_SUCCESS)
    {
        throw std::runtime_error("hiprtcGetBitcode failed");
    }

    return code;
}

// callback source code templates
namespace callback_sources
{
    // single-precision complex load callback (identity)
    const char* load_identity_c = R"(
        #include <hip/hip_runtime.h>
        
        extern "C" __device__ 
        float2 load_callback(float2* buffer, size_t offset,
                            void* callback_data, void* shared_mem)
        {
            return buffer[offset];
        }
    )";

    // double-precision complex load callback (identity)
    const char* load_identity_z = R"(
        #include <hip/hip_runtime.h>
        
        extern "C" __device__
        double2 load_callback(double2* buffer, size_t offset,
                             void* callback_data, void* shared_mem)
        {
            return buffer[offset];
        }
    )";

    // single-precision real load callback (identity)
    const char* load_identity_r = R"(
        #include <hip/hip_runtime.h>
        
        extern "C" __device__
        float load_callback(float* buffer, size_t offset,
                           void* callback_data, void* shared_mem)
        {
            return buffer[offset];
        }
    )";

    // double-precision real load callback (identity)
    const char* load_identity_d = R"(
        #include <hip/hip_runtime.h>
        
        extern "C" __device__
        double load_callback(double* buffer, size_t offset,
                            void* callback_data, void* shared_mem)
        {
            return buffer[offset];
        }
    )";

    // single-precision complex load callback (scaling)
    const char* load_scale_c = R"(
        #include <hip/hip_runtime.h>
        
        struct callback_data_t {
            float scale;
        };
        
        extern "C" __device__
        float2 load_callback(float2* buffer, size_t offset,
                            void* callback_data, void* shared_mem)
        {
            auto data = static_cast<callback_data_t*>(callback_data);
            float2 val = buffer[offset];
            val.x *= data->scale;
            val.y *= data->scale;
            return val;
        }
    )";

    // double-precision complex load callback (scaling)
    const char* load_scale_z = R"(
        #include <hip/hip_runtime.h>
        
        struct callback_data_t {
            double scale;
        };
        
        extern "C" __device__
        double2 load_callback(double2* buffer, size_t offset,
                             void* callback_data, void* shared_mem)
        {
            auto data = static_cast<callback_data_t*>(callback_data);
            double2 val = buffer[offset];
            val.x *= data->scale;
            val.y *= data->scale;
            return val;
        }
    )";

    // single-precision complex store callback (identity)
    const char* store_identity_c = R"(
        #include <hip/hip_runtime.h>
        
        extern "C" __device__
        void store_callback(float2* buffer, size_t offset,
                           float2 element, void* callback_data,
                           void* shared_mem)
        {
            buffer[offset] = element;
        }
    )";

    // double-precision complex store callback (identity)
    const char* store_identity_z = R"(
        #include <hip/hip_runtime.h>
        
        extern "C" __device__
        void store_callback(double2* buffer, size_t offset,
                           double2 element, void* callback_data,
                           void* shared_mem)
        {
            buffer[offset] = element;
        }
    )";
}

#endif // JIT_CALLBACK_HELPERS_H