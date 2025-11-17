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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// check if the primary callback SPIR-V file exists
static bool rtc_available()
{
    static bool checked   = false;
    static bool available = false;

    if(!checked)
    {
        std::ifstream f("load_callback.spv", std::ios::binary);
        available = f.good();
        checked   = true;

        if(!available)
        {
            std::cerr
                << "Warning: SPIR-V callback file 'load_callback.spv' not found.\n"
                << "         Make sure CMake compiled and copied it to the executable directory.\n";
        }
    }
    return available;
}

// compile a callback from a .hip source file using hipRTC
static std::vector<char> compile_callback(const std::string& filename)
{
    // read source file
    std::ifstream f(filename);
    if(!f.is_open())
    {
        throw std::runtime_error("compile_callback: failed to open source file '" + filename
                                 + "'. "
                                   "Make sure CMake copied it next to rocfft-test.");
    }

    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    if(src.empty())
        throw std::runtime_error("compile_callback: source file '" + filename + "' is empty");

    // create hipRTC program
    hiprtcProgram prog;
    hiprtcResult  hr
        = hiprtcCreateProgram(&prog, src.c_str(), filename.c_str(), 0, nullptr, nullptr);
    if(hr != HIPRTC_SUCCESS)
    {
        throw std::runtime_error(std::string("hiprtcCreateProgram failed: ")
                                 + hiprtcGetErrorString(hr));
    }

    // build compile options
    int        device = 0;
    hipError_t herr   = hipGetDevice(&device);
    if(herr != hipSuccess)
    {
        hiprtcDestroyProgram(&prog);
        throw std::runtime_error(std::string("hipGetDevice failed: ") + hipGetErrorString(herr));
    }

    hipDeviceProp_t props{};
    herr = hipGetDeviceProperties(&props, device);
    if(herr != hipSuccess)
    {
        hiprtcDestroyProgram(&prog);
        throw std::runtime_error(std::string("hipGetDeviceProperties failed: ")
                                 + hipGetErrorString(herr));
    }

    std::vector<std::string> opt_strings;
    std::vector<const char*> opts;

    opt_strings.emplace_back(std::string("--offload-arch=") + props.gcnArchName);
    // make device code linkable if rocFFT links it with other objects:
    opt_strings.emplace_back("-fgpu-rdc");

    for(auto& s : opt_strings)
        opts.push_back(s.c_str());

    // compile
    hr = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if(hr != HIPRTC_SUCCESS)
    {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(prog, &log_size);

        std::string log;
        if(log_size > 0)
        {
            log.resize(log_size);
            hiprtcGetProgramLog(prog, &log[0]);
        }

        hiprtcDestroyProgram(&prog);

        throw std::runtime_error(std::string("hiprtcCompileProgram failed: ")
                                 + hiprtcGetErrorString(hr) + "\nhipRTC log:\n" + log);
    }

    // extract generated code
    size_t code_size = 0;
    hr               = hiprtcGetCodeSize(prog, &code_size);
    if(hr != HIPRTC_SUCCESS || code_size == 0)
    {
        hiprtcDestroyProgram(&prog);
        throw std::runtime_error("hiprtcGetCodeSize failed or returned zero size");
    }

    std::vector<char> code(code_size);
    hr = hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    if(hr != HIPRTC_SUCCESS)
    {
        throw std::runtime_error(std::string("hiprtcGetCode failed: ") + hiprtcGetErrorString(hr));
    }

    return code;
}

// load a SPIR-V binary from disk into a byte vector,
// this is what we pass to rocfft_plan_description_set_load_callback()
// or rocfft_plan_description_set_store_callback()
static std::vector<char> load_spirv(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if(!file.is_open())
    {
        throw std::runtime_error(
            "Failed to open SPIR-V file: " + filename
            + "\n"
              "Make sure the file was compiled by CMake and copied to the executable directory.");
    }

    const std::streamsize size = file.tellg();
    if(size <= 0)
    {
        throw std::runtime_error("SPIR-V file is empty or invalid: " + filename);
    }

    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if(!file.read(data.data(), size))
    {
        throw std::runtime_error("Failed to read SPIR-V file: " + filename);
    }

    return data;
}

// verify that a loaded binary has a valid SPIR-V magic number.
// SPIR-V magic: 0x07230203 (little-endian)
static bool is_valid_spirv(const std::vector<char>& code)
{
    if(code.size() < 4)
        return false;

    const uint32_t magic = *reinterpret_cast<const uint32_t*>(code.data());
    return magic == 0x07230203;
}

// get a human-readable description of the binary format.
static std::string spirv_format_info(const std::vector<char>& code)
{
    if(code.size() < 4)
        return "Invalid (file too small, < 4 bytes)";

    const uint32_t magic = *reinterpret_cast<const uint32_t*>(code.data());

    char buf[128];
    snprintf(buf, sizeof(buf), "0x%08x", magic);
    std::string result = buf;

    // check common formats
    if(magic == 0x07230203)
    {
        result += " (SPIR-V correct format)";
    }
    else if(magic == 0x464c457f)
    {
        result += " (Use SPIR-V not ELF code object)";
    }
    else if(magic == 0x4243c0de || (magic & 0xFFFF) == 0x4342)
    {
        result += " ( Use SPIR-V, not LLVM bitcode)";
    }
    else
    {
        result += " (Unknown format!. Expected SPIR-V with magic 0x07230203)";
    }

    return result;
}

// callback source code templates
namespace callback_sources
{
    static const char* load_identity_c = R"(
#include <hip/hip_runtime.h>

// Identity load callback for single-precision complex (float2)
extern "C" __device__
float2 load_callback(float2* buffer, size_t offset,
                     void* callback_data, void* shared_memory)
{
    return buffer[offset];
}
)";

    static const char* load_identity_z = R"(
#include <hip/hip_runtime.h>

// Identity load callback for double-precision complex (double2)
extern "C" __device__
double2 load_callback(double2* buffer, size_t offset,
                      void* callback_data, void* shared_memory)
{
    return buffer[offset];
}
)";

    static const char* load_identity_r = R"(
#include <hip/hip_runtime.h>

// Identity load callback for single-precision real (float)
extern "C" __device__
float load_callback(float* buffer, size_t offset,
                    void* callback_data, void* shared_memory)
{
    return buffer[offset];
}
)";

    static const char* load_identity_d = R"(
#include <hip/hip_runtime.h>

// Identity load callback for double-precision real (double)
extern "C" __device__
double load_callback(double* buffer, size_t offset,
                     void* callback_data, void* shared_memory)
{
    return buffer[offset];
}
)";

    static const char* load_scale_c = R"(
#include <hip/hip_runtime.h>

// Scale callback that multiplies input by 2
extern "C" __device__
float2 load_callback(float2* buffer, size_t offset,
                     void* callback_data, void* shared_memory)
{
    float2 val = buffer[offset];
    val.x *= 2.0f;
    val.y *= 2.0f;
    return val;
}
)";

}

#endif // JIT_CALLBACK_HELPERS_H