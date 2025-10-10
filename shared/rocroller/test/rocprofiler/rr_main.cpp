// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include <rocprofiler-sdk-roctx/roctx.h>

#include "hip/hip_runtime.h"

// rocRoller includes
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Utilities/Generator.hpp>

using namespace rocRoller;

#define HIP_API_CALL(CALL)   \
    if((CALL) != hipSuccess) \
    {                        \
        abort();             \
    }

std::shared_ptr<ExecutableKernel> createRocRollerKernel()
{
    auto context = Context::ForDefaultHipDevice("hello_world");

    auto k = context->kernel();

    k->setKernelDimensions(1);

    k->addArgument(
        {"ptr", {DataType::Float, PointerType::PointerGlobal}, DataDirection::WriteOnly});
    k->addArgument({"val", {DataType::Float}});

    context->schedule(k->preamble());
    context->schedule(k->prolog());

    auto kb = [&]() -> Generator<Instruction> {
        Register::ValuePtr s_ptr, s_value;
        co_yield context->argLoader()->getValue("ptr", s_ptr);
        co_yield context->argLoader()->getValue("val", s_value);

        auto v_ptr = Register::Value::Placeholder(
            context, Register::Type::Vector, {DataType::Float, PointerType::PointerGlobal}, 1);
        auto v_value
            = Register::Value::Placeholder(context, Register::Type::Vector, DataType::Float, 1);

        co_yield v_ptr->allocate();
        co_yield context->copier()->copy(v_ptr, s_ptr, "Move pointer");

        co_yield v_value->allocate();
        co_yield context->copier()->copy(v_value, s_value, "Move value");

        co_yield context->mem()->storeGlobal(v_ptr, v_value, 0, 4);
    };

    context->schedule(kb());

    context->schedule(k->postamble());
    context->schedule(k->amdgpu_metadata());

    return context->instructions()->getExecutableKernel();
}

int main(int /*argc*/, char** /*argv*/)
{
    // Follows GPU_WholeKernel from KernelTest.cpp
    auto kernel = createRocRollerKernel();

    float* d_ptr = nullptr;
    HIP_API_CALL(hipMalloc(&d_ptr, sizeof(float)));

    HIP_API_CALL(hipMemset(d_ptr, 0, sizeof(float)));

    roctxProfilerResume(0);

    KernelArguments kargs;
    kargs.append("ptr", static_cast<void*>(d_ptr));
    kargs.append("val", 6.0f);

    KernelInvocation invocation;
    kernel->executeKernel(kargs, invocation);

    HIP_API_CALL(hipDeviceSynchronize());

    float h_result = 0.0f;
    HIP_API_CALL(hipMemcpy(&h_result, d_ptr, sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "rocRoller kernel first execution result: " << h_result << " (expected: 6.0)"
              << std::endl;

    KernelArguments kargs2;
    kargs2.append("ptr", static_cast<void*>(d_ptr));
    kargs2.append("val", 7.5f);

    kernel->executeKernel(kargs2, invocation);

    HIP_API_CALL(hipDeviceSynchronize());

    roctxProfilerPause(0);

    HIP_API_CALL(hipMemcpy(&h_result, d_ptr, sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "rocRoller kernel second execution result: " << h_result << " (expected: 7.5)"
              << std::endl;

    HIP_API_CALL(hipFree(d_ptr));

    return 0;
}
