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
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

using namespace rocRoller;

#define HIP_API_CALL(CALL)   \
    if((CALL) != hipSuccess) \
    {                        \
        abort();             \
    }

int main(int /*argc*/, char** /*argv*/)
{
    // Follows GPU_WholeKernel from KernelTest.cpp
    auto context = Context::ForDefaultHipDevice("hello_world");

    // Create command
    auto command = std::make_shared<Command>();

    // Define variable types
    VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
    VariableType floatVal{DataType::Float, PointerType::Value};

    // Allocate tags and arguments
    auto ptrTag  = command->allocateTag();
    auto ptr_arg = command->allocateArgument(floatPtr, ptrTag, ArgumentType::Value);
    auto valTag  = command->allocateTag();
    auto val_arg = command->allocateArgument(floatVal, valTag, ArgumentType::Value);

    // Create expressions for the arguments
    auto ptr_exp = std::make_shared<Expression::Expression>(ptr_arg);
    auto val_exp = std::make_shared<Expression::Expression>(val_arg);

    auto k = context->kernel();

    k->setKernelDimensions(1);

    const auto one = std::make_shared<Expression::Expression>(1u);
    k->setWorkgroupSize({256, 1, 1});
    k->setWorkitemCount({std::make_shared<Expression::Expression>(256 * 256), one, one});

    k->addArgument(
        {"ptr", {DataType::Float, PointerType::PointerGlobal}, DataDirection::WriteOnly, ptr_exp});
    k->addArgument({"val", {DataType::Float}, DataDirection::ReadOnly, val_exp});

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

    for(auto instr : kb())
    {
        context->schedule(instr);
        // TODO: push_back to check InstructionStatus
    }

    context->schedule(k->postamble());
    context->schedule(k->amdgpu_metadata());

    // Create CommandKernel
    CommandKernel commandKernel;
    commandKernel.setContext(context);
    commandKernel.generateKernel();

    float* d_ptr = nullptr;
    HIP_API_CALL(hipMalloc(&d_ptr, sizeof(float)));

    HIP_API_CALL(hipMemset(d_ptr, 0, sizeof(float)));

    // Create command arguments and launch kernel
    CommandArguments commandArgs = command->createArguments();
    commandArgs.setArgument(ptrTag, ArgumentType::Value, d_ptr);
    commandArgs.setArgument(valTag, ArgumentType::Value, 6.0f);

    commandKernel.launchKernel(commandArgs.runtimeArguments());

    HIP_API_CALL(hipDeviceSynchronize());

    float h_result = 0.0f;
    HIP_API_CALL(hipMemcpy(&h_result, d_ptr, sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "rocRoller kernel first execution result: " << h_result << " (expected: 6.0)"
              << std::endl;

    HIP_API_CALL(hipFree(d_ptr));

    return 0;
}
