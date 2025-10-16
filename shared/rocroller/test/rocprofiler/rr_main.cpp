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

#include "agent.hpp"

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

#include "hip/hip_runtime.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ranges>
#include <vector>

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

    std::vector<Instruction> instrs;

    const auto captureInstrAndSchedule = [&](auto gen) {
        for(auto instr : gen)
        {
            context->schedule(instr);
            instrs.push_back(std::move(instr));
        }
    };

    context->schedule(k->preamble());
    captureInstrAndSchedule(k->prolog());

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

    captureInstrAndSchedule(kb());
    instrs.push_back({"s_endpgm", {}, {}, {}, ""}); // postamble() too much extra stuff

    context->schedule(k->postamble());
    context->schedule(k->amdgpu_metadata());

    // Create CommandKernel
    CommandKernel commandKernel;
    commandKernel.setContext(context);
    commandKernel.generateKernel();

    float* d_ptr = nullptr; // TODO: use smart pointers
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
    HIP_API_CALL(hipFree(d_ptr));

    std::cout << "rocRoller kernel first execution result: " << h_result << " (expected: 6.0)"
              << std::endl;

    const auto& latencies = rocroller_profiler::get_instruction_latencies();

    std::stringstream ss;
    for(const auto& [pc, data] : latencies)
    {
        uint64_t avg_latency = data.hitcount ? (data.latency / data.hitcount) : 0;
        ss << "\"" << data.instruction << "\", " << data.latency << ", " << data.hitcount << ", "
           << avg_latency << std::endl;
    }

    instrs.erase(std::remove_if(instrs.begin(),
                                instrs.end(),
                                [](const auto& instr) {
                                    // isCommentOnly() doesn't work
                                    return instr.toString(LogLevel::Critical).empty();
                                }),
                 instrs.end());

    { // Tests
        AssertFatal(latencies.size() == 7, ss.str());
        rocroller_profiler::InstructionLatencyMap latenciesNoWaitcnt = latencies;
        for(auto it = latenciesNoWaitcnt.begin(); it != latenciesNoWaitcnt.end();)
        {
            if(it->second.instruction.find("s_waitcnt") != std::string::npos)
            {
                it = latenciesNoWaitcnt.erase(it);
            }
            else
            {
                ++it;
            }
        }

        for(size_t i = 0; i < instrs.size(); ++i)
        {
            std::cout << i << ": \"" << instrs[i].toString(LogLevel::Critical) << "\"" << std::endl;
            std::cout << instrs[i].getWaitCount() << std::endl;
        }

        for(const auto& [pc, data] : latenciesNoWaitcnt)
        {
            std::cout << data.instruction << std::endl;
        }

        AssertFatal(latenciesNoWaitcnt.size() == instrs.size(),
                    ShowValue(instrs.size()),
                    ShowValue(latenciesNoWaitcnt.size()),
                    ss.str());

        for(size_t i = 0; i < instrs.size(); ++i)
        {
            const auto& instr      = instrs[i];
            const auto& [pc, data] = *std::next(latenciesNoWaitcnt.begin(), i);

            auto other = instr.toString(LogLevel::Critical);

            // delete from last comma onwards
            auto pos = other.find_last_of(',');
            if(pos != std::string::npos)
            {
                other = other.substr(0, pos); // rocprofiler-sdk uses hex encoding on offsets
            }

            AssertFatal(data.instruction.find(other) != std::string::npos,
                        ShowValue(data.instruction),
                        ShowValue(other));
        }
    }

    return 0;
}
