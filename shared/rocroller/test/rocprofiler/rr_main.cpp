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

#include "agent.hpp"

#include <common/SourceMatcher.hpp>
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
#include <rocRoller/Utilities/HipUtils.hpp>

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hip/hip_runtime.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ranges>
#include <vector>

using namespace rocRoller;

namespace RocprofilerTest
{

    TEST_CASE("RocProfiler kernel execution and latency tracking", "[rocprofiler]")
    {
        // Follows GPU_WholeKernel from KernelTest.cpp
        auto context = Context::ForDefaultHipDevice("hello_world");

        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};

        auto ptrTag  = command->allocateTag();
        auto ptr_arg = command->allocateArgument(floatPtr, ptrTag, ArgumentType::Value);
        auto valTag  = command->allocateTag();
        auto val_arg = command->allocateArgument(floatVal, valTag, ArgumentType::Value);

        auto ptr_exp = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp = std::make_shared<Expression::Expression>(val_arg);

        auto k = context->kernel();

        k->setKernelDimensions(1);

        const auto one = std::make_shared<Expression::Expression>(1u);
        k->setWorkgroupSize({256, 1, 1});
        k->setWorkitemCount({std::make_shared<Expression::Expression>(256 * 256), one, one});

        k->addArgument({"ptr",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
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

        CommandKernel commandKernel;
        commandKernel.setContext(context);
        commandKernel.generateKernel();

        float* d_ptr = nullptr; // TODO: use smart pointers
        HIP_CHECK(hipMalloc(&d_ptr, sizeof(float)));
        HIP_CHECK(hipMemset(d_ptr, 0, sizeof(float)));

        CommandArguments commandArgs = command->createArguments();
        commandArgs.setArgument(ptrTag, ArgumentType::Value, d_ptr);
        commandArgs.setArgument(valTag, ArgumentType::Value, 6.0f);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        HIP_CHECK(hipDeviceSynchronize());

        float h_result = 0.0f;
        HIP_CHECK(hipMemcpy(&h_result, d_ptr, sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_ptr));

        CHECK(h_result == 6.0f);

        const auto latencies = rocroller_profiler::getInstructionData();

        std::stringstream ss;
        ss << "Instruction, Total Latency, Hit Count, Average Latency" << std::endl;
        for(const auto& data : latencies)
        {
            uint64_t avg_latency = data.hitcount ? (data.latency / data.hitcount) : 0;
            ss << "\"" << data.instruction << "\", " << data.latency << ", " << data.hitcount
               << ", " << avg_latency << std::endl;
        }

        instrs.erase(std::remove_if(instrs.begin(),
                                    instrs.end(),
                                    [](const auto& instr) {
                                        // isCommentOnly() doesn't work
                                        return instr.toString(LogLevel::Critical).empty();
                                    }),
                     instrs.end());

        INFO(ss.str());
        CHECK(latencies.size() == 7);
        { // First instruction
            const auto& data  = *(latencies.begin());
            const auto& instr = *(instrs.begin());
            CHECK(data.instruction == "s_load_dwordx2 s[4:5], s[0:1], 0x0");
            CHECK(NormalizedSource(instr.toString(LogLevel::Critical))
                  == NormalizedSource("s_load_dwordx2 s[4:5], s[0:1], 0"));
            CHECK(data.hitcount == 4);
            CHECK(data.latency >= data.hitcount * 4);
        }
    }

} // namespace RocprofilerTest
