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

#include "Agent.hpp"

#include <common/SourceMatcher.hpp>
#include <common/Utilities.hpp>
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

#include "../catch/TestContext.hpp"

#include <fmt/format.h>

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "hip/hip_runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ranges>
#include <thread>
#include <vector>

using namespace rocRoller;

namespace RocprofilerTest
{
    struct KernelSetup
    {
        CommandKernel             kernel;
        std::shared_ptr<uint32_t> d_ptr;
        CommandArguments          commandArgs;
        std::vector<Instruction>  instrs;
    };

    KernelSetup
        createKernel(std::shared_ptr<Context> context, uint32_t literal, uint32_t commandArg)
    {
        auto command = std::make_shared<Command>();

        VariableType uintPtr{DataType::UInt32, PointerType::PointerGlobal};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptrTag  = command->allocateTag();
        auto ptr_arg = command->allocateArgument(uintPtr, ptrTag, ArgumentType::Value);
        auto valTag  = command->allocateTag();
        auto val_arg = command->allocateArgument(uintVal, valTag, ArgumentType::Value);

        auto ptr_exp = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp = std::make_shared<Expression::Expression>(val_arg);

        auto k = context->kernel();

        k->setKernelDimensions(1);

        const auto one = std::make_shared<Expression::Expression>(1u);
        k->setWorkgroupSize({256, 1, 1});
        // more waves for rocprofiler, see Troubleshooting in
        // https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-thread-trace.html#troubleshooting
        k->setWorkitemCount({std::make_shared<Expression::Expression>(256 * 256 * 4), one, one});

        k->addArgument({"ptr",
                        {DataType::UInt32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val", {DataType::UInt32}, DataDirection::ReadOnly, val_exp});

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
                context, Register::Type::Vector, {DataType::UInt32, PointerType::PointerGlobal}, 1);
            auto v_value = Register::Value::Placeholder(
                context, Register::Type::Vector, DataType::UInt32, 1);

            co_yield v_ptr->allocate();
            co_yield context->copier()->copy(v_ptr, s_ptr, "Move pointer");

            co_yield v_value->allocate();
            co_yield Expression::generate(v_value, Expression::literal(literal), context);
            co_yield Expression::generate(
                v_value, v_value->expression() + s_value->expression(), context);

            co_yield context->mem()->storeGlobal(v_ptr, v_value, 0, 4);
        };

        captureInstrAndSchedule(kb());

        context->schedule(k->postamble());
        context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(context);
        commandKernel.generateKernel();

        auto d_ptr = make_shared_device<uint32_t>(1, 0);

        CommandArguments commandArgs = command->createArguments();
        commandArgs.setArgument(ptrTag, ArgumentType::Value, d_ptr.get());
        commandArgs.setArgument(valTag, ArgumentType::Value, commandArg);

        return {std::move(commandKernel), d_ptr, commandArgs, instrs};
    }

    TEST_CASE("Rocprofiler kernels with different literals in assembly", "[rocprofiler]")
    {
        /*
        The rocprofiler agent uses a dispatch system. Due to the callback-based nature of rocprofiler,
        this test ensures the correct dispatch is matched to the correct kernel when multiple kernels
        are launched in sequence.
        Here, kernels with different "v_mov_b32_e32 v1, 0x<literal>" are launch to distinguish them.
        */

        auto literal    = GENERATE(0xdeadbeef, 0x12345678, 0xabcdef00);
        auto commandArg = GENERATE(7, 21, 331);

        DYNAMIC_SECTION("literal: 0x" << std::hex << literal << ", value: " << std::dec
                                      << commandArg)
        {
            std::string const testName = fmt::format("const_0x{:x}_value_{}", literal, commandArg);

            auto context = TestContext::ForTestDevice({}, testName);

            INFO("Testing " << testName);

            auto kernelSetup = createKernel(context.get(), literal, commandArg);

            kernelSetup.kernel.launchKernel(kernelSetup.commandArgs.runtimeArguments());
            HIP_CHECK(hipDeviceSynchronize());

            const auto latencies = rocroller_profiler::getInstructionData();

            { // Verify device result
                uint32_t h_result = 0;
                HIP_CHECK(hipMemcpy(
                    &h_result, kernelSetup.d_ptr.get(), sizeof(uint32_t), hipMemcpyDeviceToHost));

                uint32_t expectedResult = commandArg + literal;
                CHECK(h_result == expectedResult);
            }

            std::stringstream ss;
            ss << "Instruction, Total Latency, Hit Count, Average Latency" << std::endl;
            for(const auto& data : latencies)
            {
                uint64_t avg_latency = data.hitcount ? (data.latency / data.hitcount) : 0;
                ss << "\"" << data.instruction << "\", " << data.latency << ", " << data.hitcount
                   << ", " << avg_latency << std::endl;
            }
            INFO(ss.str());
            REQUIRE(latencies.size() >= 8); // gfx12 has 9, others have 8

            { // Ensure instructions exist in expected quanities in the profile data
                std::string const instructionsStr = [&]() {
                    std::stringstream ss;
                    streamJoin(ss,
                               std::views::transform(latencies,
                                                     [](const auto& d) { return d.instruction; }),
                               "\n");
                    return ss.str();
                }();
                INFO("Instructions:\n" << instructionsStr);
                CHECK(1
                      == countSubstring(instructionsStr,
                                        fmt::format("v_mov_b32_e32 v1, 0x{:x}", literal)));
            }
        }
    }

    KernelSetup createSimpleMovKernel(std::shared_ptr<Context> context, uint32_t literal)
    {
        auto command = std::make_shared<Command>();

        auto k = context->kernel();

        k->setKernelDimensions(1);

        const auto one = std::make_shared<Expression::Expression>(1u);
        k->setWorkgroupSize({256, 1, 1});
        // more waves for rocprofiler, see Troubleshooting in
        // https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-thread-trace.html#troubleshooting
        k->setWorkitemCount({std::make_shared<Expression::Expression>(256 * 256 * 16), one, one});

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
            auto v_value = Register::Value::Placeholder(
                context, Register::Type::Vector, DataType::UInt32, 1);

            co_yield v_value->allocate();
            co_yield Expression::generate(v_value, Expression::literal(literal), context);
        };

        captureInstrAndSchedule(kb());

        context->schedule(k->postamble());
        context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(context);
        commandKernel.generateKernel();

        CommandArguments commandArgs = command->createArguments();

        return {std::move(commandKernel), nullptr, commandArgs, instrs};
    }

    TEST_CASE("Rocprofiler agent race conditions", "[rocprofiler]")
    {
        /*
        There were race conditions between the dispatch and shader data callbacks.
        This test ensures that the fix works as intended.
        Multiple simple kernels with different literals are launched various orders.
        Ensures the profiler returns instructions from the last launched kernel.
        */

        std::vector<uint32_t>    literals = {0xdeadbeef, 0x12345678, 0xabcdef00};
        std::vector<KernelSetup> kernelSetups;

        for(uint32_t literal : literals)
        {
            std::string const testName = fmt::format("simple_mov_0x{:x}", literal);
            auto              context  = TestContext::ForTestDevice({}, testName);
            kernelSetups.push_back(createSimpleMovKernel(context.get(), literal));
        }

        SECTION("Order 1")
        {
            std::vector<size_t> order = {0, 1, 2, 1};
            for(size_t idx : order)
            {
                kernelSetups[idx].kernel.launchKernel(
                    kernelSetups[idx].commandArgs.runtimeArguments());
            }
            HIP_CHECK(hipDeviceSynchronize());
            const auto        latencies  = rocroller_profiler::getInstructionData();
            std::string const literalHex = fmt::format("0x{:x}", literals[order.back()]);
            INFO("Expecting literal: " << literalHex);
            REQUIRE(latencies.size() == 2);
            CHECK(1 == countSubstring(latencies[0].instruction, literalHex));
            CHECK(latencies[1].instruction == "s_endpgm");
        }

        SECTION("Order 2")
        {
            std::vector<size_t> order = {1, 2};
            for(size_t idx : order)
            {
                kernelSetups[idx].kernel.launchKernel(
                    kernelSetups[idx].commandArgs.runtimeArguments());
            }
            HIP_CHECK(hipDeviceSynchronize());
            const auto        latencies  = rocroller_profiler::getInstructionData();
            std::string const literalHex = fmt::format("0x{:x}", literals[order.back()]);
            INFO("Expecting literal: " << literalHex);
            REQUIRE(latencies.size() == 2);
            CHECK(1 == countSubstring(latencies[0].instruction, literalHex));
            CHECK(latencies[1].instruction == "s_endpgm");
        }

        SECTION("With profiler calls")
        {
            std::vector<size_t> order = {1, 2};
            for(size_t idx : order)
            {
                kernelSetups[idx].kernel.launchKernel(
                    kernelSetups[idx].commandArgs.runtimeArguments());
                HIP_CHECK(hipDeviceSynchronize());
                rocroller_profiler::getInstructionData();
            }
            const auto        latencies  = rocroller_profiler::getInstructionData();
            std::string const literalHex = fmt::format("0x{:x}", literals[order.back()]);
            INFO("Expecting literal: " << literalHex);
            REQUIRE(latencies.size() == 2);
            CHECK(1 == countSubstring(latencies[0].instruction, literalHex));
            CHECK(latencies[1].instruction == "s_endpgm");
        }

        // FAIL();
    }

    TEST_CASE("Rocprofiler simple", "[rocprofiler]")
    {
        auto literal    = GENERATE(0x11223344);
        auto commandArg = GENERATE(7);

        std::string const testName = fmt::format("const_0x{:x}_value_{}", literal, commandArg);

        auto context = TestContext::ForTestDevice({}, testName);

        INFO("Testing " << testName);

        auto kernelSetup = createKernel(context.get(), literal, commandArg);

        kernelSetup.kernel.launchKernel(kernelSetup.commandArgs.runtimeArguments());
        HIP_CHECK(hipDeviceSynchronize());

        const auto latencies = rocroller_profiler::getInstructionData();

        { // Verify device result
            uint32_t h_result = 0;
            HIP_CHECK(hipMemcpy(
                &h_result, kernelSetup.d_ptr.get(), sizeof(uint32_t), hipMemcpyDeviceToHost));

            uint32_t expectedResult = commandArg + literal;
            CHECK(h_result == expectedResult);
        }

        std::stringstream ss;
        ss << "Instruction, Total Latency, Hit Count, Average Latency" << std::endl;
        for(const auto& data : latencies)
        {
            uint64_t avg_latency = data.hitcount ? (data.latency / data.hitcount) : 0;
            ss << "\"" << data.instruction << "\", " << data.latency << ", " << data.hitcount
               << ", " << avg_latency << std::endl;
        }
        INFO(ss.str());
        CHECK(latencies.size() >= 8); // gfx12 has 9, others have 8

        { // Ensure instructions exist in expected quanities in the profile data
            std::string const instructionsStr = [&]() {
                std::stringstream ss;
                streamJoin(
                    ss,
                    std::views::transform(latencies, [](const auto& d) { return d.instruction; }),
                    "\n");
                return ss.str();
            }();
            INFO("Instructions:\n" << instructionsStr);
            CHECK(1
                  == countSubstring(instructionsStr,
                                    fmt::format("v_mov_b32_e32 v1, 0x{:x}", literal)));
        }
    }
} // namespace RocprofilerTest
