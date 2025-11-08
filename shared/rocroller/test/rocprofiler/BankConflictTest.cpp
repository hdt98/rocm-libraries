/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Scheduling/LDSBankModel.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>
#include <rocRoller/Scheduling/RoundRobinScheduler.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>

#include "../catch/TestContext.hpp"
#include "Agent.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace rocRoller;

namespace rocRollerTest
{
    const int NUM_RUNS = 5; // Should be odd, as median is used

    template <typename T>
    T median_of_odd_elements(std::vector<T> values)
    {
        AssertFatal(!values.empty(), "median_of_odd_elements: vector must not be empty");
        AssertFatal(values.size() % 2 == 1, "median_of_odd_elements: vector size must be odd");

        std::sort(values.begin(), values.end());

        return values[values.size() / 2];
    }

    // Helper function to calculate deltas (differences) between consecutive latency values
    std::vector<int64_t> calculateLatencyDeltas(const std::vector<uint64_t>& latencies)
    {
        std::vector<int64_t> deltas;
        for(size_t i = 1; i < latencies.size(); ++i)
        {
            int64_t delta
                = static_cast<int64_t>(latencies[i]) - static_cast<int64_t>(latencies[i - 1]);
            deltas.push_back(delta);
        }
        return deltas;
    }

    std::vector<size_t>
        generateLDSAddresses(size_t workgroupSize, size_t strideMultiplier, size_t instrDwords)
    {
        std::vector<size_t> addresses;
        for(size_t workitemId = 0; workitemId < workgroupSize; ++workitemId)
        {
            size_t address = workitemId * (4 * strideMultiplier * instrDwords);
            addresses.push_back(address);
        }
        return addresses;
    }

    TEST_CASE("Rocprofiler LDS Microkernel", "[rocprofiler]")
    {
        using namespace Scheduling::LDSBankModel;

        constexpr int  ITERS         = 16;
        constexpr auto workgroupSize = 64u;

        const std::vector<int>  instrSizes      = {1, 2, 4}; // b32, b64, b128
        const std::vector<int>  strides         = {1, 2, 4, 8, 16, 32, 64, 128}; // between threads
        const std::vector<bool> writeOperations = {false, true};

        for(auto instrDwords : instrSizes)
        {
            for(auto strideMultiplier : strides)
            {
                for(auto write : writeOperations)
                {
                    const auto name = "lds_microkernel_" + std::to_string(instrDwords * 32)
                                      + "b_stride" + std::to_string(strideMultiplier) + "_"
                                      + (write ? "write" : "read");

                    DYNAMIC_SECTION(name)
                    {
                        rocRoller::profiler::reset();

                        auto context = TestContext::ForTestDevice({}, name);

                        if(not context->targetArchitecture().target().isCDNA35GPU())
                        {
                            SKIP("LDS Bank Model only implemented for CDNA 3.5 GPUs");
                        }

                        auto command = std::make_shared<Command>();
                        auto k       = context->kernel();

                        k->setKernelDimensions(1);

                        const auto one  = std::make_shared<Expression::Expression>(1u);
                        const auto zero = std::make_shared<Expression::Expression>(0u);

                        auto workitemCount = Expression::literal(workgroupSize * 256 * 32);
                        k->setWorkgroupSize({workgroupSize, 1, 1});
                        k->setWorkitemCount({workitemCount, one, one});
                        k->setDynamicSharedMemBytes(zero);

                        context->schedule(k->preamble());
                        context->schedule(k->prolog());

                        auto kb = [&]() -> Generator<Instruction> {
                            const auto alignment = instrDwords;
                            const auto regCount  = 256 - 8; // leave a few

                            auto dst = Register::Value::Placeholder(
                                context.get(),
                                Register::Type::Vector,
                                DataType::Raw32,
                                regCount,
                                Register::AllocationOptions{
                                    .contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                    .alignment            = alignment,
                                });
                            co_yield dst->allocate();

                            auto lds = Register::Value::AllocateLDS(
                                context.get(),
                                DataType::Raw32,
                                context->targetArchitecture().GetCapability(
                                    GPUCapability::MaxLdsSize)
                                    / 4);
                            auto ldsWithOffset = Register::Value::Placeholder(
                                context.get(), Register::Type::Vector, DataType::UInt32, 1);
                            auto workitemIndex = context->kernel()->workitemIndex()[0];
                            co_yield Expression::generate(
                                ldsWithOffset,
                                Expression::literal(lds->getLDSAllocation()->offset())
                                    + workitemIndex->expression()
                                          * Expression::literal((4 * strideMultiplier * alignment)
                                                                % lds->getLDSAllocation()->size()),
                                context.get());

                            auto getSubset
                                = [](size_t n, size_t m, size_t i) -> std::pair<size_t, size_t> {
                                // If run out of registers, wrap around
                                size_t num_complete_chunks = n / m;
                                if(num_complete_chunks == 0)
                                {
                                    return {0, 0};
                                }
                                size_t chunk_index = i % num_complete_chunks;
                                size_t start       = chunk_index * m;
                                return {start, start + m};
                            };

                            co_yield context->mem()->barrier({});

                            for(int i = 0; i < ITERS; ++i)
                            {
                                const auto [start, end] = getSubset(regCount, instrDwords, i);
                                const auto numBytes     = instrDwords * 4;

                                if(write)
                                {
                                    co_yield context->mem()->storeLocal(
                                        ldsWithOffset,
                                        dst->subset(Generated(iota(start, end))),
                                        0,
                                        numBytes);
                                }
                                else
                                {
                                    co_yield context->mem()->loadLocal(
                                        dst->subset(Generated(iota(start, end))),
                                        ldsWithOffset,
                                        0,
                                        numBytes);
                                }
                            }
                            co_yield Instruction::Wait(
                                WaitCount::DSCnt(context->targetArchitecture(), 1));
                            co_yield Instruction::Wait(
                                WaitCount::DSCnt(context->targetArchitecture(), 0));
                        };

                        context->schedule(kb());

                        context->schedule(k->postamble());
                        context->schedule(k->amdgpu_metadata());

                        CommandKernel commandKernel;
                        commandKernel.setContext(context.get());
                        commandKernel.generateKernel();

                        CommandArguments commandArgs = command->createArguments();

                        std::vector<std::vector<rocRoller::profiler::InstructionProfile>>
                            allLatencies;

                        for(int run = 0; run < NUM_RUNS; ++run)
                        {
                            const auto latencies
                                = rocRoller::profiler::loopUntilDispatchData([&]() {
                                      commandKernel.launchKernel(commandArgs.runtimeArguments());
                                  });
                            allLatencies.push_back(latencies);

                            INFO("Run " << (run + 1) << ": " << toString(latencies));
                            Log::debug("Run " + std::to_string(run + 1) + ": "
                                       + toString(latencies));

                            REQUIRE(latencies.size() == 21);
                        }

                        GPUArchitectureGFX gfx = context->targetArchitecture().target().gfx;

                        auto baseAddresses
                            = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

                        RuntimeLDSInstruction ldsinstr;
                        ldsinstr.memoryOp.direction
                            = write ? LdsDirection::Write : LdsDirection::Read;
                        ldsinstr.dwords        = instrDwords;
                        ldsinstr.baseAddresses = baseAddresses;

                        uint predictedCycles = getInstructionCycles(ldsinstr, gfx);

                        uint issueCycles
                            = getInstructionIssueCycles(ldsinstr.memoryOp, ldsinstr.dwords);
                        uint dataCycles = getInstructionDataCycles(ldsinstr, gfx);

                        std::stringstream info;

                        info << fmt::format("dwords {}, stride {}, {}\n",
                                            instrDwords,
                                            strideMultiplier,
                                            write ? "write" : "read");

                        std::vector<uint64_t> ldsInstrCyclesPerRun;
                        std::vector<uint64_t> sWaitcntCyclesPerRun;

                        for(int run = 0; run < NUM_RUNS; ++run)
                        {
                            uint64_t maxLdsInstrCycles  = 0;
                            uint64_t lastSWaitcntCycles = 0;

                            for(const auto& data : allLatencies[run])
                            {
                                if((write && data.instruction.find("ds_write") != std::string::npos)
                                   || (!write
                                       && data.instruction.find("ds_read") != std::string::npos))
                                {
                                    maxLdsInstrCycles
                                        = std::max(maxLdsInstrCycles, data.meanLatency());
                                }
                                else if(data.instruction.find("s_waitcnt") != std::string::npos)
                                {
                                    lastSWaitcntCycles = data.meanLatency();
                                }
                            }

                            ldsInstrCyclesPerRun.push_back(maxLdsInstrCycles);
                            sWaitcntCyclesPerRun.push_back(lastSWaitcntCycles);

                            info << fmt::format("  Run {}: LDS Cycles: {}, s_waitcnt Cycles: {}\n",
                                                run + 1,
                                                maxLdsInstrCycles,
                                                lastSWaitcntCycles);
                        }

                        uint64_t actualMaxLdsInstrCycles
                            = median_of_odd_elements(ldsInstrCyclesPerRun);
                        uint64_t actualLastSWaitcntCycles
                            = median_of_odd_elements(sWaitcntCyclesPerRun);
                        info << fmt::format("  Median s_waitcnt Cycles: {}\n",
                                            actualLastSWaitcntCycles);
                        info << fmt::format("  Median LDS Instruction Cycles: {}\n",
                                            actualMaxLdsInstrCycles);
                        info << fmt::format("  Model Predicted Cycles: {}\n", predictedCycles);
                        info << fmt::format("    Issue Cycles: {}\n", issueCycles);
                        info << fmt::format("    Data Cycles: {}\n", dataCycles);

                        INFO(info.str());
                        Log::debug(info.str());

                        CHECK(actualLastSWaitcntCycles == predictedCycles);
                        if(write && instrDwords == 4)
                            // ds_write_b128 requires queue info
                            CHECK_THAT(actualMaxLdsInstrCycles,
                                       Catch::Matchers::WithinAbs(predictedCycles, 12ul));
                        else
                            CHECK_THAT(actualMaxLdsInstrCycles,
                                       Catch::Matchers::WithinAbs(predictedCycles, 4ul));
                    }
                }
            }
        }
    }

    TEST_CASE("Rocprofiler LDS Observer", "[rocprofiler]")
    {
        using namespace Scheduling::LDSBankModel;

        constexpr int  ITERS         = 16;
        constexpr auto workgroupSize = 64u;

        const std::vector<int>  instrSizes      = {4};
        const std::vector<int>  strides         = {4}; // between threads
        const std::vector<bool> writeOperations = {true};

        for(auto instrDwords : instrSizes)
        {
            for(auto strideMultiplier : strides)
            {
                for(auto write : writeOperations)
                {
                    const auto name = "lds_microkernel_" + std::to_string(instrDwords * 32)
                                      + "b_stride" + std::to_string(strideMultiplier) + "_"
                                      + (write ? "write" : "read");

                    DYNAMIC_SECTION(name)
                    {
                        rocRoller::profiler::reset();

                        auto context = TestContext::ForTestDevice({}, name);

                        if(not context->targetArchitecture().target().isCDNA35GPU())
                        {
                            SKIP("LDS Bank Model only implemented for CDNA 3.5 GPUs");
                        }

                        auto command = std::make_shared<Command>();
                        auto k       = context->kernel();

                        k->setKernelDimensions(1);

                        const auto one  = std::make_shared<Expression::Expression>(1u);
                        const auto zero = std::make_shared<Expression::Expression>(0u);

                        auto workitemCount = Expression::literal(workgroupSize * 256 * 32);
                        k->setWorkgroupSize({workgroupSize, 1, 1});
                        k->setWorkitemCount({workitemCount, one, one});
                        k->setDynamicSharedMemBytes(zero);

                        context->schedule(k->preamble());
                        context->schedule(k->prolog());

                        auto kb = [&]() -> Generator<Instruction> {
                            const auto alignment = instrDwords;
                            const auto regCount  = 256 - 8; // leave a few

                            auto dst = Register::Value::Placeholder(
                                context.get(),
                                Register::Type::Vector,
                                DataType::Raw32,
                                regCount,
                                Register::AllocationOptions{
                                    .contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                    .alignment            = alignment,
                                });
                            co_yield dst->allocate();

                            auto lds = Register::Value::AllocateLDS(
                                context.get(),
                                DataType::Raw32,
                                context->targetArchitecture().GetCapability(
                                    GPUCapability::MaxLdsSize)
                                    / 4);
                            auto ldsWithOffset = Register::Value::Placeholder(
                                context.get(), Register::Type::Vector, DataType::UInt32, 1);
                            auto workitemIndex = context->kernel()->workitemIndex()[0];
                            co_yield Expression::generate(
                                ldsWithOffset,
                                Expression::literal(lds->getLDSAllocation()->offset())
                                    + workitemIndex->expression()
                                          * Expression::literal((4 * strideMultiplier * alignment)
                                                                % lds->getLDSAllocation()->size()),
                                context.get());

                            auto getSubset
                                = [](size_t n, size_t m, size_t i) -> std::pair<size_t, size_t> {
                                // If run out of registers, wrap around
                                size_t num_complete_chunks = n / m;
                                if(num_complete_chunks == 0)
                                {
                                    return {0, 0};
                                }
                                size_t chunk_index = i % num_complete_chunks;
                                size_t start       = chunk_index * m;
                                return {start, start + m};
                            };

                            co_yield context->mem()->barrier({});

                            for(int i = 0; i < ITERS; ++i)
                            {
                                const auto [start, end] = getSubset(regCount, instrDwords, i);
                                const auto numBytes     = instrDwords * 4;

                                if(write)
                                {
                                    co_yield context->mem()->storeLocal(
                                        ldsWithOffset,
                                        dst->subset(Generated(iota(start, end))),
                                        0,
                                        numBytes);
                                }
                                else
                                {
                                    co_yield context->mem()->loadLocal(
                                        dst->subset(Generated(iota(start, end))),
                                        ldsWithOffset,
                                        0,
                                        numBytes);
                                }
                            }
                            co_yield Instruction::Wait(
                                WaitCount::DSCnt(context->targetArchitecture(), 1));
                            co_yield Instruction::Wait(
                                WaitCount::DSCnt(context->targetArchitecture(), 0));
                        };

                        const auto baseAddresses
                            = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

                        std::vector<Instruction> instructions;
                        for(auto inst : kb())
                        {
                            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                                inst.setAddresses(baseAddresses);
                            context->schedule(inst);
                            instructions.push_back(inst);
                        }

                        context->schedule(k->postamble());
                        context->schedule(k->amdgpu_metadata());

                        CommandKernel commandKernel;
                        commandKernel.setContext(context.get());
                        commandKernel.generateKernel();

                        CommandArguments commandArgs = command->createArguments();

                        std::vector<std::vector<rocRoller::profiler::InstructionProfile>>
                            allLatencies;

                        for(int run = 0; run < NUM_RUNS; ++run)
                        {
                            const auto latencies
                                = rocRoller::profiler::loopUntilDispatchData([&]() {
                                      commandKernel.launchKernel(commandArgs.runtimeArguments());
                                  });
                            allLatencies.push_back(latencies);

                            INFO("Run " << (run + 1) << ": " << toString(latencies));
                            Log::debug("Run " + std::to_string(run + 1) + ": "
                                       + toString(latencies));

                            REQUIRE(latencies.size() == 21);
                        }

                        std::vector<std::vector<uint64_t>> ldsLatenciesPerIteration(ITERS);

                        // Extract LDS latencies for each iteration across all runs
                        for(const auto& runLatencies : allLatencies)
                        {
                            int ldsInstrCount = 0;
                            for(const auto& profile : runLatencies)
                            {
                                if((write
                                    && profile.instruction.find("ds_write") != std::string::npos)
                                   || (!write
                                       && profile.instruction.find("ds_read") != std::string::npos))
                                {
                                    if(ldsInstrCount < ITERS)
                                    {
                                        ldsLatenciesPerIteration[ldsInstrCount].push_back(
                                            profile.meanLatency());
                                        ldsInstrCount++;
                                    }
                                }
                            }
                        }

                        std::vector<uint64_t> medianLatencyPerIteration;
                        for(const auto& iterLatencies : ldsLatenciesPerIteration)
                        {
                            if(!iterLatencies.empty())
                            {
                                auto sortedLatencies = iterLatencies;
                                std::sort(sortedLatencies.begin(), sortedLatencies.end());
                                medianLatencyPerIteration.push_back(
                                    sortedLatencies[sortedLatencies.size() / 2]);
                            }
                        }

                        if(medianLatencyPerIteration.empty())
                        {
                            INFO("No LDS latency data collected");
                            continue;
                        }

                        auto deltas = calculateLatencyDeltas(medianLatencyPerIteration);

                        // Find first increase
                        int firstIncreaseIteration = -1;

                        for(size_t i = 0; i < deltas.size(); ++i)
                        {
                            if(deltas[i] > 0)
                            {
                                firstIncreaseIteration
                                    = i + 1; // +1 because deltas[0] is between iter 0 and 1
                                break;
                            }
                        }

                        // Find where latency stabilizes (delta == 0)
                        int latencyStabilizedIteration = -1;

                        if(firstIncreaseIteration != -1)
                        {
                            for(size_t i = firstIncreaseIteration; i < deltas.size(); ++i)
                            {
                                if(deltas[i] == 0)
                                {
                                    latencyStabilizedIteration = i + 1;
                                    break;
                                }
                            }
                        }

                        std::stringstream analysis;
                        analysis << "\nLDS Latency Analysis:\n";

                        if(firstIncreaseIteration != -1)
                        {
                            int64_t increaseAmount = deltas[firstIncreaseIteration - 1];
                            analysis
                                << "First latency increase at iteration: " << firstIncreaseIteration
                                << " (increased by " << increaseAmount << " cycles)\n";
                        }
                        if(latencyStabilizedIteration != -1)
                        {
                            analysis
                                << "Latency stabilized at iteration: " << latencyStabilizedIteration
                                << "\n";
                        }

                        analysis << "\nPer-iteration median latencies:\n";
                        for(size_t i = 0; i < medianLatencyPerIteration.size(); ++i)
                        {
                            analysis << "  Iteration " << i << ": " << medianLatencyPerIteration[i]
                                     << " cycles";
                            if(i > 0 && i <= deltas.size())
                            {
                                int64_t delta = deltas[i - 1];
                                if(delta != 0)
                                {
                                    analysis << " (" << (delta > 0 ? "+" : "") << delta << ")";
                                }
                            }
                            analysis << "\n";
                        }

                        Log::info(analysis.str());

                        std::stringstream instructionsInfo;
                        for(const auto& inst : instructions)
                        {
                            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                                instructionsInfo << inst.toString(LogLevel::Terse)
                                                 << inst.peekedStatus().toString() << "\n";
                        }
                        Log::info("Instructions:\n" + instructionsInfo.str());
                    }
                }
            }
        }
    }

    TEST_CASE("Scheduler with LDS and ALU instruction streams", "[rocprofiler][scheduler]")
    {
        using namespace Scheduling::LDSBankModel;

        Settings::getInstance()->set(Settings::SchedulerCost,
                                     Scheduling::CostFunction::LinearWeightedSimple);

        constexpr int  ITERS         = 32;
        constexpr auto workgroupSize = 64u;

        const auto name = "lds_alu_scheduler_test";

        rocRoller::profiler::reset();

        auto context = TestContext::ForTestDevice({}, name);

        if(not context->targetArchitecture().target().isCDNA35GPU())
        {
            SKIP("LDS Bank Model only implemented for CDNA 3.5 GPUs");
        }

        auto command = std::make_shared<Command>();
        auto k       = context->kernel();

        k->setKernelDimensions(1);

        const auto one  = std::make_shared<Expression::Expression>(1u);
        const auto zero = std::make_shared<Expression::Expression>(0u);

        auto workitemCount = Expression::literal(workgroupSize * 256 * 32);
        k->setWorkgroupSize({workgroupSize, 1, 1});
        k->setWorkitemCount({workitemCount, one, one});
        k->setDynamicSharedMemBytes(zero);

        context->schedule(k->preamble());
        context->schedule(k->prolog());

        // ds_read_b64 with stride multiplier of 4
        const auto instrDwords      = 4;
        const auto strideMultiplier = 4;
        const auto baseAddresses
            = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

        auto ldsData = Register::Value::AllocateLDS(
            context.get(),
            DataType::Raw32,
            context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize) / 4);

        auto ldsWithOffset = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        auto workitemIndex = context->kernel()->workitemIndex()[0];

        context->schedule(Expression::generate(
            ldsWithOffset,
            Expression::literal(ldsData->getLDSAllocation()->offset())
                + workitemIndex->expression()
                      * Expression::literal((4 * strideMultiplier * instrDwords)
                                            % ldsData->getLDSAllocation()->size()),
            context.get()));

        auto ldsStream = [&]() -> Generator<Instruction> {
            auto ldsDst = Register::Value::Placeholder(
                context.get(),
                Register::Type::Vector,
                DataType::Raw32,
                2 * ITERS,
                Register::AllocationOptions{.contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                            .alignment            = instrDwords});
            co_yield ldsDst->allocate();

            co_yield context->mem()->barrier({});

            for(int i = 0; i < ITERS; ++i)
            {
                auto dstRegs = ldsDst->subset(Generated(iota(i * 2, (i + 1) * 2)));

                co_yield context->mem()->storeLocal(ldsWithOffset,
                                                    dstRegs,
                                                    i * 8, // offset in bytes
                                                    8 // 64 bits = 8 bytes
                );
            }
        };

        auto aluStream = [&]() -> Generator<Instruction> {
            auto v0 = Register::Value::Placeholder(
                context.get(), Register::Type::Vector, DataType::Int32, 1);
            auto v1 = Register::Value::Placeholder(
                context.get(), Register::Type::Vector, DataType::Int32, 1);

            co_yield v0->allocate();
            co_yield v1->allocate();

            for(int i = 0; i < 2 * ITERS; ++i)
            {
                co_yield generateOp<Expression::Add>(v0, v0, v1);
            }
        };

        auto scheduler = Component::GetNew<Scheduling::Scheduler>(
            std::make_tuple(Scheduling::SchedulerProcedure::RoundRobin,
                            Scheduling::CostFunction::None,
                            context.get()));

        std::vector<Generator<Instruction>> streams;
        streams.push_back(ldsStream());
        streams.push_back(aluStream());

        for(auto inst : (*scheduler)(streams))
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                inst.setAddresses(baseAddresses);
            context->schedule(inst);
        }

        context->schedule(k->postamble());
        context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(context.get());
        commandKernel.generateKernel();

        CommandArguments commandArgs = command->createArguments();

        std::vector<std::vector<rocRoller::profiler::InstructionProfile>> allLatencies;

        for(int run = 0; run < NUM_RUNS; ++run)
        {
            const auto latencies = rocRoller::profiler::loopUntilDispatchData(
                [&]() { commandKernel.launchKernel(commandArgs.runtimeArguments()); });
            allLatencies.push_back(latencies);
        }

        Log::info(context.output());

        Log::info(toString(allLatencies[0]));
    }
}
