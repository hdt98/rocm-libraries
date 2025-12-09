/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <common/Scheduling.hpp>

#include "../catch/TestContext.hpp"
#include "../catch/TestKernels.hpp"
#include "Agent.hpp"
#include "Utils.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace rocRoller;

const int NUM_RUNS = 5; // Should be odd, as median is used

TEST_CASE("Weave LDS and waitcnt", "[rocprofiler][scheduler][lds-model]")
{
    /*
    v_lshlrev_b32_e32 v1, 5, v0, model 4, profiler 4, delta 0
    s_barrier, model 4, profiler 4, delta 0
    ds_read_b128 v[4:7], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 52, profiler 52, delta 0
    ds_read_b128 v[8:11], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 52, profiler 52, delta 0
    ds_read_b128 v[12:15], v1, model 4, profiler 4, delta 0
    ...
    */

    using namespace Scheduling::LDSBankModel;

    constexpr auto workgroupSize = 64u;

    int instrDwords;
    int strideMultiplier;
    int write;

    constexpr auto testIndividual = false;
    if(testIndividual)
    {
        instrDwords      = GENERATE(4);
        strideMultiplier = GENERATE(2);
        write            = GENERATE(false);
    }
    else
    {
        instrDwords      = GENERATE(1, 2, 4);
        strideMultiplier = GENERATE(1, 2, 4, 8);
        write            = GENERATE(true, false);
    }
    constexpr auto iters = 8;

    const auto baseAddresses = generateLDSAddresses(64, strideMultiplier, instrDwords);

    const auto name = fmt::format("lds_weave_waitcnt_{}_b{}_stride{}",
                                  write ? "write" : "read",
                                  instrDwords * 32,
                                  strideMultiplier);

    rocRoller::profiler::reset();

    auto context = TestContext::ForTestDevice({}, name);

    if(not context->targetArchitecture().target().isCDNA35GPU())
    {
        SKIP("Currently only testing on gfx950");
    }

    SECTION(name)
    {
        auto command = std::make_shared<Command>();
        auto k       = context->kernel();
        CAPTURE(name);

        k->setKernelDimensions(1);

        const auto one  = std::make_shared<Expression::Expression>(1u);
        const auto zero = std::make_shared<Expression::Expression>(0u);

        auto workitemCount = Expression::literal(workgroupSize * 256);
        k->setWorkgroupSize({workgroupSize, 1, 1});
        k->setWorkitemCount({workitemCount, one, one});
        k->setDynamicSharedMemBytes(zero);

        context->schedule(k->preamble());
        context->schedule(k->prolog());

        auto ldsData = Register::Value::AllocateLDS(
            context.get(),
            DataType::Raw32,
            context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize) / 4);

        auto ldsWithOffset = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        auto workitemIndex = context->kernel()->workitemIndex()[0];

        auto kb = [&]() -> Generator<Instruction> {
            co_yield Expression::generate(
                ldsWithOffset,
                Expression::literal(ldsData->getLDSAllocation()->offset())
                    + workitemIndex->expression()
                          * Expression::literal((4 * strideMultiplier * instrDwords)
                                                % ldsData->getLDSAllocation()->size()),
                context.get());

            auto ldsDst = Register::Value::Placeholder(
                context.get(),
                Register::Type::Vector,
                DataType::Raw32,
                248,
                Register::AllocationOptions{.contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                            .alignment            = instrDwords});
            co_yield ldsDst->allocate();

            co_yield context->mem()->barrier({});

            int counter = 0;
            for(int i = 0; i < iters; ++i)
            {
                const auto [start, end]
                    = getAlignedSubset(ldsDst->registerCount(), instrDwords, counter++);
                auto dstRegs = ldsDst->subset(Generated(iota(start, end)));

                if(write)
                    co_yield context->mem()->storeLocal(ldsWithOffset, dstRegs, 0, 4 * instrDwords);
                else
                    co_yield context->mem()->loadLocal(dstRegs, ldsWithOffset, 0, 4 * instrDwords);

                co_yield Instruction::Wait(WaitCount::DSCnt(context->targetArchitecture(), 0));
            }
        };

        auto scheduler = Component::GetNew<Scheduling::Scheduler>(
            std::make_tuple(Scheduling::SchedulerProcedure::Priority,
                            Scheduling::CostFunction::LinearWeighted,
                            context.get()));

        std::vector<Instruction> instructions;
        for(auto inst : kb())
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                inst.setAddresses(baseAddresses);
            context->schedule(inst);
            instructions.push_back(inst);
        }
        instructions.push_back(Instruction("s_endpgm", {}, {}, {}, ""));

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

        // Filter instructions and verify alignment with profiler data
        const auto filteredInstructions
            = filterAndVerifyInstructions(instructions, allLatencies[0]);

        const auto infoStr = formatLatencyComparison(filteredInstructions, allLatencies[0]);
        INFO(infoStr);

        size_t expectedSize = allLatencies[0].size();
        REQUIRE(std::all_of(
            allLatencies.begin(), allLatencies.end(), [expectedSize](const auto& latencies) {
                return latencies.size() == expectedSize;
            }));

        std::vector<std::tuple<std::string, size_t>> medianLatencies;
        for(size_t i = 0; i < filteredInstructions.size(); ++i)
        {
            std::vector<uint64_t> latenciesPerRun;
            for(const auto& runLatencies : allLatencies)
            {
                latenciesPerRun.push_back(runLatencies[i].meanLatency());
            }
            auto       medianLatency = median_of_odd_elements(latenciesPerRun);
            const auto instrString   = allLatencies[0][i].instruction;
            medianLatencies.push_back(std::make_tuple(instrString, medianLatency));
        }

        if(testIndividual)
        {
            Log::info(infoStr);
            Log::info(context.output());
        }

        int totalAbsoluteDelta       = 0;
        int totalDelta               = 0;
        int incorrectPredictionCount = 0;
        int waitcntInstructionCount  = 0;

        for(size_t i = 0; i < filteredInstructions.size() - 1; ++i) // exclude s_endpgm
        {
            const auto& inst = filteredInstructions[i];
            using namespace Scheduling::LDSBankModel;

            int modelLatency = inst.totalCycles() * 4;

            int actualLatency = std::get<1>(medianLatencies[i]);
            int delta         = actualLatency - modelLatency;

            totalAbsoluteDelta += std::abs(delta);
            totalDelta += delta;

            if(delta != 0)
            {
                incorrectPredictionCount++;
            }

            if(inst.getWaitCount().dscnt() >= 0)
            {
                waitcntInstructionCount++;
            }
        }

        INFO(fmt::format(
            "Total absolute delta: {}, Incorrect predictions: {}/{}, Waitcnt instructions: {}",
            totalAbsoluteDelta,
            incorrectPredictionCount,
            filteredInstructions.size() - 1,
            waitcntInstructionCount));

        CHECK(waitcntInstructionCount == iters);

        if(strideMultiplier == 1)
        {
            /* In case of no bank conflicts, profiler reports ~1 quadcycle slower than model predicts -- not sure why.
            Examples:
            1)
                ds_write_b128 v1, v[20:23], model 20, profiler 24, delta 4
                s_waitcnt lgkmcnt(0), model 60, profiler 48, delta -12
            2)
                ds_write_b128 v1, v[16:19], model 20, profiler 20, delta 0
                s_waitcnt lgkmcnt(0), model 60, profiler 52, delta -8
            3)
                ds_read_b64 v[2:3], v1, model 4, profiler 8, delta 4
                s_waitcnt lgkmcnt(0), model 48, profiler 44, delta -4
            */

            CHECK(totalAbsoluteDelta <= 16 * iters); // for case 1)
            CHECK(totalDelta >= -8 * waitcntInstructionCount); // for case 1)/2)
        }
        else
        {
            /*  Sometimes get this:
                ds_read_b32 v2, v1, model 4, profiler 8, delta 4
                s_waitcnt lgkmcnt(0), model 52, profiler 48, delta -4
            */
            CHECK(totalAbsoluteDelta <= 4 * 2 * iters);
            CHECK(totalDelta == 0);
        }
    }
}

TEST_CASE("Weave LDS and s_add", "[rocprofiler][scheduler][lds-model]")
{
    /*
    ...
    ds_read_b128 v[72:75], v1, model 16, profiler 16, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    ds_read_b128 v[76:79], v1, model 12, profiler 12, delta 0
    ds_read_b128 v[80:83], v1, model 16, profiler 16, delta 0
    ds_read_b128 v[84:87], v1, model 16, profiler 16, delta 0
    ds_read_b128 v[88:91], v1, model 16, profiler 16, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    ds_read_b128 v[92:95], v1, model 8, profiler 8, delta 0
    ds_read_b128 v[96:99], v1, model 16, profiler 16, delta 0
    ds_read_b128 v[100:103], v1, model 16, profiler 16, delta 0
    ds_read_b128 v[104:107], v1, model 16, profiler 16, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    ds_read_b128 v[108:111], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[112:115], v1, model 16, profiler 16, delta 0
    ds_read_b128 v[116:119], v1, model 16, profiler 16, delta 0
    ds_read_b128 v[120:123], v1, model 16, profiler 16, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    ds_read_b128 v[124:127], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[128:131], v1, model 12, profiler 12, delta 0
    ds_read_b128 v[132:135], v1, model 16, profiler 16, delta 0
    ds_read_b128 v[136:139], v1, model 16, profiler 16, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    s_add_u32 s3, s3, s4, model 4, profiler 4, delta 0
    ds_read_b128 v[140:143], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[144:147], v1, model 8, profiler 8, delta 0
    ds_read_b128 v[148:151], v1, model 16, profiler 16, delta 0
    ds_read_b128 v[152:155], v1, model 16, profiler 16, delta 0
    ...
    */

    using namespace Scheduling::LDSBankModel;

    constexpr auto workgroupSize = 64u;

    int instrDwords;
    int strideMultiplier;
    int write;

    constexpr auto testIndividual = false;
    if(testIndividual)
    {
        instrDwords      = GENERATE(4);
        strideMultiplier = GENERATE(4);
        write            = GENERATE(false);
    }
    else
    {
        instrDwords      = GENERATE(1, 2, 4);
        strideMultiplier = GENERATE(1, 2, 4, 8);
        write            = GENERATE(true, false);
    }

    const auto baseAddresses = generateLDSAddresses(64, strideMultiplier, instrDwords);

    const auto name = fmt::format(
        "lds_weave_{}_b{}_stride{}", write ? "write" : "read", instrDwords * 32, strideMultiplier);

    rocRoller::profiler::reset();

    auto context = TestContext::ForTestDevice({}, name);

    if(not context->targetArchitecture().target().isCDNA35GPU())
    {
        SKIP("Currently only testing on gfx950");
    }

    SECTION(name)
    {
        auto command = std::make_shared<Command>();
        auto k       = context->kernel();

        k->setKernelDimensions(1);

        const auto one  = std::make_shared<Expression::Expression>(1u);
        const auto zero = std::make_shared<Expression::Expression>(0u);

        auto workitemCount = Expression::literal(workgroupSize * 256);
        k->setWorkgroupSize({workgroupSize, 1, 1});
        k->setWorkitemCount({workitemCount, one, one});
        k->setDynamicSharedMemBytes(zero);

        context->schedule(k->preamble());
        context->schedule(k->prolog());

        auto ldsData = Register::Value::AllocateLDS(
            context.get(),
            DataType::Raw32,
            context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize) / 4);

        auto ldsWithOffset = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        auto workitemIndex = context->kernel()->workitemIndex()[0];

        auto kb = [&]() -> Generator<Instruction> {
            co_yield Expression::generate(
                ldsWithOffset,
                Expression::literal(ldsData->getLDSAllocation()->offset())
                    + workitemIndex->expression()
                          * Expression::literal((4 * strideMultiplier * instrDwords)
                                                % ldsData->getLDSAllocation()->size()),
                context.get());

            auto ldsDst = Register::Value::Placeholder(
                context.get(),
                Register::Type::Vector,
                DataType::Raw32,
                248, // Leave a few leftover for indexing
                Register::AllocationOptions{.contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                            .alignment            = instrDwords});
            co_yield ldsDst->allocate();

            auto s0 = Register::Value::Placeholder(
                context.get(), Register::Type::Scalar, DataType::UInt32, 1);
            auto s1 = Register::Value::Placeholder(
                context.get(), Register::Type::Scalar, DataType::UInt32, 1);
            co_yield s0->allocate();
            co_yield s1->allocate();

            co_yield context->mem()->barrier({});

            int counter = 0;
            for(int i = 0; i < 14; ++i)
            {
                const auto [start, end]
                    = getAlignedSubset(ldsDst->registerCount(), instrDwords, counter++);
                auto dstRegs = ldsDst->subset(Generated(iota(start, end)));
                if(write)
                    co_yield context->mem()->storeLocal(ldsWithOffset, dstRegs, 0, 4 * instrDwords);
                else
                    co_yield context->mem()->loadLocal(dstRegs, ldsWithOffset, 0, 4 * instrDwords);
            }

            for(int i = 1; i < 8; ++i)
            {
                for(int k = 0; k < 4; ++k)
                {
                    const auto [start, end]
                        = getAlignedSubset(ldsDst->registerCount(), instrDwords, counter++);
                    auto dstRegs = ldsDst->subset(Generated(iota(start, end)));
                    if(write)
                        co_yield context->mem()->storeLocal(
                            ldsWithOffset, dstRegs, 0, 4 * instrDwords);
                    else
                        co_yield context->mem()->loadLocal(
                            dstRegs, ldsWithOffset, 0, 4 * instrDwords);
                }
                for(int j = 0; j < i; ++j)
                {
                    co_yield generateOp<Expression::Add>(s0, s0, s1);
                }
            }
        };

        auto scheduler = Component::GetNew<Scheduling::Scheduler>(
            std::make_tuple(Scheduling::SchedulerProcedure::Priority,
                            Scheduling::CostFunction::LinearWeighted,
                            context.get()));

        std::vector<Instruction> instructions;
        for(auto inst : kb())
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                inst.setAddresses(baseAddresses);
            context->schedule(inst);
            instructions.push_back(inst);
        }
        instructions.push_back(
            Instruction("s_endpgm", {}, {}, {}, "")); // postamble too much other stuff

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

        // Filter instructions and verify alignment with profiler data
        const auto filteredInstructions
            = filterAndVerifyInstructions(instructions, allLatencies[0]);

        const auto infoStr = formatLatencyComparison(filteredInstructions, allLatencies[0]);
        INFO(infoStr);

        { // All latencies have same number of instructions
            size_t expectedSize = allLatencies[0].size();
            REQUIRE(std::all_of(
                allLatencies.begin(), allLatencies.end(), [expectedSize](const auto& latencies) {
                    return latencies.size() == expectedSize;
                }));
        }

        std::vector<std::tuple<std::string, size_t>> medianLatencies;
        for(size_t i = 0; i < filteredInstructions.size(); ++i)
        {
            std::vector<uint64_t> latenciesPerRun;
            for(const auto& runLatencies : allLatencies)
            {
                latenciesPerRun.push_back(runLatencies[i].meanLatency());
            }
            auto       medianLatency = median_of_odd_elements(latenciesPerRun);
            const auto instrString   = allLatencies[0][i].instruction;
            medianLatencies.push_back(std::make_tuple(instrString, medianLatency));
        }

        if(testIndividual)
        {
            Log::info(infoStr);
            Log::info(context.output());
        }

        int totalAbsoluteDelta       = 0;
        int totalDelta               = 0;
        int incorrectPredictionCount = 0;

        for(size_t i = 0; i < filteredInstructions.size() - 1; ++i) // exclude s_endpgm
        {
            const auto& inst = filteredInstructions[i];
            using namespace Scheduling::LDSBankModel;

            int modelLatency = inst.totalCycles() * 4;

            int actualLatency = std::get<1>(medianLatencies[i]);
            int delta         = actualLatency - modelLatency;

            if(write && instrDwords == 4)
            { // ds_write_b128 cycles between +0/+12 cycles at steady state
                if(delta > 12)
                {
                    incorrectPredictionCount++;
                    totalDelta += delta;
                }
                totalAbsoluteDelta += std::max(0, std::abs(delta) - 12);
            }
            else
            {
                totalDelta += delta;
                totalAbsoluteDelta += std::abs(delta);
                if(delta != 0)
                {
                    incorrectPredictionCount++;
                }
            }
        }

        INFO(fmt::format("Total absolute delta: {}, Incorrect predictions: {}/{}",
                         totalAbsoluteDelta,
                         incorrectPredictionCount,
                         filteredInstructions.size() - 1));

        /* Sometimes as steady state is reached, there are deltas during transition
        ds_read_b128 v[32:35], v1, model 4, profiler 4, delta 0
        ds_read_b128 v[36:39], v1, model 8, profiler 4, delta -4
        ds_read_b128 v[40:43], v1, model 32, profiler 24, delta -8
        ds_read_b128 v[44:47], v1, model 32, profiler 32, delta 0
        */
        CHECK(totalAbsoluteDelta <= 20);
        CHECK_THAT(totalDelta, Catch::Matchers::WithinAbs(0, 20));
        CHECK(incorrectPredictionCount <= 4);
    }
}
