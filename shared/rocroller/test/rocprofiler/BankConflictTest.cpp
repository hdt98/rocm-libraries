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

#include <common/Scheduling.hpp>

#include "../catch/TestContext.hpp"
#include "../catch/TestKernels.hpp"
#include "Agent.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace rocRoller;

const int NUM_RUNS = 5; // Should be odd, as median is used

class LDSBankConflictTestKernel : public AssemblyTestKernel
{
public:
    LDSBankConflictTestKernel(ContextPtr context,
                              uint32_t   workgroupSize,
                              size_t     instrDwords,
                              size_t     strideMultiplier,
                              bool       write)
        : AssemblyTestKernel(context)
        , m_workgroupSize(workgroupSize)
        , m_instrDwords(instrDwords)
        , m_strideMultiplier(strideMultiplier)
        , m_write(write)
    {
        auto k = m_context->kernel();
        k->setKernelDimensions(1);

        const auto one  = std::make_shared<Expression::Expression>(1u);
        const auto zero = std::make_shared<Expression::Expression>(0u);

        auto workitemCount = Expression::literal(m_workgroupSize * 256 * 32);
        k->setWorkgroupSize({m_workgroupSize, 1, 1});
        k->setWorkitemCount({workitemCount, one, one});
        k->setDynamicSharedMemBytes(zero);
    }

    void operator()()
    {
        KernelInvocation invocation{{m_workgroupSize * 256 * 32, 1, 1}, {m_workgroupSize, 1, 1}, 0};
        AssemblyTestKernel::operator()(invocation);
    }

protected:
    void generate() override
    {
        auto k = m_context->kernel();

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            const auto alignment = static_cast<int>(m_instrDwords);
            const auto regCount  = 256 - 8; // leave a few

            auto dst = Register::Value::Placeholder(
                m_context,
                Register::Type::Vector,
                DataType::Raw32,
                regCount,
                Register::AllocationOptions{
                    .contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                    .alignment            = alignment,
                });
            co_yield dst->allocate();

            auto lds = Register::Value::AllocateLDS(
                m_context,
                DataType::Raw32,
                m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize) / 4);
            auto ldsWithOffset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            auto workitemIndex = m_context->kernel()->workitemIndex()[0];
            co_yield Expression::generate(
                ldsWithOffset,
                Expression::literal(lds->getLDSAllocation()->offset())
                    + workitemIndex->expression()
                          * Expression::literal((4 * m_strideMultiplier * alignment)
                                                    % lds->getLDSAllocation()->size(),
                                                resultType(workitemIndex->expression()).varType),
                m_context);

            co_yield m_context->mem()->barrier({});

            for(int i = 0; i < ITERS; ++i)
            {
                const auto [start, end] = getAlignedSubset(regCount, m_instrDwords, i);
                const auto numBytes     = m_instrDwords * 4;

                if(m_write)
                {
                    co_yield m_context->mem()->storeLocal(
                        ldsWithOffset, dst->subset(Generated(iota(start, end))), 0, numBytes);
                }
                else
                {
                    co_yield m_context->mem()->loadLocal(
                        dst->subset(Generated(iota(start, end))), ldsWithOffset, 0, numBytes);
                }
            }
            co_yield Instruction::Wait(WaitCount::DSCnt(m_context->targetArchitecture(), 1));
            co_yield Instruction::Wait(WaitCount::DSCnt(m_context->targetArchitecture(), 0));
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

private:
    static constexpr int ITERS = 32;

    uint32_t m_workgroupSize;
    size_t   m_instrDwords;
    size_t   m_strideMultiplier;
    bool     m_write;
};

TEST_CASE("LDS bank model with bank conflicts", "[rocprofiler][gpu]")
{
    using namespace Scheduling::LDSBankModel;

    constexpr auto workgroupSize = 64u;

    const std::vector<int>  instrSizes      = {4}; // b32, b64, b128
    const std::vector<int>  strides         = {1}; // between threads
    const std::vector<bool> writeOperations = {false};

    for(auto instrDwords : instrSizes)
    {
        for(auto strideMultiplier : strides)
        {
            for(auto write : writeOperations)
            {
                const auto name = "lds_microkernel_" + std::to_string(instrDwords * 32) + "b_stride"
                                  + std::to_string(strideMultiplier) + "_"
                                  + (write ? "write" : "read");

                DYNAMIC_SECTION(name)
                {
                    rocRoller::profiler::reset();

                    auto context = TestContext::ForTestDevice({}, name);

                    if(not context->targetArchitecture().target().isCDNA35GPU())
                    {
                        SKIP("LDS Bank Model only implemented for CDNA 3.5 GPUs");
                    }

                    LDSBankConflictTestKernel kernel(
                        context.get(), workgroupSize, instrDwords, strideMultiplier, write);

                    std::vector<std::vector<rocRoller::profiler::InstructionProfile>> allLatencies;

                    for(int run = 0; run < NUM_RUNS; ++run)
                    {
                        const auto latencies
                            = rocRoller::profiler::loopUntilDispatchData([&]() { kernel(); });
                        allLatencies.push_back(latencies);

                        INFO("Run " << (run + 1) << ": " << toString(latencies));
                        Log::debug("Run " + std::to_string(run + 1) + ": " + toString(latencies));

                        REQUIRE(latencies.size() == 37);
                    }

                    GPUArchitectureGFX gfx = context->targetArchitecture().target().gfx;

                    auto baseAddresses
                        = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

                    RuntimeLDSInstruction ldsinstr;
                    ldsinstr.memoryOp.direction = write ? LdsDirection::Write : LdsDirection::Read;
                    ldsinstr.dwords             = instrDwords;
                    ldsinstr.baseAddresses      = baseAddresses;

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
                               || (!write && data.instruction.find("ds_read") != std::string::npos))
                            {
                                maxLdsInstrCycles = std::max(maxLdsInstrCycles, data.meanLatency());
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

                    uint64_t actualMaxLdsInstrCycles = median_of_odd_elements(ldsInstrCyclesPerRun);
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

                    CHECK_THAT(actualLastSWaitcntCycles,
                               Catch::Matchers::WithinAbs(predictedCycles, 1ul));
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

TEST_CASE("Weave LDS and nops", "[rocprofiler][scheduler]")
{
    using namespace Scheduling::LDSBankModel;

    constexpr auto workgroupSize = 64u;

    int instrDwords;
    int strideMultiplier;
    int write;

    constexpr auto testIndividual = false;
    if(testIndividual)
    {
        instrDwords      = GENERATE(1);
        strideMultiplier = GENERATE(8);
        write            = GENERATE(false);
    }
    else
    {
        instrDwords      = GENERATE(1, 2, 4);
        strideMultiplier = GENERATE(1, 2, 4, 8);
        write            = GENERATE(true, false);
    }

    const auto baseAddresses = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

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

        // Filter for instructions with opcodes
        std::vector<Instruction> filteredInstructions;
        for(const auto& inst : instructions)
        {
            if(not inst.toString(LogLevel::Terse).empty()) // isCommentOnly did not work
                filteredInstructions.push_back(inst);
        }
        {
            std::stringstream deltas;
            for(size_t i = 0; i < std::max(filteredInstructions.size(), allLatencies[0].size());
                ++i)
            {
                const auto& inst
                    = i < filteredInstructions.size() ? filteredInstructions[i] : Instruction();
                const auto& profile = i < allLatencies[0].size()
                                          ? allLatencies[0][i]
                                          : rocRoller::profiler::InstructionProfile();

                deltas << fmt::format(
                    "{}: filtered {}, profiler {}\n", i, inst.getOpCode(), profile.instruction);
            }
            INFO(deltas.str());
            REQUIRE(filteredInstructions.size() == allLatencies[0].size());
        }

        std::stringstream infoMessage;
        for(size_t i = 0; i < std::min(filteredInstructions.size(), allLatencies[0].size()); ++i)
        {
            const auto& inst    = filteredInstructions[i];
            const auto& profile = allLatencies[0][i];

            using namespace Scheduling::LDSBankModel;

            int modelLatency = inst.totalCycles() * 4;
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                modelLatency = getInstructionIssueCycles(write ? MemoryOpLDS{LdsDirection::Write}
                                                               : MemoryOpLDS{LdsDirection::Read},
                                                         instrDwords)
                               + inst.peekedStatus().stallCycles * 4;

            infoMessage << fmt::format("{}, model {}, profiler {}, delta {}\n",
                                       profile.instruction,
                                       modelLatency,
                                       profile.meanLatency(),
                                       static_cast<int>(profile.meanLatency()) - modelLatency);
        }
        const auto infoStr = testIndividual ? infoMessage.str() : "";
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

        for(size_t i = 0; i < filteredInstructions.size(); ++i)
        {
            const auto& inst = filteredInstructions[i];
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
            {
                using namespace Scheduling::LDSBankModel;

                const auto medianLatency = medianLatencies[i];
                int        modelLatency  = inst.totalCycles() * 4;
                modelLatency = getInstructionIssueCycles(write ? MemoryOpLDS{LdsDirection::Write}
                                                               : MemoryOpLDS{LdsDirection::Read},
                                                         instrDwords)
                               + inst.peekedStatus().stallCycles * 4;

                INFO(fmt::format("{}, profiler {}, model {}, delta {}",
                                 std::get<0>(medianLatency),
                                 std::get<1>(medianLatency),
                                 modelLatency,
                                 static_cast<int>(std::get<1>(medianLatency)) - modelLatency));

                if(write && instrDwords == 4)
                {
                    CHECK_THAT(std::get<1>(medianLatency),
                               Catch::Matchers::WithinAbs(modelLatency, 12ul));
                }
                else
                {
                    CHECK_THAT(std::get<1>(medianLatency),
                               Catch::Matchers::WithinAbs(modelLatency, 0ul));
                }
            }
        }
        if(testIndividual)
            Log::trace(context.output());
    }
}

TEST_CASE("Scheduling LDS", "[rocprofiler][scheduler]")
{
    using namespace Scheduling::LDSBankModel;

    Settings::getInstance()->set(Settings::SchedulerCost,
                                 Scheduling::CostFunction::LinearWeightedSimple);

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

    constexpr auto workgroupSize = 64u;
    auto           workitemCount = Expression::literal(workgroupSize * 256);
    k->setWorkgroupSize({workgroupSize, 1, 1});
    k->setWorkitemCount({workitemCount, one, one});
    k->setDynamicSharedMemBytes(zero);

    context->schedule(k->preamble());
    context->schedule(k->prolog());

    // ds_read_b64 with stride multiplier of 4
    const auto instrDwords      = 4;
    const auto strideMultiplier = 4;
    const auto baseAddresses    = generateLDSAddresses(64, strideMultiplier, instrDwords);

    auto ldsData = Register::Value::AllocateLDS(
        context.get(),
        DataType::Raw32,
        context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize) / 4);

    auto ldsWithOffset
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::UInt32, 1);
    auto workitemIndex = context->kernel()->workitemIndex()[0];

    auto ldsStream = [&]() -> Generator<Instruction> {
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
            64,
            Register::AllocationOptions{.contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                        .alignment            = instrDwords});
        co_yield ldsDst->allocate();

        co_yield context->mem()->barrier({});

        for(int i = 0; i < 16; ++i)
        {
            auto dstRegs = ldsDst->subset(Generated(iota(i * 2, (i + 1) * 2)));
            co_yield context->mem()->storeLocal(ldsWithOffset, dstRegs, 0, 8);
        }
    };

    auto aluStream = [&]() -> Generator<Instruction> {
        auto v0 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        auto v1 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        co_yield v0->allocate();
        co_yield v1->allocate();

        for(int i = 0; i < 32; ++i)
        {
            co_yield generateOp<Expression::Add>(v0, v0, v1);
        }
    };

    auto scheduler = Component::GetNew<Scheduling::Scheduler>(
        std::make_tuple(Scheduling::SchedulerProcedure::Priority,
                        Scheduling::CostFunction::LinearWeightedSimple,
                        context.get()));

    auto withAddresses = [&](Generator<Instruction> stream) -> Generator<Instruction> {
        for(auto inst : stream)
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                inst.setAddresses(baseAddresses);
            co_yield inst;
        }
    };

    std::vector<Generator<Instruction>> streams;
    streams.push_back(withAddresses(ldsStream()));
    streams.push_back(aluStream());

    std::vector<Instruction> instructions;
    for(auto inst : (*scheduler)(streams))
    {
        context->schedule(inst);
        instructions.push_back(inst);
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

    // Filter for instructions with opcodes
    std::vector<Instruction> filteredInstructions;
    for(const auto& inst : instructions)
    {
        if(not inst.toString(LogLevel::Terse).empty())
            filteredInstructions.push_back(inst);
    }

    // Show weaved together
    for(size_t i = 0; i < std::min(filteredInstructions.size(), allLatencies[0].size()); ++i)
    {
        const auto& inst    = filteredInstructions[i];
        const auto& profile = allLatencies[0][i];

        Log::info(fmt::format("{}: model {}, profiler mean {}, with NONE {}",
                              profile.instruction,
                              inst.peekedStatus().stallCycles * 4,
                              profile.meanLatency(),
                              profile.meanLatencyWithPrecedingNone()));
    }
}
