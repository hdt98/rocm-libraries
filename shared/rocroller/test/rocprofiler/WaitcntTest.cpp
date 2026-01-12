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
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>
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

class LDSWaitcntWeaveKernel : public LDSTestKernelBase
{
public:
    LDSWaitcntWeaveKernel(ContextPtr                 context,
                          uint32_t                   workgroupSize,
                          size_t                     instrDwords,
                          size_t                     strideMultiplier,
                          const std::vector<size_t>& baseAddresses,
                          bool                       write)
        : LDSTestKernelBase(
            context, workgroupSize, instrDwords, strideMultiplier, baseAddresses, write)
    {
    }

protected:
    Generator<Instruction> generateKernelBody() override
    {
        int counter = 0;
        // Too many iterations, then latency of s_barrier gets noticeable
        for(int i = 1; i <= 4; ++i)
        {
            for(int k = 0; k < i; ++k)
            {
                const auto [start, end]
                    = getAlignedSubset(m_ldsDst->registerCount(), m_instrDwords, counter++);
                auto dstRegs = m_ldsDst->subset(Generated(iota(start, end)));
                if(m_write)
                    co_yield m_context->mem()->storeLocal(
                        m_ldsWithOffset, dstRegs, 0, 4 * m_instrDwords);
                else
                    co_yield m_context->mem()->loadLocal(
                        dstRegs, m_ldsWithOffset, 0, 4 * m_instrDwords);
            }
            co_yield Instruction::Wait(WaitCount::DSCnt(m_context->targetArchitecture(), 0));
            co_yield m_context->mem()->barrier({});
        }
    }
};

TEST_CASE("Weave multiple LDS and waitcnt 0",
          "[rocprofiler][lds-model][lds-model-waitcnt][lds-model-waitcnt-not-steady][gpu]")
{
    // Expect 391 passed : 4 failed
    // Mainly affected by waitcnt queue values
    /*
    ds_read_b128 v[60:63], v1, model 16, profiler 16, delta 0
    s_waitcnt lgkmcnt(0), model 168, profiler 164, delta -4
    ds_read_b128 v[64:67], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[68:71], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 72, profiler 72, delta 0
    ds_read_b128 v[72:75], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[76:79], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[80:83], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 84, profiler 84, delta 0
    ds_read_b128 v[84:87], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[88:91], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[92:95], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[96:99], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 96, profiler 96, delta 0
    ds_read_b128 v[100:103], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[104:107], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[108:111], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[112:115], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[116:119], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 108, profiler 108, delta 0
    */
    using namespace Scheduling::LDSBankModel;
    Settings::getInstance()->set(Settings::DSObserver, DSObserverType::WeightlessDSMemObserver);

    const auto workgroupSize = GENERATE(64u, 128u, 256u);

    int instrDwords      = GENERATE(1, 2, 4);
    int strideMultiplier = GENERATE(1, 2, 4, 8, 16);
    int write            = GENERATE(true, false);

    const auto baseAddresses = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

    rocRoller::profiler::reset();

    auto context = TestContext::ForTestDevice({}, "");

    if(not context->targetArchitecture().target().isCDNA35GPU())
    {
        SKIP("Currently only testing on gfx950");
    }

    LDSWaitcntWeaveKernel kernel(
        context.get(), workgroupSize, instrDwords, strideMultiplier, baseAddresses, write);

    SECTION(kernel.getSectionName())
    {

        auto result = runKernelAndCollectLatencies(context, kernel);
        INFO(result.infoStr);
        const auto& filteredInstructions = result.filteredInstructions;
        const auto& medianLatencies      = result.medianLatencies;

        auto analysis = analyzeLatencyDeltas(filteredInstructions, medianLatencies);

        INFO(fmt::format("Total delta: {}, Total absolute delta: {}, Incorrect predictions: {}/{}",
                         analysis.totalDelta,
                         analysis.totalAbsoluteDelta,
                         analysis.incorrectPredictionCount,
                         filteredInstructions.size() - 1));

        if(workgroupSize == 64u)
        {
            if(write && instrDwords == 2 && strideMultiplier == 1)
            {
                CHECK((analysis.incorrectPredictionCount <= 8
                       || std::abs(analysis.totalDelta) <= 35));
            }
            else if(write && instrDwords == 4)
            {
                CHECK((analysis.incorrectPredictionCount <= 20
                       || std::abs(analysis.totalDelta) <= 180));
            }
            else if(instrDwords == 1 && strideMultiplier == 1)
            {
                CHECK((analysis.incorrectPredictionCount <= 9
                       || std::abs(analysis.totalDelta) <= 32));
            }
            else if(instrDwords == 2 && strideMultiplier == 1)
            {
                CHECK((analysis.incorrectPredictionCount <= 8
                       || std::abs(analysis.totalDelta) <= 35));
            }
            else if(instrDwords == 4 && strideMultiplier == 1)
            {
                CHECK((analysis.incorrectPredictionCount <= 8
                       || std::abs(analysis.totalDelta) <= 35));
            }
            else
            {
                CHECK(
                    (analysis.incorrectPredictionCount <= 4 || std::abs(analysis.totalDelta) <= 0));
            }
        }
        else
        {
            CHECK_THAT(analysis.totalDelta, Catch::Matchers::WithinAbs(0, 500));
        }
    }
}

class WeaveLdsAndNonzeroWaitcnt : public LDSTestKernelBase
{
public:
    WeaveLdsAndNonzeroWaitcnt(ContextPtr                 context,
                              uint32_t                   workgroupSize,
                              size_t                     instrDwords,
                              size_t                     strideMultiplier,
                              const std::vector<size_t>& baseAddresses,
                              bool                       write)
        : LDSTestKernelBase(
            context, workgroupSize, instrDwords, strideMultiplier, baseAddresses, write)
    {
    }

protected:
    Generator<Instruction> generateKernelBody() override
    {
        int counter = 0;
        for(int i = 0; i < 8; ++i)
        {
            for(int k = 0; k < 16; ++k)
            {
                const auto [start, end]
                    = getAlignedSubset(m_ldsDst->registerCount(), m_instrDwords, counter++);
                auto dstRegs = m_ldsDst->subset(Generated(iota(start, end)));
                if(m_write)
                    co_yield m_context->mem()->storeLocal(
                        m_ldsWithOffset, dstRegs, 0, 4 * m_instrDwords);
                else
                    co_yield m_context->mem()->loadLocal(
                        dstRegs, m_ldsWithOffset, 0, 4 * m_instrDwords);
            }
            co_yield Instruction::Wait(WaitCount::DSCnt(m_context->targetArchitecture(), i));
        }
    }
};

TEST_CASE("Weave LDS and waitcnt at steady state",
          "[rocprofiler][lds-model][lds-model-waitcnt][gpu]")
{
    // Expect 527 passed : 18 failed
    /*
    ...
    s_waitcnt lgkmcnt(2), model 136, profiler 132, delta -4
    ds_read_b128 v[124:127], v1, model 4, profiler 4, delta 0
    ...
    ds_read_b128 v[160:163], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(3), model 120, profiler 116, delta -4
    ds_read_b128 v[164:167], v1, model 4, profiler 4, delta 0
    ...
    ds_read_b128 v[200:203], v1, model 4, profiler 8, delta 4
    s_waitcnt lgkmcnt(4), model 104, profiler 100, delta -4
    ...
    */
    using namespace Scheduling::LDSBankModel;
    Settings::getInstance()->set(Settings::DSObserver, DSObserverType::WeightlessDSMemObserver);

    const auto workgroupSize = 64u;

    int instrDwords      = GENERATE(1, 2, 4);
    int strideMultiplier = GENERATE(1, 2, 4, 8, 16);
    int write            = GENERATE(true, false);

    const auto baseAddresses = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

    rocRoller::profiler::reset();

    auto context = TestContext::ForTestDevice({}, "");

    if(not context->targetArchitecture().target().isCDNA35GPU())
    {
        SKIP("Currently only testing on gfx950");
    }

    WeaveLdsAndNonzeroWaitcnt kernel(
        context.get(), workgroupSize, instrDwords, strideMultiplier, baseAddresses, write);

    SECTION(kernel.getSectionName())
    {

        auto result = runKernelAndCollectLatencies(context, kernel);
        INFO(result.infoStr);
        const auto& filteredInstructions = result.filteredInstructions;
        const auto& medianLatencies      = result.medianLatencies;

        auto analysis = analyzeLatencyDeltas(filteredInstructions, medianLatencies);

        INFO(fmt::format("Total delta: {}, Total absolute delta: {}, Incorrect predictions: {}/{}",
                         analysis.totalDelta,
                         analysis.totalAbsoluteDelta,
                         analysis.incorrectPredictionCount,
                         filteredInstructions.size() - 1));

        if(!write && instrDwords == 1 && strideMultiplier >= 4)
        {
            CHECK((analysis.incorrectPredictionCount <= 21 || std::abs(analysis.totalDelta) <= 85));
        }
        else if(!write && instrDwords == 2 && strideMultiplier >= 8)
        {
            CHECK(
                (analysis.incorrectPredictionCount <= 40 || std::abs(analysis.totalDelta) <= 1100));
        }
        else if(!write && instrDwords == 2 && strideMultiplier >= 4)
        {
            CHECK((analysis.incorrectPredictionCount <= 21 || std::abs(analysis.totalDelta) <= 85));
        }
        else if(!write && instrDwords == 2 && strideMultiplier == 1)
        {
            CHECK((analysis.incorrectPredictionCount <= 9 || std::abs(analysis.totalDelta) <= 50));
        }
        else if(!write && instrDwords == 4 && strideMultiplier >= 8)
        {
            CHECK(
                (analysis.incorrectPredictionCount <= 25 || std::abs(analysis.totalDelta) <= 920));
        }
        else if(!write && instrDwords == 4 && strideMultiplier >= 2)
        {
            CHECK(
                (analysis.incorrectPredictionCount <= 32 || std::abs(analysis.totalDelta) <= 230));
        }
        else if(!write && instrDwords == 4 && strideMultiplier == 1)
        {
            CHECK(
                (analysis.incorrectPredictionCount <= 20 || std::abs(analysis.totalDelta) <= 110));
        }
        else if(write && instrDwords == 4)
        {
            CHECK(
                (analysis.incorrectPredictionCount <= 80 || std::abs(analysis.totalDelta) <= 1300));
        }
        else if(write)
        {
            CHECK(
                (analysis.incorrectPredictionCount <= 43 || std::abs(analysis.totalDelta) <= 500));
        }
        else if(instrDwords == 1 && strideMultiplier == 1)
        {
            CHECK((analysis.incorrectPredictionCount <= 8 || std::abs(analysis.totalDelta) <= 50));
        }
        else
        {
            CHECK((analysis.incorrectPredictionCount <= 4 || std::abs(analysis.totalDelta) <= 0));
        }
    }
}

class WeaveLdsAndWaitcntZeroNoSaturation : public LDSTestKernelBase
{
public:
    WeaveLdsAndWaitcntZeroNoSaturation(ContextPtr                 context,
                                       uint32_t                   workgroupSize,
                                       size_t                     instrDwords,
                                       size_t                     strideMultiplier,
                                       const std::vector<size_t>& baseAddresses,
                                       bool                       write)
        : LDSTestKernelBase(
            context, workgroupSize, instrDwords, strideMultiplier, baseAddresses, write)
    {
    }

protected:
    Generator<Instruction> generateKernelBody() override
    {
        int counter = 0;

        const auto scheduleLds = [&]() -> Generator<Instruction> {
            const auto [start, end]
                = getAlignedSubset(m_ldsDst->registerCount(), m_instrDwords, counter++);
            auto dstRegs = m_ldsDst->subset(Generated(iota(start, end)));
            if(m_write)
                co_yield m_context->mem()->storeLocal(
                    m_ldsWithOffset, dstRegs, 0, 4 * m_instrDwords);
            else
                co_yield m_context->mem()->loadLocal(
                    dstRegs, m_ldsWithOffset, 0, 4 * m_instrDwords);
        };

        for(int i = 1; i <= 8; ++i)
        {
            for(int k = 0; k < i; ++k)
            {
                co_yield scheduleLds();
            }
            for(int w = 0; w < i; ++w)
            {
                co_yield Instruction::Wait(
                    WaitCount::DSCnt(m_context->targetArchitecture(), i - w - 1));
            }
        }
    }
};

TEST_CASE("Weave LDS and waitcnt not steady state",
          "[rocprofiler][lds-model][lds-model-waitcnt][lds-model-waitcnt-not-steady][gpu]")
{
    // Expect 541 passed : 4 failed
    /*
    ds_write_b32 v1, v2, model 8, profiler 8, delta 0
  * s_waitcnt lgkmcnt(0), model 60, profiler 52, delta -8
    ds_write_b32 v1, v3, model 8, profiler 8, delta 0
    ds_write_b32 v1, v4, model 8, profiler 8, delta 0
  * s_waitcnt lgkmcnt(1), model 52, profiler 44, delta -8
  * s_waitcnt lgkmcnt(0), model 16, profiler 8, delta -8
    ds_write_b32 v1, v5, model 8, profiler 8, delta 0
    ds_write_b32 v1, v6, model 8, profiler 8, delta 0
    ds_write_b32 v1, v7, model 8, profiler 8, delta 0
  * s_waitcnt lgkmcnt(2), model 44, profiler 36, delta -8
  * s_waitcnt lgkmcnt(1), model 16, profiler 8, delta -8
  * s_waitcnt lgkmcnt(0), model 16, profiler 8, delta -8
    ds_write_b32 v1, v8, model 8, profiler 8, delta 0
    ds_write_b32 v1, v9, model 8, profiler 8, delta 0
    ds_write_b32 v1, v10, model 8, profiler 8, delta 0
    ds_write_b32 v1, v11, model 8, profiler 8, delta 0
  * s_waitcnt lgkmcnt(3), model 36, profiler 28, delta -8
  * s_waitcnt lgkmcnt(2), model 16, profiler 8, delta -8
  * s_waitcnt lgkmcnt(1), model 16, profiler 8, delta -8
  * s_waitcnt lgkmcnt(0), model 16, profiler 8, delta -8
    */
    using namespace Scheduling::LDSBankModel;
    Settings::getInstance()->set(Settings::DSObserver, DSObserverType::WeightlessDSMemObserver);

    const auto workgroupSize = 64u;

    int instrDwords      = GENERATE(1, 2, 4);
    int strideMultiplier = GENERATE(1, 2, 4, 8, 16);
    int write            = GENERATE(true, false);

    const auto baseAddresses = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

    rocRoller::profiler::reset();

    auto context = TestContext::ForTestDevice({}, "");

    if(not context->targetArchitecture().target().isCDNA35GPU())
    {
        SKIP("Currently only testing on gfx950");
    }

    WeaveLdsAndWaitcntZeroNoSaturation kernel(
        context.get(), workgroupSize, instrDwords, strideMultiplier, baseAddresses, write);

    SECTION(kernel.getSectionName())
    {

        auto result = runKernelAndCollectLatencies(context, kernel);
        INFO(result.infoStr);
        const auto& filteredInstructions = result.filteredInstructions;
        const auto& medianLatencies      = result.medianLatencies;

        auto analysis = analyzeLatencyDeltas(filteredInstructions, medianLatencies);

        INFO(fmt::format("Total delta: {}, Total absolute delta: {}, Incorrect predictions: {}/{}",
                         analysis.totalDelta,
                         analysis.totalAbsoluteDelta,
                         analysis.incorrectPredictionCount,
                         filteredInstructions.size() - 1));

        if(write && instrDwords == 2 && strideMultiplier == 1)
        {
            CHECK((analysis.incorrectPredictionCount <= 9 || std::abs(analysis.totalDelta) <= 35));
        }
        else if(write && instrDwords == 4)
        {
            CHECK(
                (analysis.incorrectPredictionCount <= 34 || std::abs(analysis.totalDelta) <= 240));
        }
        else if(!write && instrDwords == 2 && strideMultiplier == 1)
        {
            CHECK((analysis.incorrectPredictionCount <= 10 || std::abs(analysis.totalDelta) <= 45));
        }
        else if(!write && instrDwords == 4 && strideMultiplier == 1)
        {
            CHECK((analysis.incorrectPredictionCount <= 10 || std::abs(analysis.totalDelta) <= 45));
        }
        else if(instrDwords == 1 && strideMultiplier == 1)
        {
            CHECK((analysis.incorrectPredictionCount <= 9 || std::abs(analysis.totalDelta) <= 40));
        }
        else
        {
            CHECK((analysis.incorrectPredictionCount <= 4 || std::abs(analysis.totalDelta) <= 0));
        }
    }
}
