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

class LDSWaitcntTestKernel : public LDSTestKernelBase
{
public:
    LDSWaitcntTestKernel(ContextPtr                 context,
                         uint32_t                   workgroupSize,
                         size_t                     instrDwords,
                         size_t                     strideMultiplier,
                         const std::vector<size_t>& baseAddresses,
                         bool                       write,
                         int                        iterations = 8)
        : LDSTestKernelBase(
            context, workgroupSize, instrDwords, strideMultiplier, baseAddresses, write)
        , m_iterations(iterations)
    {
    }

protected:
    Generator<Instruction> generateKernelBody() override
    {
        int counter = 0;
        for(int i = 0; i < m_iterations; ++i)
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

            co_yield Instruction::Wait(WaitCount::DSCnt(m_context->targetArchitecture(), 0));
        }
    }

private:
    int m_iterations;
};

class LDSArithmeticWeaveTestKernel : public LDSTestKernelBase
{
public:
    LDSArithmeticWeaveTestKernel(ContextPtr                 context,
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
        auto s0
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::UInt32, 1);
        auto s1
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::UInt32, 1);
        co_yield s0->allocate();
        co_yield s1->allocate();

        int counter = 0;
        for(int i = 0; i < 14; ++i)
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

        for(int i = 1; i < 8; ++i)
        {
            for(int k = 0; k < 4; ++k)
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
            for(int j = 0; j < i; ++j)
            {
                co_yield generateOp<Expression::Add>(s0, s0, s1);
            }
        }
    }
};

TEST_CASE("Weave LDS and waitcnt", "[rocprofiler][scheduler][lds-model][gpu]")
{
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
        CAPTURE(name);

        LDSWaitcntTestKernel kernel(context.get(),
                                    workgroupSize,
                                    instrDwords,
                                    strideMultiplier,
                                    baseAddresses,
                                    write,
                                    iters);

        auto [filteredInstructions, medianLatencies]
            = runKernelAndCollectLatencies(context, kernel, testIndividual);

        int totalAbsoluteDelta       = 0;
        int totalDelta               = 0;
        int incorrectPredictionCount = 0;
        int waitcntInstructionCount  = 0;

        for(size_t i = 0; i < filteredInstructions.size() - 1; ++i)
        {
            const auto& inst = filteredInstructions[i];

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

TEST_CASE("Weave LDS and s_add", "[rocprofiler][scheduler][lds-model][gpu]")
{
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
        LDSArithmeticWeaveTestKernel kernel(
            context.get(), workgroupSize, instrDwords, strideMultiplier, baseAddresses, write);

        auto [filteredInstructions, medianLatencies]
            = runKernelAndCollectLatencies(context, kernel, testIndividual);

        int totalAbsoluteDelta       = 0;
        int totalDelta               = 0;
        int incorrectPredictionCount = 0;

        for(size_t i = 0; i < filteredInstructions.size() - 1; ++i)
        {
            const auto& inst = filteredInstructions[i];

            int modelLatency = inst.totalCycles() * 4;

            int actualLatency = std::get<1>(medianLatencies[i]);
            int delta         = actualLatency - modelLatency;

            if(write && instrDwords == 4)
            {
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
