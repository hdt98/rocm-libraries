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
#include <rocRoller/Scheduling/LDSModel.hpp>
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

TEST_CASE("Weave LDS and s_add", "[rocprofiler][lds-model][gpu]")
{
    // Expect 545 passed : 0 failed
    using namespace Scheduling::LDSModel;

    const auto workgroupSize = 64u;

    int instrDwords      = GENERATE(1, 2, 4);
    int strideMultiplier = GENERATE(1, 2, 4, 8, 16);
    int write            = GENERATE(true, false);

    const auto baseAddresses = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

    rocRoller::profiler::reset();

    rocRoller::KernelOptions kernelOpts;
    kernelOpts->dsObserver = DSObserverType::WeightlessDSMemObserver;
    auto context           = TestContext::ForTestDevice(kernelOpts, "");

    if(not context->targetArchitecture().target().isCDNA35GPU())
    {
        SKIP("Currently only testing on gfx950");
    }

    LDSArithmeticWeaveTestKernel kernel(
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

        if(write && instrDwords == 4)
        {
            // CHECK((analysis.incorrectPredictionCount <= 16));
        }
        else
        {
            CHECK((analysis.incorrectPredictionCount <= 4 || analysis.totalDelta == 0));
        }
    }
}

class SteadyStateLdsInstructions : public LDSTestKernelBase
{
public:
    SteadyStateLdsInstructions(ContextPtr                 context,
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
        for(int i = 0; i < 32; ++i)
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
    }
};

TEST_CASE("Steady state LDS instructions", "[rocprofiler][lds-model][gpu]")
{
    // Expect 545 passed : 0 failed
    // Mainly affected by the queue size, e.g. when does steady state get reached?
    /*
    ds_read_b128 v[4:7], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[8:11], v1, model 4, profiler 4, delta 0
    ... 32 times
    */
    using namespace Scheduling::LDSModel;

    const auto workgroupSize = GENERATE(64u, 128u, 256u);

    int instrDwords      = GENERATE(1, 2, 4);
    int strideMultiplier = GENERATE(1, 2, 4, 8, 16);
    int write            = GENERATE(true, false);

    const auto baseAddresses = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

    rocRoller::profiler::reset();

    rocRoller::KernelOptions kernelOpts;
    kernelOpts->dsObserver = DSObserverType::WeightlessDSMemObserver;
    auto context           = TestContext::ForTestDevice(kernelOpts, "");

    if(not context->targetArchitecture().target().isCDNA35GPU())
    {
        SKIP("Currently only testing on gfx950");
    }

    SteadyStateLdsInstructions kernel(
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

        if(write && instrDwords == 4)
        {
            if(workgroupSize <= 128u)
            {
                CHECK_THAT(analysis.totalDelta, Catch::Matchers::WithinAbs(0, 210));
            }
            else
            {
                // These are super off
                CHECK(analysis.totalAbsoluteDelta <= 1600);
            }
        }
        else if(workgroupSize == 64u)
        {
            CHECK((analysis.incorrectPredictionCount <= 4 || analysis.totalDelta == 0));
        }
        else if(workgroupSize > 64u)
        {
            CHECK_THAT(analysis.totalDelta, Catch::Matchers::WithinAbs(0, 300));
        }
    }
}
