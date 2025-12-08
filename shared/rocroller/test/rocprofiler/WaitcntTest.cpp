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

class WaitcntTestKernel : public AssemblyTestKernel
{
public:
    WaitcntTestKernel(ContextPtr                 context,
                      uint32_t                   workgroupSize,
                      size_t                     instrDwords,
                      size_t                     strideMultiplier,
                      const std::vector<size_t>& baseAddresses,
                      bool                       write)
        : AssemblyTestKernel(context)
        , m_workgroupSize(workgroupSize)
        , m_instrDwords(instrDwords)
        , m_strideMultiplier(strideMultiplier)
        , m_baseAddresses(baseAddresses)
        , m_write(write)
    {
        auto k = m_context->kernel();
        k->setKernelDimensions(1);

        const auto one  = std::make_shared<Expression::Expression>(1u);
        const auto zero = std::make_shared<Expression::Expression>(0u);

        auto workitemCount = Expression::literal(m_workgroupSize * 256);
        k->setWorkgroupSize({m_workgroupSize, 1, 1});
        k->setWorkitemCount({workitemCount, one, one});
        k->setDynamicSharedMemBytes(zero);
    }

    void operator()()
    {
        KernelInvocation    invocation{{m_workgroupSize * 256, 1, 1}, {m_workgroupSize, 1, 1}, 0};
        AssemblyTestKernel::operator()(invocation);
    }

    const std::vector<Instruction>& getInstructions() const
    {
        return m_instructions;
    }

protected:
    void generate() override
    {
        auto k = m_context->kernel();

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            const auto alignment = static_cast<int>(m_instrDwords);
            const auto regCount  = 248; // Leave some registers for other operations

            auto readDst = Register::Value::Placeholder(
                m_context,
                Register::Type::Vector,
                DataType::Raw32,
                regCount,
                Register::AllocationOptions{
                    .contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                    .alignment            = alignment,
                });
            co_yield readDst->allocate();

            auto lds = Register::Value::AllocateLDS(
                m_context,
                DataType::Raw32,
                m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize) / 4);

            auto ldsAddr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            auto workitemIndex = m_context->kernel()->workitemIndex()[0];

            co_yield Expression::generate(
                ldsAddr,
                Expression::literal(lds->getLDSAllocation()->offset())
                    + workitemIndex->expression()
                          * Expression::literal((4 * m_strideMultiplier * m_instrDwords)
                                                    % lds->getLDSAllocation()->size(),
                                                resultType(workitemIndex->expression()).varType),
                m_context);

            co_yield m_context->mem()->barrier({});

            for(int i = 0; i < 20; i++)
            {
                const auto [start, end] = getAlignedSubset(regCount, m_instrDwords, i);
                const auto numBytes     = m_instrDwords * 4;

                if(m_write)
                {
                    co_yield m_context->mem()->storeLocal(
                        ldsAddr, readDst->subset(Generated(iota(start, end))), 0, numBytes);
                }
                else
                {
                    co_yield m_context->mem()->loadLocal(
                        readDst->subset(Generated(iota(start, end))), ldsAddr, 0, numBytes);
                }
            }

            {
                int i = 18;
                do
                {
                    i /= 2;
                    co_yield Instruction::Wait(
                        WaitCount::DSCnt(m_context->targetArchitecture(), i));
                } while(i > 0);
            }
        };

        m_instructions.clear();
        for(auto inst : kb())
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                inst.setAddresses(m_baseAddresses);
            m_context->schedule(inst);
            m_instructions.push_back(inst);
        }
        m_instructions.push_back(Instruction("s_endpgm", {}, {}, {}, ""));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

private:
    uint32_t                 m_workgroupSize;
    size_t                   m_instrDwords;
    size_t                   m_strideMultiplier;
    std::vector<size_t>      m_baseAddresses;
    std::vector<Instruction> m_instructions;
    bool                     m_write;
};

TEST_CASE("Cycle count predictions with waitcnt", "[rocprofiler][gpu][lds-model]")
{
    using namespace Scheduling::LDSBankModel;

    constexpr auto workgroupSize    = 64u;
    constexpr auto strideMultiplier = 8u;

    auto instrDwords = GENERATE(2);
    auto write       = GENERATE(true);

    SECTION("Waitcnt sequence 7 to 0, dwords " + std::to_string(instrDwords) + ", "
            + (write ? "write" : "read"))
    {
        rocRoller::profiler::reset();

        auto context = TestContext::ForTestDevice({}, "waitcnt_sequence");

        if(not context->targetArchitecture().target().isCDNAGPU())
        {
            SKIP("Test designed for CDNA architectures");
        }

        const auto baseAddresses
            = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

        WaitcntTestKernel kernel(
            context.get(), workgroupSize, instrDwords, strideMultiplier, baseAddresses, write);

        const auto latencies = rocRoller::profiler::loopUntilDispatchData([&]() { kernel(); });

        const auto& instructions = kernel.getInstructions();

        const auto filteredInstructions = filterAndVerifyInstructions(instructions, latencies);

        const auto comparisonStr = formatLatencyComparison(filteredInstructions, latencies);
        INFO(comparisonStr);

        int dsReadCount  = 0;
        int dsWriteCount = 0;
        int waitcntCount = 0;

        for(const auto& inst : filteredInstructions)
        {
            const auto& opcode = inst.getOpCode();
            if(opcode.find("ds_read") != std::string::npos)
            {
                dsReadCount++;
            }
            else if(opcode.find("ds_write") != std::string::npos)
            {
                dsWriteCount++;
            }
            else if(opcode.find("s_waitcnt") != std::string::npos)
            {
                waitcntCount++;
            }
        }

        if(write)
        {
            CHECK(dsReadCount == 0);
            CHECK(dsWriteCount == 20);
        }
        else
        {
            CHECK(dsReadCount == 20);
            CHECK(dsWriteCount == 0);
        }
        CHECK(waitcntCount == 8);

        Log::info(context.output());
    }
}
