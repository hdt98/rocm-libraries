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

#include "Utils.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sstream>

#include <common/Scheduling.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelArguments.hpp>

namespace rocRoller
{
    std::string formatLatencyComparison(const std::vector<Instruction>& filteredInstructions,
                                        const std::vector<profiler::InstructionProfile>& profiles)
    {
        std::stringstream infoMessage;
        for(size_t i = 0; i < std::min(filteredInstructions.size(), profiles.size()); ++i)
        {
            const auto& inst    = filteredInstructions[i];
            const auto& profile = profiles[i];

            int modelLatency = inst.totalCycles() * 4;

            infoMessage << fmt::format("{}, model {}, profiler {}, delta {}\n",
                                       profile.instruction,
                                       modelLatency,
                                       profile.meanLatency(),
                                       static_cast<int>(profile.meanLatency()) - modelLatency);
        }
        return infoMessage.str();
    }

    std::vector<Instruction>
        filterAndVerifyInstructions(const std::vector<Instruction>&                  instructions,
                                    const std::vector<profiler::InstructionProfile>& latencies)
    {
        std::vector<Instruction> filteredInstructions;
        for(const auto& inst : instructions)
        {
            if(not inst.toString(LogLevel::Terse).empty())
                filteredInstructions.push_back(inst);
        }

        std::stringstream deltas;
        for(size_t i = 0; i < std::max(filteredInstructions.size(), latencies.size()); ++i)
        {
            const auto& inst
                = i < filteredInstructions.size() ? filteredInstructions[i] : Instruction();
            const auto& profile
                = i < latencies.size() ? latencies[i] : profiler::InstructionProfile();

            deltas << fmt::format(
                "{}: filtered {}, profiler {}\n", i, inst.getOpCode(), profile.instruction);
        }
        INFO(deltas.str());

        REQUIRE(filteredInstructions.size() == latencies.size());

        return filteredInstructions;
    }

    LDSTestKernelBase::LDSTestKernelBase(ContextPtr                 context,
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

    void LDSTestKernelBase::operator()()
    {
        KernelInvocation    invocation{{m_workgroupSize * 256, 1, 1}, {m_workgroupSize, 1, 1}, 0};
        AssemblyTestKernel::operator()(invocation);
    }

    const std::vector<Instruction>& LDSTestKernelBase::getInstructions() const
    {
        return m_instructions;
    }

    void LDSTestKernelBase::generate()
    {
        auto k = m_context->kernel();

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto ldsData = Register::Value::AllocateLDS(
            m_context,
            DataType::Raw32,
            m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize) / 4);

        m_ldsWithOffset
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::UInt32, 1);
        m_workitemIndex = m_context->kernel()->workitemIndex()[0];

        auto kb = [&]() -> Generator<Instruction> {
            co_yield Expression::generate(
                m_ldsWithOffset,
                Expression::literal(ldsData->getLDSAllocation()->offset())
                    + m_workitemIndex->expression()
                          * Expression::literal((4 * m_strideMultiplier * m_instrDwords)
                                                    % ldsData->getLDSAllocation()->size(),
                                                resultType(m_workitemIndex->expression()).varType),
                m_context);

            m_ldsDst = Register::Value::Placeholder(
                m_context,
                Register::Type::Vector,
                DataType::Raw32,
                248,
                Register::AllocationOptions{.contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                            .alignment = static_cast<int>(m_instrDwords)});
            co_yield m_ldsDst->allocate();

            co_yield m_context->mem()->barrier({});

            co_yield generateKernelBody();
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

    std::tuple<std::vector<Instruction>, std::vector<std::tuple<std::string, size_t>>>
        runKernelAndCollectLatencies(TestContext&       context,
                                     LDSTestKernelBase& kernel,
                                     bool               testIndividual)
    {
        constexpr int NUM_RUNS = 5; // Should be odd, as median is used

        std::vector<std::vector<rocRoller::profiler::InstructionProfile>> allLatencies;

        for(int run = 0; run < NUM_RUNS; ++run)
        {
            const auto latencies = rocRoller::profiler::loopUntilDispatchData([&]() { kernel(); });
            allLatencies.push_back(latencies);
        }

        const auto& instructions = kernel.getInstructions();

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
            Log::info(context.output());
            Log::info(infoStr);
        }

        return std::make_tuple(filteredInstructions, medianLatencies);
    }

} // namespace rocRoller
