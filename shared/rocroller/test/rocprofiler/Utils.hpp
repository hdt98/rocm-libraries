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

#pragma once

#include "../catch/TestContext.hpp"
#include "../catch/TestKernels.hpp"
#include "Agent.hpp"
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <common/Scheduling.hpp>
#include <string>
#include <tuple>
#include <vector>

namespace rocRoller
{
    /**
     * @brief Formats a comparison between model predictions and profiler measurements
     * 
     * @param filteredInstructions The list of instructions to compare
     * @param profiles The profiler measurements for each instruction
     * @return Formatted string comparing model predictions with profiler measurements
     */
    std::string formatLatencyComparison(const std::vector<Instruction>& filteredInstructions,
                                        const std::vector<profiler::InstructionProfile>& profiles);

    /**
     * @brief Filters instructions to exclude comments and verifies alignment with profiler data
     * 
     * @param instructions Raw instructions from the kernel
     * @param latencies Profiler latencies to verify against
     * @return Vector of filtered instructions (non-comment only)
     */
    std::vector<Instruction>
        filterAndVerifyInstructions(const std::vector<Instruction>&                  instructions,
                                    const std::vector<profiler::InstructionProfile>& latencies);

    /**
     * @brief Base class for LDS test kernels with shared functionality
     */
    class LDSTestKernelBase : public AssemblyTestKernel
    {
    public:
        LDSTestKernelBase(ContextPtr                 context,
                          uint32_t                   workgroupSize,
                          size_t                     instrDwords,
                          size_t                     strideMultiplier,
                          const std::vector<size_t>& baseAddresses,
                          bool                       write);

        void operator()();

        const std::vector<Instruction>& getInstructions() const;

    protected:
        void generate() override;

        virtual Generator<Instruction>
            generateKernelBody(std::shared_ptr<Register::Value> ldsData,
                               std::shared_ptr<Register::Value> ldsWithOffset,
                               std::shared_ptr<Register::Value> workitemIndex)
            = 0;

    protected:
        uint32_t                 m_workgroupSize;
        size_t                   m_instrDwords;
        size_t                   m_strideMultiplier;
        std::vector<size_t>      m_baseAddresses;
        bool                     m_write;
        std::vector<Instruction> m_instructions;
    };

    /**
     * @brief Helper function for profiling and collecting median latencies
     */
    template <typename KernelType>
    std::tuple<std::vector<Instruction>, std::vector<std::tuple<std::string, size_t>>>
        runKernelAndCollectLatencies(TestContext& context, KernelType& kernel, bool testIndividual)
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
