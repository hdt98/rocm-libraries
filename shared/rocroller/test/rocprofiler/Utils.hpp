/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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

#include "Agent.hpp"
#include <rocRoller/CodeGen/Instruction.hpp>

#include <string>
#include <tuple>
#include <vector>

namespace rocRoller
{
    /**
     * @brief Number of runs for profiling tests
     *
     * Should be odd, as median is used for statistical analysis
     */
    constexpr int NUM_RUNS = 5;

    /**
     * @brief Formats a comparison between model predictions and profiler measurements
     *
     * @param filteredInstructions The list of instructions to compare
     * @param latencies The median latencies for each instruction
     * @return Formatted string comparing model predictions with profiler measurements
     */
    std::string
        formatLatencyComparison(const std::vector<Instruction>& filteredInstructions,
                                const std::vector<std::tuple<std::string, size_t>>& latencies);

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

} // namespace rocRoller
