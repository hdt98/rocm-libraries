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

namespace rocRoller
{
    std::string
        formatLatencyComparison(const std::vector<Instruction>& filteredInstructions,
                                const std::vector<std::tuple<std::string, size_t>>& latencies)
    {
        std::stringstream infoMessage;
        for(size_t i = 0; i < std::min(filteredInstructions.size(), latencies.size()); ++i)
        {
            const auto& inst                 = filteredInstructions[i];
            const auto& [inst_name, latency] = latencies[i];

            int const modelLatency = inst.totalCycles() * 4;
            int const delta        = static_cast<int>(latency) - modelLatency;

            infoMessage << fmt::format("{} {}, model {}, profiler {}, delta {}\n",
                                       delta != 0 ? "*" : " ",
                                       inst_name,
                                       modelLatency,
                                       latency,
                                       delta);
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

} // namespace rocRoller
