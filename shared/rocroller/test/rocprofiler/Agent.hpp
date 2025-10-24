// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/experimental/thread-trace/dispatch.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace rocRoller
{
    namespace profiler
    {
        struct InstructionProfile
        {
            uint64_t    totalLatency{0}; // Total latency in cycles
            uint64_t    hitcount{0}; // Number of times instruction was executed
            std::string instruction; // Disassembled instruction text

            uint64_t    meanLatency() const;
            std::string toString() const;
        };

        std::string toString(std::vector<InstructionProfile> const& profiles);

        struct pc_comparator
        {
            bool operator()(const rocprofiler_thread_trace_decoder_pc_t& a,
                            const rocprofiler_thread_trace_decoder_pc_t& b) const
            {
                if(a.code_object_id == b.code_object_id)
                    return a.address < b.address;
                return a.code_object_id < b.code_object_id;
            }
        };

        using InstructionLatencyMap
            = std::map<rocprofiler_thread_trace_decoder_pc_t, InstructionProfile, pc_comparator>;

        /**
         * @brief Wait for a specific number of dispatches and get the instruction latency data
         * 
         * This function increments the expected dispatch counter by n and waits for the
         * dispatch data to be available before returning it.
         * 
         * @param n Number of dispatches to wait for
         * @return optional vector of InstructionProfile from the dispatch, or nullopt if no data available
         */
        std::optional<std::vector<InstructionProfile>> waitForDispatchData(int n);

        /**
         * @brief Call a function that dispatches a kernel and attempts to collect data
         * 
         * When the agent is enabled, this function will repeatedly call the dispatch function
         * until instruction profiling data is successfully collected. This is useful for 
         * smaller kernels that may not launch with enough waves on the first attempt.
         * 
         * @param n Number of dispatches to wait for
         * @param dispatch Function that dispatches kernel(s)
         * @return Vector of InstructionProfile from the dispatch. Returns empty vector if 
         *         agent is disabled (ENABLE_AGENT is false).
         */
        std::vector<InstructionProfile> loopUntilDispatchData(int                   n,
                                                              std::function<void()> dispatch);

    } // namespace profiler
} // namespace rocRoller
