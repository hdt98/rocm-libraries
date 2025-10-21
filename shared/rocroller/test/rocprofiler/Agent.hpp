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
#include <map>
#include <string>
#include <vector>

namespace rocRoller
{
    namespace profiler
    {
        struct InstructionData
        {
            uint64_t    latency{0}; // Total latency in cycles
            uint64_t    hitcount{0}; // Number of times instruction was executed
            std::string instruction; // Disassembled instruction text
        };

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
            = std::map<rocprofiler_thread_trace_decoder_pc_t, InstructionData, pc_comparator>;

        /**
         * @brief Get the instruction latency data from the most recent dispatch
         * 
         * This function returns a vector containing instruction latency
         * information from the most recent dispatch. Each element contains
         * latency, hit count, and instruction disassembly for each instruction.
         * 
         * @return vector of InstructionData from the most recent dispatch
         */
        std::vector<InstructionData> getMostRecentDispatchData();

        /**
         * @brief Prepare to wait for a specific number of dispatch callbacks
         * 
         * This function sets up the profiler to wait for exactly n dispatch
         * callbacks before getMostRecentDispatchData() returns. It also clears any
         * previously collected dispatch data.
         * 
         * @param n Number of dispatches to expect
         */
        void expect_dispatches(int n);

    } // namespace profiler
} // namespace rocRoller
