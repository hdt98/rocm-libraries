////////////////////////////////////////////////////////////////////////////////
//
// MIT License
//
// Copyright 2025 AMD ROCm(TM) Software
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
// ies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
// PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
// CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <rocRoller/Utilities/Error.hpp>
#include <utility>
#include <vector>

namespace rocRoller
{
    /**
     * @brief Generates LDS addresses for a workgroup based on stride and instruction size
     * 
     * @param workgroupSize Number of work items in the workgroup
     * @param strideMultiplier Stride multiplier for address calculation
     * @param instrDwords Instruction size in dwords
     * @return std::vector<size_t> Vector of calculated LDS addresses
     */
    std::vector<size_t>
        generateLDSAddresses(size_t workgroupSize, size_t strideMultiplier, size_t instrDwords);

    /**
     * @brief Finds the median value of an odd-sized vector
     * 
     * @tparam T Type of elements in the vector
     * @param values Vector with odd number of elements
     * @return T Median value
     */
    template <typename T>
    T median_of_odd_elements(std::vector<T> values)
    {
        AssertFatal(!values.empty(), "median_of_odd_elements: vector must not be empty");
        AssertFatal(values.size() % 2 == 1, "median_of_odd_elements: vector size must be odd");

        std::sort(values.begin(), values.end());

        return values[values.size() / 2];
    }

    /**
     * @brief Calculates differences between consecutive latency values
     * 
     * @param latencies Vector of latency values
     * @return std::vector<int64_t> Vector of deltas between consecutive latencies
     */
    std::vector<int64_t> calculateLatencyDeltas(const std::vector<uint64_t>& latencies);

    /**
     * @brief Computes aligned register subset with wraparound
     * 
     * @param totalRegs Total number of registers available
     * @param requestedRegCount Number of registers requested
     * @param position Position index
     * @return std::pair<size_t, size_t> Start and end indices of the aligned subset
     */
    std::pair<size_t, size_t>
        getAlignedSubset(size_t totalRegs, size_t requestedRegCount, size_t position);

} // namespace rocRoller
