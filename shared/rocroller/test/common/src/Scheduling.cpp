////////////////////////////////////////////////////////////////////////////////
//
// MIT License
//
// Copyright 2025-2026 AMD ROCm(TM) Software
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

#include "common/Scheduling.hpp"

namespace rocRoller
{
    std::vector<size_t>
        generateLDSAddresses(size_t count, size_t strideMultiplier, size_t instrDwords)
    {
        std::vector<size_t> addresses;
        for(size_t workitemId = 0; workitemId < count; ++workitemId)
        {
            size_t address = workitemId * (4 * strideMultiplier * instrDwords);
            addresses.push_back(address);
        }
        return addresses;
    }

    std::pair<size_t, size_t>
        getAlignedSubset(size_t totalRegs, size_t requestedRegCount, size_t position)
    {
        // If run out of registers, wrap around
        size_t num_complete_chunks = totalRegs / requestedRegCount;
        if(num_complete_chunks == 0)
        {
            return {0, 0};
        }
        size_t chunk_index = position % num_complete_chunks;
        size_t start       = chunk_index * requestedRegCount;
        return {start, start + requestedRegCount};
    }

} // namespace rocRoller
