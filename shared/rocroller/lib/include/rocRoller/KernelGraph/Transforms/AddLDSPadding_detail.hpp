/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding.hpp>

#include <iosfwd>
#include <string>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace AddLDSPaddingDetail
        {
            /**
             * @brief Information about LDS padding.
             *
             * Information about LDS padding that is being added to
             * the graph.
             */
            struct LDSPaddingInfo
            {
                int ldsTag; //< LDS coordinate.
                int upstreamEdge; //< Edge immediately upstream (usually a Flatten) of ldsTag.
                int downstreamEdge; //< Edge immediately downstream (usually a Tile) of ldsTag.
                std::array<int, 2> upstreamTags; //< Coordinates upstream of upstreamEdge.
                std::array<int, 2> downstreamTags; //< Coordinates downstream of downstreamEdge.
                DataType           dataType; //< DataType of the data in LDS.
                LayoutType         layoutType; //< LayoutType of the data in LDS.
                uint
                    loadInstructionByteWidth; //< Byte-width of the instructions used to load data destined for LDS.
                uint loadLaneWidth; //< Number of lanes that should be considered contiguous.
            };

            /**
             * @brief Calculate automatic contiguous block size for LDS padding.
             */
            uint CalculateAutomaticContiguousBlockSize(LDSPaddingInfo const& info);

            /**
             * @brief Convert LDSPaddingInfo to string representation.
             */
            std::string toString(LDSPaddingInfo const& info);

            /**
             * @brief Stream output operator for LDSPaddingInfo.
             */
            std::ostream& operator<<(std::ostream& stream, LDSPaddingInfo const& info);
        }
    }
}
