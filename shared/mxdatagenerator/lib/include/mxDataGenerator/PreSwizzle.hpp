/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace DGen
{
    /**
     * @brief Swizzle mode for scale data
     */
    enum class ScaleSwizzleMode
    {
        None,      // No swizzle
        RocRoller, // RocRoller kernel pattern (existing preSwizzle)
        AITER      // AITER kernel pattern
    };

    /**
     * @brief Helper to compute product of elements in a range
     */
    template <typename T>
    inline size_t product(std::vector<T> const& x)
    {
        return std::accumulate(x.begin(), x.end(), size_t(1), std::multiplies<size_t>());
    }

    /**
     * @brief Compute strides for column-major layout given sizes
     */
    inline std::vector<size_t> computeStrides(std::vector<size_t> const& sizes)
    {
        std::vector<size_t> strides(sizes.size());
        if(sizes.empty())
            return strides;

        strides[0] = 1;
        for(size_t i = 1; i < sizes.size(); ++i)
            strides[i] = strides[i - 1] * sizes[i - 1];

        return strides;
    }

    /**
     * @brief Compute shuffled strides given dimension order
     */
    inline std::vector<size_t> computeShuffledStrides(std::vector<size_t> const& sizes,
                                                       std::vector<size_t> const& dimOrder)
    {
        std::vector<size_t> strides(sizes.size(), 0);
        size_t              stride = 1;
        for(auto idx : dimOrder)
        {
            strides.at(idx) = stride;
            stride *= sizes.at(idx);
        }
        return strides;
    }

    /**
     * @brief Shuffle data according to dimension reordering
     *
     * This performs a dimension shuffle where:
     * - input is arranged according to srcStrides
     * - output is arranged according to dstStrides
     * - both have the same dimension sizes
     */
    template <typename T>
    inline std::vector<T> shuffleDims(std::vector<T> const&      input,
                                      std::vector<size_t> const& sizes,
                                      std::vector<size_t> const& dstStrides,
                                      std::vector<size_t> const& srcStrides)
    {
        if(sizes.size() != dstStrides.size() || sizes.size() != srcStrides.size())
            throw std::runtime_error("shuffleDims: size/stride dimension mismatch");

        if(sizes.size() < 2)
            throw std::runtime_error("shuffleDims: need at least 2 dimensions");

        size_t totalElements = product(sizes);
        if(input.size() != totalElements)
        {
            std::ostringstream msg;
            msg << "shuffleDims: input size " << input.size() << " doesn't match expected "
                << totalElements;
            throw std::runtime_error(msg.str());
        }

        std::vector<T> output(input.size());

        // Compute total number of coordinates
        size_t totalCoords = 1;
        for(size_t i = 0; i < sizes.size(); ++i)
            totalCoords *= sizes[i];

#pragma omp parallel for
        for(size_t coordNum = 0; coordNum < totalCoords; ++coordNum)
        {
            // Convert coordNum to N-D coordinates
            std::vector<size_t> coord(sizes.size());
            size_t              remaining = coordNum;
            for(size_t i = 0; i < sizes.size(); ++i)
            {
                coord[i] = remaining % sizes[i];
                remaining /= sizes[i];
            }

            // Compute source and destination indices using strides
            size_t srcIdx = 0;
            size_t dstIdx = 0;
            for(size_t i = 0; i < sizes.size(); ++i)
            {
                srcIdx += coord[i] * srcStrides[i];
                dstIdx += coord[i] * dstStrides[i];
            }

            output[dstIdx] = input[srcIdx];
        }

        return output;
    }

    /**
     * @brief Pre-swizzle and optionally pre-tile the input.
     *
     * This function rearranges tensor data according to swizzle and tile configurations.
     * The incoming data should be in row-major order with the 0 dimension being the
     * fastest (smallest stride).
     *
     * @param input The input data vector
     * @param sizes The dimension sizes {size0, size1}
     * @param preSwizzleSize The swizzle configuration {tileMN, tileK, subTileK}, or empty
     * @param preTileSize The pre-tile configuration {tileSize0, tileSize1}, or empty
     * @return The pre-swizzled/pre-tiled data
     */
    template <typename T>
    inline std::vector<T> preSwizzle(std::vector<T> const&      input,
                                     std::vector<size_t> const& sizes,
                                     std::vector<size_t> const& preSwizzleSize,
                                     std::vector<size_t> const& preTileSize)
    {
        if(!preSwizzleSize.empty())
        {
            if(preSwizzleSize.size() != 3)
            {
                std::ostringstream msg;
                msg << "preSwizzle: preSwizzleSize must have 3 elements, got "
                    << preSwizzleSize.size();
                throw std::runtime_error(msg.str());
            }
        }

        if(sizes.size() != 2)
        {
            std::ostringstream msg;
            msg << "preSwizzle: Batch dimension not yet supported. sizes.size()=" << sizes.size();
            throw std::runtime_error(msg.str());
        }

        size_t totalElements = product(sizes);
        if(totalElements != input.size())
        {
            std::ostringstream msg;
            msg << "preSwizzle: input size " << input.size() << " doesn't match sizes product "
                << totalElements;
            throw std::runtime_error(msg.str());
        }

        std::vector<size_t> srcSizes, dimOrder;

        if((!preSwizzleSize.empty()) && (preTileSize.empty()))
        {
            auto tileMN   = preSwizzleSize[0];
            auto tileK    = preSwizzleSize[1];
            auto subTileK = preSwizzleSize[2];

            if(tileMN != 64 && tileMN != 32)
            {
                std::ostringstream msg;
                msg << "preSwizzle: tileMN must be 32 or 64, got " << tileMN;
                throw std::runtime_error(msg.str());
            }

            if(tileK % 4 != 0)
            {
                std::ostringstream msg;
                msg << "preSwizzle: tileK must be a multiple of 4, got " << tileK;
                throw std::runtime_error(msg.str());
            }

            size_t nLanesPerSIMD   = 16;
            size_t nSIMDsPerWave   = 4;
            size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
            size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
            size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
            size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
            size_t nSIMDIndexBlock = nVGPRIndex;
            size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

            if(nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK)
            {
                std::ostringstream msg;
                msg << "preSwizzle: nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK";
                throw std::runtime_error(msg.str());
            }

            if(nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN)
            {
                std::ostringstream msg;
                msg << "preSwizzle: nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN";
                throw std::runtime_error(msg.str());
            }

            srcSizes = {nVGPRIndex,
                        nVGPRBlock,
                        nSIMDBlock,
                        sizes[0] / (tileK),
                        nLanesPerSIMD,
                        nSIMDIndexIndex,
                        nSIMDIndexBlock,
                        sizes[1] / (tileMN)};

            if(tileMN == 64)
            {
                // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
                dimOrder = {6, 1, 2, 3, 4, 5, 0, 7};
            }
            else if(tileMN == 32 && subTileK == 4)
            {
                // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
                //              swap nSIMDBlock (2) and nVGPRBlock (1)
                dimOrder = {6, 2, 1, 3, 4, 5, 0, 7};
            }
            else if(tileMN == 32 && subTileK == 2)
            {
                // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
                dimOrder = {1, 2, 0, 3, 4, 5, 6, 7};
            }
        }
        else if((preSwizzleSize.empty()) && (!preTileSize.empty()))
        {
            srcSizes = {preTileSize[0],
                        sizes[0] / preTileSize[0],
                        preTileSize[1],
                        sizes[1] / preTileSize[1]};

            // Pre-tiling: 1 and 3 are pushed to the back (they become the slowest)
            dimOrder = {0, 2, 1, 3};
        }
        else
        {
            auto tileMN   = preSwizzleSize[0];
            auto tileK    = preSwizzleSize[1];
            auto subTileK = preSwizzleSize[2];

            if(tileMN != 64 && tileMN != 32)
            {
                std::ostringstream msg;
                msg << "preSwizzle: tileMN must be 32 or 64, got " << tileMN;
                throw std::runtime_error(msg.str());
            }

            if(tileK % 4 != 0)
            {
                std::ostringstream msg;
                msg << "preSwizzle: tileK must be a multiple of 4, got " << tileK;
                throw std::runtime_error(msg.str());
            }

            size_t ptTileSizeK     = preTileSize[0];
            size_t ptTileSizeMN    = preTileSize[1];
            size_t nLanesPerSIMD   = 16;
            size_t nSIMDsPerWave   = 4;
            size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
            size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
            size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
            size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
            size_t nSIMDIndexBlock = nVGPRIndex;
            size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

            if(ptTileSizeK / tileK == 0)
            {
                std::ostringstream msg;
                msg << "preSwizzle: ptTileSizeK / tileK == 0, ptTileSizeK=" << ptTileSizeK
                    << ", tileK=" << tileK;
                throw std::runtime_error(msg.str());
            }

            if(ptTileSizeMN / tileMN == 0)
            {
                std::ostringstream msg;
                msg << "preSwizzle: ptTileSizeMN / tileMN == 0, ptTileSizeMN=" << ptTileSizeMN
                    << ", tileMN=" << tileMN;
                throw std::runtime_error(msg.str());
            }

            if(nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK)
            {
                std::ostringstream msg;
                msg << "preSwizzle: nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK";
                throw std::runtime_error(msg.str());
            }

            if(nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN)
            {
                std::ostringstream msg;
                msg << "preSwizzle: nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN";
                throw std::runtime_error(msg.str());
            }

            srcSizes = {nVGPRIndex,
                        nVGPRBlock,
                        nSIMDBlock,
                        ptTileSizeK / tileK,
                        sizes[0] / ptTileSizeK,
                        nLanesPerSIMD,
                        nSIMDIndexIndex,
                        nSIMDIndexBlock,
                        ptTileSizeMN / tileMN,
                        sizes[1] / ptTileSizeMN};

            if(tileMN == 64)
            {
                // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
                // Pre tile: push workgroup tiles (4 and 9) to the end
                dimOrder = {7, 1, 2, 3, 5, 6, 0, 8, 4, 9};
            }
            else if(tileMN == 32 && subTileK == 4)
            {
                // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
                //              swap nSIMDBlock (2) and nVGPRBlock (1)
                // Pre tile: push workgroup tiles (4 and 9) to the end
                dimOrder = {7, 2, 1, 3, 5, 6, 0, 8, 4, 9};
            }
            else if(tileMN == 32 && subTileK == 2)
            {
                // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
                // Pre tile: push workgroup tiles (4 and 9) to the end
                dimOrder = {1, 2, 0, 3, 5, 6, 7, 8, 4, 9};
            }
        }

        if(product(srcSizes) != product(sizes))
        {
            std::ostringstream msg;
            msg << "PreSwizzle size mismatch: product(srcSizes)=" << product(srcSizes)
                << " != product(sizes)=" << product(sizes);
            throw std::runtime_error(msg.str());
        }

        if(srcSizes.empty())
            throw std::runtime_error("PreSwizzle source size not populated.");

        if(dimOrder.empty())
            throw std::runtime_error("PreSwizzle permutation order not populated.");

        auto srcStrides = computeStrides(srcSizes);
        auto dstStrides = computeShuffledStrides(srcSizes, dimOrder);

        return shuffleDims(input, srcSizes, dstStrides, srcStrides);
    }

    /**
     * @brief Pre-swizzle scale data for AITER kernel.
     *
     * This implements the AITER e8m0_shuffle algorithm from:
     * https://github.com/ROCm/aiter/blob/main/aiter/utility/fp4_utils.py
     *
     * The algorithm is:
     *   scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
     *   scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
     *   scale = scale.view(sm, sn)
     *
     * For output position (outRow, outCol), we compute the source position
     * by decomposing into 6D indices and applying the inverse permutation.
     *
     * @param input The input scale data vector (row-major, M x numScaleCols)
     * @param sizes The dimension sizes {numScaleRows, numScaleCols} where numScaleRows = M
     * @return The AITER-swizzled scale data
     */
    template <typename T>
    inline std::vector<T> preSwizzleAITER(std::vector<T> const&      input,
                                          std::vector<size_t> const& sizes)
    {
        if(sizes.size() != 2)
        {
            std::ostringstream msg;
            msg << "preSwizzleAITER: sizes must have 2 elements, got " << sizes.size();
            throw std::runtime_error(msg.str());
        }

        size_t numRows = sizes[0]; // M dimension (number of scale rows)
        size_t numCols = sizes[1]; // K/32 dimension (number of scale columns)

        size_t totalElements = numRows * numCols;
        if(totalElements != input.size())
        {
            std::ostringstream msg;
            msg << "preSwizzleAITER: input size " << input.size() << " doesn't match sizes product "
                << totalElements;
            throw std::runtime_error(msg.str());
        }

        if(numRows % 32 != 0)
        {
            std::ostringstream msg;
            msg << "preSwizzleAITER: numRows must be a multiple of 32, got " << numRows;
            throw std::runtime_error(msg.str());
        }

        if(numCols % 8 != 0)
        {
            std::ostringstream msg;
            msg << "preSwizzleAITER: numCols must be a multiple of 8, got " << numCols;
            throw std::runtime_error(msg.str());
        }

        std::vector<T> output(input.size());

        // AITER shuffle algorithm:
        // view as (numRows // 32, 2, 16, numCols // 8, 2, 4)
        // permute (0, 3, 5, 2, 4, 1)
        // view as (numRows, numCols)
        //
        // For output linear index, decompose into 6D output indices,
        // then map to 6D input indices via inverse permute,
        // then compute source row/col.

        size_t numColBlocks = numCols / 8;

#pragma omp parallel for
        for(size_t outRow = 0; outRow < numRows; ++outRow)
        {
            for(size_t outCol = 0; outCol < numCols; ++outCol)
            {
                // Compute linear index in output
                size_t linear = outRow * numCols + outCol;

                // Decompose into 6D indices for output layout:
                // (numRows // 32, numCols // 8, 4, 16, 2, 2)
                // Dimensions are ordered from slowest to fastest varying
                size_t d5 = linear % 2;        linear /= 2;
                size_t d4 = linear % 2;        linear /= 2;
                size_t d3 = linear % 16;       linear /= 16;
                size_t d2 = linear % 4;        linear /= 4;
                size_t d1 = linear % numColBlocks; linear /= numColBlocks;
                size_t d0 = linear;

                // Apply inverse permutation
                // Permute was (0, 3, 5, 2, 4, 1), so:
                // out[0]=in[0], out[1]=in[3], out[2]=in[5], out[3]=in[2], out[4]=in[4], out[5]=in[1]
                // Inverse: in[0]=d0, in[1]=d5, in[2]=d3, in[3]=d1, in[4]=d4, in[5]=d2
                size_t i0 = d0;  // block32
                size_t i1 = d5;  // row2 (0 or 1, selects which half of 32)
                size_t i2 = d3;  // row16 (0-15)
                size_t i3 = d1;  // colBlock8
                size_t i4 = d4;  // col2 (0 or 1, selects which half of 8)
                size_t i5 = d2;  // col4 (0-3)

                // Compute source row and column
                size_t srcRow = i0 * 32 + i1 * 16 + i2;
                size_t srcCol = i3 * 8 + i4 * 4 + i5;

                // Bounds check
                if(srcRow < numRows && srcCol < numCols)
                {
                    output[outRow * numCols + outCol] = input[srcRow * numCols + srcCol];
                }
            }
        }

        return output;
    }

} // namespace DGen

