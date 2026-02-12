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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "CustomSections.hpp"
#include "SimpleTest.hpp"

#include <mxDataGenerator/PreSwizzle.hpp>
#include <rocRoller/Operations/BlockScale.hpp>
#include <random>

using namespace rocRoller;
using namespace DGen;
using namespace Catch::Matchers;

namespace PreSwizzleTest
{
    /**
     * @brief Generate random uint8_t data for testing
     */
    std::vector<uint8_t> generateRandomData(size_t size, unsigned seed = 42)
    {
        std::vector<uint8_t> data(size);
        std::mt19937         rng(seed);
        std::uniform_int_distribution<int> dist(0, 255);

        for(size_t i = 0; i < size; ++i)
        {
            data[i] = static_cast<uint8_t>(dist(rng));
        }

        return data;
    }

    // Note: preSwizzleScalesGFX950 and preSwizzle implement different algorithms:
    // - preSwizzleScalesGFX950: AITER e8m0_shuffle algorithm (6D view with permute)
    // - preSwizzle: Wave/SIMD decomposition algorithm (8D view)
    // They produce same result when preSwizzle uses alternative layout.

    TEST_CASE("preSwizzle with alternative layout", "[PreSwizzle]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            SECTION("Alternative layout with pre-tiling")
            {
                size_t M         = 1024;
                size_t K         = 256;
                size_t blockSize = 32;
                size_t numRows   = M;
                size_t numCols   = K / blockSize;

                auto input = generateRandomData(numRows * numCols, 99999);

                std::vector<size_t> sizes          = {numRows, numCols};
                std::vector<size_t> preSwizzleSize = {32, 8, 4};
                std::vector<size_t> preTileSize    = {8, 32}; // tileK x tileM

                // Standard layout with pre-tiling
                auto result_standard = preSwizzle(input, sizes, preSwizzleSize, preTileSize, false);

                // Alternative layout with pre-tiling
                auto result_alternative
                    = preSwizzle(input, sizes, preSwizzleSize, preTileSize, true);

                // Apply GFX950 AITER algorithm
                auto result_gfx950 = preSwizzleScalesGFX950(input, sizes);

                // Both should be valid permutations
                REQUIRE(result_standard.size() == input.size());
                REQUIRE(result_alternative.size() == input.size());
                REQUIRE(result_gfx950.size() == input.size());

                CHECK(result_gfx950 == result_alternative);
            }
        }
    }

    TEST_CASE("SubTileTranspose with alternative layout flag", "[PreSwizzle][SubTileTranspose]")
    {
        using namespace rocRoller::Operations;
        
        SECTION("SubTileTranspose creation and serialization")
        {
            // Test creating SubTileTranspose with alternative layout flag
            OperationTag input(1);
            std::vector<size_t> tileDimensions = {32, 8, 4};
            
            SECTION("Standard layout (false)")
            {
                SubTileTranspose op(input, tileDimensions, false);
                
                REQUIRE(op.input() == input);
                REQUIRE(op.tileDimensions() == tileDimensions);
                REQUIRE(op.useAlternativeLayout() == false);
            }
            
            SECTION("Alternative layout (true)")
            {
                SubTileTranspose op(input, tileDimensions, true);
                
                REQUIRE(op.input() == input);
                REQUIRE(op.tileDimensions() == tileDimensions);
                REQUIRE(op.useAlternativeLayout() == true);
            }
            
            SECTION("Default parameter (false)")
            {
                SubTileTranspose op(input, tileDimensions);
                
                REQUIRE(op.input() == input);
                REQUIRE(op.tileDimensions() == tileDimensions);
                REQUIRE(op.useAlternativeLayout() == false);
            }
        }
        
        SECTION("Invalid tile dimensions are rejected")
        {
            OperationTag input(1);
            
            // Wrong number of dimensions
            REQUIRE_THROWS(SubTileTranspose(input, {32, 8}));
            
            // Wrong total size (should be 256)
            REQUIRE_THROWS(SubTileTranspose(input, {32, 4, 4}));
            
            // Wrong K dimension (should be 2 or 4)
            REQUIRE_THROWS(SubTileTranspose(input, {32, 8, 3}));
        }
    }
}
