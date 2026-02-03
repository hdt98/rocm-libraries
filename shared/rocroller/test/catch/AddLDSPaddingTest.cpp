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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include "TestContext.hpp"

namespace rocRoller::KernelGraph
{
    void addLoadWaveTileCT(KernelGraph&                     graph,
                           std::vector<DeferredConnection>& connections,
                           int                              macTileTag,
                           int                              iMacX,
                           int                              iMacY,
                           DataType const&                  dataType,
                           int                              wavefrontSize,
                           bool                             isFromLDS,
                           std::vector<unsigned int> const& jammedTiles,
                           CommandParametersPtr             params,
                           ContextPtr                       context);
};

namespace AddLDSPaddingTest
{
    using namespace rocRoller;
    using namespace rocRoller::KernelGraph::CoordinateGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    TEST_CASE("getNumLDSElements", "[kernel-graph][utils]")
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::Expression;

        SECTION("Simple flatten")
        {
            rocRoller::KernelGraph::KernelGraph graph;

            uint sizeX = 5u;
            uint sizeY = 7u;

            auto indexX = graph.coordinates.addElement(MacroTileIndex(0, literal(sizeX), nullptr));
            auto indexY = graph.coordinates.addElement(MacroTileIndex(1, literal(sizeY), nullptr));

            auto ldsTag = graph.coordinates.addElement(LDS());

            auto flatten = graph.coordinates.addElement(Flatten(), {indexX, indexY}, {ldsTag});

            int ldsElements = GetNumLDSElements(graph, ldsTag);
            CHECK(ldsElements == sizeX * sizeY);
        }

        SECTION("Joined LDS (X)")
        {
            rocRoller::KernelGraph::KernelGraph graph;

            uint sizeX   = 5u;
            uint sizeY   = 7u;
            uint strideX = GENERATE(7u, 10u);
            uint strideY = 1u;

            auto indexX
                = graph.coordinates.addElement(MacroTileIndex(0, literal(sizeX), literal(strideX)));
            auto indexY
                = graph.coordinates.addElement(MacroTileIndex(1, literal(sizeY), literal(strideY)));

            auto ldsTag = graph.coordinates.addElement(LDS());

            auto join = graph.coordinates.addElement(Join(), {indexX, indexY}, {ldsTag});

            int ldsElements = GetNumLDSElements(graph, ldsTag);
            CHECK(ldsElements == strideX * sizeX);
        }

        SECTION("Joined LDS (Y)")
        {
            rocRoller::KernelGraph::KernelGraph graph;

            uint sizeX   = 5u;
            uint sizeY   = 7u;
            uint strideX = 1u;
            uint strideY = GENERATE(5u, 11u);

            auto indexX
                = graph.coordinates.addElement(MacroTileIndex(0, literal(sizeX), literal(strideX)));
            auto indexY
                = graph.coordinates.addElement(MacroTileIndex(1, literal(sizeY), literal(strideY)));

            auto ldsTag = graph.coordinates.addElement(LDS());

            auto join = graph.coordinates.addElement(Join(), {indexX, indexY}, {ldsTag});

            int ldsElements = GetNumLDSElements(graph, ldsTag);
            CHECK(ldsElements == strideY * sizeY);
        }
    }

    TEST_CASE("CalculateAutomaticContiguousBlockSize", "[kernel-graph][utils]")
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::AddLDSPaddingDetail;
        using namespace rocRoller::Expression;

        auto testContext = TestContext::ForDefaultTarget();
        auto context     = testContext.get();
        auto params      = std::make_shared<CommandParameters>();

        SECTION("Simple automatic contiguous block width")
        {
            auto loadWidth     = GENERATE(4u, 8u, 16u);
            auto loadLaneWidth = GENERATE(32u, 64u, 256u);

            LDSPaddingInfo info{.ldsTag                   = 0,
                                .upstreamEdge             = 1,
                                .downstreamEdge           = 2,
                                .upstreamTags             = {3, 4},
                                .downstreamTags           = {5, 6},
                                .dataType                 = DataType::Float,
                                .layoutType               = LayoutType::MATRIX_A,
                                .loadInstructionByteWidth = loadWidth,
                                .loadLaneWidth            = loadLaneWidth};

            uint expectedContiguousBytes   = loadWidth * loadLaneWidth;
            uint calculatedContiguousBytes = CalculateAutomaticContiguousBlockSize(info);
            CHECK(calculatedContiguousBytes == expectedContiguousBytes);
        }
    }

    TEST_CASE("CalculateAutomaticPaddingBytes", "[kernel-graph][utils]")
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::AddLDSPaddingDetail;

        auto makeInfo = [](DataType dt) {
            return LDSPaddingInfo{.ldsTag                   = 0,
                                  .upstreamEdge             = 1,
                                  .downstreamEdge           = 2,
                                  .upstreamTags             = {3, 4},
                                  .downstreamTags           = {5, 6},
                                  .dataType                 = dt,
                                  .layoutType               = LayoutType::MATRIX_A,
                                  .loadInstructionByteWidth = 16u,
                                  .loadLaneWidth            = 64u};
        };

        auto gcdScore = [](uint strideBytes) -> uint {
            constexpr uint LDS_NUM_BANKS      = 32;
            constexpr uint LDS_BYTES_PER_BANK = 4;
            REQUIRE(strideBytes % LDS_BYTES_PER_BANK == 0u);
            uint strideWords = strideBytes / LDS_BYTES_PER_BANK;
            return std::gcd(strideWords, LDS_NUM_BANKS);
        };

        SECTION("Aligned 128B stride gets padded, typically with a small amount")
        {
            // 1024B is 128*8. Baseline strideWords = 1024/4 = 256 => gcd(256,32)=32 (worst)
            // The heuristic should find a small padding to maximize bank rotation
            auto info            = makeInfo(DataType::Float);
            uint contiguousBytes = 1024u;

            uint paddingBytes = CalculateAutomaticPaddingBytes(info, contiguousBytes);

            CHECK(paddingBytes > 0u);
            CHECK(paddingBytes <= 64u);

            // We enforce 4B bank granularity via stepBytes = lcm(elementBytes, 4)
            CHECK(paddingBytes % 4u == 0u);

            uint baseG = gcdScore(contiguousBytes);
            uint newG  = gcdScore(contiguousBytes + paddingBytes);
            CHECK(newG < baseG);
        }

        SECTION("Non-128B stride may still get padded if bank rotation improves")
        {
            // 1008B is not a multiple of 128, but
            //   1008/4 = 252 => gcd(252,32)=4 (row starts only hit 8 banks)
            // Adding 4B gives 1012/4=253 => gcd(253,32)=1 (full 32 bank rotation)
            auto info            = makeInfo(DataType::Float);
            uint contiguousBytes = 1008u;

            uint paddingBytes = CalculateAutomaticPaddingBytes(info, contiguousBytes);

            CHECK(paddingBytes > 0u);
            CHECK(paddingBytes % 4u == 0u);

            uint baseG = gcdScore(contiguousBytes);
            uint newG  = gcdScore(contiguousBytes + paddingBytes);
            CHECK(newG < baseG);
        }

        SECTION("Non-128B stride with already good rotation keeps padding at 0")
        {
            // 1028B is not a multiple of 128 and
            //   1028/4 = 257 => gcd(257,32)=1 which is an already optimal rotation
            // So we should not add padding
            auto info            = makeInfo(DataType::Float);
            uint contiguousBytes = 1028u;

            uint paddingBytes = CalculateAutomaticPaddingBytes(info, contiguousBytes);
            CHECK(paddingBytes == 0u);
        }

        SECTION("Different data types keep padding element valid and 4B aligned")
        {
            // Use a 128B-aligned contiguousBytes so padding is definitely useful
            uint contiguousBytes = 2048u; // 128*16

            // FP16
            {
                auto info         = makeInfo(DataType::Half);
                uint paddingBytes = CalculateAutomaticPaddingBytes(info, contiguousBytes);

                CHECK(paddingBytes > 0u);
                CHECK(paddingBytes <= 64u);

                // Due to bank granularity, padding is in 4B steps even for 2B elements
                CHECK(paddingBytes % 4u == 0u);

                // Still valid element, divisible by elementBytes
                CHECK(paddingBytes % 2u == 0u);
            }
            //FP32
            {
                auto info         = makeInfo(DataType::Float);
                uint paddingBytes = CalculateAutomaticPaddingBytes(info, contiguousBytes);

                CHECK(paddingBytes > 0u);
                CHECK(paddingBytes <= 64u);
                CHECK(paddingBytes % 4u == 0u);
            }
            //FP8
            {
                auto info         = makeInfo(DataType::FP8);
                uint paddingBytes = CalculateAutomaticPaddingBytes(info, contiguousBytes);

                CHECK(paddingBytes > 0u);
                CHECK(paddingBytes <= 64u);

                // Bank granularity forces 4B steps even for 1B elements
                CHECK(paddingBytes % 4u == 0u);
            }
        }

        SECTION("Padding respects maximum bounds")
        {
            auto info            = makeInfo(DataType::Float);
            uint contiguousBytes = 8192u; // 128 * 64

            uint paddingBytes = CalculateAutomaticPaddingBytes(info, contiguousBytes);

            CHECK(paddingBytes <= 64u);
        }
    }
    TEST_CASE("CalculateAutomaticPaddingBytes additional invariants", "[kernel-graph][utils]")
    {
        using namespace rocRoller::KernelGraph::AddLDSPaddingDetail;

        auto makeInfo = [](DataType dt) {
            return LDSPaddingInfo{.ldsTag                   = 0,
                                  .upstreamEdge             = 1,
                                  .downstreamEdge           = 2,
                                  .upstreamTags             = {3, 4},
                                  .downstreamTags           = {5, 6},
                                  .dataType                 = dt,
                                  .layoutType               = LayoutType::MATRIX_A,
                                  .loadInstructionByteWidth = 16u,
                                  .loadLaneWidth            = 64u};
        };

        auto stepBytes = [](DataType dt) -> uint {
            constexpr uint LDS_BYTES_PER_BANK = 4;
            uint           elementBits        = DataTypeInfo::Get(dt).elementBits;
            uint           elementBytes       = elementBits / 8u;
            return std::lcm(elementBytes, LDS_BYTES_PER_BANK);
        };

        auto gcdScore = [](uint strideBytes) -> uint {
            constexpr uint LDS_NUM_BANKS      = 32;
            constexpr uint LDS_BYTES_PER_BANK = 4;
            REQUIRE(strideBytes % LDS_BYTES_PER_BANK == 0u);
            return std::gcd(strideBytes / LDS_BYTES_PER_BANK, LDS_NUM_BANKS);
        };

        SECTION("contiguousBytes==0 returns 0")
        {
            auto info = makeInfo(DataType::Float);
            CHECK(CalculateAutomaticPaddingBytes(info, 0u) == 0u);
        }

        SECTION("Returned padding is either 0 or a multiple of stepBytes and 4B aligned")
        {
            for(auto dt : {DataType::FP8, DataType::Half, DataType::Float})
            {
                auto info = makeInfo(dt);

                uint contiguousBytes = 1024u;

                uint pad  = CalculateAutomaticPaddingBytes(info, contiguousBytes);
                uint step = stepBytes(dt);

                if(pad != 0u)
                {
                    CHECK(pad % step == 0u);
                    CHECK(pad % 4u == 0u);
                }
                else
                {
                    SUCCEED("pad == 0 is allowed");
                }
            }
        }

        SECTION("If padding is non-zero, bank rotation score strictly improves")
        {
            auto info            = makeInfo(DataType::Float);
            uint contiguousBytes = 1024u; // worst case gcd at baseline, usually 32

            uint pad = CalculateAutomaticPaddingBytes(info, contiguousBytes);

            REQUIRE(pad > 0u);
            CHECK(gcdScore(contiguousBytes + pad) < gcdScore(contiguousBytes));
        }

        SECTION("Small contiguousBytes holds maxPadBytes, padding stays bounded")
        {
            auto info            = makeInfo(DataType::Float);
            uint contiguousBytes = 128u; // 128B aligned and small

            uint pad = CalculateAutomaticPaddingBytes(info, contiguousBytes);

            // maxPadBytes <= max(stepBytes, contiguousBytes/4) and also <= 64
            // For fp32, stepBytes=4, contiguousBytes/4=32 then maxPadBytes becomes 32
            CHECK(pad <= 32u);
        }

        SECTION("If already optimal rotation, then zero padding - test for odd strideWords")
        {
            auto info            = makeInfo(DataType::Float);
            uint contiguousBytes = 1028u; // 1028/4=257 => gcd(257,32)=1

            uint pad = CalculateAutomaticPaddingBytes(info, contiguousBytes);
            CHECK(pad == 0u);
        }
    }
}
