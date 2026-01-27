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
        using namespace rocRoller::Expression;

        SECTION("No padding needed for non-128-aligned blocks")
        {
            // When contiguous block size is not a multiple of 128, no
            // padding is needed as successive rows naturally hit
            // different banks
            LDSPaddingInfo info{.ldsTag                   = 0,
                                .upstreamEdge             = 1,
                                .downstreamEdge           = 2,
                                .upstreamTags             = {3, 4},
                                .downstreamTags           = {5, 6},
                                .dataType                 = DataType::Float,
                                .layoutType               = LayoutType::MATRIX_A,
                                .loadInstructionByteWidth = 16u,
                                .loadLaneWidth            = 64u};

            uint contiguousBytes = 16u * 64u; // 1024 bytes (128 * 8, so it IS aligned)
            uint paddingBytes    = CalculateAutomaticPaddingBytes(info, contiguousBytes);
            CHECK(paddingBytes > 0);

            contiguousBytes = 16u * 63u; // 1008 bytes (not a multiple of 128)
            paddingBytes    = CalculateAutomaticPaddingBytes(info, contiguousBytes);
            CHECK(paddingBytes == 0);
        }

        SECTION("Padding for 128-aligned blocks")
        {
            // When contiguous block size is a multiple of 128,
            // padding is needed
            LDSPaddingInfo info{.ldsTag                   = 0,
                                .upstreamEdge             = 1,
                                .downstreamEdge           = 2,
                                .upstreamTags             = {3, 4},
                                .downstreamTags           = {5, 6},
                                .dataType                 = DataType::Float,
                                .layoutType               = LayoutType::MATRIX_A,
                                .loadInstructionByteWidth = 16u,
                                .loadLaneWidth            = 64u};

            uint contiguousBytes = 1024u; // 128 * 8
            uint paddingBytes    = CalculateAutomaticPaddingBytes(info, contiguousBytes);

            // Padding should be non-zero
            CHECK(paddingBytes > 0);
            // Padding should be reasonable (not too large)
            CHECK(paddingBytes <= 64u);
            // Padding should be at least one element (4 bytes for Float)
            CHECK(paddingBytes >= 4u);
        }

        SECTION("Different data types")
        {
            uint contiguousBytes = 2048u; // 128 * 16

            // Float16
            LDSPaddingInfo infoF16{.ldsTag                   = 0,
                                   .upstreamEdge             = 1,
                                   .downstreamEdge           = 2,
                                   .upstreamTags             = {3, 4},
                                   .downstreamTags           = {5, 6},
                                   .dataType                 = DataType::Half,
                                   .layoutType               = LayoutType::MATRIX_A,
                                   .loadInstructionByteWidth = 16u,
                                   .loadLaneWidth            = 64u};
            uint paddingBytesF16 = CalculateAutomaticPaddingBytes(infoF16, contiguousBytes);
            CHECK(paddingBytesF16 >= 2u); // At least one Half element (2 bytes)

            // Float32
            LDSPaddingInfo infoF32{.ldsTag                   = 0,
                                   .upstreamEdge             = 1,
                                   .downstreamEdge           = 2,
                                   .upstreamTags             = {3, 4},
                                   .downstreamTags           = {5, 6},
                                   .dataType                 = DataType::Float,
                                   .layoutType               = LayoutType::MATRIX_A,
                                   .loadInstructionByteWidth = 16u,
                                   .loadLaneWidth            = 64u};
            uint paddingBytesF32 = CalculateAutomaticPaddingBytes(infoF32, contiguousBytes);
            CHECK(paddingBytesF32 >= 4u); // At least one Float element (4 bytes)

            // FP8
            LDSPaddingInfo infoFP8{.ldsTag                   = 0,
                                   .upstreamEdge             = 1,
                                   .downstreamEdge           = 2,
                                   .upstreamTags             = {3, 4},
                                   .downstreamTags           = {5, 6},
                                   .dataType                 = DataType::FP8,
                                   .layoutType               = LayoutType::MATRIX_A,
                                   .loadInstructionByteWidth = 16u,
                                   .loadLaneWidth            = 64u};
            uint paddingBytesFP8 = CalculateAutomaticPaddingBytes(infoFP8, contiguousBytes);
            CHECK(paddingBytesFP8 >= 1u); // At least one FP8 element (1 byte)
        }

        SECTION("Padding respects maximum bounds")
        {
            // Very large contiguous block
            LDSPaddingInfo info{.ldsTag                   = 0,
                                .upstreamEdge             = 1,
                                .downstreamEdge           = 2,
                                .upstreamTags             = {3, 4},
                                .downstreamTags           = {5, 6},
                                .dataType                 = DataType::Float,
                                .layoutType               = LayoutType::MATRIX_A,
                                .loadInstructionByteWidth = 16u,
                                .loadLaneWidth            = 64u};

            uint contiguousBytes = 8192u; // 128 * 64
            uint paddingBytes    = CalculateAutomaticPaddingBytes(info, contiguousBytes);

            // Padding should not exceed 64 bytes
            CHECK(paddingBytes <= 64u);
        }
    }
}
