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

/**
@class AddLDSPadding
@brief Add element padding to LDS buffers.

Padding LDS buffers is used to reduce the number of LDS bank
conflicts.

Padding is added to flat LDS buffers.  For each `LDS` node, upstream
`Flatten` edges are transformed to `Join` edges, and downstream
`Tile` edges are transformed to `Split` edges.

Recall that `Join` and `Split` edges honour the `stride` attribute of
upstream/downstream nodes.  Padding is accomplished by updating the
`stride` attributes of the upstream/downstream nodes.

In particular, the slow strides are set to the fast size plus a
padding value, and the fast strides are set to 1.

For example:

    MacroTileIndex                  MacroTileIndex
       size=256                       size=128
                 \                  /
                  ------------------
                          |
                       Flatten
                          |
                         LDS
                     size=32,768
                          |
                        Tile
                          |
                  ------------------
                 /                  \
    MacroTileIndex                 MacroTileIndex
       size=256                       size=128

will be transformed to:

    MacroTileIndex                  MacroTileIndex
       size=256                       size=128
   stride=128+padding                 stride=1
                 \                  /
                  ------------------
                          |
                        Join
                          |
                         LDS
              size=32,768 + 256*padding
                          |
                        Split
                          |
                  ------------------
                 /                  \
    MacroTileIndex                 MacroTileIndex
       size=256                       size=128
    stride=128+padding                stride=1


Note that the size of the LDS allocation is computed in `getNumLDSElements()`.

This LDS padding strategy is well-suited for direct-to-lds loads; as
the assumption is that writes into LDS are contiguous across the
workitems within a workgroup.

*/

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <fmt/format.h>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace AddLDSPaddingDetail
        {
            uint CalculateAutomaticContiguousBlockSize(LDSPaddingInfo const& info)
            {
                // Typical configuration:
                //
                // - loadInstructionByteWidth = globalReadVectorWidth * bytesPerElement
                // - loadLaneWidth = wavefrontSize (e.g., 64 lanes)
                //
                // Example: 4 elements/load * 4 bytes/element * 64 lanes = 1024 bytes

                return info.loadInstructionByteWidth * info.loadLaneWidth;
            }

            uint CalculateAutomaticPaddingBytes(LDSPaddingInfo const& info, uint contiguousBytes)
            {
                // LDS has 32 banks, each 4 bytes wide
                constexpr uint LDS_NUM_BANKS      = 32;
                constexpr uint LDS_BYTES_PER_BANK = 4;

                // A "row" of banks has 128 bytes.
                constexpr uint LDS_BANK_ROW_BYTES = LDS_NUM_BANKS * LDS_BYTES_PER_BANK;

                auto elementBytes = DataTypeInfo::Get(info.dataType).elementBits / 8u;

                // The goal is to avoid LDS bank conflicts by adding
                // padding when rows would otherwise map to the same
                // banks.
                //
                // If contiguous block size + padding is NOT a
                // multiple of 128 bytes, successive rows will hit
                // different banks, reducing conflicts.
                //

                // Calculate how the contiguous block aligns with LDS
                // banks
                uint remainder = contiguousBytes % LDS_BANK_ROW_BYTES;

                // If already not aligned to 128 bytes, bank conflicts
                // are naturally avoided
                if(remainder != 0)
                {
                    return 0;
                }

                // For 128-byte aligned blocks, we need padding to
                // shift subsequent rows:
                //
                // - loadInstructionByteWidth (for optimal load pattern)
                // - or 16 bytes

                uint paddingBytes = info.loadInstructionByteWidth;

                // However, ensure padding is at least one element and
                // reasonable
                uint minPaddingBytes = elementBytes;
                uint maxPaddingBytes = 64;

                paddingBytes = std::max(paddingBytes, minPaddingBytes);
                paddingBytes = std::min(paddingBytes, maxPaddingBytes);

                // Special case: for very small element types with large vector widths,
                // we might over-pad. Cap at a reasonable fraction of the block size.
                if(paddingBytes > contiguousBytes / 8)
                {
                    paddingBytes = std::max(contiguousBytes / 16, minPaddingBytes);
                }

                Log::debug("KernelGraph::AddLDSPadding::CalculateAutomaticPaddingBytes: "
                           "contiguousBytes={}, elementBytes={}, loadInstructionByteWidth={}, "
                           "calculated paddingBytes={}",
                           contiguousBytes,
                           elementBytes,
                           info.loadInstructionByteWidth,
                           paddingBytes);

                return paddingBytes;
            }

            std::string toString(LDSPaddingInfo const& info)
            {
                return fmt::format(
                    "LDSPaddingInfo{{ldsTag={}, upstreamEdge={}, downstreamEdge={}, "
                    "upstreamTags=[{}, {}], downstreamTags=[{}, {}], "
                    "dataType={}, layoutType={}, loadInstructionByteWidth={}, loadLaneWidth={}}}",
                    info.ldsTag,
                    info.upstreamEdge,
                    info.downstreamEdge,
                    info.upstreamTags[0],
                    info.upstreamTags[1],
                    info.downstreamTags[0],
                    info.downstreamTags[1],
                    rocRoller::toString(info.dataType),
                    rocRoller::toString(info.layoutType),
                    info.loadInstructionByteWidth,
                    info.loadLaneWidth);
            }

            std::ostream& operator<<(std::ostream& stream, LDSPaddingInfo const& info)
            {
                return stream << toString(info);
            }
        }

        using GD = Graph::Direction;

        using namespace Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        using namespace AddLDSPaddingDetail;

        /**
         * @brief Return the number of elements that the fastest thread-tile indexes.
         */
        uint GetFastThreadTileIndexElementWidth(KernelGraph const& graph, int tileTag)
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;

            auto isLoadPredicate = [&graph](auto const& x) {
                return graph.control.get<LoadTiled>(x.control).has_value()
                       || graph.control.get<LoadLDSTile>(x.control).has_value();
            };

            auto loadOp
                = only(filter(isLoadPredicate, graph.mapper.getCoordinateConnections(tileTag)))
                      .value()
                      .control;

            auto elementNumberTag = graph.mapper.get<CT::ElementNumber>(loadOp, 1);
            auto elementNumber    = graph.coordinates.get<ElementNumber>(elementNumberTag).value();

            return getUnsignedInt(evaluate(elementNumber.size));
        }

        /**
         * @brief Get the layout type and data type of the LDS node.
         *
         * This will traverse upstream from the LDS node until it finds
         * a MacroTile node, and returns the layout type and data type
         * of that MacroTile.
         *
         * If no MacroTile is found, an empty optional is returned.
         */
        std::optional<std::pair<LayoutType, DataType>>
            GetLayoutTypeAndDataType(KernelGraph const& graph, int ldsTag)
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;

            std::optional<LayoutType> layoutType;
            std::optional<DataType>   dataType;

            auto isDataFlow = [&](int tag) -> bool {
                return graph.coordinates.get<CT::DataFlow>(tag).has_value();
            };

            auto isLoadPredicate = [&graph](int x) {
                return graph.control.get<LoadTiled>(x).has_value()
                       || graph.control.get<LoadLDSTile>(x).has_value();
            };

            auto target = ldsTag;
            while(true)
            {
                if(not dataType)
                {
                    for(auto conn : graph.mapper.getCoordinateConnections(target))
                    {
                        if(isLoadPredicate(conn.control))
                            dataType = getVariableType(graph, conn.control).dataType;
                    }
                }

                auto edge = only(
                    filter(isDataFlow, graph.coordinates.getNeighbours<GD::Upstream>(target)));
                if(not edge)
                    break;
                target = only(graph.coordinates.getNeighbours<GD::Upstream>(edge.value())).value();

                if(not layoutType)
                {
                    auto maybeMacroTile = graph.coordinates.get<MacroTile>(target);
                    if(maybeMacroTile)
                        layoutType = maybeMacroTile->layoutType;
                }
            }

            if(layoutType && dataType)
                return std::pair<LayoutType, DataType>{layoutType.value(), dataType.value()};

            return {};
        }

        /**
         * @brief Get load operation info.
         *
         * This determines the load width for each load instruction,
         * and the number of lanes that "participate" in the load.
         */
        std::pair<uint, uint> GetLoadAndLaneWidth(KernelGraph const& graph,
                                                  int                ldsTag,
                                                  LayoutType         layoutType,
                                                  DataType           dataType,
                                                  ContextPtr         context)
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;

            auto isDataFlowEdge = CT::isEdge<CT::DataFlow>;
            auto tileTag
                = graph.coordinates.getInputNodeIndices(ldsTag, isDataFlowEdge).only().value();
            auto tile = graph.coordinates.get<MacroTile>(tileTag).value();

            auto isDirect2LDS = tile.memoryType == MemoryType::WAVE_Direct2LDS;

            auto workgroupSize = product(context->kernel()->workgroupSize());
            auto wavefrontSize = context->kernel()->wavefront_size();

            auto loadElementWidth = GetFastThreadTileIndexElementWidth(graph, tileTag);

            uint loadWidth = loadElementWidth * DataTypeInfo::Get(dataType).elementBits / 8u;
            uint laneWidth = isDirect2LDS ? workgroupSize : wavefrontSize;

            return {loadWidth, laneWidth};
        }

        /**
         * @brief Make padded blocks for the given tags.
         */
        void MakePaddedBlocks(KernelGraph&        graph,
                              Graph::Direction    direction,
                              int&                edge,
                              std::array<int, 2>& tags,
                              uint                contiguousBytes,
                              uint                paddingBytes,
                              uint                elementBits)
        {
            auto getCoordinateSize
                = [&](int tag) { return getSize(graph.coordinates.getNode(tag)); };

            auto slowSize = getCoordinateSize(tags[0]);
            auto fastSize = getCoordinateSize(tags[1]);

            auto one = literal(1u);

            auto linearFlat
                = graph.coordinates.addElement(Linear(simplify(slowSize * fastSize), one));

            AssertFatal(contiguousBytes * 8u % elementBits == 0,
                        "Padding mismatch: contiguous-bytes is not divisible by elementBits.",
                        ShowValue(contiguousBytes),
                        ShowValue(elementBits));
            AssertFatal(paddingBytes * 8u % elementBits == 0,
                        "Padding mismatch: padding-bytes is not divisible by elementBits.",
                        ShowValue(paddingBytes),
                        ShowValue(elementBits));

            auto newFastSize   = literal(contiguousBytes * 8u / elementBits);
            auto newSlowSize   = simplify(slowSize * fastSize / newFastSize);
            auto newSlowStride = simplify(newFastSize + literal(paddingBytes * 8u / elementBits));
            auto linearSlow    = graph.coordinates.addElement(Linear(newSlowSize, newSlowStride));
            auto linearFast    = graph.coordinates.addElement(Linear(newFastSize, one));

            Log::debug("KernelGraph::AddLDSPadding::MakePaddedBlocks: "
                       "slowSize {}, fastSize {}, newSlowSize {}, newSlowStride {}, newFastSize {}",
                       toString(slowSize),
                       toString(fastSize),
                       toString(newSlowSize),
                       toString(newSlowStride),
                       toString(newFastSize));

            // If tags are upstream...
            if(direction == GD::Upstream)
            {
                auto downstreamTag = *only(graph.coordinates.getNeighbours<GD::Downstream>(edge));
                graph.coordinates.deleteElement(edge);
                graph.coordinates.addElement(Flatten(), {tags[0], tags[1]}, {linearFlat});
                graph.coordinates.addElement(Tile(), {linearFlat}, {linearSlow, linearFast});
                edge = graph.coordinates.addElement(
                    Join(), {linearSlow, linearFast}, {downstreamTag});
            }
            else
            {
                auto upstreamTag = *only(graph.coordinates.getNeighbours<GD::Upstream>(edge));
                graph.coordinates.deleteElement(edge);
                edge = graph.coordinates.addElement(
                    Split(), {upstreamTag}, {linearSlow, linearFast});
                graph.coordinates.addElement(Flatten(), {linearSlow, linearFast}, {linearFlat});
                graph.coordinates.addElement(Tile(), {linearFlat}, {tags[0], tags[1]});
            }

            // Update tags to point to the new Linear nodes, these will be padded
            tags = {linearSlow, linearFast};
        }

        /**
         * @brief Add LDS padding transformer.
         */
        struct AddLDSPaddingVisitor
        {
            AddLDSPaddingVisitor(ContextPtr context, CommandParametersPtr params)
                : m_context(context)
                , m_params(params)
            {
            }

            void stage(KernelGraph const&, int);
            void commit(KernelGraph&);

        private:
            std::pair<uint, uint> getLDSPaddingElements(KernelGraph const&,
                                                        LDSPaddingInfo const&) const;

            ContextPtr           m_context;
            CommandParametersPtr m_params;

            std::map<int, LDSPaddingInfo> m_ldsTags;
        };

        /**
         * @brief Get the number of padding elements for a given LDS
         * node.
         *
         * If the padding is set to -1, this will compute a default
         * padding value.
         */
        std::pair<uint, uint>
            AddLDSPaddingVisitor::getLDSPaddingElements(KernelGraph const&    graph,
                                                        LDSPaddingInfo const& info) const
        {
            if(m_params->ldsPadding.contains(info.layoutType))
            {
                Log::debug("KernelGraph::AddLDSPadding: {}", toString(info));
                int contiguousBytes, paddingBytes;
                std::tie(contiguousBytes, paddingBytes) = m_params->ldsPadding.at(info.layoutType);
                if(contiguousBytes == -1)
                {
                    contiguousBytes = CalculateAutomaticContiguousBlockSize(info);
                    Log::debug("KernelGraph::AddLDSPadding: Automatic block size: {}",
                               contiguousBytes);
                }
                if(paddingBytes == -1)
                {
                    paddingBytes = CalculateAutomaticPaddingBytes(info, contiguousBytes);
                    Log::debug("KernelGraph::AddLDSPadding: Automatic padding bytes: {}",
                               paddingBytes);
                }
                return {contiguousBytes, paddingBytes};
            }
            return {0u, 0u};
        }

        /**
         * @brief Stage LDS nodes that are flattened/tiled.
         */
        void AddLDSPaddingVisitor::stage(KernelGraph const& graph, int ldsTag)
        {
            using GD            = Graph::Direction;
            auto flattenEdgeTag = GetEdgeTag<GD::Upstream, Flatten>(graph, ldsTag);
            auto tileEdgeTag    = GetEdgeTag<GD::Downstream, Tile>(graph, ldsTag);

            if((not flattenEdgeTag) or (not tileEdgeTag))
                return;

            auto maybeLayoutTypeAndDataType = GetLayoutTypeAndDataType(graph, ldsTag);
            if(!maybeLayoutTypeAndDataType)
            {
                Log::debug("KernelGraph::AddLDSPadding: "
                           "Could not determine layout type and data type for LDS tag {}",
                           ldsTag);
                return;
            }

            auto const& [loadWidth, laneWidth]
                = GetLoadAndLaneWidth(graph,
                                      ldsTag,
                                      maybeLayoutTypeAndDataType->first,
                                      maybeLayoutTypeAndDataType->second,
                                      m_context);

            auto upstreamTags
                = graph.coordinates.getNeighbours<GD::Upstream>(flattenEdgeTag.value());
            auto downstreamTags
                = graph.coordinates.getNeighbours<GD::Downstream>(tileEdgeTag.value());

            m_ldsTags[ldsTag] = LDSPaddingInfo{ldsTag,
                                               flattenEdgeTag.value(),
                                               tileEdgeTag.value(),
                                               {upstreamTags[0], upstreamTags[1]},
                                               {downstreamTags[0], downstreamTags[1]},
                                               maybeLayoutTypeAndDataType->second,
                                               maybeLayoutTypeAndDataType->first,
                                               loadWidth,
                                               laneWidth};
        }

        /**
         * @brief Commit LDS padding changes to the graph.
         *
         * This will change the upstream Flatten edge to a Join edge,
         * and the downstream Tile edge to a Split edge.
         */
        void AddLDSPaddingVisitor::commit(KernelGraph& graph)
        {
            for(auto& [ldsTag, info] : m_ldsTags)
            {
                auto [contiguousBytes, paddingBytes] = getLDSPaddingElements(graph, info);

                Log::debug("KernelGraph::AddLDSPadding: ldsTag {}, upstreamEdge {}, "
                           "downstreamEdge {}, contiguousBytes {} paddingBytes {}",
                           info.ldsTag,
                           info.upstreamEdge,
                           info.downstreamEdge,
                           contiguousBytes,
                           paddingBytes);

                if(contiguousBytes == 0)
                    continue;

                MakePaddedBlocks(graph,
                                 GD::Downstream,
                                 info.downstreamEdge,
                                 info.downstreamTags,
                                 contiguousBytes,
                                 paddingBytes,
                                 DataTypeInfo::Get(info.dataType).elementBits);
                MakePaddedBlocks(graph,
                                 GD::Upstream,
                                 info.upstreamEdge,
                                 info.upstreamTags,
                                 contiguousBytes,
                                 paddingBytes,
                                 DataTypeInfo::Get(info.dataType).elementBits);
            }
        }

        KernelGraph AddLDSPadding::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddLDSPadding");
            auto graph   = original;
            auto visitor = AddLDSPaddingVisitor(m_context, m_params);
            for(auto ldsTag : graph.coordinates.getNodes<LDS>())
                visitor.stage(graph, ldsTag);
            visitor.commit(graph);
            return graph;
        }
    }
}
