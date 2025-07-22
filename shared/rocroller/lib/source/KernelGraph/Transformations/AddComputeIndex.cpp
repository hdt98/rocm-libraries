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

#include <algorithm>
#include <variant>

#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddComputeIndex.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/KernelGraph/Transforms/LoadPacked_detail.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile_details.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;
    namespace Expression = rocRoller::Expression;
    using namespace Expression;

    using GD = Graph::Direction;

    struct ComputeIndexChainSpecification
    {
        int              target;
        std::vector<int> coords;
        int              location;
        Graph::Direction direction;
        int              forLoop          = -1;
        bool             replaceWithScope = true;
    };

    bool operator<(const ComputeIndexChainSpecification& a, const ComputeIndexChainSpecification& b)
    {
        return std::tie(a.target, a.coords, a.location, a.direction)
               < std::tie(b.target, b.coords, b.location, b.direction);
    }

    struct ComputeIndexChain
    {
        int top, bottom;

        std::vector<DeferredConnection> connections;

        int update = -1;
    };

    struct RequiredCoordinateInfo
    {
        int  coord, base, sdim;
        bool isUnroll;
        bool needsUpdate;
    };

    using BufferMap = std::map<int, int>;

    /**
     * @brief Return existing Buffer for load/stores from/to `dst`.
     *
     * Returns -1 if the operation doesn't need a buffer descriptor.
     *
     * If a Buffer edge doesn't already exist, we create a new
     * Workgroup coordinate and attach it with a Buffer edge to the
     * `dst`.
     */
    int getBuffer(KernelGraph& graph, int opTag, int dst, BufferMap& bufferMap, bool isDirect2LDS)
    {
        auto op = graph.control.getElement(opTag);
        if(isOperation<LoadLDSTile>(op) || isOperation<StoreLDSTile>(op) || isDirect2LDS)
            return -1;

        if(!bufferMap.contains(dst))
        {
            auto wg        = graph.coordinates.addElement(Workgroup());
            bufferMap[dst] = graph.coordinates.addElement(Buffer(), {wg}, {dst});
        }

        return bufferMap[dst];
    }

    /**
     * @brief True if ForLoopOp has a translate-time increment.
     */
    bool uniformForLoop(std::optional<int> maybeForLoop, KernelGraph const& kgraph)
    {
        if(!maybeForLoop)
            return false;

        auto [lhs, rhs] = getForLoopIncrement(kgraph, *maybeForLoop);
        return evaluationTimes(rhs)[EvaluationTime::Translate];
    }

    /**
     * @brief Add a ComputeIndex node and add mapper connections.
     */
    int makeComputeIndex(KernelGraph& graph,
                         int          target,
                         int          increment,
                         int          base,
                         int          offset,
                         int          stride,
                         int          buffer,
                         bool         forward,
                         DataType     valueType,
                         DataType     offsetType,
                         DataType     strideType,
                         bool         isDirect2LDS)
    {
        using CCI = Connections::ComputeIndex;
        using CCA = Connections::ComputeIndexArgument;

        auto ci = graph.control.addElement(
            ComputeIndex{forward, isDirect2LDS, valueType, offsetType, strideType});

        if(base > 0)
            graph.mapper.connect(ci, base, CCI{CCA::BASE});
        if(buffer > 0)
            graph.mapper.connect(ci, buffer, CCI{CCA::BUFFER});
        if(increment > 0)
            graph.mapper.connect(ci, increment, CCI{CCA::INCREMENT});
        if(offset > 0)
            graph.mapper.connect(ci, offset, CCI{CCA::OFFSET});
        if(stride > 0)
            graph.mapper.connect(ci, stride, CCI{CCA::STRIDE});
        if(target > 0)
            graph.mapper.connect(ci, target, CCI{CCA::TARGET});

        rocRoller::Log::getLogger()->debug(
            "KernelGraph::makeComputeIndex: ci {} {}/{} {}; {}/{}/{}",
            ci,
            target,
            increment,
            forward,
            base,
            offset,
            stride);

        return ci;
    }

    ExpressionPtr toBytes(ExpressionPtr expr, DataType valueType)
    {
        uint numBits = DataTypeInfo::Get(valueType).elementBits;
        if(numBits % 8u == 0)
            return expr * Expression::literal(numBits / 8u);
        return (expr * Expression::literal(numBits)) / Expression::literal(8u);
    }

    inline ExpressionPtr L(auto const& x)
    {
        return Expression::literal(x);
    }

    int makeAssignBase(KernelGraph& graph,
                       int          target,
                       int          base,
                       int          offset,
                       bool         forward,
                       DataType     valueType,
                       DataType     offsetType,
                       bool         maybeLDS,
                       bool         isTransposed,
                       ContextPtr   context,
                       Transformer& coords)
    {
        auto        indexExpr = forward ? coords.forward({target})[0] : coords.reverse({target})[0];
        auto const& typeInfo  = DataTypeInfo::Get(valueType);
        auto        numBits   = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;

        auto const& arch = context->targetArchitecture();
        const auto  needsPadding
            = numBits == 6 && isTransposed
              && arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes);

        ExpressionPtr paddingBytes{L(0u)};
        if(needsPadding && maybeLDS)
        {
            uint elementsPerTrLoad = bitsPerTransposeLoad(arch, numBits) / numBits;
            auto extraLdsBytes     = extraLDSBytesPerElementBlock(arch, numBits);
            paddingBytes           = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
        }

        auto assignNode
            = Assign{Register::Type::Vector, convert(offsetType, toBytes(indexExpr, valueType) + paddingBytes)};
        assignNode.variableType = offsetType;
        auto assignTag          = graph.control.addElement(assignNode);
        graph.mapper.connect(assignTag, offset, NaryArgument::DEST);

        rocRoller::Log::getLogger()->debug(
            "KernelGraph::makeAssignBase: assign {} expression {} to offset {}",
            assignTag,
            toString(assignNode.expression),
            offset);
        // std::cout << "YL: makeAssignBase tag " << assignTag << ", expression " << toString(assignNode.expression) << ", target " << target <<
        // ", offset " << offset << std::endl;
        std::cout << "YL makeAssignOffset (tag, expression, offset) " << assignTag << ", "
                  << toString(assignNode.expression) << ", " << offset << std::endl;

        return assignTag;
    }

    static std::pair<uint, uint>
        getElementBlockValues(KernelGraph const& graph, int target, const bool isTransposed)
    {
        uint elementBlockNumber = 0;
        uint elementBlockIndex  = 0;

        std::unordered_set<int> tileTags;
        using OpsAndTilesType
            = std::tuple<std::pair<int, Operation>, std::pair<int, MacroTile>, DataType>;
        std::vector<OpsAndTilesType> targetOpsAndTiles;

        for(auto conn : graph.mapper.getCoordinateConnections(target))
        {
            auto     opTag = conn.control;
            auto     op    = std::get<Operation>(graph.control.getElement(opTag));
            DataType dataType;
            if(std::visit(rocRoller::overloaded{[&](LoadTiled& load) {
                                                    dataType = load.varType.dataType;
                                                    return true;
                                                },
                                                [&](LoadLDSTile& load) {
                                                    dataType = load.varType.dataType;
                                                    return true;
                                                },
                                                [&](StoreTiled& store) {
                                                    dataType = store.varType.dataType;
                                                    return true;
                                                },
                                                [&](StoreLDSTile& store) {
                                                    dataType = store.varType.dataType;
                                                    return true;
                                                },
                                                [&](auto& other) { return false; }},
                          op))
            {
                auto [macTileTag, macTile] = graph.getDimension<MacroTile>(opTag);

                auto maybeParentTile = only(graph.coordinates.getOutputNodeIndices(
                    macTileTag, rocRoller::KernelGraph::CoordinateGraph::isEdge<Duplicate>));
                if(maybeParentTile)
                {
                    macTileTag = *maybeParentTile;
                    macTile    = *graph.coordinates.get<MacroTile>(macTileTag);
                }

                if(!tileTags.count(macTileTag))
                {
                    targetOpsAndTiles.push_back({{opTag, op}, {macTileTag, macTile}, dataType});
                }
            }
        }

        auto [tagAndOp, tagAndTile, dataType] = [](auto opsAndTiles) -> OpsAndTilesType {
            for(OpsAndTilesType& elem : opsAndTiles)
            {
                auto memType = std::get<1>(elem).second.memoryType;
                if(memType == MemoryType::WAVE || memType == MemoryType::WAVE_SWIZZLE)
                {
                    return elem;
                }
            }
            return opsAndTiles[0];
        }(targetOpsAndTiles);
        auto [opTag, op]           = tagAndOp;
        auto [macTileTag, macTile] = tagAndTile;

        if(macTile.memoryType == MemoryType::VGPR
           || (macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR
               && macTile.memoryType == MemoryType::WAVE_SPLIT))
        {
            auto [elementNumberXTag, elementNumberX] = graph.getDimension<ElementNumber>(opTag, 0);
            AssertFatal(Expression::evaluationTimes(elementNumberX.size)[EvaluationTime::Translate],
                        "Could not determine ElementNumberX size at translate-time.\n",
                        ShowValue(elementNumberX));

            auto [elementNumberYTag, elementNumberY] = graph.getDimension<ElementNumber>(opTag, 1);
            AssertFatal(Expression::evaluationTimes(elementNumberY.size)[EvaluationTime::Translate],
                        "Could not determine ElementNumber size at translate-time.\n",
                        ShowValue(elementNumberY));

            elementBlockNumber = getUnsignedInt(evaluate(elementNumberX.size));
            elementBlockIndex  = getUnsignedInt(evaluate(elementNumberY.size));
        }
        else if(macTile.memoryType == MemoryType::WAVE
                || macTile.memoryType == MemoryType::WAVE_SWIZZLE)
        {
            auto [vgprBlockNumberTag, vgprBlockNumber]
                = graph.getDimension<VGPRBlockNumber>(opTag, 0);
            AssertFatal(
                Expression::evaluationTimes(vgprBlockNumber.size)[EvaluationTime::Translate],
                "Could not determine VGPRBlockNumber size at translate-time.\n",
                ShowValue(vgprBlockNumber));

            auto [vgprBlockIndexTag, vgprBlockIndex] = graph.getDimension<VGPRBlockIndex>(opTag, 0);
            AssertFatal(Expression::evaluationTimes(vgprBlockIndex.size)[EvaluationTime::Translate],
                        "Could not determine VGPRBlockIndex size at translate-time.\n",
                        ShowValue(vgprBlockIndex));

            elementBlockNumber = getUnsignedInt(evaluate(vgprBlockNumber.size));
            elementBlockIndex  = getUnsignedInt(evaluate(vgprBlockIndex.size));
            if(isScaleType(dataType))
            {
                // Scales are another special case here. For Scales we need
                // to get VGPR coordinate instead of VGPRBlockNumber/Index
                // (see addLoadSwizzleTileCT).
                auto [vgprTag, vgpr] = graph.getDimension<VGPR>(opTag, 0);
                AssertFatal(Expression::evaluationTimes(vgpr.size)[EvaluationTime::Translate],
                            "Could not determine VGPR size at translate-time.\n",
                            ShowValue(vgpr));
                // Multiplying by elementBlockNumber here forces the use
                // of the widest load/store possible
                elementBlockIndex = elementBlockNumber * getUnsignedInt(evaluate(vgpr.size));
            }

            if((!LowerTileDetails::isTileOfSubDwordTypeWithNonContiguousVGPRBlocks(
                    dataType,
                    {.m = macTile.subTileSizes[0],
                     .n = macTile.subTileSizes[1],
                     .k = macTile.subTileSizes[2]})
                || isScaleType(dataType))
               && !isTransposed)
            {
                // For Scales and other kinds of tiles, VGPRBlockIndex holds
                // number of VGPR per block and not elements per VGPRBlock.
                elementBlockIndex *= packingFactorForDataType(dataType);
            }
        }
        else
        {
            Throw<FatalError>(
                "Could not find ElementNumber or VGPRBlockNumber/Index coordinates.\n",
                ShowValue(op),
                ShowValue(macTile));
        }

        AssertFatal(elementBlockNumber > 0 && elementBlockIndex > 0,
                    "elemementBlockNumber & elementBlockIndex must be greater than zero. ",
                    ShowValue(elementBlockNumber),
                    ShowValue(elementBlockIndex));
        return {elementBlockNumber, elementBlockIndex};
    }

    int makeAssignStride(KernelGraph& graph,
                         int          target,
                         int          stride,
                         int          increment,
                         bool         forward,
                         DataType     valueType,
                         DataType     strideType,
                         bool         maybeLDS,
                         bool         isTransposed,
                         ContextPtr   context,
                         Transformer& coords)
    {
        auto indexExpr = forward
                             ? coords.forwardStride(increment, Expression::literal(1), {target})[0]
                             : coords.reverseStride(increment, Expression::literal(1), {target})[0];

        bool unitStride = false;
        if(Expression::evaluationTimes(indexExpr)[EvaluationTime::Translate])
        {
            if(getUnsignedInt(evaluate(indexExpr)) == 1u)
                unitStride = true;
        }

        uint          elementBlockSize = 0;
        ExpressionPtr elementBlockStride;
        ExpressionPtr trLoadPairStride;
        ExpressionPtr elementBlockStridePaddingBytes{L(0u)};
        ExpressionPtr trLoadPairStridePaddingBytes{L(0u)};
        ExpressionPtr indexExprPaddingBytes{L(0u)};

        auto const& typeInfo = DataTypeInfo::Get(valueType);
        auto        numBits  = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;

        if(numBits == 16 || numBits == 8 || numBits == 6 || numBits == 4)
        {
            auto [elementBlockNumber, elementBlockIndex]
                = getElementBlockValues(graph, target, isTransposed);

            elementBlockSize = elementBlockIndex;

            auto const& arch = context->targetArchitecture();
            if(isTransposed)
            {
                // See addLoadWaveTileCTF8F6F4 in LowerTile.cpp
                const auto wfs        = arch.GetCapability(GPUCapability::DefaultWavefrontSize);
                uint const numVBlocks = wfs == 64 ? (numBits == 8 ? 2 : 1) : (numBits == 8 ? 4 : 2);
                elementBlockSize      = (elementBlockNumber / numVBlocks) * elementBlockSize;
            }
            AssertFatal(elementBlockSize > 0, "Invalid elementBlockSize: ", elementBlockSize);

            const auto needsPadding
                = numBits == 6 && isTransposed
                  && arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes);

            // Padding is added after every 16 elements, thus for F6 datatypes that will
            // be transpose loaded from LDS elementBlockSize is set to 16 instead of 32.
            if(needsPadding)
            {
                elementBlockSize = 16;
            }

            elementBlockStride
                = forward ? coords.forwardStride(increment, L(elementBlockSize), {target})[0]
                          : coords.reverseStride(increment, L(elementBlockSize), {target})[0];

            uint elementsPerTrLoad = elementBlockIndex;
            trLoadPairStride
                = forward ? coords.forwardStride(increment, L(elementsPerTrLoad), {target})[0]
                          : coords.reverseStride(increment, L(elementsPerTrLoad), {target})[0];

            if(needsPadding && maybeLDS)
            {
                uint elementsPerTrLoad = bitsPerTransposeLoad(arch, numBits) / numBits;
                auto extraLdsBytes     = extraLDSBytesPerElementBlock(arch, numBits);
                elementBlockStridePaddingBytes
                    = elementBlockStride / L(elementsPerTrLoad) * L(extraLdsBytes);
                trLoadPairStridePaddingBytes
                    = trLoadPairStride / L(elementsPerTrLoad) * L(extraLdsBytes);
                indexExprPaddingBytes = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
            }
        }

        auto assignNode
            = Assign{Register::Type::Vector, toBytes(indexExpr, valueType) + indexExprPaddingBytes};
        assignNode.variableType = strideType;
        assignNode.strideExpressionAttributes
            = {strideType,
               unitStride,
               elementBlockSize,
               toBytes(elementBlockStride, valueType) + elementBlockStridePaddingBytes,
               toBytes(trLoadPairStride, valueType) + trLoadPairStridePaddingBytes};
        auto assignTag = graph.control.addElement(assignNode);
        graph.mapper.connect(assignTag, stride, NaryArgument::DEST);

        rocRoller::Log::getLogger()->debug(
            "KernelGraph::makeAssignStride: assign {} expression {} to stride {}",
            assignTag,
            toString(assignNode.expression),
            stride);
        std::cout << "YL makeAssignStride (tag, expression, stride) " << assignTag << ", "
                  << toString(assignNode.expression) << ", " << stride << std::endl;
        return assignTag;
    }

    /**
     * @brief Get coordinates in `path` attached to `coordinate` via a
     * CoordinateTransformEdge.
     */
    int getNeighbourNodeInPath(int                            coordinate,
                               Graph::Direction               direction,
                               std::unordered_set<int> const& path,
                               KernelGraph const&             graph)
    {
        auto neighbourNodes
            = (direction == Graph::Direction::Upstream)
                  ? graph.coordinates
                        .getOutputNodeIndices(coordinate,
                                              rocRoller::KernelGraph::CoordinateGraph::isEdge<
                                                  CoordinateTransformEdge>)
                        .to<std::unordered_set>()
                  : graph.coordinates
                        .getInputNodeIndices(coordinate,
                                             rocRoller::KernelGraph::CoordinateGraph::isEdge<
                                                 CoordinateTransformEdge>)
                        .to<std::unordered_set>();

        for(auto tag : neighbourNodes)
        {
            if(path.contains(tag))
                return tag;
        }

        return -1;
    }

    /**
     * @brief Get list of required coordinates, and how they relate to
     * each other.
     *
     * Builds a list of coordinates, slow-to-fast, that need
     * offset/strides for operation `op`.
     */
    std::vector<RequiredCoordinateInfo> getRequiredCoordinatesInfo(int                op,
                                                                   int                location,
                                                                   KernelGraph const& graph,
                                                                   bool isDirect2LDS = false)
    {
        auto [target, direction] = getOperationTarget(op, graph, isDirect2LDS);
        auto [required, path]    = findRequiredCoordinates(target, direction, graph);
        auto codegen             = getCodeGeneratorCoordinates(graph, op, isDirect2LDS);

        std::set<int>    isForLoop, isUnroll;
        std::vector<int> ordered;

        // If location is a ForLoop, its coordinate is the slowest.
        if(location != -1)
        {
            auto maybeForLoop = graph.control.get<ForLoopOp>(location);
            if(maybeForLoop)
            {
                auto forLoopCoord = graph.mapper.get<ForLoop>(location);
                auto coord        = getNeighbourNodeInPath(forLoopCoord, direction, path, graph);

                if(coord != -1)
                {
                    ordered.push_back(coord);
                    isForLoop.insert(coord);
                }
            }
        }

        // Next, consider Unroll coordinates.
        auto unrolls = filterCoordinates<Unroll>(required, graph);
        for(auto unroll : unrolls)
        {
            std::vector<int> neighbourNodes;
            if(direction == Graph::Direction::Upstream)
                neighbourNodes = graph.coordinates.childNodes(unroll).to<std::vector>();
            else
                neighbourNodes = graph.coordinates.parentNodes(unroll).to<std::vector>();
            for(auto neighbourNode : neighbourNodes)
            {
                if(path.contains(neighbourNode) && !isForLoop.contains(neighbourNode))
                {
                    auto it = std::find(codegen.cbegin(), codegen.cend(), neighbourNode);
                    if(it == codegen.cend())
                    {
                        ordered.push_back(neighbourNode);
                        isUnroll.insert(neighbourNode);
                    }
                }
            }
        }

        // Finally, the code-gen coordinates are the fastest moving.
        for(auto x : codegen)
            ordered.push_back(x);

        // Now build list... the slowest dimension doesn't have a
        // "base"; subsequent dimensions use the previous one as their
        // base.
        std::vector<RequiredCoordinateInfo> rv;

        int base = -1;
        for(auto coord : ordered)
        {
            // Compute the sub-dimension for code-gen coordinates.
            // TODO Slow to fast; lift this from Tensor directly
            int sdim = -1;
            {
                auto it = std::find(codegen.cbegin(), codegen.cend(), coord);
                if(it != codegen.cend())
                    sdim = std::distance(codegen.cbegin(), it);
            }

            if(isDirect2LDS)
            {
                sdim += ordered.size();
            }

            if(!isUnroll.contains(coord))
            {
                auto needsUpdate = isForLoop.contains(coord) && uniformForLoop(location, graph);
                rv.push_back({coord, base, sdim, false, needsUpdate});
                base = coord;
            }
            else
            {
                rv.push_back({coord, -1, -1, true, false});
            }
        }

        return rv;
    }

    /**
     * @brief Return datatype that should be used for the offset when
     * generating `op`.
     */
    DataType getOffsetDataType(int op, KernelGraph const& graph, bool direct2LDS)

    {
        DataType rv = DataType::UInt64;
        auto     s  = graph.control.get<StoreTiled>(op);
        auto     l  = graph.control.get<LoadTiled>(op);
        auto     ll = graph.control.get<LoadLDSTile>(op);
        auto     sl = graph.control.get<StoreLDSTile>(op);
        if(s || l || ll || sl || direct2LDS)
        {
            rv = DataType::UInt32;
        }
        return rv;
    }

    /**
     * @brief Add ComputeIndex nodes required for `op`.
     */
    ComputeIndexChain addComputeIndex(KernelGraph&  graph,
                                      int           op,
                                      ExpressionPtr step,
                                      int           location,
                                      BufferMap&    bufferMap,
                                      bool          isDirect2LDS,
                                      ContextPtr    context)
    {
        rocRoller::Log::getLogger()->debug(
            "KernelGraph::AddComputeIndex()::genericComputeIndex(): op {} location {}",
            op,
            location);

        auto dtype = getDataType(graph.control.getNode(op));

        auto [target, direction] = getOperationTarget(op, graph, isDirect2LDS);

        int                             update = -1;
        std::vector<int>                chain;
        std::vector<DeferredConnection> connections;
        std::map<int, int>              offsetOfCoord;

        auto [coords, remainingCoords] = LoadPackedDetail::getFakeTransformerForControlNode(
            op, graph, context, isDirect2LDS, true);

        for(auto coord : remainingCoords)
        {
            coords.setCoordinate(coord, Expression::literal(0u));
        }

        for(auto info : getRequiredCoordinatesInfo(op, location, graph, isDirect2LDS))
        {
            // Add ComputeIndex operation
            int offset = -1, stride = -1, buffer = -1;
            if(direction == Graph::Direction::Downstream)
            {
                if(!info.isUnroll)
                    offset = graph.coordinates.addElement(Offset(), {target}, {info.coord});
                stride = graph.coordinates.addElement(Stride(), {target}, {info.coord});
                if(info.base == -1 && offset != -1)
                    buffer = getBuffer(graph, op, target, bufferMap, isDirect2LDS);
            }
            else
            {
                if(!info.isUnroll)
                    offset = graph.coordinates.addElement(Offset(), {info.coord}, {target});
                stride = graph.coordinates.addElement(Stride(), {info.coord}, {target});
                if(info.base == -1 && offset != -1)
                    buffer = getBuffer(graph, op, target, bufferMap, isDirect2LDS);
            }

            offsetOfCoord[info.coord] = offset;

            int base = (info.base == -1) ? -1 : offsetOfCoord.at(info.base);

            // For future: choose type based on buffer or non-buffer
            auto offsetDataType = getOffsetDataType(op, graph, isDirect2LDS);
            auto strideDataType = DataType::UInt64;

            if(info.isUnroll)
            {
                offsetDataType = DataType::Int64;
                strideDataType = DataType::Int64;
            }

            auto ci = makeComputeIndex(graph,
                                             target,
                                             info.coord,
                                             base,
                                             offset,
                                             stride,
                                             buffer,
                                             direction == Graph::Direction::Upstream,
                                             dtype,
                                             offsetDataType,
                                             strideDataType,
                                             isDirect2LDS);

            chain.push_back(ci);

            // if (isDirect2LDS)
            // {
            //     std::cout << "YL: node " << op << " target " << target << std::endl;
            // }

            // determine if target is LDS
            auto maybeLDS  = graph.coordinates.get<LDS>(target).has_value();
            auto newTarget = target;
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of computing indexes,
                // use the parent LDS as the target instead.
                auto maybeParentLDS = only(graph.coordinates.getOutputNodeIndices(
                    newTarget, rocRoller::KernelGraph::CoordinateGraph::isEdge<Duplicate>));
                if(maybeParentLDS)
                    newTarget = *maybeParentLDS;
            }
            maybeLDS = graph.coordinates.get<LDS>(newTarget).has_value();

            // if (isDirect2LDS)
            // {
            //     std::cout << "YL: node " << op << " new target" << newTarget << std::endl;
            // }

            // determin if the operation is tranpose load
            auto isLoad       = graph.control.get<LoadTiled>(op).has_value();
            auto isLoadLDS    = graph.control.get<LoadLDSTile>(op).has_value();
            auto isTransposed = false;
            if(isLoad)
            {
                auto tile    = graph.control.get<LoadTiled>(op).value();
                isTransposed = tile.isTransposedTile;
            }
            else if(isLoadLDS)
            {
                auto tile    = graph.control.get<LoadLDSTile>(op).value();
                isTransposed = tile.isTransposedTile;
            }

            // make Assign base expression
            // Set the zero-coordinates to zero
            // auto coords           = Transformer(&graph.coordinates);
            auto increment        = info.coord;
            auto fullStop         = [&](int ciTag) { return ciTag == increment; };
            auto [required, path] = findRequiredCoordinates(newTarget, direction, fullStop, graph);

            for(auto ciTag : required)
                if((ciTag != increment) && (!coords.hasCoordinate(ciTag)))
                    coords.setCoordinate(ciTag, Expression::literal(0u));

            // Set the increment coordinate to zero if it doesn't
            // already have a value
            bool initializeIncrement
                = !coords.hasPath({newTarget}, direction == Graph::Direction::Upstream);
            if(initializeIncrement)
            {
                coords.setCoordinate(increment, Expression::literal(0u));
            }

            // Assign base expression
            // if(base < 0 && offset > 0)
            // {
            //     chain.push_back(makeAssignBase(graph,
            //                                    newTarget,
            //                                    base,
            //                                    offset,
            //                                    direction == Graph::Direction::Upstream,
            //                                    dtype,
            //                                    offsetDataType,
            //                                    maybeLDS,
            //                                    isTransposed,
            //                                    context,
            //                                    coords));
            // }

            if(stride > 0)
            {
                chain.push_back(makeAssignStride(graph,
                                                 newTarget,
                                                 stride,
                                                 increment,
                                                 direction == Graph::Direction::Upstream,
                                                 dtype,
                                                 strideDataType,
                                                 maybeLDS,
                                                 isTransposed,
                                                 context,
                                                 coords));
            }

            // if (isDirect2LDS)
            // {
            //     std::cout << "YL: node " << op << " target " << target << std::endl;
            // }

            // determine if target is LDS
            auto maybeLDS  = graph.coordinates.get<LDS>(target).has_value();
            auto newTarget = target;
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of computing indexes,
                // use the parent LDS as the target instead.
                auto maybeParentLDS = only(graph.coordinates.getOutputNodeIndices(
                    newTarget, rocRoller::KernelGraph::CoordinateGraph::isEdge<Duplicate>));
                if(maybeParentLDS)
                    newTarget = *maybeParentLDS;
            }
            maybeLDS = graph.coordinates.get<LDS>(newTarget).has_value();

            // if (isDirect2LDS)
            // {
            //     std::cout << "YL: node " << op << " new target" << newTarget << std::endl;
            // }

            // determin if the operation is tranpose load
            auto isLoad       = graph.control.get<LoadTiled>(op).has_value();
            auto isLoadLDS    = graph.control.get<LoadLDSTile>(op).has_value();
            auto isTransposed = false;
            if(isLoad)
            {
                auto tile    = graph.control.get<LoadTiled>(op).value();
                isTransposed = tile.isTransposedTile;
            }
            else if(isLoadLDS)
            {
                auto tile    = graph.control.get<LoadLDSTile>(op).value();
                isTransposed = tile.isTransposedTile;
            }

            // make Assign base expression
            // Set the zero-coordinates to zero
            // auto coords           = Transformer(&graph.coordinates);
            auto increment        = info.coord;
            auto fullStop         = [&](int ciTag) { return ciTag == increment; };
            auto [required, path] = findRequiredCoordinates(newTarget, direction, fullStop, graph);

            for(auto ciTag : required)
                if((ciTag != increment) && (!coords.hasCoordinate(ciTag)))
                    coords.setCoordinate(ciTag, Expression::literal(0u));

            // Set the increment coordinate to zero if it doesn't
            // already have a value
            bool initializeIncrement
                = !coords.hasPath({newTarget}, direction == Graph::Direction::Upstream);
            if(initializeIncrement)
            {
                coords.setCoordinate(increment, Expression::literal(0u));
            }

            // Assign base expression
            if(base < 0)
            {
                chain.push_back(makeAssignBase(graph,
                                               newTarget,
                                               base,
                                               offset,
                                               direction == Graph::Direction::Upstream,
                                               dtype,
                                               offsetDataType,
                                               maybeLDS,
                                               isTransposed,
                                               context,
                                               coords));
            }

            if(stride > 0)
            {
                chain.push_back(makeAssignStride(graph,
                                                 newTarget,
                                                 stride,
                                                 increment,
                                                 direction == Graph::Direction::Upstream,
                                                 dtype,
                                                 strideDataType,
                                                 maybeLDS,
                                                 isTransposed,
                                                 context,
                                                 coords));
            }

            // Add connections for register allocate, and so tracer
            // can determine correct lifetimes
            if(offset != -1)
                connections.push_back(DC<Offset>(offset, info.sdim));
            if(stride != -1)
                connections.push_back(DC<Stride>(stride, info.sdim));
            if(buffer != -1)
                connections.push_back(DC<Buffer>(buffer));

            if(info.needsUpdate)
            {
                auto offsetExpr = std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{offset, Register::Type::Vector, offsetDataType});
                auto strideExpr = std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{stride, Register::Type::Scalar, strideDataType});

                if(step == nullptr)
                    update = graph.control.addElement(Assign{
                        Register::Type::Vector, convert(offsetDataType, offsetExpr + strideExpr)});
                else
                    update = graph.control.addElement(
                        Assign{Register::Type::Vector,
                               convert(offsetDataType, offsetExpr + step * strideExpr)});
                graph.mapper.connect(update, offset, NaryArgument::DEST);
            }
        }

        for(int i = 1; i < chain.size(); ++i)
            graph.control.addElement(Sequence(), {chain[i - 1]}, {chain[i]});

        return {chain.front(), chain.back(), connections, update};
    }

    /**
     * @brief Add ComputeIndex operations.
     *
     * Adding ComputeIndex operations to the control graph is done in
     * two phases: staging and committing.
     *
     * During the staging phase, we look at all load/store operations
     * in the control graph and "stage" the addition of ComputeIndex
     * operations.  During the staging phase, we are able to detect
     * when two or more load/store operations would result in the same
     * chain of ComputeIndex operations, and eliminate any
     * redundancies.
     *
     * Usually ComputeIndex operations come in sequential groups of
     * two or more operations, and hence we call them "compute index
     * chains".
     *
     * During the commit stage, we add ComputeIndex operations to the
     * graphs, and add connections for load/store operations to the
     * newly created Base, Offset, and Stride elements of the
     * coordinate graph.
     *
     * For each candidate load/store operation:
     *
     * 1. The type of ComputeIndex chain is determined.
     *
     * 2. The required location of the ComputeIndex chain is
     *    determined.
     *
     * 3. The chain is staged.
     *
     * To determined where the chain should be placed:
     *
     * 1. Find all required coordinates by querying the Coordinate
     *    Transform graph.
     *
     * 2. If one-or-more Unroll dimension(s) are required:
     *
     *    a. Find SetCoordinate operations above the candidate and
     *       record the values of required Unroll dimensions.
     *
     *    b. Find the earliest matching set of SetCoordinate
     *       operations that are identical (ie, Unroll dimension and
     *       value) to the required Unroll dimensions.
     *
     *    c. The chain is added below the SetCoordinate operation from
     *       (b).
     *
     * 3. If a ForLoop dimension is required, find the containing
     *    ForLoop operation.  The chain is added above the ForLoop
     *    operation.
     *
     * 4. If both ForLoop and Unroll dimensions are required, the
     *    chain is added above the containing ForLoop.
     */
    struct AddComputeIndexer
    {
        void stageChain(KernelGraph const& graph,
                        int                target,
                        int                candidate,
                        int                location,
                        Graph::Direction   direction,
                        bool               isDirect2LDS     = false,
                        int                forLoop          = -1,
                        bool               replaceWithScope = true)
        {
            std::vector<int> specCoords;
            for(auto info : getRequiredCoordinatesInfo(candidate, location, graph, isDirect2LDS))
            {
                specCoords.push_back(info.coord);
            }

            ComputeIndexChainSpecification spec{
                target, specCoords, location, direction, forLoop, replaceWithScope};
            m_chains[spec].push_back(candidate);
        }

        void stage(KernelGraph const& kgraph, int candidate, bool isDirect2LDS)
        {
            auto log = rocRoller::Log::getLogger();

            auto node = kgraph.control.getNode<Operation>(candidate);
            log->debug("KernelGraph::addComputeIndex({}): {}", candidate, toString(node));

            auto [target, direction] = getOperationTarget(candidate, kgraph, isDirect2LDS);
            auto [required, path]    = findRequiredCoordinates(target, direction, kgraph);
            auto forLoopCoordinates  = filterCoordinates<ForLoop>(required, kgraph);
            auto unrollCoordinates   = filterCoordinates<Unroll>(required, kgraph);

            log->debug("  target: {}", target);
            for(auto r : required)
            {
                log->debug("  required: {}: {}", r, toString(kgraph.coordinates.getNode(r)));
            }

            auto maybeForLoop  = findContainingOperation<ForLoopOp>(candidate, kgraph);
            auto maybeScope    = findContainingOperation<Scope>(candidate, kgraph);
            auto hasForLoop    = !forLoopCoordinates.empty();
            auto hasUnroll     = !unrollCoordinates.empty();
            auto isUniformLoop = maybeForLoop && uniformForLoop(maybeForLoop, kgraph);

            if(hasForLoop && isUniformLoop)
            {
                log->debug("  staged as: hasForLoop and isUniformLoop, location {} forLoopOp {}",
                           *maybeForLoop,
                           *maybeForLoop);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeForLoop,
                           GD::Upstream,
                           isDirect2LDS,
                           *maybeForLoop);
                return;
            }

            // Prefetching
            // Find all children ForLoopOps. If any forLoopCoordinates are associated with the
            // children ForLoopOps, this is a prefetch.
            auto allChildForLoops
                = kgraph.control
                      .findNodes(
                          getTopSetCoordinate(kgraph, candidate),
                          [&](int tag) -> bool {
                              return isOperation<ForLoopOp>(kgraph.control.getElement(tag));
                          },
                          GD::Downstream)
                      .to<std::vector>();

            if(hasForLoop
               && std::any_of(allChildForLoops.begin(), allChildForLoops.end(), [&](auto tag) {
                      return forLoopCoordinates.count(kgraph.mapper.get<ForLoop>(tag)) > 0;
                  }))
            {
                log->debug("  staged as: hasForLoop and requiresDownstreamForLoop, location {} "
                           "forLoopOp {}",
                           *maybeForLoop,
                           *maybeForLoop);
                stageChain(kgraph, target, candidate, *maybeScope, GD::Upstream, isDirect2LDS, -1);
                return;
            }

            if(maybeForLoop && !isUniformLoop && hasUnroll)
            {
                auto maybeTopOfLoop = findTopOfContainingOperation<ForLoopOp>(candidate, kgraph);
                log->debug("  staged as: hasForLoop and not isUniformLoop, location {}, {}",
                           *maybeForLoop,
                           *maybeTopOfLoop);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeTopOfLoop,
                           GD::Upstream,
                           isDirect2LDS,
                           -1,
                           false);
                return;
            }

            if(hasUnroll)
            {
                log->debug("  staged as: hasUnroll");

                auto kernel = *kgraph.control.roots().begin();
                stageChain(kgraph, target, candidate, kernel, GD::Downstream, isDirect2LDS, -1);
                return;
            }

            if(isUniformLoop)
            {
                auto forLoop = *maybeForLoop;
                log->debug("  staged as: uniformForLoop, forLoopOp {}", forLoop);

                stageChain(kgraph, target, candidate, forLoop, GD::Upstream, isDirect2LDS, forLoop);
                return;
            }

            log->debug("  staged as: immediate");
            stageChain(kgraph, target, candidate, candidate, GD::Upstream, isDirect2LDS);
        }

        KernelGraph commit(KernelGraph const& original, ContextPtr context) const
        {
            auto               kgraph = original;
            std::map<int, int> scopes;
            BufferMap          bufferMap;

            for(auto const& [spec, candidates] : m_chains)
            {
                ExpressionPtr step = Expression::literal(1u);
                if(spec.forLoop > 0)
                {
                    auto [lhs, rhs] = getForLoopIncrement(kgraph, spec.forLoop);
                    step            = simplify(rhs);
                }

                auto isDirect2LDS
                    = (original.control.get<LoadTileDirect2LDS>(candidates[0]).has_value()
                       && original.coordinates.get<LDS>(spec.target).has_value());

                // Use first candidate to compute indexes
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::AddComputeIndex()::commit({}) isDirect2LDS({})",
                    candidates[0],
                    isDirect2LDS);

                auto chain = addComputeIndex(
                    kgraph, candidates[0], step, spec.location, bufferMap, isDirect2LDS, context);

                if(spec.direction == GD::Downstream)
                {
                    // Add ComputeIndexes to an Initialize block below target
                    kgraph.control.addElement(Initialize(), {spec.location}, {chain.top});
                }
                else
                {
                    if(spec.replaceWithScope)
                    {
                        // Add ComputeIndexes in a Scope above target. Only the location
                        // is within the scope.
                        if(!scopes.contains(spec.location))
                        {
                            scopes[spec.location] = replaceWith(
                                kgraph, spec.location, kgraph.control.addElement(Scope()), false);
                        }
                        auto scope = scopes[spec.location];
                        if(m_serializeComputeIndex)
                        {
                            auto isScope = kgraph.control.get<Scope>(scope).has_value();
                            kgraph.control.addElement(isScope ? ControlEdge(Body())
                                                              : ControlEdge(Sequence()),
                                                      {scope},
                                                      {chain.top});
                            kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
                            scopes[spec.location] = chain.bottom;
                        }
                        else
                        {
                            kgraph.control.addElement(Body(), {scope}, {chain.top});
                            kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
                        }
                    }
                    else
                    {
                        // Add ComputeIndexes in a Scope above target. Everything underneath
                        // the location is within the scope.
                        if(!scopes.contains(spec.location))
                        {
                            scopes[spec.location] = kgraph.control.addElement(Scope());
                            insertWithBody(kgraph, spec.location, scopes[spec.location]);
                        }
                        std::cout << "Insert chain.bottom " << chain.bottom << " before location "
                                  << spec.location << std::endl;
                        insertBefore(kgraph, spec.location, chain.top, chain.bottom);
                    }
                }

                // If the chain has an update but no containing
                // ForLoopOp, it is from a pre-fetch
                if(chain.update > 0 && spec.forLoop < 0)
                {
                    kgraph.control.deleteElement(chain.update);
                    kgraph.mapper.purge(chain.update);
                    chain.update = -1;
                }

                // Attach increment to associate ForLoop
                if(chain.update > 0)
                {
                    kgraph.control.addElement(ForLoopIncrement(), {spec.forLoop}, {chain.update});
                }

                // Add deferred connections
                for(auto candidate : candidates)
                {
                    for(auto const& dc : chain.connections)
                    {
                        kgraph.mapper.connect(candidate, dc.coordinate, dc.connectionSpec);
                    }
                }
            }
            // // YL: test
            // kgraph.control.deleteElement(35);
            return kgraph;
        }

    private:
        std::map<ComputeIndexChainSpecification, std::vector<int>> m_chains;

        bool m_serializeComputeIndex = true;
    };

    KernelGraph AddComputeIndex::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::AddComputeIndex");

        AddComputeIndexer indexer;

        for(auto candidate :
            findComputeIndexCandidates(original, *original.control.roots().begin()))
        {
            indexer.stage(original, candidate, false);
            auto isDirect2LDS = original.control.get<LoadTileDirect2LDS>(candidate).has_value();
            if(isDirect2LDS)
                indexer.stage(original, candidate, true);
        }

        return indexer.commit(original, m_context);
    }
}
