// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerLinear.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        namespace LowerLinearLoopDetail
        {
            struct LoopHelper
            {
                ExpressionPtr                m_loopSize;
                ExpressionPtr                m_loopStride;
                int                          m_loopDimTag = -1;
                std::unordered_map<int, int> m_loopDims;

                LoopHelper(ExpressionPtr loopSize)
                    : m_loopSize(loopSize)
                    , m_loopStride(Expression::literal(1))
                {
                }

                int loopIndexDim(KernelGraph& graph)
                {
                    if(m_loopDimTag < 0)
                        m_loopDimTag
                            = graph.coordinates.addElement(Linear(m_loopSize, m_loopStride));
                    return m_loopDimTag;
                }

                int getLoop(int workgroupTag, KernelGraph& graph)
                {
                    if(m_loopDims.count(workgroupTag))
                        return m_loopDims.at(workgroupTag);

                    auto loopIndex = loopIndexDim(graph);
                    auto loop = graph.coordinates.addElement(ForLoop(m_loopSize, m_loopStride));
                    graph.coordinates.addElement(DataFlow(), {loopIndex}, {loop});

                    m_loopDims.emplace(workgroupTag, loop);
                    return loop;
                }

                void
                    addLoopDst(KernelGraph& graph, int edgeTag, CoordinateTransformEdge const& edge)
                {
                    auto outgoing
                        = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(edgeTag);
                    int dstTag = -1;
                    for(auto out : outgoing)
                    {
                        if(isDimension<Workgroup>(graph.coordinates.getNode(out)))
                        {
                            dstTag = out;
                            break;
                        }
                    }
                    AssertFatal(dstTag > 0, "addLoopDst: Workgroup dimension not found");

                    auto loop = getLoop(dstTag, graph);
                    outgoing.insert(outgoing.begin(), loop);

                    auto incoming
                        = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(edgeTag);
                    graph.coordinates.deleteElement(edgeTag);
                    graph.coordinates.addElement(edgeTag, edge, incoming, outgoing);
                }

                void
                    addLoopSrc(KernelGraph& graph, int edgeTag, CoordinateTransformEdge const& edge)
                {
                    auto incoming
                        = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(edgeTag);
                    int srcTag = -1;
                    for(auto in : incoming)
                    {
                        if(isDimension<Workgroup>(graph.coordinates.getNode(in)))
                        {
                            srcTag = in;
                            break;
                        }
                    }
                    AssertFatal(srcTag > 0, "addLoopSrc: Workgroup dimension not found");

                    auto loop = getLoop(srcTag, graph);
                    incoming.insert(incoming.begin(), loop);

                    auto outgoing
                        = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(edgeTag);
                    graph.coordinates.deleteElement(edgeTag);
                    graph.coordinates.addElement(edgeTag, edge, incoming, outgoing);
                }
            };
        } // namespace LowerLinearLoopDetail

        KernelGraph LowerLinearLoop::apply(KernelGraph const& original)
        {
            using namespace LowerLinearLoopDetail;

            KernelGraph graph = original;
            LoopHelper  helper(m_loopSize);

            // Create the loop index dimension and ForLoopOp under the Kernel node.
            auto iterTag   = helper.loopIndexDim(graph);
            auto kernelTag = *graph.control.getNodes<Kernel>().begin();

            auto loopVarExp = std::make_shared<Expression::Expression>(
                DataFlowTag{iterTag, Register::Type::Scalar, DataType::Int32});
            auto loopOp = graph.control.addElement(ForLoopOp{loopVarExp < m_loopSize, ""});
            graph.control.addElement(Body(), {kernelTag}, {loopOp});

            auto zero   = literal(0);
            auto initOp = graph.control.addElement(Assign{Register::Type::Scalar, zero});
            graph.control.addElement(Initialize(), {loopOp}, {initOp});

            auto incOp = graph.control.addElement(
                Assign{Register::Type::Scalar, loopVarExp + helper.m_loopStride});
            graph.control.addElement(ForLoopIncrement(), {loopOp}, {incOp});

            graph.mapper.connect<Dimension>(loopOp, iterTag);
            graph.mapper.connect(initOp, iterTag, NaryArgument::DEST);
            graph.mapper.connect(incOp, iterTag, NaryArgument::DEST);

            // Modify coordinate transform edges to include loop dimensions.
            for(auto tag : original.coordinates.getEdges<CoordinateTransformEdge>())
            {
                auto edge = original.coordinates.getEdge<CoordinateTransformEdge>(tag);
                if(auto* tile = std::get_if<Tile>(&edge))
                    helper.addLoopDst(graph, tag, *tile);
                else if(auto* inherit = std::get_if<Inherit>(&edge))
                    helper.addLoopDst(graph, tag, *inherit);
                else if(auto* flatten = std::get_if<Flatten>(&edge))
                {
                    auto incoming
                        = original.coordinates.getNeighbours<Graph::Direction::Upstream>(tag);
                    if(incoming.size() > 1)
                        helper.addLoopSrc(graph, tag, *flatten);
                }
                else if(auto* forget = std::get_if<Forget>(&edge))
                    helper.addLoopSrc(graph, tag, *forget);
            }

            // Reparent LoadVGPR nodes under the loop operation.
            for(auto tag : original.control.getNodes<LoadVGPR>())
            {
                auto incomingEdges = graph.control.getNeighbours<Graph::Direction::Upstream>(tag);
                AssertFatal(incomingEdges.size() == 1, "one parent edge expected");
                auto incomingEdge = graph.control.getElement(incomingEdges[0]);
                graph.control.deleteElement(incomingEdges[0]);
                graph.control.addElement(incomingEdges[0],
                                         incomingEdge,
                                         std::vector<int>{loopOp},
                                         std::vector<int>{tag});
            }

            return graph;
        }
    }
}
