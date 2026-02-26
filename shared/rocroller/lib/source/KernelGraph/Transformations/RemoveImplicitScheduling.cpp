/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/Transforms/RemoveImplicitScheduling.hpp>

#include <rocRoller/KernelGraph/NodeSchedulingUtils.hpp>

#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    namespace RemoveImplicitSchedulingDetail
    {
        void breakupNodes(KernelGraph& graph, std::vector<int> const& nodes)
        {
            auto dependenceDAG = NodeScheduling::ConstructDataDependenceDAG(graph);

            // Pull every Multiply node next to its body parent
            for(auto node : nodes)
            {
                auto parent = bodyParents(node, graph).take(1).only();
                AssertFatal(parent.has_value(), "Node has no body parent", ShowValue(node));

                auto bodyEdge = graph.control.findEdge(parent.value(), node);
                // if node is not directly connected to its parent via Body edge
                if(!bodyEdge.has_value())
                {
                    auto replaceOp = graph.control.addElement(ControlGraph::NOP());
                    replaceWith(graph, node, replaceOp, false);
                    // add the body edge from parent to node
                    graph.control.addElement(ControlGraph::Body(), {parent.value()}, {node});
                }
            }

            // add Sequence edges between dependent nodes
            for(auto node : nodes)
            {
                for(auto sourceNode :
                    dependenceDAG.getInputNodeIndices<ControlGraph::Sequence>(node))
                {
                    auto topOp   = getTopSetCoordinate(graph, sourceNode);
                    auto seqEdge = graph.control.findEdge(topOp, node);
                    if(!seqEdge.has_value())
                    {
                        graph.control.addElement(ControlGraph::Sequence(), {topOp}, {node});
                    }
                }

                for(auto destNode :
                    dependenceDAG.getOutputNodeIndices<ControlGraph::Sequence>(node))
                {
                    auto topOp   = getTopSetCoordinate(graph, destNode);
                    auto seqEdge = graph.control.findEdge(node, topOp);
                    if(!seqEdge.has_value())
                    {
                        graph.control.addElement(ControlGraph::Sequence(), {node}, {topOp});
                    }
                }
            }

            for(auto nodeA : nodes)
            {
                for(auto nodeB : nodes)
                {
                    if(nodeA == nodeB)
                        continue;

                    auto pathAToB = dependenceDAG
                                        .path<Graph::Direction::Downstream>(std::vector<int>{nodeA},
                                                                            std::vector<int>{nodeB})
                                        .to<std::vector>();
                    auto pathBToA = dependenceDAG
                                        .path<Graph::Direction::Downstream>(std::vector<int>{nodeB},
                                                                            std::vector<int>{nodeA})
                                        .to<std::vector>();
                    AssertFatal(pathAToB.empty() || pathBToA.empty(),
                                "Dependence DAG has a cycle!");

                    auto order = graph.control.compareNodes(UseCacheIfAvailable, nodeA, nodeB);

                    AssertFatal(
                        (pathAToB.empty() && pathBToA.empty()
                         && order == ControlGraph::NodeOrdering::Undefined)
                            || (!pathAToB.empty()
                                && (order == ControlGraph::NodeOrdering::LeftFirst
                                    || order == ControlGraph::NodeOrdering::RightInBodyOfLeft))
                            || (!pathBToA.empty()
                                && (order == ControlGraph::NodeOrdering::RightFirst
                                    || order == ControlGraph::NodeOrdering::LeftInBodyOfRight)),
                        ShowValue(nodeA),
                        ShowValue(nodeB),
                        ShowValue(pathAToB.empty()),
                        ShowValue(pathBToA.empty()),
                        ShowValue(order));
                }
            }
        }
    }

    KernelGraph RemoveImplicitScheduling::apply(KernelGraph const& original)
    {
        auto rv = original;

        // grouped by immediate body-parent
        auto groupedMultiplyNodes = NodeScheduling::getGroupedNodes<ControlGraph::Multiply>(rv);

        for(auto& [parent, nodes] : groupedMultiplyNodes)
        {
            RemoveImplicitSchedulingDetail::breakupNodes(rv, nodes);
        }

        return rv;
    }
}
