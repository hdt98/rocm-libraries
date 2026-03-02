// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/Transforms/RemoveImplicitScheduling.hpp>

#include <rocRoller/KernelGraph/NodeSchedulingUtils.hpp>

#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    namespace RemoveImplicitSchedulingDetail
    {
        void breakupNodes(KernelGraph& graph, std::vector<int> const& nodes)
        {
            auto getLoopOp = [&](int op) -> std::optional<int> {
                auto stack = controlStack(op, graph);

                for(auto parent : std::views::reverse(stack))
                {
                    if(graph.control.get<ControlGraph::ForLoopOp>(parent))
                        return parent;
                }

                return std::nullopt;
            };

            auto notMultiply = [&graph](int idx) {
                if(graph.control.getElementType(idx) != Graph::ElementType::Node)
                    return false;

                return !(graph.control.get<ControlGraph::Multiply>(idx).has_value());
            };

            std::map<int, int> connectionsToKeep;

            for(auto node : nodes)
            {
                auto upstreamNode
                    = graph.control.breadthFirstVisit(node, Graph::Direction::Upstream)
                          .filter(notMultiply)
                          .take(1)
                          .only();

                AssertFatal(upstreamNode.has_value(), ShowValue(node));

                AssertFatal(getLoopOp(node) == getLoopOp(*upstreamNode), ShowValue(node));

                connectionsToKeep[node] = *upstreamNode;
            }

            Log::debug("Got connections.");

            auto dependenceDAG = NodeScheduling::ConstructDataDependenceDAG(graph);

            for(auto nodeA : nodes)
            {
                for(auto nodeB : nodes)
                {
                    if(nodeA == nodeB)
                        continue;

                    auto depEdge = dependenceDAG.findEdge(nodeA, nodeB);
                    auto seqEdge = graph.control.findEdge(nodeA, nodeB);

                    if(depEdge.has_value())
                    {
                        if(!seqEdge.has_value())
                        {
                            graph.control.addElement(ControlGraph::Sequence(), {nodeA}, {nodeB});
                        }
                    }
                    else if(seqEdge.has_value())
                    {
                        auto upstream = connectionsToKeep.at(nodeB);
                        auto order
                            = graph.control.compareNodes(UseCacheIfAvailable, upstream, nodeB);
                        AssertFatal(order == ControlGraph::NodeOrdering::LeftFirst
                                        || order == ControlGraph::NodeOrdering::RightInBodyOfLeft,
                                    ShowValue(order),
                                    ShowValue(upstream),
                                    ShowValue(nodeB),
                                    ShowValue(seqEdge.value()));
                        graph.control.deleteElement(seqEdge.value());
                        if(order == ControlGraph::NodeOrdering::LeftFirst)
                            graph.control.chain<ControlGraph::Sequence>(upstream, nodeB);
                        else
                            graph.control.chain<ControlGraph::Body>(upstream, nodeB);
                    }
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
