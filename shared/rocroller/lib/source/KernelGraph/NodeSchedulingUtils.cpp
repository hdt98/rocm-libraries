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

#include <rocRoller/KernelGraph/NodeSchedulingUtils.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph::NodeScheduling
{
    ControlGraph::ControlGraph createSubGraph(KernelGraph const&      graph,
                                              std::vector<int> const& nodes)
    {
        ControlGraph::ControlGraph subGraph;

        for(auto node : nodes)
        {
            subGraph.setElement(node, graph.control.getElement(node));
        }

        for(auto iterA = nodes.begin(); iterA != nodes.end(); ++iterA)
        {
            for(auto iterB = iterA + 1; iterB != nodes.end(); ++iterB)
            {
                auto order = graph.control.compareNodes(UpdateCache, *iterA, *iterB);

                switch(order)
                {
                case ControlGraph::NodeOrdering::LeftFirst:
                    subGraph.addElement(ControlGraph::Sequence{}, {*iterA}, {*iterB});
                    break;
                case ControlGraph::NodeOrdering::RightFirst:
                    subGraph.addElement(ControlGraph::Sequence{}, {*iterB}, {*iterA});
                    break;
                case ControlGraph::NodeOrdering::LeftInBodyOfRight:
                    subGraph.addElement(ControlGraph::Body{}, {*iterB}, {*iterA});
                    break;
                case ControlGraph::NodeOrdering::RightInBodyOfLeft:
                    subGraph.addElement(ControlGraph::Body{}, {*iterA}, {*iterB});
                    break;
                case ControlGraph::NodeOrdering::Undefined:
                    break;
                case ControlGraph::NodeOrdering::Count:
                    Throw<FatalError>("Should not get here!");
                    break;
                }
            }
        }

        removeRedundantSequenceEdges(subGraph);

        return subGraph;
    }

    ControlGraph::ControlGraph constructDataDependenceDAG(KernelGraph const& graph)
    {
        ControlGraph::ControlGraph dependenceDAG;

        // Insert all control graph nodes into the data dependence DAG
        for(auto node : graph.control.getNodes())
        {
            dependenceDAG.setElement(node, graph.control.getElement(node));
        }

        // Create tracer and get all the trace records from it.
        auto tracer  = ControlFlowRWTracer(graph);
        auto records = tracer.coordinatesReadWrite();

        std::map<int, int>                     latestWriteToCoord;
        std::map<int, std::unordered_set<int>> latestReadsToCoord;

        // This assumes that the trace is ordered and records for the
        // same control operation are consecutive.
        auto it = records.begin();
        while(it != records.end())
        {
            int currentControl = it->control;
            while(it != records.end() && it->control == currentControl)
            {
                auto coord = it->coordinate;

                // adds WW(output dep) and WR(flow dep) edges
                if(latestWriteToCoord.contains(coord))
                {
                    AssertFatal(latestWriteToCoord[coord] != currentControl,
                                ShowValue(it->control),
                                ShowValue(it->coordinate),
                                ShowValue(it->rw));

                    auto sourceNodeBodyParent
                        = bodyParents(getTopSetCoordinate(graph, latestWriteToCoord[coord]), graph)
                              .take(1)
                              .only();
                    AssertFatal(sourceNodeBodyParent.has_value(),
                                "Source node has no body parent",
                                ShowValue(latestWriteToCoord[coord]));

                    auto destNodeBodyParent
                        = bodyParents(getTopSetCoordinate(graph, currentControl), graph)
                              .take(1)
                              .only();
                    AssertFatal(destNodeBodyParent.has_value(),
                                "Dest node has no body parent",
                                ShowValue(currentControl));

                    if(sourceNodeBodyParent.value() == destNodeBodyParent.value())
                    {
                        auto order = graph.control.compareNodes(
                            UseCacheIfAvailable, latestWriteToCoord[coord], currentControl);
                        AssertFatal(order == ControlGraph::NodeOrdering::LeftFirst
                                        || order == ControlGraph::NodeOrdering::RightInBodyOfLeft,
                                    ShowValue(latestWriteToCoord[coord]),
                                    ShowValue(currentControl),
                                    ShowValue(order));

                        dependenceDAG.addElement(ControlGraph::Sequence(),
                                                 {latestWriteToCoord[coord]},
                                                 {currentControl});
                    }
                }

                if(it->rw == ControlFlowRWTracer::WRITE || it->rw == ControlFlowRWTracer::READWRITE)
                {
                    // adds RW(anti dep) edges
                    for(auto const read : latestReadsToCoord[coord])
                    {
                        if(read == currentControl)
                            continue;

                        auto sourceNodeBodyParent
                            = bodyParents(getTopSetCoordinate(graph, read), graph).take(1).only();
                        AssertFatal(sourceNodeBodyParent.has_value(),
                                    "Source node has no body parent",
                                    ShowValue(read));

                        auto destNodeBodyParent
                            = bodyParents(getTopSetCoordinate(graph, currentControl), graph)
                                  .take(1)
                                  .only();
                        AssertFatal(destNodeBodyParent.has_value(),
                                    "Dest node has no body parent",
                                    ShowValue(currentControl));

                        if(sourceNodeBodyParent.value() == destNodeBodyParent.value())
                        {
                            auto order = graph.control.compareNodes(
                                UseCacheIfAvailable, read, currentControl);
                            AssertFatal(order == ControlGraph::NodeOrdering::LeftFirst
                                            || order
                                                   == ControlGraph::NodeOrdering::RightInBodyOfLeft,
                                        ShowValue(read),
                                        ShowValue(currentControl),
                                        ShowValue(order));

                            dependenceDAG.addElement(
                                ControlGraph::Sequence(), {read}, {currentControl});
                        }
                    }

                    // Since the current control node writes into this coord,
                    // the latest reads info needs to be reset.
                    latestReadsToCoord[coord].clear();

                    // update the latest write to coord
                    latestWriteToCoord[coord] = currentControl;
                }

                if(it->rw == ControlFlowRWTracer::READ || it->rw == ControlFlowRWTracer::READWRITE)
                {
                    latestReadsToCoord[coord].insert(currentControl);
                }

                it++;
            }
        }
        return dependenceDAG;
    }
}
