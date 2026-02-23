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

#include <map>
#include <unordered_map>
#include <unordered_set>

namespace rocRoller::KernelGraph::NodeScheduling
{
    namespace
    {
        using ReadWrite = ControlFlowRWTracer::ReadWrite;

        struct DependenceState
        {
            std::map<int, int>                     latestWriteToCoord;
            std::map<int, std::unordered_set<int>> latestReadsToCoord;
        };

        int GetBodyParentForControl(KernelGraph const&            graph,
                                    int                           control,
                                    std::unordered_map<int, int>& bodyParentCache)
        {
            if(auto iter = bodyParentCache.find(control); iter != bodyParentCache.end())
                return iter->second;

            auto topSetCoordinate = getTopSetCoordinate(graph, control);
            auto bodyParent       = bodyParents(topSetCoordinate, graph).take(1).only();
            AssertFatal(bodyParent.has_value(),
                        "Control node has no body parent",
                        ShowValue(control),
                        ShowValue(topSetCoordinate));

            bodyParentCache.emplace(control, bodyParent.value());
            return bodyParent.value();
        }

        void AddDependenceEdge(KernelGraph const&            graph,
                               ControlGraph::ControlGraph&   dependenceDAG,
                               int                           sourceControl,
                               int                           destControl,
                               std::unordered_map<int, int>& bodyParentCache)
        {
            if(sourceControl == destControl)
                return;

            auto sourceBodyParent = GetBodyParentForControl(graph, sourceControl, bodyParentCache);
            auto destBodyParent   = GetBodyParentForControl(graph, destControl, bodyParentCache);

            if(sourceBodyParent != destBodyParent)
                return;

            auto order
                = graph.control.compareNodes(UseCacheIfAvailable, sourceControl, destControl);
            AssertFatal(order == ControlGraph::NodeOrdering::LeftFirst
                            || order == ControlGraph::NodeOrdering::RightInBodyOfLeft,
                        ShowValue(sourceControl),
                        ShowValue(destControl),
                        ShowValue(sourceBodyParent),
                        ShowValue(destBodyParent),
                        ShowValue(order));

            if(!dependenceDAG.findEdge(sourceControl, destControl).has_value())
            {
                dependenceDAG.addElement(ControlGraph::Sequence(), {sourceControl}, {destControl});
            }
        }

        void ProcessCoordinateAccess(KernelGraph const&            graph,
                                     ControlGraph::ControlGraph&   dependenceDAG,
                                     DependenceState&              state,
                                     std::unordered_map<int, int>& bodyParentCache,
                                     int                           currentControl,
                                     int                           coordinate,
                                     ReadWrite                     rw)
        {
            AssertFatal(rw != ReadWrite::Count,
                        ShowValue(currentControl),
                        ShowValue(coordinate),
                        ShowValue(rw));

            if(auto writeIter = state.latestWriteToCoord.find(coordinate);
               writeIter != state.latestWriteToCoord.end())
            {
                AssertFatal(writeIter->second != currentControl,
                            ShowValue(writeIter->second),
                            ShowValue(currentControl),
                            ShowValue(coordinate),
                            ShowValue(rw));

                // adds WW(output dep) and WR(flow dep) edges
                AddDependenceEdge(
                    graph, dependenceDAG, writeIter->second, currentControl, bodyParentCache);
            }

            if(rw == ReadWrite::WRITE || rw == ReadWrite::READWRITE)
            {
                for(auto const readControl : state.latestReadsToCoord[coordinate])
                {
                    if(readControl == currentControl)
                        continue;

                    // adds RW(anti dep) edges
                    AddDependenceEdge(
                        graph, dependenceDAG, readControl, currentControl, bodyParentCache);
                }

                // Since the current control node writes into this coord,
                // the latest reads info needs to be reset.
                state.latestReadsToCoord[coordinate].clear();
                // update the latest write to coord
                state.latestWriteToCoord[coordinate] = currentControl;
            }

            if(rw == ReadWrite::READ || rw == ReadWrite::READWRITE)
            {
                state.latestReadsToCoord[coordinate].insert(currentControl);
            }
        }
    }

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

    ControlGraph::ControlGraph ConstructDataDependenceDAG(KernelGraph const& graph)
    {
        ControlGraph::ControlGraph dependenceDAG;

        // Insert all control graph nodes into the data dependence DAG
        for(auto node : graph.control.getNodes())
        {
            dependenceDAG.setElement(node, graph.control.getElement(node));
        }

        auto tracer  = ControlFlowRWTracer(graph);
        auto records = tracer.coordinatesReadWrite();

        DependenceState              state;
        std::unordered_map<int, int> bodyParentCache;

        // This assumes that the trace is ordered and records for the
        // same control operation are consecutive.
        for(auto iter = records.begin(); iter != records.end();)
        {
            auto currentControl = iter->control;

            for(; iter != records.end() && iter->control == currentControl; ++iter)
            {
                ProcessCoordinateAccess(graph,
                                        dependenceDAG,
                                        state,
                                        bodyParentCache,
                                        currentControl,
                                        iter->coordinate,
                                        iter->rw);
            }
        }

        return dependenceDAG;
    }
}
