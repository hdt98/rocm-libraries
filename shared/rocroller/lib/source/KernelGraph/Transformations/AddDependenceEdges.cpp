#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDependenceEdges.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;
    using namespace CoordinateGraph;
    using AdjacencyList = std::map<int, std::unordered_set<int>>;

    AdjacencyList constructDataDependenceDAG(KernelGraph const& graph)
    {
        // Create tracer and get all the trace records from it.
        auto tracer  = ControlFlowRWTracer(graph);
        auto records = tracer.coordinatesReadWrite();

        // adjacency list for the dependence graph
        AdjacencyList                          dependenceDAG;
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
                        AssertFatal(order == NodeOrdering::LeftFirst
                                        || order == NodeOrdering::RightInBodyOfLeft,
                                    ShowValue(latestWriteToCoord[coord]),
                                    ShowValue(currentControl),
                                    ShowValue(order));

                        dependenceDAG[latestWriteToCoord[coord]].insert(currentControl);
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
                            AssertFatal(order == NodeOrdering::LeftFirst
                                            || order == NodeOrdering::RightInBodyOfLeft,
                                        ShowValue(read),
                                        ShowValue(currentControl),
                                        ShowValue(order));

                            dependenceDAG[read].insert(currentControl);
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

    void AddDepEdges(KernelGraph& graph, AdjacencyList const& dependenceDAG)
    {
        auto const root = graph.control.roots().only().value();

        for(auto const sourceNode :
            filter(graph.control.isElemType<Operation>(), graph.control.depthFirstVisit(root)))
        {
            auto it = dependenceDAG.find(sourceNode);
            if(it != dependenceDAG.end())
            {
                auto sourceNodeBodyParent
                    = bodyParents(getTopSetCoordinate(graph, sourceNode), graph).take(1).only();
                AssertFatal(sourceNodeBodyParent.has_value(),
                            "Source node has no body parent",
                            ShowValue(sourceNode));

                for(const auto& destNode : it->second)
                {
                    auto destNodeBodyParent
                        = bodyParents(getTopSetCoordinate(graph, destNode), graph).take(1).only();
                    AssertFatal(destNodeBodyParent.has_value(),
                                "Dest node has no body parent",
                                ShowValue(destNode));
                    AssertFatal(sourceNodeBodyParent.value() == destNodeBodyParent.value(),
                                ShowValue(sourceNodeBodyParent.value()),
                                ShowValue(destNodeBodyParent.value()));

                    graph.control.addElement(Dependence(), {sourceNode}, {destNode});
                }
            }
        }
    }

    KernelGraph AddDependenceEdges::apply(KernelGraph const& original)
    {
        auto newGraph = original;

        auto dependenceDAG = constructDataDependenceDAG(newGraph);
        AddDepEdges(newGraph, dependenceDAG);

        return newGraph;
    }
}
