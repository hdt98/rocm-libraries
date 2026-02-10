#include <rocRoller/KernelGraph/Transforms/RemoveScheduling.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;
    using namespace CoordinateGraph;
    using AdjacencyList = std::map<int, std::unordered_set<int>>;

    AdjacencyList constructDataDependenceDAG(KernelGraph const& graph, int forLoop)
    {
        // Create tracer, build dependencies and get all the trace records from it.
        auto tracer = ControlFlowRWTracer(graph, forLoop);
        auto records = tracer.coordinatesReadWrite();

        // adjacency list for the dependence graph
	AdjacencyList dependenceDAG;
	std::map<int, int> latestWriteToCoord;
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
		    AssertFatal(latestWriteToCoord[coord] != currentControl, ShowValue(it->control), ShowValue(it->coordinate), ShowValue(it->rw));
                    AssertFatal(graph.control.compareNodes(UseCacheIfAvailable, latestWriteToCoord[coord], currentControl)
                                == NodeOrdering::LeftFirst);

		    dependenceDAG[latestWriteToCoord[coord]].insert(currentControl);
		}
	        
		if(it->rw == ControlFlowRWTracer::READ)
		{
		    latestReadsToCoord[coord].insert(currentControl);
		}
		else if(it->rw == ControlFlowRWTracer::WRITE || it->rw == ControlFlowRWTracer::READWRITE)
	        {
		    // adds RW(anti dep) edges
		    for(auto const read : latestReadsToCoord[coord])
		    {
		        if(read == currentControl)
			    continue;

                        AssertFatal(graph.control.compareNodes(UseCacheIfAvailable, read, currentControl)
                                == NodeOrdering::LeftFirst);

		        dependenceDAG[read].insert(currentControl);
		    }

                    // Since the current control node writes into this coord,
		    // the latest reads info needs to be reset.
                    latestReadsToCoord[coord].clear();
		    latestReadsToCoord[coord].insert(currentControl);

                    // update the latest write to coord
		    latestWriteToCoord[coord] = currentControl;

		}

		it++;
            }
	}
	return dependenceDAG;
    }

    void AddDependencyEdges(KernelGraph& graph, AdjacencyList const& dependenceDAG, int forLoop)   
    {
        for(auto const sourceNode : filter(graph.control.isElemType<Operation>(), 
	                         graph.control.depthFirstVisit(forLoop)))
	{
	    auto it = dependenceDAG.find(sourceNode);
	    if(it != dependenceDAG.end())
	    {
	        for(const auto& destNode : it->second)
	        {
	            graph.control.addElement(Dependency(), {sourceNode}, {destNode});
                }
            }
	}
    }

    KernelGraph RemoveScheduling::apply(KernelGraph const& original)
    {
        auto newGraph = original;

        auto root = *newGraph.control.roots().only();

        std::optional<int> maybeKLoop;
        for(auto const loop : filter(newGraph.control.isElemType<ForLoopOp>(),
                                     newGraph.control.depthFirstVisit(root)))
        {
            auto forloop = *newGraph.control.get<ForLoopOp>(loop);
            if(forloop.loopName == rocRoller::KLOOP)
            {
                maybeKLoop = loop;
                break;
            }
        }

        if(!maybeKLoop.has_value())
	    return original;

        auto kLoop = maybeKLoop.value();

	auto dependenceDAG = constructDataDependenceDAG(newGraph, kLoop);
	AddDependencyEdges(newGraph, dependenceDAG, kLoop);

        return newGraph;
    }
}

