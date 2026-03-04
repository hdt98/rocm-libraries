
#pragma once

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller::KernelGraph::DataDependenceDAG::Detail
{
    class DataDependenceDAGDetail
    {
    public:
        DataDependenceDAGDetail(KernelGraph const& graph);

        int  getBodyParent(int control);
        void addDependenceEdge(int sourceControl, int destControl);
        void processReadWriteRecord(ControlFlowRWTracer::ReadWriteRecord const& record);
        void constructDataDependenceDAG();
        ControlGraph::ControlGraph getDataDependenceDAG();

    private:
        KernelGraph const&                     m_graph;
        ControlGraph::ControlGraph             m_dependenceDAG;
        std::unordered_map<int, int>           m_bodyParentCache;
        std::map<int, int>                     m_latestWriteToCoord;
        std::map<int, std::unordered_set<int>> m_latestReadsToCoord;
    };
}
