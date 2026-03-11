// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller::KernelGraph::DataDependenceDAG::Detail
{
    class DataDependenceDAGDetail
    {
    public:
        /**
         * Initializes the data dependence DAG structure (`m_dependenceDAG`)
         * with only the control nodes from the given kernel graph.
         */
        DataDependenceDAGDetail(KernelGraph const& graph);

        /**
         * Returns the body-parent for the given node in the control graph.
         */
        int getBodyParent(int control);
        /**
         * Adds a dependence edge(represented via `Sequence`) between the given
         * source and destination nodes in the dependence DAG (`m_dependenceDAG),
         * if both the nodes have the same body-parent.
         */
        void addDependenceEdge(int source, int dest);
        /**
         * Makes necessary updates to `m_latestWriteToCoord`, `m_latestReadsToCoord`
         * and `m_dependenceDAG` structures for the given `ReadWriteRecord`.
         */
        void processReadWriteRecord(ControlFlowRWTracer::ReadWriteRecord const& record);
        /**
         * Builds the data dependence DAG by populating the `m_dependenceDAG` structure
         * with required dependence edges based on the trace generated using `ControlFlowRWTracer`
         * for `m_graph`.
         */
        void constructDataDependenceDAG();
        /**
         * Returns the data dependence DAG (`m_dependenceDAG`).
         */
        ControlGraph::ControlGraph getDataDependenceDAG();

    private:
        KernelGraph const&                     m_graph;
        ControlGraph::ControlGraph             m_dependenceDAG;
        std::unordered_map<int, int>           m_bodyParentCache;
        std::map<int, int>                     m_latestWriteToCoord;
        std::map<int, std::unordered_set<int>> m_latestReadsToCoord;
    };
}
