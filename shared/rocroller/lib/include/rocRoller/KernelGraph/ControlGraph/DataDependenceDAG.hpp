// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>

namespace rocRoller::KernelGraph::DataDependenceDAG
{
    /**
     * Builds the data dependence graph for the given kernel graph,
     * specifically based on its control graph portion. It explicitly
     * represents only the data dependence (flow, anti, and output)
     * between the control graph nodes at the region level i.e.
     * between the node sharing the same body parent in `graph.control`.
     *
     * The data dependence graph has no cycles in it, therefore it
     * is termed as data dependence DAG (directed acyclic graph).
     * Since this graph only represents data dependence and can
     * involve multiple regions, it may not be a connected graph
     * at the moment. In the future, we can make it as a connected
     * graph by transforming it into a program dependence graph
     * (which includes both data and control dependence information).
     *
     * It describes the dependency relationship between the control graph nodes,
     * so it makes sense to use the existing `rocRoller::KernelGraph::ControlGraph`
     * structure, where the data dependence is indicated using `Sequence` edges.
     */
    ControlGraph::ControlGraph ConstructDataDependenceDAG(KernelGraph const& graph);
}
