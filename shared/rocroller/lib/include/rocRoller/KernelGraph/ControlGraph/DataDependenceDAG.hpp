// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>

namespace rocRoller::KernelGraph::DataDependenceDAG
{
    /**
     * Builds the data dependence graph for the given kernel graph,
     * specifically based on its control graph portion. It explicitly
     * represents only the data dependences(flow, anti, and output) between
     * the control graph nodes at the basic-block level.
     *
     * The data dependence graph has no cycles in it, therefore it
     * is termed as data dependence DAG (directed acyclic graph).
     * Since it involves multiple basic-blocks, the data dependence graph
     * may not be a connected graph at the moment. In the future,
     * we can make it as a connected graph by transforming it into a
     * program dependence graph (which includes both data and control
     * dependence information).
     *
     * It describes the dependency relationship between the control graph nodes,
     * so it makes sense to use the existing `rocRoller::KernelGraph::ControlGraph`
     * structure for its represenatation.
     *
     * The dependences are indicated using `Sequence` edges at each basic-block
     * level i.e. between the nodes sharing the same immediate body-parent
     * in `graph.control`.
     */
    ControlGraph::ControlGraph ConstructDataDependenceDAG(KernelGraph const& graph);
}
