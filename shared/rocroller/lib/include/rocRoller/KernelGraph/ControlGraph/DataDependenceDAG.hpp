
#pragma once

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>

namespace rocRoller::KernelGraph::DataDependenceDAG
{
    ControlGraph::ControlGraph ConstructDataDependenceDAG(KernelGraph const& graph);
}
