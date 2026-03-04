// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>

namespace rocRoller::KernelGraph::DataDependenceDAG
{
    ControlGraph::ControlGraph ConstructDataDependenceDAG(KernelGraph const& graph);
}
