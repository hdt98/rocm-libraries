// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AssignVariableType.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        KernelGraph AssignVariableType::apply(KernelGraph const& original)
        {
            auto kgraph = original;

            namespace CG = rocRoller::KernelGraph::ControlGraph;

            for(auto tag : kgraph.control.getNodes<CG::Assign>())
            {
                auto node = kgraph.control.get<CG::Assign>(tag);
                if(node.has_value() && !node->variableType.has_value())
                {
                    auto vt = Expression::resultVariableType(node->expression);
                    if(vt.dataType != DataType::None && vt.dataType != DataType::Count)
                    {
                        node->variableType = vt;
                        kgraph.control.setElement(tag, std::move(*node));
                    }
                }
            }

            return kgraph;
        }
    }
}
