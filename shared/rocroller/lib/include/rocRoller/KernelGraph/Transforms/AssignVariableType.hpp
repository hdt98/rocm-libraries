// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        class AssignVariableType : public GraphTransform
        {
        public:
            AssignVariableType() {}

            KernelGraph apply(KernelGraph const& original) override;

            std::string name() const override
            {
                return "AssignVariableType";
            }
        };
    }
}
