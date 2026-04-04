// File: include/rocRoller/KernelGraph/Transforms/CrossOperationCSE.hpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>
namespace rocRoller
{
    namespace KernelGraph
    {
        class CrossOperationCSE : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "CrossOperationCSE";
            }
        };
    }
}
