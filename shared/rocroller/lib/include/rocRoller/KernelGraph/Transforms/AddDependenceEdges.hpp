#pragma once

#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        class AddDependenceEdges : public GraphTransform
        {
        public:
            AddDependenceEdges() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddDependenceEdges";
            }
        };
    }
}
