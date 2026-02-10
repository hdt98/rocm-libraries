#pragma once

#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        class RemoveScheduling : public GraphTransform
        {
        public:
            RemoveScheduling() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "RemoveScheduling";
            }
        };
    }
}

