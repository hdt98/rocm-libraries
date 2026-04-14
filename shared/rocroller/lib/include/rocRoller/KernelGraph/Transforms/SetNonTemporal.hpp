// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        /**
         * @brief Propagate non-temporal load flags from CommandParameters onto MacroTile nodes.
         *
         * For each MacroTile node whose layoutType is MATRIX_A or MATRIX_B, sets
         * MacroTile::nonTemporal from CommandParameters::nonTemporalA / nonTemporalB.
         *
         * Applied after AddLDSPadding so that all MacroTile nodes have their final
         * layoutType assigned.
         *
         * @ingroup Transformations
         */
        class SetNonTemporal : public GraphTransform
        {
        public:
            SetNonTemporal(CommandParametersPtr params)
                : m_params(params)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string  name() const override
            {
                return "SetNonTemporal";
            }

        private:
            CommandParametersPtr m_params;
        };
    }
}
