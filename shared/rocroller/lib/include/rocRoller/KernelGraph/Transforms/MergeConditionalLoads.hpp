// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        /**
         * @brief Merge pairs of A-side and B-side LoadTiled operations into one conditional load.
         *
         * In scaled GEMM kernels (FP8/FP4 with block scales), each ForLoopK iteration has
         * separate LoadTiled ops for aTensor, bTensor, aScale, and bScale — consuming 4 SGPR
         * buffer descriptors (16 SGPRs total).
         *
         * This transform merges each qualifying (MATRIX_A, MATRIX_B) pair into a single
         * LoadTiled with WaveGroupBranch mapper connections:
         * - WaveGroupBranch{0}: A-side (loaded by waves where waveID%2 == 0)
         * - WaveGroupBranch{1}: B-side (loaded by waves where waveID%2 == 1)
         *
         * AddLDS subsequently creates 2 LDS nodes + 1 internal VGPR tile for the merged op.
         * AddDirect2LDS further merges LoadTiled+StoreLDSTile into LoadTileDirect2LDS.
         *
         * Prerequisites:
         * - Both ops use the same memoryType (WAVE_Direct2LDS, WAVE_LDS, or LDS)
         * - Both ops have equal MacroTile sizes and element byte width
         * - Both ops are in the same ForLoopOp
         * - One has layoutType MATRIX_A, the other MATRIX_B
         *
         * This transform runs BEFORE AddLDS, where the original MacroTile memoryType and
         * layoutType are still directly readable.
         *
         * @ingroup Transformations
         */
        class MergeConditionalLoads : public GraphTransform
        {
        public:
            MergeConditionalLoads(ContextPtr context, CommandParametersPtr params)
                : m_context(context)
                , m_params(params)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "MergeConditionalLoads";
            }

        private:
            ContextPtr           m_context;
            CommandParametersPtr m_params;
        };
    }
}
