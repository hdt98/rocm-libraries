// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @class SetNonTemporal
 * @brief Propagate non-temporal load flags onto MacroTile nodes.
 *
 * Iterates over all MacroTile nodes in the coordinate graph and sets
 * MacroTile::nonTemporal based on CommandParameters::nonTemporalA (for
 * MATRIX_A tiles) and CommandParameters::nonTemporalB (for MATRIX_B tiles).
 * Tiles with other layout types are left unchanged.
 */

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/SetNonTemporal.hpp>
#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;

        KernelGraph SetNonTemporal::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::SetNonTemporal");

            if(!m_params->nonTemporalA && !m_params->nonTemporalB)
                return original;

            auto graph = original;

            for(auto tag : graph.coordinates.getNodes<MacroTile>())
            {
                auto tile = graph.coordinates.get<MacroTile>(tag).value();

                if(tile.layoutType == LayoutType::MATRIX_A)
                {
                    if(tile.nonTemporal != m_params->nonTemporalA)
                    {
                        tile.nonTemporal = m_params->nonTemporalA;
                        graph.coordinates.setElement(tag, tile);
                        Log::debug("KernelGraph::SetNonTemporal: "
                                   "Set MacroTile {} (MATRIX_A) nonTemporal={}",
                                   tag,
                                   tile.nonTemporal);
                    }
                }
                else if(tile.layoutType == LayoutType::MATRIX_B)
                {
                    if(tile.nonTemporal != m_params->nonTemporalB)
                    {
                        tile.nonTemporal = m_params->nonTemporalB;
                        graph.coordinates.setElement(tag, tile);
                        Log::debug("KernelGraph::SetNonTemporal: "
                                   "Set MacroTile {} (MATRIX_B) nonTemporal={}",
                                   tag,
                                   tile.nonTemporal);
                    }
                }
            }

            return graph;
        }
    }
}
