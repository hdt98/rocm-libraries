// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
@class AddLDS
@brief Add load/store through LDS to the graph; and prefetching.

# Overview

Load/store operations inside loops, and tagged with MemoryType LDS
or WAVE_LDS, are transformed.

An entire tile is loaded once per loop iteration (which may be
unrolled) into LDS.  Subsequent loads in the loop read from LDS.

*/

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDS.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using GD = rocRoller::Graph::Direction;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;
        using namespace Register;

        template <Graph::Direction Dir>
        void insertInstead(rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph& coordinates,
                           int                                                       newTag,
                           int                                                       oldTag)
        {
            auto edgeTags = coordinates.getNeighbours<Dir>(oldTag);
            for(auto edgeTag : edgeTags)
            {
                auto edge = coordinates.getElement(edgeTag);

                auto upDirTags   = coordinates.getNeighbours<Graph::opposite(Dir)>(edgeTag);
                auto downDirTags = coordinates.getNeighbours<Dir>(edgeTag);

                std::replace(upDirTags.begin(), upDirTags.end(), oldTag, newTag);

                coordinates.deleteElement(edgeTag);
                if constexpr(Dir == Graph::Direction::Upstream)
                {
                    coordinates.addElement(edge, downDirTags, upDirTags);
                }
                else
                {
                    coordinates.addElement(edge, upDirTags, downDirTags);
                }
            }
        }

        /**
         * Add LDS transformer.
         *
         * Splits LoadTiled operations into:
         * - LoadTiled
         * - StoreLDSTile
         * - LoadLDSTile
         *
         * Similarly for StoreTiled operations.
         */
        struct AddLDSVisitor
        {
            AddLDSVisitor(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            void stage(KernelGraph const&, int);
            void commit(KernelGraph&);

        private:
            std::set<int> m_operations;

            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };

        KernelGraph AddLDS::apply(KernelGraph const& original)
        {
            auto graph = original;

            auto visitor = AddLDSVisitor(m_params, m_context);
            for(auto const& loadTag : graph.control.getNodes<LoadTiled>())
                visitor.stage(graph, loadTag);
            for(auto const& storeTag : graph.control.getNodes<StoreTiled>())
                visitor.stage(graph, storeTag);

            visitor.commit(graph);

            return graph;
        }

        ConstraintStatus NoLDSTiles(const KernelGraph& graph)
        {
            TIMER(t, "Constraint::NoLDSTiles");

            ConstraintStatus retval;
            for(auto tag : graph.coordinates.getNodes<MacroTile>())
            {
                auto tile = *graph.coordinates.get<MacroTile>(tag);
                if(tile.memoryType == MemoryType::LDS || tile.memoryType == MemoryType::WAVE_LDS
                   || tile.memoryType == MemoryType::WAVE_LDS_FROM_GLOBAL)
                {
                    retval.combine(false, concatenate("Tile has LDS memory type: ", tag));
                }
            }
            return retval;
        }

        std::vector<GraphConstraint> AddLDS::postConstraints() const
        {
            return {NoLDSTiles};
        }

        void AddLDSVisitor::stage(KernelGraph const& k, int opTag)
        {
            // Detect merged (conditional) LoadTiled: has WaveGroupBranch{0} User connection.
            // For merged ops, the normal getDimension<User> won't work (no TypeAndSubDimension),
            // so we check for WaveGroupBranch{0} first and fall back to the normal path.
            int  userTag    = -1;
            int  tileTag    = -1;
            bool isMerged   = false;

            int wgUserA = k.mapper.getWaveGroup<User>(opTag, /*waveGroup=*/0);
            if(wgUserA != -1)
            {
                // Merged (conditional) LoadTiled
                isMerged = true;
                userTag  = wgUserA;
                tileTag  = k.mapper.get<MacroTile>(opTag);
                AssertFatal(tileTag != -1,
                            "MergeConditionalLoads: merged LoadTiled missing MacroTile.",
                            ShowValue(opTag));
            }
            else
            {
                // Normal LoadTiled — existing path
                auto [ut, u] = k.getDimension<User>(opTag);
                auto [tt, t] = k.getDimension<MacroTile>(opTag);
                userTag      = ut;
                tileTag      = tt;
            }

            auto tile = k.coordinates.getNode<MacroTile>(tileTag);

            if(not(tile.memoryType == MemoryType::WAVE_LDS || tile.memoryType == MemoryType::LDS
                   || tile.memoryType == MemoryType::WAVE_Direct2LDS
                   || tile.memoryType == MemoryType::WAVE_LDS_FROM_GLOBAL))
                return;

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::AddLDS()::stage({}): User {}, MacroTile {}{}", opTag, userTag, tileTag,
                isMerged ? " [merged/conditional]" : "");

            m_operations.insert(opTag);
        }

        void AddLDSVisitor::commit(KernelGraph& k)
        {
            //
            // Commit: Create operations and DataFlow
            //
            for(auto opTag : m_operations)
            {
                Log::debug("KernelGraph::AddLDS()::commit({})", opTag);

                auto isLoad = k.control.get<LoadTiled>(opTag).has_value();

                // Detect merged (conditional) op: WaveGroupBranch{0} User connection present.
                int wgUserA = k.mapper.getWaveGroup<User>(opTag, /*waveGroup=*/0);
                bool isMerged = (wgUserA != -1) && isLoad;

                if(isMerged)
                {
                    // -- Conditional (merged) LoadTiled path --
                    // Retrieve both A-side and B-side User tags and the shared MacroTile.
                    int userATag = wgUserA;
                    int userBTag = k.mapper.getWaveGroup<User>(opTag, /*waveGroup=*/1);
                    AssertFatal(userBTag != -1,
                                "MergeConditionalLoads: merged LoadTiled missing B-side User.",
                                ShowValue(opTag));

                    int tileTag = k.mapper.get<MacroTile>(opTag);
                    AssertFatal(tileTag != -1,
                                "MergeConditionalLoads: merged LoadTiled missing MacroTile.",
                                ShowValue(opTag));

                    auto varType = getVariableType(k, opTag);
                    auto tile    = k.coordinates.getNode<MacroTile>(tileTag);

                    const bool isDirect2LDS = tile.memoryType == MemoryType::WAVE_Direct2LDS;

                    // Create 2 separate LDS nodes (one per side)
                    auto ldsTagA = k.coordinates.addElement(LDS());
                    auto ldsTagB = k.coordinates.addElement(LDS());

                    // Create 1 shared internal VGPR tile (equal sizes → single tile)
                    auto internalTag = createInternalTile(
                        k, varType, tileTag, {1, 1}, /*splitStore=*/false, m_params, m_context);

                    // Connect coordinate DataFlow edges for the A-side LDS
                    insertInstead<Graph::Direction::Downstream>(k.coordinates, internalTag, tileTag);
                    k.coordinates.addElement(DataFlow(), {internalTag}, {ldsTagA});
                    k.coordinates.addElement(DataFlow(), {ldsTagA}, {tileTag});
                    // B-side LDS: also connects ldsTagB → tileTag (both sides feed the same tile)
                    k.coordinates.addElement(DataFlow(), {internalTag}, {ldsTagB});
                    k.coordinates.addElement(DataFlow(), {ldsTagB}, {tileTag});

                    // Create StoreLDSTile (1 shared, writes conditionally to ldsA or ldsB)
                    auto storeLDSOp = k.control.addElement(StoreLDSTile(varType.dataType));
                    // Create 2 LoadLDSTile ops (A reads ldsA, B reads ldsB)
                    auto loadLDSOpA = k.control.addElement(LoadLDSTile(varType));
                    auto loadLDSOpB = k.control.addElement(LoadLDSTile(varType));

                    // Update external tile memoryType (same logic as normal path)
                    if(tile.memoryType == MemoryType::WAVE_Direct2LDS)
                        tile.memoryType = MemoryType::WAVE;
                    else if(tile.memoryType == MemoryType::WAVE_LDS)
                        tile.memoryType = MemoryType::WAVE;
                    else if(tile.memoryType == MemoryType::LDS)
                        tile.memoryType = MemoryType::VGPR;
                    else if(tile.memoryType == MemoryType::WAVE_LDS_FROM_GLOBAL)
                        tile.memoryType = MemoryType::WAVE_FROM_GLOBAL;
                    k.coordinates.setElement(tileTag, tile);

                    // Chain: opTag → storeLDSOp → loadLDSOpA → loadLDSOpB → (successors)
                    // insertAfter(opTag, storeLDSOp, loadLDSOpB) wires opTag→storeLDSOp
                    // and rewires existing successors of opTag to come after loadLDSOpB.
                    k.control.addElement(Sequence(), {storeLDSOp}, {loadLDSOpA});
                    k.control.addElement(Sequence(), {loadLDSOpA}, {loadLDSOpB});
                    insertAfter(k, opTag, storeLDSOp, loadLDSOpB);

                    // Rebuild LoadTiled mapper connections using WaveGroupBranch
                    k.mapper.purge(opTag);
                    k.mapper.connectWaveGroup<User>(opTag, userATag, /*waveGroup=*/0);
                    k.mapper.connectWaveGroup<User>(opTag, userBTag, /*waveGroup=*/1);
                    k.mapper.connect<MacroTile>(opTag, internalTag);
                    // For WAVE_Direct2LDS: both LDS connections (both sides) on LoadTiled
                    if(isDirect2LDS)
                    {
                        k.mapper.connectWaveGroup<LDS>(opTag, ldsTagA, /*waveGroup=*/0);
                        k.mapper.connectWaveGroup<LDS>(opTag, ldsTagB, /*waveGroup=*/1);
                    }

                    // StoreLDSTile: conditional LDS connections (both sides)
                    k.mapper.connectWaveGroup<User>(storeLDSOp, userATag, /*waveGroup=*/0);
                    k.mapper.connectWaveGroup<User>(storeLDSOp, userBTag, /*waveGroup=*/1);
                    k.mapper.connect<MacroTile>(storeLDSOp, internalTag);
                    k.mapper.connectWaveGroup<LDS>(storeLDSOp, ldsTagA, /*waveGroup=*/0);
                    k.mapper.connectWaveGroup<LDS>(storeLDSOp, ldsTagB, /*waveGroup=*/1);

                    // LoadLDSTile A reads from ldsA
                    k.mapper.connect<User>(loadLDSOpA, userATag);
                    k.mapper.connect<LDS>(loadLDSOpA, ldsTagA);
                    k.mapper.connect<MacroTile>(loadLDSOpA, tileTag);

                    // LoadLDSTile B reads from ldsB
                    k.mapper.connect<User>(loadLDSOpB, userBTag);
                    k.mapper.connect<LDS>(loadLDSOpB, ldsTagB);
                    k.mapper.connect<MacroTile>(loadLDSOpB, tileTag);

                    Log::debug("KernelGraph::AddLDS()::commit({}): conditional — "
                               "ldsA={}, ldsB={}, internalTag={}, storeLDS={}, "
                               "loadLDSA={}, loadLDSB={}",
                               opTag, ldsTagA, ldsTagB, internalTag,
                               storeLDSOp, loadLDSOpA, loadLDSOpB);

                    continue; // done with this merged op
                }

                // -- Normal (non-merged) LoadTiled / StoreTiled path (unchanged) --
                auto tileTag = k.mapper.get<MacroTile>(opTag);
                auto userTag = k.mapper.get<User>(opTag);
                auto varType = getVariableType(k, opTag);
                auto tile    = k.coordinates.getNode<MacroTile>(tileTag);

                // TODO: enable SwizzleScale when store D via LDS
                auto isStoreD = tile.layoutType == LayoutType::MATRIX_ACCUMULATOR;
                if(!isLoad && isStoreD)
                    AssertFatal(!m_params->swizzleScale,
                                "Store D via LDS is not supported by SwizzleScale");

                // Create new coordinates
                auto              ldsTag      = k.coordinates.addElement(LDS());
                std::vector<uint> jammedTiles = {1, 1};
                bool              splitStore  = false;

                if(!isLoad)
                {
                    // For StoreTiled operations, the waves are
                    // "serialized" in OrderEpilogueBlocks, so we can
                    // use a smaller internal tile.
                    jammedTiles = m_params->getWaveTilesPerWavefront();
                    splitStore  = m_params->getSplitStoreTileIntoWaveBlocks();
                }

                auto internalTag = createInternalTile(
                    k, varType, tileTag, jammedTiles, splitStore, m_params, m_context);

                // Connect coordinates with DataFlow edges
                insertInstead<Graph::Direction::Downstream>(k.coordinates, internalTag, tileTag);
                k.coordinates.addElement(DataFlow(), {internalTag}, {ldsTag});
                k.coordinates.addElement(DataFlow(), {ldsTag}, {tileTag});

                // Create new operations and update old operation
                auto loadLDSOp  = k.control.addElement(LoadLDSTile(varType));
                auto storeLDSOp = k.control.addElement(StoreLDSTile(varType.dataType));

                const auto isDirect2LDS = tile.memoryType == MemoryType::WAVE_Direct2LDS;

                // Update tile
                if(tile.memoryType == MemoryType::WAVE_Direct2LDS)
                    tile.memoryType = MemoryType::WAVE;
                if(tile.memoryType == MemoryType::WAVE_LDS)
                    tile.memoryType = MemoryType::WAVE;
                if(tile.memoryType == MemoryType::LDS)
                    tile.memoryType = MemoryType::VGPR;
                if(tile.memoryType == MemoryType::WAVE_LDS_FROM_GLOBAL)
                    tile.memoryType = MemoryType::WAVE_FROM_GLOBAL;
                k.coordinates.setElement(tileTag, tile);

                if(!isLoad)
                {
                    // For StoreTiled operations, the waves are
                    // "serialized" in OrderEpilogueBlocks, so we can
                    // use a smaller tile size.
                    tile.sizes[0] /= m_params->getWaveTilesPerWavefront()[0];
                    tile.sizes[1] /= m_params->getWaveTilesPerWavefront()[1];
                    k.coordinates.setElement(tileTag, tile);
                }

                // Update connections and connect operations
                k.control.addElement(Sequence(), {storeLDSOp}, {loadLDSOp});

                k.mapper.purge(opTag);
                k.mapper.connect<User>(opTag, userTag);
                k.mapper.connect<User>(storeLDSOp, userTag);
                k.mapper.connect<LDS>(storeLDSOp, ldsTag);
                k.mapper.connect<User>(loadLDSOp, userTag);
                k.mapper.connect<LDS>(loadLDSOp, ldsTag);
                // Connection to LDS is required as LoadTileDirect2LDS acts as both
                // LoadTile & StoreLDSTile and this connection is used create pre-operation
                // barriers to syncronize LDS accesses when dependencies are loop-carried.
                if(isDirect2LDS)
                    k.mapper.connect<LDS>(opTag, ldsTag, 0);

                if(isLoad)
                {
                    insertAfter(k, opTag, storeLDSOp, loadLDSOp);

                    k.mapper.connect<MacroTile>(opTag, internalTag);
                    k.mapper.connect<MacroTile>(storeLDSOp, internalTag);
                    k.mapper.connect<MacroTile>(loadLDSOp, tileTag);
                }
                else
                {
                    insertBefore(k, opTag, storeLDSOp, loadLDSOp);

                    k.mapper.connect<MacroTile>(storeLDSOp, tileTag);
                    k.mapper.connect<MacroTile>(loadLDSOp, internalTag);
                    k.mapper.connect<MacroTile>(opTag, internalTag);
                }
            }
        }
    }
}
