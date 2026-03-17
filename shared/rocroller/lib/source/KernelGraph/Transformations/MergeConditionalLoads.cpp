// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file MergeConditionalLoads.cpp
 *
 * Implements the MergeConditionalLoads transform.
 *
 * For each qualifying (MATRIX_A, MATRIX_B) pair of LoadTiled operations in the same
 * ForLoopK, the transform merges them into a single LoadTiled with WaveGroupBranch
 * mapper connections:
 *   WaveGroupBranch{0} → A-side (userA / ldsA for waves where waveID%2==0)
 *   WaveGroupBranch{1} → B-side (userB / ldsB for waves where waveID%2==1)
 *
 * The B-side LoadTiled is replaced with a NOP. AddLDS subsequently creates 2 LDS nodes
 * + 1 internal VGPR tile for the merged op.
 *
 * In the prolog (before the ForLoopK), we insert:
 *   Assign{Scalar, (threadIdx.x / wavefrontSize) % 2}
 * which computes the waveGroup index (0=A-side, 1=B-side) into a scalar SGPR.
 * The resulting Adhoc coordinate tag is stored in each merged LoadTiled's mapper
 * via connect<Adhoc>(mergedLoad, waveGroupCoordTag) so that AssignIndexExpressions
 * and LoadStoreTileGenerator can retrieve it.
 *
 * See specs/conditional-load/design.md for the full design.
 */

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/Transforms/MergeConditionalLoads.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        /**
         * Describes a qualifying pair of LoadTiled operations that can be merged.
         */
        struct MergeablePair
        {
            int loadA;  ///< Control tag of MATRIX_A LoadTiled
            int loadB;  ///< Control tag of MATRIX_B LoadTiled
            int tileA;  ///< Coordinate tag of A-side MacroTile
            int tileB;  ///< Coordinate tag of B-side MacroTile
            int userA;  ///< Coordinate tag of A-side User
            int userB;  ///< Coordinate tag of B-side User
        };

        /**
         * Search for qualifying (MATRIX_A, MATRIX_B) pairs of LoadTiled ops in the same ForLoopK.
         *
         * Runs before AddLDS — the original MacroTile with full memoryType/layoutType is visible.
         */
        static std::vector<MergeablePair> searchCandidates(KernelGraph const& kgraph)
        {
            std::vector<MergeablePair> results;

            // Collect all LoadTiled ops
            auto loadTiledTags = kgraph.control.getNodes<LoadTiled>().to<std::vector>();

            // For each A-side op, search for a matching B-side op in the same ForLoopK
            for(size_t i = 0; i < loadTiledTags.size(); ++i)
            {
                int tagA = loadTiledTags[i];

                // Already merged (WaveGroupBranch{0} connection means already processed A-side)
                if(kgraph.mapper.getWaveGroup<User>(tagA, 0) != -1)
                    continue;

                auto tileATag = kgraph.mapper.get<MacroTile>(tagA);
                if(tileATag == -1)
                    continue;
                auto tileAOpt = kgraph.coordinates.get<MacroTile>(tileATag);
                if(!tileAOpt)
                    continue;
                auto const& tileA = *tileAOpt;

                // Must be an LDS-type memoryType and MATRIX_A layout
                if(tileA.layoutType != LayoutType::MATRIX_A)
                    continue;
                if(tileA.memoryType != MemoryType::WAVE_Direct2LDS
                   && tileA.memoryType != MemoryType::WAVE_LDS
                   && tileA.memoryType != MemoryType::LDS)
                    continue;

                int userATag = kgraph.mapper.get<User>(tagA);
                if(userATag == -1)
                    continue;

                // Find ForLoopK ancestor for tagA
                auto forLoopA = only(kgraph.control.getInputNodeIndices<Body>(tagA));
                if(!forLoopA)
                    continue;

                // Search for a B-side partner
                for(size_t j = 0; j < loadTiledTags.size(); ++j)
                {
                    if(i == j)
                        continue;

                    int tagB = loadTiledTags[j];

                    // Already merged
                    if(kgraph.mapper.getWaveGroup<User>(tagB, 0) != -1)
                        continue;

                    auto tileBTag = kgraph.mapper.get<MacroTile>(tagB);
                    if(tileBTag == -1)
                        continue;
                    auto tileBOpt = kgraph.coordinates.get<MacroTile>(tileBTag);
                    if(!tileBOpt)
                        continue;
                    auto const& tileB = *tileBOpt;

                    // B-side must be MATRIX_B layout
                    if(tileB.layoutType != LayoutType::MATRIX_B)
                        continue;

                    // Same memoryType
                    if(tileA.memoryType != tileB.memoryType)
                        continue;

                    // Same tile sizes
                    if(tileA.sizes != tileB.sizes)
                        continue;

                    // Same element byte width
                    auto varTypeA = getVariableType(kgraph, tagA);
                    auto varTypeB = getVariableType(kgraph, tagB);
                    if(varTypeA.dataType != varTypeB.dataType)
                        continue;

                    int userBTag = kgraph.mapper.get<User>(tagB);
                    if(userBTag == -1)
                        continue;

                    // Same ForLoopK parent
                    auto forLoopB = only(kgraph.control.getInputNodeIndices<Body>(tagB));
                    if(!forLoopB || *forLoopA != *forLoopB)
                        continue;

                    results.push_back({tagA, tagB, tileATag, tileBTag, userATag, userBTag});
                    break; // Each A-side matches at most one B-side
                }
            }

            return results;
        }

        /**
         * Insert the waveGroup prolog Assign before the ForLoopK.
         *
         * Creates:
         *   waveGroupAdhoc = Adhoc("waveGroup") coordinate
         *   workitem0      = Workitem(0) coordinate
         *   waveGroupExpr  = (DataFlowTag(workitem0, Vector, UInt32) / wavefrontSz) % 2
         *   assignOp       = Assign{Scalar, waveGroupExpr} → connected to waveGroupAdhoc
         *
         * The assignOp is inserted before the first ForLoopOp in the control graph (the
         * outermost scope body, before ForLoopK).
         *
         * Returns the waveGroupAdhoc coordinate tag (or -1 if no ForLoopOp found).
         */
        static int insertWaveGroupPrologAssign(KernelGraph& graph, ContextPtr context)
        {
            using namespace Expression;

            // Find the first ForLoopOp in the graph (ForLoopK)
            auto forLoopTags = graph.control.getNodes<ForLoopOp>().to<std::vector>();
            if(forLoopTags.empty())
                return -1;

            // Use the first ForLoopOp (the outermost K loop)
            int forLoopTag = forLoopTags.front();

            // Create a fresh Workitem(0) coordinate (represents threadIdx.x / v0)
            auto workitem0Tag = graph.coordinates.addElement(Workitem(0));

            // Build: waveGroupExpr = (v0 / wavefrontSize) % 2
            auto wavefrontSz = literal(static_cast<unsigned int>(
                context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize)));
            auto workitem0DF = std::make_shared<rocRoller::Expression::Expression>(
                DataFlowTag{workitem0Tag, Register::Type::Vector, DataType::UInt32});
            auto waveGroupExpr = (workitem0DF / wavefrontSz) % literal(2u);

            // Create the Adhoc coordinate that will hold the waveGroup SGPR result
            auto waveGroupAdhocTag
                = graph.coordinates.addElement(Adhoc("waveGroup", nullptr, nullptr));

            // Create Assign{Scalar, waveGroupExpr} connected to the Adhoc coordinate
            auto assignNode   = Assign{Register::Type::Scalar, waveGroupExpr};
            auto assignOpTag  = graph.control.addElement(assignNode);
            graph.mapper.connect(assignOpTag, waveGroupAdhocTag, NaryArgument::DEST);

            // Insert the assign before the ForLoopOp
            insertBefore(graph, forLoopTag, assignOpTag, assignOpTag);

            Log::debug("MergeConditionalLoads: inserted waveGroup Assign (op={}) before "
                       "ForLoopK ({}). waveGroupAdhoc={}, workitem0={}.",
                       assignOpTag,
                       forLoopTag,
                       waveGroupAdhocTag,
                       workitem0Tag);

            return waveGroupAdhocTag;
        }

        KernelGraph MergeConditionalLoads::apply(KernelGraph const& original)
        {
            auto graph = original;

            auto pairs = searchCandidates(graph);
            if(pairs.empty())
                return graph;

            Log::debug("MergeConditionalLoads: found {} mergeable pairs", pairs.size());

            // Insert waveGroup Assign in the prolog (before the first ForLoopK).
            // The returned Adhoc coordinate tag is stored on each merged LoadTiled so that
            // AssignIndexExpressions and LoadStoreTileGenerator can retrieve the waveGroup SGPR.
            int waveGroupAdhocTag = insertWaveGroupPrologAssign(graph, m_context);
            AssertFatal(waveGroupAdhocTag != -1,
                        "MergeConditionalLoads: could not find a ForLoopOp to insert waveGroup "
                        "Assign before.");

            for(auto const& pair : pairs)
            {
                Log::debug(
                    "MergeConditionalLoads: merging loadA={} (tileA={}) with loadB={} (tileB={})",
                    pair.loadA,
                    pair.tileA,
                    pair.loadB,
                    pair.tileB);

                // The merged LoadTiled reuses loadA's slot in the control graph.
                // Its mapper connections become WaveGroupBranch{0} for A-side and
                // WaveGroupBranch{1} for B-side.
                //
                // We purge loadA's existing connections and rebuild with WaveGroupBranch.
                // loadB is replaced with a NOP.

                // Rebuild loadA's mapper connections using WaveGroupBranch
                graph.mapper.purge(pair.loadA);
                graph.mapper.connectWaveGroup<User>(pair.loadA, pair.userA, /*waveGroup=*/0);
                graph.mapper.connectWaveGroup<User>(pair.loadA, pair.userB, /*waveGroup=*/1);
                graph.mapper.connect<MacroTile>(pair.loadA, pair.tileA);
                // Store the waveGroup coordinate so downstream transforms can retrieve it
                graph.mapper.connect<Adhoc>(pair.loadA, waveGroupAdhocTag);

                // Remove tileB from the coordinate graph (no longer directly referenced)
                // First remove its incident DataFlow edges
                {
                    auto edgeTags = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(
                        pair.tileB);
                    for(auto et : edgeTags)
                        graph.coordinates.deleteElement(et);
                    auto upEdgeTags
                        = graph.coordinates
                              .getNeighbours<Graph::Direction::Upstream>(pair.tileB);
                    for(auto et : upEdgeTags)
                        graph.coordinates.deleteElement(et);
                    graph.coordinates.deleteElement(pair.tileB);
                }

                // Replace loadB with a NOP (preserves control flow edges)
                auto nopTag = graph.control.addElement(NOP());
                replaceWith(graph, pair.loadB, nopTag, /*includeBody=*/false);

                Log::debug("MergeConditionalLoads: merged pair — loadA={} now has "
                           "WaveGroupBranch{{0}}=userA={}, WaveGroupBranch{{1}}=userB={}, "
                           "MacroTile=tileA={}, waveGroupAdhoc={}. "
                           "loadB={} replaced by NOP={}.",
                           pair.loadA,
                           pair.userA,
                           pair.userB,
                           pair.tileA,
                           waveGroupAdhocTag,
                           pair.loadB,
                           nopTag);
            }

            return graph;
        }

    } // namespace KernelGraph
} // namespace rocRoller
