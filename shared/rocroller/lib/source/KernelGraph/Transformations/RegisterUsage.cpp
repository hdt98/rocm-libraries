// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <queue>

#include <rocRoller/Graph/GraphUtilities.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/AliasDataFlowTags_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using TagExtent = AliasDataFlowTagsDetail::TagExtent;

        enum class Liveness
        {
            Alive = 0,
            MaybeAlive,
            Dead,
            Count
        };

        struct LivenessClassification
        {
            // Store TagExtent:
            //
            //  Use tag extents to estimate register usage
            //
            // Use register allocator as gold reference in LowerFromKernelGraph to figure out
            // the actual usage and do a comparison
            //
            std::vector<TagExtent> alive;
            std::vector<TagExtent> maybeAlive;
            std::vector<TagExtent> dead;
        };

        /**
         * Given an op and an extent, determine if the op overlap with
         * the extent or not.
         */
        Liveness queryLiveness(KernelGraph const& kgraph, int op, TagExtent const& tagExtent)
        {
            // Is node2 after node1 ?
            auto const isAfter = [&](int node1, int node2) {
                auto const order
                    = kgraph.control.compareNodes(rocRoller::UpdateCache, node1, node2);
                return order == ControlGraph::NodeOrdering::LeftFirst;
            };

            // If the op comes BEFORE the first op of the extent, it is dead (no overlap)
            if(std::ranges::all_of(tagExtent.extent.begin,
                                   [&](auto const node) { return isAfter(op, node); }))
                return Liveness::Dead;

            // If the op comes AFTER the last op of the extent, it is dead (no overlap)
            if(std::ranges::all_of(tagExtent.extent.end,
                                   [&](auto const node) { return isAfter(node, op); }))
                return Liveness::Dead;

            // At this point:
            //    op would be after or ambiguous with one of the begins,
            //    op might be before or ambiguous with one of the ends.

            // if the op is after one of the begin nodes AND before one of the end nodes
            //  then this is Alive
            if(std::ranges::any_of(tagExtent.extent.begin,
                                   [&](auto const node) { return isAfter(node, op); })
               && std::ranges::any_of(tagExtent.extent.end,
                                      [&](auto const node) { return isAfter(op, node); }))
                return Liveness::Alive;

            // Unknown
            return Liveness::MaybeAlive;
        }

        /**
         * Compute extents for ALL coordinates traced by ControlFlowRWTracer.
         * Returns a map from coordinate tag -> TagExtent.
         */
        std::map<int, TagExtent> getAllCoordExtents(KernelGraph const& kgraph)
        {
            using namespace AliasDataFlowTagsDetail;
            using Record = ControlFlowRWTracer::ReadWriteRecord;

            // Create the tracer --- it walks the entire control graph and records
            // every READ/WRITE for every coordinate.
            ControlFlowRWTracer tracer(kgraph);

            // Get all trace records.
            auto allRecords = tracer.coordinatesReadWrite();

            // Group records by coordinate tag.
            std::map<int, std::vector<Record>> recordsByCoord;
            for(auto const& rec : allRecords)
            {
                recordsByCoord[rec.coordinate].push_back(rec);
            }

            std::map<int, TagExtent> result;
            // Compute extent for each coordinate.
            for(auto& [coordTag, records] : recordsByCoord)
            {
                //auto extent = getCoordExtent(kgraph, records);
                auto extent = getExtent(kgraph, records);
                if(not extent.empty())
                {
                    result[coordTag] = std::move(extent);
                }
            }
            return result;
        }

        std::unordered_map<int, LivenessClassification>
            buildLivenessClassifications(KernelGraph const& kgraph)
        {
            auto coordExtents = getAllCoordExtents(kgraph);

            // For each op, classify all coordinates based on
            // their liveness
            std::unordered_map<int, LivenessClassification> classifications;
            for(auto op : kgraph.control.getNodes())
            {
                // We probably can skip ForLoop/Conditional/Assert

                auto& classification = classifications[op];
                for(auto const& [coord, extent] : coordExtents)
                {
                    auto liveness = queryLiveness(kgraph, op, extent);
                    switch(liveness)
                    {
                    case Liveness::Alive:
                        classification.alive.push_back(extent);
                        break;

                    case Liveness::MaybeAlive:
                        classification.maybeAlive.push_back(extent);
                        break;

                    case Liveness::Dead:
                        classification.dead.push_back(extent);
                        break;

                    case Liveness::Count:
                        AssertFatal(false, "Invalid liveness");
                        break;
                    }
                }
            }

            return classifications;
        }
    }
}
