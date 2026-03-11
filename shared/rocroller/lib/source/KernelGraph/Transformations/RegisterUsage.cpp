// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <queue>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/AliasDataFlowTags_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/Graph/GraphUtilities.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using TagExtent = AliasDataFlowTagsDetail::TagExtent;

        //TagExtent getCoordExtent(KernelGraph const& kgraph,
        //        std::vector<ControlFlowRWTracer::ReadWriteRecord> const& records)
        //{
        //    using namespace AliasDataFlowTagsDetail;

        //    using Tracer = ControlFlowRWTracer;
        //    TagExtent rv;
        //    if(records.empty())
        //        return rv;
        //    rv.baseTag = records.front().coordinate;

        //    // Step 1: Build the ordering graph for these records.
        //    auto ordering = getOrdering(kgraph, records);
        //    auto getControlNode = [&](int idx) {
        //        return ordering.getNode(idx).control;
        //    };

        //    // Step 2: Overall extent = roots..leaves (same as existing code).
        //    rv.extent.begin = ordering.roots().map(getControlNode).to<std::set>();
        //    rv.extent.end   = ordering.leaves().map(getControlNode).to<std::set>();

        //    // Step 3: Identify individual WRITE->READ lifespans.
        //    //
        //    // Approach: In the ordering graph, a lifespan starts at each WRITE
        //    // node and extends to the "last READs" before the next WRITE (or end).
        //    //
        //    // For simplicity, we're ignoring ForLoop/Branch for now,
        //    //   - ForLoop: how to handle loop carried depedencies?
        //    //
        //    auto isRead  = [&](int idx) { return ordering.getNode(idx).rw == Tracer::READ; };
        //    auto isWrite = [&](int idx) { return ordering.getNode(idx).rw == Tracer::WRITE; };
        //    auto isReadWrite = [&](int idx) {
        //        return ordering.getNode(idx).rw == Tracer::READWRITE;
        //    };
        //    auto writes = ordering.getNodes().filter([&](int idx) {
        //            return isWrite(idx) || isReadWrite(idx);
        //            }).to<std::vector>();

        //    for(int writeIdx : writes)
        //    {
        //        int writeControl = getControlNode(writeIdx);

        //        // Traversal from writeIdx, collecting reachable READ nodes,
        //        // stopping at WRITE/READWRITE nodes.
        //        // "Last READs" = READ nodes in this reachable set that have
        //        // no outgoing edge to another READ in the set.
        //        std::set<int> reachableReads;
        //        std::queue<int> frontier;

        //        // Start from writeIdx's successors
        //        auto successors = ordering.getOutputNodeIndices<Edge>(writeIdx);
        //        for(int succ : successors)
        //            frontier.push(succ);

        //        // For each WRITE node, find the "frontier" of last READs before
        //        // the next WRITE. We do this by collecting all downstream nodes
        //        // from this WRITE, stopping at the next WRITE nodes.
        //        while(!frontier.empty())
        //        {
        //            int current = frontier.front();
        //            frontier.pop();
        //            if(isWrite(current))
        //            {
        //                // Stop: this is the next WRITE, don't go further.
        //                continue;
        //            }

        //            if(isReadWrite(current))
        //            {
        //                // READWRITE acts as a boundary. It reads (end of this
        //                // lifespan) and writes (start of next). Include it as
        //                // a read endpoint but don't traverse further.
        //                reachableReads.insert(current);
        //                continue;
        //            }

        //            // It's a READ node.
        //            reachableReads.insert(current);

        //            // Continue to its successors.
        //            auto nexts = ordering.getOutputNodeIndices<Edge>(current);
        //            for(int next : nexts)
        //                frontier.push(next);
        //        }

        //        if(reachableReads.empty())
        //        {
        //            // Special case:
        //            //   WRITE with no subsequent READs --- the value is never read.
        //            //   We still record it as a zero-length lifespan.
        //            CoordLifespan lifespan;
        //            lifespan.begin = {writeControl};
        //            lifespan.end   = {writeControl};
        //            rv.lifespans.push_back(std::move(lifespan));
        //            continue;
        //        }

        //        // All reachable READs go into lifespan.end. The lifespan is considered done only after
        //        // ALL of them complete --- which is exactly how GraphExtent::isWithin
        //        // already works.
        //        //
        //        // If we need a single "join point", we can compute the earliest
        //        // common successor of all nodes in lastReadControls using the
        //        // control graph. For now, the set representation suffices.
        //        CoordLifespan lifespan;
        //        lifespan.begin = {writeControl};
        //        lifespan.end   = reachableReads;
        //        rv.lifespans.push_back(std::move(lifespan));
        //    }
        //    return rv;
        //}


        enum class Liveness
        {
            Alive,
            MaybeAlive,
            Dead,
        };

        Liveness queryLiveness(KernelGraph const& kgraph, int op, TagExtent const& tagExtent)
        {
            // Is node2 after node1 ?
            auto const isAfter = [&](int node1, int node2){
                auto const order = kgraph.control.compareNodes(
                        rocRoller::UpdateCache, node1, node2);
                return order == ControlGraph::NodeOrdering::LeftFirst;
            };

            auto const isWithin = [&](AliasDataFlowTagsDetail::GraphExtent const& gap, int const op) {
                return std::ranges::any_of(gap.begin, [&](int node){ return not isAfter(node, op); }) &&
                    std::ranges::any_of(gap.end, [&](int node){ return not isAfter(op, node); });
            };

            if(std::ranges::any_of(tagExtent.gaps, [&](auto const& gap) { return isWithin(gap, op); }))
                return Liveness::Alive;

            if(std::ranges::all_of(tagExtent.extent.begin, [&](auto const node) { return isAfter(op, node); }))
                return Liveness::Dead;

            if(std::ranges::all_of(tagExtent.extent.end, [&](auto const node) { return isAfter(node, op); }))
                return Liveness::Dead;

            return Liveness::MaybeAlive;
        }

        /**
         * Compute lifespans for ALL coordinates traced by ControlFlowRWTracer.
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
    }
}
