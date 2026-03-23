// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <queue>

#include <rocRoller/Graph/GraphUtilities.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/AliasDataFlowTags_detail.hpp>
#include <rocRoller/KernelGraph/Transforms/RegisterUsageAnalysis.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace RegisterUsageAnalysis_Detail
        {
            namespace CG = ControlGraph;
            namespace CT = CoordinateGraph;

            // --------------------------------------------------------
            // Helpers
            // --------------------------------------------------------
            // Get the string name of a coordinate graph edge.
            // Works for both CoordinateTransformEdge and DataFlowEdge.
            static std::string coordLabel(KernelGraph const& kgraph, int tag)
            {
                auto elemType = kgraph.coordinates.getElementType(tag);
                if(elemType == Graph::ElementType::Node)
                {
                    // Dimension node: MacroTile, VGPR, LDS, Offset dim, etc.
                    auto dim = kgraph.coordinates.getNode(tag);
                    return CT::toString(dim);
                }
                // Edge element: Stride, Offset, Buffer, DataFlow, Alias, etc.
                // The tracer tracks these because trackOffsetAndStride() and
                // trackBuffer() register edge tags as "coordinates."
                //
                // CT::toString(Edge) dispatches through the variant to produce
                // the concrete type name.
                auto        edge         = kgraph.coordinates.getEdge(tag);
                std::string edgeTypeName = CT::toString(edge);
                // Also show the source and destination nodes of this edge
                // so the user can see what it connects.
                auto               loc = kgraph.coordinates.getLocation(tag);
                std::ostringstream ss;
                ss << edgeTypeName << "{";
                bool first = true;
                for(int src : loc.incoming)
                {
                    if(!first)
                        ss << ",";
                    ss << src;
                    first = false;
                }
                ss << "->";
                first = true;
                for(int dst : loc.outgoing)
                {
                    if(!first)
                        ss << ",";
                    ss << dst;
                    first = false;
                }
                ss << "}";
                return ss.str();
            }

            static std::string opLabel(KernelGraph const& kgraph, int tag)
            {
                auto op = kgraph.control.getNode(tag);
                return CG::toString(op);
            }
            static std::string regTypeStr(Register::Type rt)
            {
                switch(rt)
                {
                case Register::Type::Vector:
                    return "VGPR";
                case Register::Type::Accumulator:
                    return "AGPR";
                case Register::Type::Scalar:
                    return "SGPR";
                default:
                    return Register::toString(rt);
                }
            }
            static Register::Type coordRegType(std::map<int, CoordTypeInfo> const& allTypes,
                                               int                                 tag)
            {
                auto it = allTypes.find(tag);
                if(it != allTypes.end() && it->second.resolved
                   && it->second.regType != Register::Type::Count)
                    return it->second.regType;
                return Register::Type::Vector;
            }
            static int coordRegCount(std::map<int, CoordRegisterEstimate> const& regEstimates,
                                     int                                         tag)
            {
                auto it = regEstimates.find(tag);
                if(it != regEstimates.end() && it->second.resolved)
                    return it->second.registerCount;
                return -1;
            }
            static std::string dotEscape(std::string const& s)
            {
                std::string out;
                out.reserve(s.size());
                for(char c : s)
                {
                    switch(c)
                    {
                    case '<':
                        out += "&lt;";
                        break;
                    case '>':
                        out += "&gt;";
                        break;
                    case '&':
                        out += "&amp;";
                        break;
                    case '"':
                        out += "&quot;";
                        break;
                    default:
                        out += c;
                    }
                }
                return out;
            }
            // Get the string name of a coordinate graph edge.
            // Works for both CoordinateTransformEdge and DataFlowEdge.
            static std::string coordEdgeLabel(KernelGraph const& kgraph, int edgeTag)
            {
                auto edge = kgraph.coordinates.getEdge(edgeTag);
                return CT::toString(edge);
            }

            // ============================================================
            // dumpControlGraphLiveness
            // ============================================================
            std::string
                dumpControlGraphLiveness(KernelGraph const&                           kgraph,
                                         std::map<int, CoordExtent> const&            allExtents,
                                         std::map<int, CoordTypeInfo> const&          allTypes,
                                         std::map<int, CoordRegisterEstimate> const&  regEstimates,
                                         std::map<int, OperationRegisterUsage> const& allUsage)
            {
                std::ostringstream ss;
                // Collect and sort operation tags for deterministic output.
                std::vector<int> opTags;
                for(auto opTag : kgraph.control.getNodes())
                    opTags.push_back(opTag);
                std::sort(opTags.begin(), opTags.end());
                for(int opTag : opTags)
                {
                    ss << "=== Operation [" << opTag << "] " << opLabel(kgraph, opTag) << " ===\n";
                    // Show register summary from OperationRegisterUsage.
                    auto usageIt = allUsage.find(opTag);
                    if(usageIt != allUsage.end())
                    {
                        auto const& u = usageIt->second;
                        ss << "  Registers:";
                        for(auto const& [rt, cnt] : u.regCountByType)
                            ss << "  " << regTypeStr(rt) << "=" << cnt;
                        ss << "  Total=" << u.totalRegisters << "\n";
                    }
                    // Classify each coordinate.
                    std::vector<std::pair<int, CoordLiveness>> alive, maybeAlive;
                    //for(auto const& [coordTag, ext] : allExtents)
                    for(auto const& [coordTag, ext] : usageIt->second.overlappedCoords)
                    {
                        auto lv = getCoordLiveness(kgraph, opTag, ext);
                        if(lv == CoordLiveness::Alive)
                            alive.emplace_back(coordTag, lv);
                        else if(lv == CoordLiveness::MaybeAlive)
                            maybeAlive.emplace_back(coordTag, lv);
                    }
                    auto printCoordList = [&](auto const& list) {
                        for(auto const& [cTag, lv] : list)
                        {
                            int  regs = coordRegCount(regEstimates, cTag);
                            auto rt   = coordRegType(allTypes, cTag);
                            ss << "    coord [" << cTag << "] " << coordLabel(kgraph, cTag) << "  "
                               << regTypeStr(rt);
                            if(regs >= 0)
                                ss << "  regs=" << regs;
                            else
                                ss << "  regs=?";
                            ss << "\n";
                        }
                    };
                    if(!alive.empty())
                    {
                        ss << "  Alive (" << alive.size() << "):\n";
                        printCoordList(alive);
                    }
                    if(!maybeAlive.empty())
                    {
                        ss << "  MaybeAlive (" << maybeAlive.size() << "):\n";
                        printCoordList(maybeAlive);
                    }
                    if(alive.empty() && maybeAlive.empty())
                        ss << "  (no live coordinates)\n";
                    ss << "\n";
                }
                return ss.str();
            }
            // ============================================================
            // dumpCoordinateAccessMap
            // ============================================================
            std::string
                dumpCoordinateAccessMap(KernelGraph const&                          kgraph,
                                        std::map<int, CoordExtent> const&           allExtents,
                                        std::map<int, CoordTypeInfo> const&         allTypes,
                                        std::map<int, CoordRegisterEstimate> const& regEstimates,
                                        std::map<int, std::vector<Record>> const&   recordsByCoord)
            {
                std::ostringstream ss;
                // Sort coordinate tags for deterministic output.
                std::vector<int> coordTags;
                for(auto const& [tag, _] : recordsByCoord)
                    coordTags.push_back(tag);
                std::sort(coordTags.begin(), coordTags.end());
                for(int cTag : coordTags)
                {
                    int  regs = coordRegCount(regEstimates, cTag);
                    auto rt   = coordRegType(allTypes, cTag);
                    ss << "=== Coordinate [" << cTag << "] " << coordLabel(kgraph, cTag)
                       << " ===\n";
                    ss << "  RegType=" << regTypeStr(rt);
                    if(regs >= 0)
                        ss << "  Regs=" << regs;
                    else
                        ss << "  Regs=?";
                    // Type info.
                    auto typeIt = allTypes.find(cTag);
                    if(typeIt != allTypes.end() && typeIt->second.resolved)
                        ss << "  DataType=" << toString(typeIt->second.varType);
                    ss << "\n";
                    // Extent.
                    auto extIt = allExtents.find(cTag);
                    if(extIt != allExtents.end() && !extIt->second.empty())
                    {
                        ss << "  Extent begin={";
                        bool first = true;
                        for(int b : extIt->second.extent.begin)
                        {
                            if(!first)
                                ss << ",";
                            ss << b;
                            first = false;
                        }
                        ss << "}  end={";
                        first = true;
                        for(int e : extIt->second.extent.end)
                        {
                            if(!first)
                                ss << ",";
                            ss << e;
                            first = false;
                        }
                        ss << "}\n";
                    }
                    // Accesses.
                    auto recIt = recordsByCoord.find(cTag);
                    if(recIt != recordsByCoord.end())
                    {
                        ss << "  Accesses (" << recIt->second.size() << "):\n";
                        for(auto const& rec : recIt->second)
                        {
                            ss << "    op [" << rec.control << "] " << opLabel(kgraph, rec.control)
                               << "  " << toString(rec.rw) << "\n";
                        }
                    }
                    ss << "\n";
                }
                return ss.str();
            }
            // ============================================================
            // exportControlGraphLivenessDot
            // ============================================================
            void exportControlGraphLivenessDot(
                std::ostream&                                out,
                KernelGraph const&                           kgraph,
                std::map<int, CoordExtent> const&            allExtents,
                std::map<int, CoordTypeInfo> const&          allTypes,
                std::map<int, CoordRegisterEstimate> const&  regEstimates,
                std::map<int, OperationRegisterUsage> const& allUsage)
            {
                out << "digraph ControlGraphLiveness {\n";
                out << "  rankdir=TB;\n";
                out << "  node [shape=plaintext fontname=\"Courier\"];\n";
                out << "  edge [fontname=\"Courier\" fontsize=10];\n";
                out << "\n";
                // Collect operation tags.
                std::vector<int> opTags;
                for(auto opTag : kgraph.control.getNodes())
                    opTags.push_back(opTag);
                std::sort(opTags.begin(), opTags.end());
                for(int opTag : opTags)
                {
                    std::string label = opLabel(kgraph, opTag);
                    // Build an HTML-like label with a table.
                    std::ostringstream html;
                    html << "<<TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\">\n";
                    // Header row: operation name.
                    html << "<TR><TD COLSPAN=\"3\" BGCOLOR=\"lightblue\">"
                         << "<B>[" << opTag << "] " << dotEscape(label) << "</B></TD></TR>\n";
                    // Register summary row.
                    auto usageIt = allUsage.find(opTag);
                    if(usageIt != allUsage.end())
                    {
                        html << "<TR><TD COLSPAN=\"3\" BGCOLOR=\"lightyellow\">";
                        for(auto const& [rt, cnt] : usageIt->second.regCountByType)
                            html << regTypeStr(rt) << "=" << cnt << " ";
                        html << "Total=" << usageIt->second.totalRegisters;
                        html << "</TD></TR>\n";
                    }
                    // Alive coordinates.
                    for(auto const& [coordTag, ext] : allExtents)
                    {
                        auto lv = getCoordLiveness(kgraph, opTag, ext);
                        if(lv != CoordLiveness::Alive && lv != CoordLiveness::MaybeAlive)
                            continue;
                        int         regs = coordRegCount(regEstimates, coordTag);
                        auto        rt   = coordRegType(allTypes, coordTag);
                        std::string bg   = (lv == CoordLiveness::Alive) ? "#c8ffc8" // green
                                                                        : "#ffffc8"; // yellow
                        html << "<TR>"
                             << "<TD BGCOLOR=\"" << bg << "\">"
                             << dotEscape(coordLabel(kgraph, coordTag)) << "</TD>"
                             << "<TD BGCOLOR=\"" << bg << "\">" << regTypeStr(rt) << "</TD>"
                             << "<TD BGCOLOR=\"" << bg << "\">"
                             << (regs >= 0 ? std::to_string(regs) : "?") << "</TD></TR>\n";
                    }
                    html << "</TABLE>>";
                    out << "  op_" << opTag << " [label=" << html.str() << "];\n";
                }
                out << "\n";
                // Edges from the control graph.
                for(auto edgeTag : kgraph.control.getEdges())
                {
                    auto        loc     = kgraph.control.getLocation(edgeTag);
                    auto        edgeStr = CG::toString(kgraph.control.getEdge(edgeTag));
                    std::string style   = "solid";
                    std::string color   = "black";
                    if(edgeStr == "Body" || edgeStr == "Else" || edgeStr == "Initialize"
                       || edgeStr == "ForLoopIncrement")
                    {
                        style = "dashed";
                        color = "blue";
                    }
                    for(int src : loc.incoming)
                    {
                        for(int dst : loc.outgoing)
                        {
                            out << "  op_" << src << " -> op_" << dst << " [label=\"" << edgeStr
                                << "\""
                                << " style=" << style << " color=" << color << "];\n";
                        }
                    }
                }
                out << "}\n";
            }
            // ============================================================
            // exportCoordinateAccessDot
            // ============================================================
            void exportCoordinateAccessDot(std::ostream&                               out,
                                           KernelGraph const&                          kgraph,
                                           std::map<int, CoordExtent> const&           allExtents,
                                           std::map<int, CoordTypeInfo> const&         allTypes,
                                           std::map<int, CoordRegisterEstimate> const& regEstimates,
                                           std::map<int, std::vector<Record>> const& recordsByCoord)
            {
                out << "digraph CoordinateAccessMap {\n";
                out << "  rankdir=LR;\n";
                out << "  node [fontname=\"Courier\"];\n";
                out << "  edge [fontname=\"Courier\" fontsize=10];\n";
                out << "\n";
                // Collect all referenced operation tags.
                std::set<int> opTagsUsed;
                for(auto const& [cTag, records] : recordsByCoord)
                    for(auto const& rec : records)
                        opTagsUsed.insert(rec.control);
                // Collect tracked coordinate tags.
                std::set<int>    trackedCoords;
                std::vector<int> coordTags;
                for(auto const& [tag, _] : recordsByCoord)
                {
                    coordTags.push_back(tag);
                    trackedCoords.insert(tag);
                }
                std::sort(coordTags.begin(), coordTags.end());
                // --- Subgraph: operation nodes ---
                out << "  subgraph cluster_ops {\n";
                out << "    label=\"Operations\";\n";
                out << "    style=dashed;\n";
                out << "    color=gray;\n";
                for(int opTag : opTagsUsed)
                {
                    out << "    op_" << opTag << " [shape=box style=filled fillcolor=lightblue"
                        << " label=\"[" << opTag << "] " << dotEscape(opLabel(kgraph, opTag))
                        << "\"];\n";
                }
                out << "  }\n\n";
                // --- Subgraph: coordinate nodes ---
                out << "  subgraph cluster_coords {\n";
                out << "    label=\"Coordinates\";\n";
                out << "    style=dashed;\n";
                out << "    color=gray;\n";
                for(int cTag : coordTags)
                {
                    int                regs = coordRegCount(regEstimates, cTag);
                    auto               rt   = coordRegType(allTypes, cTag);
                    std::ostringstream lbl;
                    lbl << "[" << cTag << "] " << dotEscape(coordLabel(kgraph, cTag)) << "\\n"
                        << regTypeStr(rt);
                    if(regs >= 0)
                        lbl << " regs=" << regs;
                    else
                        lbl << " regs=?";
                    out << "    coord_" << cTag
                        << " [shape=ellipse style=filled fillcolor=lightyellow"
                        << " label=\"" << lbl.str() << "\"];\n";
                }
                out << "  }\n\n";
                // --- Edges: operation -> coordinate (R/W access) ---
                out << "  // Operation -> Coordinate access edges\n";
                for(auto const& [cTag, records] : recordsByCoord)
                {
                    for(auto const& rec : records)
                    {
                        std::string rwStr = toString(rec.rw);
                        std::string color = "black";
                        if(rec.rw == ControlFlowRWTracer::WRITE)
                            color = "red";
                        else if(rec.rw == ControlFlowRWTracer::READ)
                            color = "darkgreen";
                        else
                            color = "purple";
                        out << "  op_" << rec.control << " -> coord_" << cTag << " [label=\""
                            << rwStr << "\""
                            << " color=" << color << "];\n";
                    }
                }
                out << "\n";
                // --- Edges: coordinate -> coordinate (coord graph edges) ---
                //
                // Iterate all edges in the coordinate graph.  For each edge
                // whose source AND destination are both tracked coordinates,
                // emit a DOT edge labeled with the edge type name
                // (DataFlow, Alias, PassThrough, Tile, Flatten, etc.).
                out << "  // Coordinate graph edges\n";
                for(auto edgeTag : kgraph.coordinates.getEdges())
                {
                    auto loc = kgraph.coordinates.getLocation(edgeTag);
                    // Check if at least one source and one dest are tracked.
                    bool anySrcTracked = false;
                    bool anyDstTracked = false;
                    for(int n : loc.incoming)
                        if(trackedCoords.count(n))
                            anySrcTracked = true;
                    for(int n : loc.outgoing)
                        if(trackedCoords.count(n))
                            anyDstTracked = true;
                    if(!anySrcTracked && !anyDstTracked)
                        continue;
                    std::string edgeName = coordEdgeLabel(kgraph, edgeTag);
                    // For edges that involve untracked coordinates,
                    // create lightweight placeholder nodes so the edge
                    // is still visible.
                    for(int src : loc.incoming)
                    {
                        if(!trackedCoords.count(src))
                        {
                            out << "  coord_" << src << " [shape=ellipse style=dotted"
                                << " label=\"[" << src << "] " << dotEscape(coordLabel(kgraph, src))
                                << "\"];\n";
                            trackedCoords.insert(src);
                        }
                    }
                    for(int dst : loc.outgoing)
                    {
                        if(!trackedCoords.count(dst))
                        {
                            out << "  coord_" << dst << " [shape=ellipse style=dotted"
                                << " label=\"[" << dst << "] " << dotEscape(coordLabel(kgraph, dst))
                                << "\"];\n";
                            trackedCoords.insert(dst);
                        }
                    }
                    // Emit edges.  Coordinate graph edges are hyperedges
                    // ({src1,src2,...} -> {dst1,dst2,...}).  For DOT we
                    // draw one arrow per src-dst pair, all with the same
                    // label.
                    for(int src : loc.incoming)
                    {
                        for(int dst : loc.outgoing)
                        {
                            out << "  coord_" << src << " -> coord_" << dst << " [label=\""
                                << edgeName << "\""
                                << " style=dashed color=gray40"
                                << " fontcolor=gray30];\n";
                        }
                    }
                }
                out << "}\n";
            }

            // ================================================================
            // Per-work-item element count helpers for MacroTile
            // ================================================================
            //
            // AMD GPU register model:
            //   VGPRs and AGPRs -> per-work-item (per lane in a wave).
            //   SGPRs           -> per-wave (shared across all lanes).
            //   vgpr_count in assembly = VGPRs each work-item needs.
            //
            // MacroTile.sizes are workgroup-level tile dimensions.
            // The per-work-item register count depends on how the data
            // is distributed:
            //
            //   WAVE memory + MATRIX layout -> WaveTile decomposition.
            //     Per lane = waveTile.elements() / wavefrontSize.
            //     (See LowerTile.cpp line 568-570.)
            //
            //   VGPR memory (any layout) -> ThreadTile decomposition.
            //     Per work-item = product(subTileSizes[0..rank-1]).
            //     (See ThreadTile::ThreadTile(MacroTile) in Dimension.cpp:282.)
            //
            //   LDS memory -> 0 GPRs (shared memory).
            //
            // Important: memoryType determines the decomposition, NOT
            // layoutType.  A VGPR MacroTile with MATRIX_A layoutType
            // (from LDS->VGPR conversion in LowerTile.cpp:1424) still
            // uses the ThreadTile path (loadMacroTile_VGPR), not WaveTile.
            namespace detail
            {
                // Returns true for memory types that use WaveTile
                // decomposition (data distributed across lanes of a wave).
                bool isWaveLikeMemory(MemoryType mt)
                {
                    switch(mt)
                    {
                    case MemoryType::WAVE:
                    case MemoryType::WAVE_LDS:
                    case MemoryType::WAVE_SPLIT:
                    case MemoryType::WAVE_FROM_GLOBAL:
                    case MemoryType::WAVE_LDS_FROM_GLOBAL:
                    case MemoryType::WAVE_SWIZZLE:
                    case MemoryType::WAVE_Direct2LDS:
                        return true;
                    default:
                        return false;
                    }
                }
                bool isMatrixLayout(LayoutType lt)
                {
                    return lt == LayoutType::MATRIX_A || lt == LayoutType::MATRIX_B
                           || lt == LayoutType::MATRIX_ACCUMULATOR;
                }
                // Element count for ONE wave tile, derived from
                // MacroTile.subTileSizes.
                //
                // From WaveTile::WaveTile(MacroTile) in Dimension.cpp:
                //   MATRIX_A:   sizes = {waveM, waveK} -> waveM * waveK
                //   MATRIX_B:   sizes = {waveK, waveN} -> waveK * waveN
                //   MATRIX_ACC: sizes = {waveM, waveN} -> waveM * waveN
                //
                // where waveM = subTileSizes[0], waveN = subTileSizes[1],
                //       waveK = subTileSizes[2].
                int waveTileElements(CT::MacroTile const& mt)
                {
                    if(mt.subTileSizes.size() < 3)
                        return 0;
                    switch(mt.layoutType)
                    {
                    case LayoutType::MATRIX_A:
                        return mt.subTileSizes[0] * mt.subTileSizes[2];
                    case LayoutType::MATRIX_B:
                        return mt.subTileSizes[2] * mt.subTileSizes[1];
                    case LayoutType::MATRIX_ACCUMULATOR:
                        return mt.subTileSizes[0] * mt.subTileSizes[1];
                    default:
                        return 0;
                    }
                }
                // Compute per-work-item element count for a MacroTile.
                // Returns 0 when the count cannot be determined or the
                // tile does not reside in GPRs (LDS, Global).
                int perWorkitemElements(CT::MacroTile const& mt, int wavefrontSize)
                {
                    // LDS: data in shared memory -> 0 GPRs.
                    if(mt.memoryType == MemoryType::LDS)
                        return 0;
                    // WAVE-like memory + MATRIX layout: WaveTile path.
                    //
                    // Each lane (work-item) in a wave holds
                    // waveTileElements / wavefrontSize values.
                    //
                    // Example: MATRIX_A with subTileSizes=(16,16,32,1),
                    //   wavefrontSize=64:
                    //   waveTileElements = 16*32 = 512
                    //   perWorkitem = 512/64 = 8
                    if(isWaveLikeMemory(mt.memoryType) && isMatrixLayout(mt.layoutType))
                    {
                        int wte = waveTileElements(mt);
                        if(wte > 0 && wavefrontSize > 0)
                            return wte / wavefrontSize;
                        return 0;
                    }
                    // Everything else (VGPR memory, AGPR memory,
                    // non-MATRIX WAVE tiles): ThreadTile path.
                    //
                    // Per-work-item = product of first `rank` subTileSizes.
                    //
                    // For a VGPR MacroTile with sizes={M,N},
                    // subTileSizes={m,n}:
                    //   ThreadTile.sizes = {m, n}
                    //   wsizes = {M/m, N/n}
                    //   Each work-item owns m*n elements.
                    //
                    // For a VGPR MacroTile with MATRIX_A layout (from
                    // LDS->VGPR conversion, subTileSizes has 4 elements
                    // but rank=2):
                    //   product(subTileSizes[:2]) = waveM * waveN.
                    //   Correct for ThreadTile path (loadMacroTile_VGPR).
                    if(!mt.subTileSizes.empty() && mt.rank > 0)
                    {
                        int count = 1;
                        int dims  = std::min(static_cast<int>(mt.subTileSizes.size()), mt.rank);
                        for(int i = 0; i < dims; ++i)
                        {
                            if(mt.subTileSizes[i] <= 0)
                                return 0;
                            count *= mt.subTileSizes[i];
                        }
                        return count;
                    }
                    return 0;
                }
            } // namespace detail
            // ================================================================
            // CoordExtent
            // ================================================================
            bool CoordExtent::empty() const
            {
                return extent.begin.empty() || extent.end.empty();
            }
            // Merge records that share the same control node but have
            // different ReadWrite types (e.g., x = x + 1 produces both
            // READ and WRITE for the same coordinate).  Uses combine()
            // to merge: READ + WRITE -> READWRITE.
            std::vector<Record> deduplicateRecords(std::vector<Record> const& records)
            {
                std::map<int, Record> merged;
                for(auto const& rec : records)
                {
                    auto it = merged.find(rec.control);
                    if(it == merged.end())
                        merged.emplace(rec.control, rec);
                    else
                        it->second.rw = combine(it->second.rw, rec.rw);
                }
                std::vector<Record> result;
                result.reserve(merged.size());
                for(auto& [ctrl, rec] : merged)
                    result.push_back(rec);
                return result;
            }
            // Build an ordering graph for records, correctly handling body
            // relationships (LeftInBodyOfRight, RightInBodyOfLeft).
            //
            // Body relationships are interpreted as execution order edges:
            //   LeftInBodyOfRight  -> right starts before left
            //   RightInBodyOfLeft  -> left starts before right
            AliasDataFlowTagsDetail::TagRWGraph
                getOrderingWithBodies(KernelGraph const& kgraph, std::vector<Record> const& records)
            {
                AliasDataFlowTagsDetail::TagRWGraph ordering;
                std::vector<int>                    graphIndices;
                graphIndices.reserve(records.size());
                for(auto const& rec : records)
                    graphIndices.push_back(ordering.addElement(rec));
                for(size_t a = 0; a < records.size(); a++)
                {
                    for(size_t b = a + 1; b < records.size(); b++)
                    {
                        if(records[a].control == records[b].control)
                            continue;
                        auto order = kgraph.control.compareNodes(
                            rocRoller::UpdateCache, records[a].control, records[b].control);
                        auto aIdx = graphIndices[a];
                        auto bIdx = graphIndices[b];
                        switch(order)
                        {
                        case CG::NodeOrdering::LeftFirst:
                            ordering.addElement(CG::Sequence{}, {aIdx}, {bIdx});
                            break;
                        case CG::NodeOrdering::LeftInBodyOfRight:
                            ordering.addElement(CG::Sequence{}, {bIdx}, {aIdx});
                            break;
                        case CG::NodeOrdering::Undefined:
                            break;
                        case CG::NodeOrdering::RightInBodyOfLeft:
                            ordering.addElement(CG::Sequence{}, {aIdx}, {bIdx});
                            break;
                        case CG::NodeOrdering::RightFirst:
                            ordering.addElement(CG::Sequence{}, {bIdx}, {aIdx});
                            break;
                        case CG::NodeOrdering::Count:
                            Throw<FatalError>("Unexpected NodeOrdering::Count");
                        }
                    }
                }
                {
                    auto truePred = [](int) { return true; };
                    Graph::removeRedundantEdges(ordering, truePred);
                }
                return ordering;
            }
            // Build the extent (lifespan) for a single coordinate from
            // its raw trace records.  The extent's begin set is the
            // roots of the ordering graph; the end set is the leaves.
            CoordExtent getCoordExtent(KernelGraph const&         kgraph,
                                       int                        coordTag,
                                       std::vector<Record> const& rawRecords)
            {
                CoordExtent result;
                result.coordTag = coordTag;
                if(rawRecords.empty())
                    return result;
                for(auto const& rec : rawRecords)
                    result.allAccessNodes.insert(rec.control);
                auto records        = deduplicateRecords(rawRecords);
                auto ordering       = getOrderingWithBodies(kgraph, records);
                auto getControl     = [&](int idx) { return ordering.getNode(idx).control; };
                result.extent.begin = ordering.roots().map(getControl).to<std::set>();
                result.extent.end   = ordering.leaves().map(getControl).to<std::set>();
                return result;
            }
            // Build extents for ALL coordinates tracked by ControlFlowRWTracer.
            std::map<int, CoordExtent> buildAllCoordExtents(KernelGraph const& kgraph)
            {
                ControlFlowRWTracer                tracer(kgraph);
                auto                               allRecords = tracer.coordinatesReadWrite();
                std::map<int, std::vector<Record>> recordsByCoord;
                for(auto const& rec : allRecords)
                    recordsByCoord[rec.coordinate].push_back(rec);
                std::map<int, CoordExtent> result;
                for(auto& [coordTag, records] : recordsByCoord)
                    result[coordTag] = getCoordExtent(kgraph, coordTag, records);
                return result;
            }
            // ================================================================
            // Liveness classification
            // ================================================================
            //
            // Dead:       O is before ALL begin nodes OR after ALL end nodes.
            // Alive:      O is after ANY begin node AND before ANY end node.
            // MaybeAlive: Otherwise (ordering is ambiguous).
            //
            // Special case: if O is in allAccessNodes (i.e., O directly
            // accesses this coordinate), it is immediately Alive.  This
            // also avoids calling compareNodes(O, O) which asserts.
            CoordLiveness
                getCoordLiveness(KernelGraph const& kgraph, int O, CoordExtent const& coordExtent)
            {
                if(coordExtent.empty())
                    return CoordLiveness::Dead;
                if(coordExtent.allAccessNodes.contains(O))
                    return CoordLiveness::Alive;
                auto const& ext = coordExtent.extent;
                // Check: is O strictly before ALL begin nodes?
                {
                    bool beforeAllBegin = true;
                    for(int b : ext.begin)
                    {
                        if(O == b)
                        {
                            beforeAllBegin = false;
                            break;
                        }
                        auto ord = kgraph.control.compareNodes(rocRoller::UpdateCache, O, b);
                        if(ord != CG::NodeOrdering::LeftFirst)
                        {
                            beforeAllBegin = false;
                            break;
                        }
                    }
                    if(beforeAllBegin)
                        return CoordLiveness::Dead;
                }
                // Check: is O strictly after ALL end nodes?
                {
                    bool afterAllEnd = true;
                    for(int e : ext.end)
                    {
                        if(O == e)
                        {
                            afterAllEnd = false;
                            break;
                        }
                        auto ord = kgraph.control.compareNodes(rocRoller::UpdateCache, e, O);
                        if(ord != CG::NodeOrdering::LeftFirst)
                        {
                            afterAllEnd = false;
                            break;
                        }
                    }
                    if(afterAllEnd)
                        return CoordLiveness::Dead;
                }
                // Check: is O after ANY begin AND before ANY end?
                // Body relationships count as overlapping (alive).
                {
                    auto isBeforeOrOverlapping = [](CG::NodeOrdering order) {
                        return order == CG::NodeOrdering::LeftFirst
                               || order == CG::NodeOrdering::LeftInBodyOfRight
                               || order == CG::NodeOrdering::RightInBodyOfLeft;
                    };
                    bool anyBeginBeforeO = false;
                    for(int b : ext.begin)
                    {
                        if(O == b)
                        {
                            anyBeginBeforeO = true;
                            break;
                        }
                        if(isBeforeOrOverlapping(
                               kgraph.control.compareNodes(rocRoller::UpdateCache, b, O)))
                        {
                            anyBeginBeforeO = true;
                            break;
                        }
                    }
                    bool oBeforeAnyEnd = false;
                    for(int e : ext.end)
                    {
                        if(O == e)
                        {
                            oBeforeAnyEnd = true;
                            break;
                        }
                        if(isBeforeOrOverlapping(
                               kgraph.control.compareNodes(rocRoller::UpdateCache, O, e)))
                        {
                            oBeforeAnyEnd = true;
                            break;
                        }
                    }
                    if(anyBeginBeforeO && oBeforeAnyEnd)
                        return CoordLiveness::Alive;
                }
                return CoordLiveness::MaybeAlive;
            }
            // Classify all coordinates' liveness at a single operation.
            std::map<int, CoordLiveness>
                getLivenessAtOperation(KernelGraph const&                kgraph,
                                       int                               operationTag,
                                       std::map<int, CoordExtent> const& allExtents)
            {
                std::map<int, CoordLiveness> result;
                for(auto const& [coordTag, ext] : allExtents)
                    result[coordTag] = getCoordLiveness(kgraph, operationTag, ext);
                return result;
            }
            std::string toString(CoordLiveness liveness)
            {
                switch(liveness)
                {
                case CoordLiveness::Dead:
                    return "Dead";
                case CoordLiveness::Alive:
                    return "Alive";
                case CoordLiveness::MaybeAlive:
                    return "MaybeAlive";
                }
                return "Unknown";
            }

            enum class CoordRegAllocation
            {
                NoRegisters, // guaranteed 0 physical registers
                ExpressionOnly, // stored as expression; may become literal (0) or runtime (small)
                PhysicalRegister // will need actual register allocation
            };

            CoordRegAllocation
                classifyCoordinate(KernelGraph const& kgraph,
                                   int                coordTag,
                                   std::vector<ControlFlowRWTracer::ReadWriteRecord> const& records)
            {
                // 1. Check if it's a coordinate graph edge (Stride, Offset, Buffer, etc.)
                auto elemType = kgraph.coordinates.getElementType(coordTag);
                if(elemType == Graph::ElementType::Node)
                {
                    auto dim = kgraph.coordinates.getNode(coordTag);
                    // Dimension types that never consume GPRs:
                    bool noRegs = std::visit(
                        overloaded{
                            [](CoordinateGraph::User const&) { return true; },
                            [](CoordinateGraph::LDS const&) { return true; },
                            [](CoordinateGraph::Workgroup const&) { return true; },
                            [](CoordinateGraph::Workitem const&) { return true; },
                            [](CoordinateGraph::Wavefront const&) { return true; },
                            [](CoordinateGraph::Lane const&) { return true; },
                            [](CoordinateGraph::ForLoop const&) { return true; },
                            [](CoordinateGraph::Unroll const&) { return true; },
                            [](CoordinateGraph::SubDimension const&) { return true; },
                            [](CoordinateGraph::MacroTileIndex const&) { return true; },
                            [](CoordinateGraph::MacroTileNumber const&) { return true; },
                            [](CoordinateGraph::WaveTileIndex const&) { return true; },
                            [](CoordinateGraph::WaveTileNumber const&) { return true; },
                            [](CoordinateGraph::ThreadTileIndex const&) { return true; },
                            [](CoordinateGraph::ThreadTileNumber const&) { return true; },
                            [](CoordinateGraph::JammedWaveTileNumber const&) { return true; },
                            [](CoordinateGraph::ElementNumber const&) { return true; },
                            [](auto const&) { return false; }},
                        dim);
                    if(noRegs)
                        return CoordRegAllocation::NoRegisters;
                    // Remaining: MacroTile, VGPR, Linear, ThreadTile, WaveTile,
                    //            Adhoc, VGPRBlockNumber, VGPRBlockIndex
                    // These are storage dimensions -> PhysicalRegister
                    // (but the actual count depends on type resolution)
                }
                // 2. For edge coordinates (Stride, Offset, Buffer) or storage nodes,
                //    find the Assign operation that writes to this coordinate.
                for(auto const& rec : records)
                {
                    if(rec.coordinate != coordTag)
                        continue;
                    if(rec.rw != ControlFlowRWTracer::WRITE
                       && rec.rw != ControlFlowRWTracer::READWRITE)
                        continue;
                    auto op     = kgraph.control.getNode(rec.control);
                    auto assign = std::get_if<ControlGraph::Assign>(&op);
                    if(!assign)
                        continue; // non-Assign write (e.g., LoadTiled)
                            // Signal 1: strideExpressionAttributes -> expression-only
                    if(assign->strideExpressionAttributes.has_value())
                    {
                        auto evalTimes = Expression::evaluationTimes(assign->expression);
                        if(evalTimes[Expression::EvaluationTime::Translate])
                            return CoordRegAllocation::NoRegisters;
                        else
                            return CoordRegAllocation::ExpressionOnly;
                    }
                    // Signal 2: Expression fully evaluable at translate time -> literal
                    auto evalTimes = Expression::evaluationTimes(assign->expression);
                    if(evalTimes[Expression::EvaluationTime::Translate])
                    {
                        return CoordRegAllocation::NoRegisters;
                    }
                }
                // 3. Default: needs physical registers
                return CoordRegAllocation::PhysicalRegister;
            }

            // ================================================================
            // Type resolution -- Priorities 1-3 (operation-based)
            // ================================================================
            //
            // P1: Assign.variableType -- most explicit source.  Set by
            //     LowerTensorContraction (accType), AssignIndexExpressions
            //     (offsetType, strideType), PrefetchScale (copy Assign).
            //
            // P2: resultVariableType(assign.expression) -- handles Convert
            //     expressions and typed DataFlowTags.
            //
            // P3: getDataType(operation) -- for Load/Store/Exchange that
            //     have a varType member.
            CoordTypeInfo resolveCoordTypeFromOperations(KernelGraph const&         kgraph,
                                                         int                        coordTag,
                                                         std::vector<Record> const& records)
            {
                CoordTypeInfo info;

                bool isBuffer = false;
                if(kgraph.coordinates.getElementType(coordTag) == Graph::ElementType::Edge)
                {
                    if(std::holds_alternative<CoordinateGraph::DataFlowEdge>(
                           kgraph.coordinates.getEdge(coordTag)))
                        if(std::holds_alternative<CoordinateGraph::Buffer>(
                               std::get<CoordinateGraph::DataFlowEdge>(
                                   kgraph.coordinates.getEdge(coordTag))))
                            isBuffer = true;
                }

                // Priority 1: Assign.variableType
                for(auto const& rec : records)
                {
                    auto  op     = kgraph.control.getNode(rec.control);
                    auto* assign = std::get_if<CG::Assign>(&op);
                    if(!assign)
                        continue;
                    if(assign->variableType.has_value()
                       && assign->variableType->dataType != DataType::None)
                    {
                        info.varType    = assign->variableType.value();
                        info.regType    = assign->regType;
                        info.valueCount = assign->valueCount;
                        info.resolved   = true;
                        if(isBuffer)
                            info.regType = Register::Type::Scalar;
                        return info;
                    }
                }
                // Priority 2: resultVariableType(assign.expression)
                for(auto const& rec : records)
                {
                    auto  op     = kgraph.control.getNode(rec.control);
                    auto* assign = std::get_if<CG::Assign>(&op);
                    if(!assign)
                        continue;
                    auto vt = Expression::resultVariableType(assign->expression);
                    if(vt.dataType != DataType::None && vt.dataType != DataType::Count)
                    {
                        info.varType    = vt;
                        info.regType    = assign->regType;
                        info.valueCount = assign->valueCount;
                        info.resolved   = true;
                        if(isBuffer)
                            info.regType = Register::Type::Scalar;
                        return info;
                    }
                }
                // Priority 3: getDataType from non-Assign operations
                for(auto const& rec : records)
                {
                    auto op = kgraph.control.getNode(rec.control);
                    auto dt = CG::getDataType(op);
                    if(dt != DataType::None && dt != DataType::Count)
                    {
                        info.varType  = VariableType(dt);
                        info.resolved = true;
                        auto getRT    = [](auto const& o) -> Register::Type {
                            if constexpr(requires { o.regType; })
                                return o.regType;
                            return Register::Type::Vector;
                        };
                        info.regType = std::visit(getRT, op);
                        if(isBuffer)
                            info.regType = Register::Type::Scalar;
                        return info;
                    }
                }
                return info; // resolved == false
            }
            // ================================================================
            // Type resolution -- Priority 4 (DataFlow neighbor propagation)
            // ================================================================
            //
            // For coordinates still unresolved after P1-P3, propagate types
            // through DataFlow edges in the coordinate graph.  Iterate to
            // a fixpoint.  Handles XOp fusion chains where intermediate
            // MacroTiles have None type but their DataFlow neighbors are
            // resolved.
            std::map<int, CoordTypeInfo>
                resolveAllCoordTypes(KernelGraph const&                        kgraph,
                                     std::map<int, std::vector<Record>> const& recordsByCoord)
            {
                std::map<int, CoordTypeInfo> result;
                // Pass 1: operation-based resolution (P1-P3).
                for(auto const& [coordTag, records] : recordsByCoord)
                    result[coordTag] = resolveCoordTypeFromOperations(kgraph, coordTag, records);
                // Pass 2: DataFlow neighbor propagation (P4).
                auto isDataFlowEdge
                    = [](auto const& edge) { return CT::isEdge<CT::DataFlowEdge>(edge); };
                bool changed = true;
                while(changed)
                {
                    changed = false;
                    for(auto& [coordTag, info] : result)
                    {
                        if(info.resolved)
                            continue;
                        if(kgraph.coordinates.getElementType(coordTag) != Graph::ElementType::Node)
                            continue;
                        // Try upstream neighbors.
                        for(auto neighbor :
                            kgraph.coordinates.getConnectedNodeIndices<Graph::Direction::Upstream>(
                                coordTag, isDataFlowEdge))
                        {
                            auto it = result.find(neighbor);
                            if(it != result.end() && it->second.resolved)
                            {
                                info.varType  = it->second.varType;
                                info.regType  = it->second.regType;
                                info.resolved = true;
                                changed       = true;
                                break;
                            }
                        }
                        if(info.resolved)
                            continue;
                        // Try downstream neighbors.
                        for(auto neighbor :
                            kgraph.coordinates
                                .getConnectedNodeIndices<Graph::Direction::Downstream>(
                                    coordTag, isDataFlowEdge))
                        {
                            auto it = result.find(neighbor);
                            if(it != result.end() && it->second.resolved)
                            {
                                info.varType  = it->second.varType;
                                info.regType  = it->second.regType;
                                info.resolved = true;
                                changed       = true;
                                break;
                            }
                        }
                    }
                }
                return result;
            }
            // ================================================================
            // Per-work-item register count estimation
            // ================================================================
            //
            // For MacroTile coordinates:
            //   1. Try geometry-based per-work-item element count
            //      (detail::perWorkitemElements).  This is the primary
            //      and most accurate method.
            //   2. Fallback: Assign.valueCount > 1 (explicitly set
            //      per-work-item count from LowerTensorContraction etc.).
            //   3. Otherwise: unresolved.
            //
            //   Physical GPRs = (perWorkitemElements * registerCount) / packing.
            //   This matches LowerFromKernelGraph.cpp line 661-690:
            //     valueCount /= packing;
            //     physicalRegisterCount = valueCount * registerCount;
            //
            // For non-MacroTile coordinates:
            //   1 logical value per entity -> registerCount GPRs.
            //   (1 for 32-bit types, 2 for 64-bit types.)
            CoordRegisterEstimate estimateCoordRegisterCount(KernelGraph const&   kgraph,
                                                             int                  coordTag,
                                                             CoordTypeInfo const& typeInfo,
                                                             int                  wavefrontSize)
            {
                CoordRegisterEstimate est;
                // LDS coordinate dimension -> 0 GPRs.
                if(kgraph.coordinates.get<CT::LDS>(coordTag).has_value())
                {
                    est.resolved = true;
                    return est;
                }
                if(!typeInfo.resolved)
                    return est;
                auto const& dtInfo      = DataTypeInfo::Get(typeInfo.varType);
                int         regPerValue = dtInfo.registerCount;
                int         packing     = dtInfo.packing;
                auto        maybeMT     = kgraph.coordinates.get<CT::MacroTile>(coordTag);
                if(maybeMT.has_value())
                {
                    auto const& mt = maybeMT.value();
                    // Primary: geometry-based per-work-item element count.
                    //
                    // For WAVE+MATRIX tiles:
                    //   perWorkitem = waveTileElements / wavefrontSize
                    //   e.g., MATRIX_A (16,16,32,1), wfs=64 -> 16*32/64 = 8
                    //
                    // For VGPR tiles:
                    //   perWorkitem = product(subTileSizes[:rank])
                    //   e.g., subTileSizes={4,2}, rank=2 -> 4*2 = 8
                    //
                    // For LDS tiles:
                    //   perWorkitem = 0
                    int perWorkitem = detail::perWorkitemElements(mt, wavefrontSize);
                    if(perWorkitem > 0)
                    {
                        est.registerCount = (perWorkitem * regPerValue) / packing;
                        est.resolved      = true;
                        return est;
                    }
                    // Fallback: Assign.valueCount.
                    // Only trust values > 1 to avoid using the default
                    // of 1 (which would underestimate for MacroTiles that
                    // didn't explicitly set valueCount).
                    if(typeInfo.valueCount > 1)
                    {
                        est.registerCount = (typeInfo.valueCount * regPerValue) / packing;
                        est.resolved      = true;
                        return est;
                    }
                    // Cannot determine per-work-item register count.
                    return est;
                }
                // Non-MacroTile: 1 logical value per entity.
                // (Per-work-item for VGPR, per-wave for SGPR.)
                est.registerCount = regPerValue;
                est.resolved      = true;
                return est;
            }
            // ================================================================
            // Top-level: estimateRegisterUsage
            // ================================================================
            //
            // For every control operation in the kernel graph, estimates the
            // number of live registers classified by Register::Type.
            //
            // Steps:
            //   1. Trace all (control, coordinate, readwrite) records.
            //   2. Build liveness extents for each coordinate.
            //   3. Resolve types for all coordinates (P1-P4).
            //   4. Compute per-work-item register estimate per coordinate.
            //   5. Identify aliased-inner coordinates (share registers
            //      with alias target; must not double-count).
            //   6. For each operation:
            //      a. Classify each coordinate as Dead/Alive/MaybeAlive.
            //      b. For Alive and MaybeAlive (excluding aliased inners),
            //         sum per-work-item register counts by Register::Type.
            //
            // Output interpretation:
            //   regCountByType[Vector]      -> per-work-item VGPR count
            //   regCountByType[Accumulator] -> per-work-item AGPR count
            //   regCountByType[Scalar]      -> per-wave SGPR count
            std::map<int, OperationRegisterUsage> estimateRegisterUsage(KernelGraph const& kgraph,
                                                                        int wavefrontSize)
            {
                // Step 1: Trace all coordinate read/write records.
                ControlFlowRWTracer                tracer(kgraph);
                auto                               allRecords = tracer.coordinatesReadWrite();
                std::map<int, std::vector<Record>> recordsByCoord;
                for(auto const& rec : allRecords)
                    recordsByCoord[rec.coordinate].push_back(rec);
                // Step 2: Build liveness extents for every coordinate.
                std::map<int, CoordExtent> allExtents;
                for(auto& [coordTag, records] : recordsByCoord)
                    allExtents[coordTag] = getCoordExtent(kgraph, coordTag, records);
                // Step 3: Resolve types for all coordinates.
                auto allTypes = resolveAllCoordTypes(kgraph, recordsByCoord);

                std::unordered_set<int> coordNoRegister;
                for(auto& [coordTag, records] : recordsByCoord)
                {
                    auto const result = classifyCoordinate(kgraph, coordTag, allRecords);
                    if(result == CoordRegAllocation::NoRegisters)
                    {
                        coordNoRegister.insert(coordTag);
                    }
                    if(coordTag == 314)
                    {
                        //if(result == CoordRegAllocation::NoRegisters)
                        //    Log::info("no reg");
                        //if(result == CoordRegAllocation::ExpressionOnly)
                        //    Log::info("expr only");
                        //if(result == CoordRegAllocation::PhysicalRegister)
                        //    Log::info("phy");
                        //exit(0);
                    }
                }

                // Step 4: Per-work-item register estimates.
                std::map<int, CoordRegisterEstimate> regEstimates;
                for(auto const& [coordTag, typeInfo] : allTypes)
                    regEstimates[coordTag]
                        = estimateCoordRegisterCount(kgraph, coordTag, typeInfo, wavefrontSize);

                // Step 5: Identify aliased-inner coordinates.
                //
                // An Alias edge goes from inner (source) to outer (target).
                // The inner borrows the outer's register allocation,
                // so the inner must not be counted separately.
                std::set<int> aliasedInners;
                {
                    auto isAlias = [&kgraph](int idx) {
                        auto edge = kgraph.coordinates.getEdge(idx);
                        if(!std::holds_alternative<CT::DataFlowEdge>(edge))
                            return false;
                        return std::holds_alternative<CT::Alias>(std::get<CT::DataFlowEdge>(edge));
                    };
                    for(auto edgeTag : kgraph.coordinates.getEdges().filter(isAlias))
                    {
                        auto loc = kgraph.coordinates.getLocation(edgeTag);
                        for(int src : loc.incoming)
                            aliasedInners.insert(src);
                    }
                }
                // Step 6: Per-operation register usage aggregation.
                std::map<int, OperationRegisterUsage>             result;
                std::unordered_map<int, std::vector<CoordExtent>> controlExtents;
                for(auto opTag : kgraph.control.getNodes())
                {
                    OperationRegisterUsage usage;
                    usage.operationTag = opTag;
                    for(auto const& [coordTag, ext] : allExtents)
                    {
                        // Skip aliased-inner coordinates.
                        if(aliasedInners.contains(coordTag))
                            continue;

                        // Skip if no register for this coordinate (e.g., constant)
                        if(coordNoRegister.contains(coordTag))
                            continue;

                        auto liveness = getCoordLiveness(kgraph, opTag, ext);
                        // Dead coordinates don't use registers at this point.
                        if(liveness == CoordLiveness::Dead or liveness == CoordLiveness::MaybeAlive)
                            continue;

                        // Look up the pre-computed register estimate.
                        auto estIt = regEstimates.find(coordTag);
                        if(estIt == regEstimates.end() || !estIt->second.resolved)
                            continue;

                        int regs = estIt->second.registerCount;
                        if(regs <= 0)
                            continue;

                        // Determine which Register::Type bucket this belongs to.
                        // Default to Vector if regType is unknown.
                        Register::Type rType = Register::Type::Vector;
                        auto           tIt   = allTypes.find(coordTag);
                        if(tIt != allTypes.end() && tIt->second.resolved
                           && tIt->second.regType != Register::Type::Count)
                        {
                            rType = tIt->second.regType;
                        }

                        usage.regCountByType[rType] += regs;
                        usage.totalRegisters += regs;
                        controlExtents[opTag].push_back(ext);

                        usage.overlappedCoords.insert({coordTag, ext});
                    }
                    result[opTag] = std::move(usage);
                }
                //exit(0);

                // --- 1. ASCII text: control graph liveness ---
                std::string controlText
                    = dumpControlGraphLiveness(kgraph, allExtents, allTypes, regEstimates, result);
                Log::info("Control Graph Liveness:\n{}", controlText);
                // --- 2. ASCII text: coordinate access map ---
                std::string coordText = dumpCoordinateAccessMap(
                    kgraph, allExtents, allTypes, regEstimates, recordsByCoord);
                Log::info("Coordinate Access Map:\n{}", coordText);
                // // --- 3. DOT: control graph liveness ---
                // {
                //     std::ofstream dot("control_liveness.dot");
                //     exportControlGraphLivenessDot(
                //         dot, kgraph, allExtents, allTypes, regEstimates, result);
                // }
                // // Render: python3 scripts/render_register_usage.py control_liveness.dot
                // // --- 4. DOT: coordinate access map ---
                // {
                //     std::ofstream dot("coord_access.dot");
                //     exportCoordinateAccessDot(
                //         dot, kgraph, allExtents, allTypes, regEstimates, recordsByCoord);
                // }

                //exit(0);
                return result;
            }

        } // namespace RegisterUsageAnalysis

        KernelGraph RegisterUsageAnalysis::apply(KernelGraph const& original)
        {
            auto rv = original;

            using namespace RegisterUsageAnalysis_Detail;

            auto usage = estimateRegisterUsage(rv, 64);
            for(auto const& [op, u] : usage)
            {
                Log::info("op = {} ({})", toString(rv.control.getNode(op)), op);
                for(auto const& [type, count] : u.regCountByType)
                {
                    Log::info("{}  ->  {}", toString(type), count);
                }
                Log::info("");
            }

            return rv;
        }

    }
}
