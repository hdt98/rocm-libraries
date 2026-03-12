// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iomanip>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>
namespace rocRoller::KernelGraph::ControlGraph
{
    //=========================================================================
    // Anonymous helpers (file-local, used only by cache build/read methods).
    //=========================================================================
    namespace
    {
        // Edge variant indices matching ControlEdge = variant<Sequence, Initialize, ForLoopIncrement, Body, Else>.
        constexpr uint8_t kSeq  = 0;
        constexpr uint8_t kInit = 1;
        constexpr uint8_t kInc  = 2;
        constexpr uint8_t kBody = 3;
        constexpr uint8_t kElse = 4;
        // How many uint64_t words are needed to hold `bits` bits.
        inline size_t wordsForBits(size_t bits)
        {
            return (bits + 63) >> 6;
        }
        // Pointer to the beginning of row `r` in a row-major bit matrix.
        inline uint64_t* matRow(std::vector<uint64_t>& mat, size_t words, int r)
        {
            return mat.data() + static_cast<size_t>(r) * words;
        }
        inline uint64_t const* matRow(std::vector<uint64_t> const& mat, size_t words, int r)
        {
            return mat.data() + static_cast<size_t>(r) * words;
        }
        // Bitwise OR src into dst (both length `words`).
        inline void rowOr(uint64_t* dst, uint64_t const* src, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                dst[i] |= src[i];
        }
        // True if any bit is set.
        inline bool rowAny(uint64_t const* row, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                if(row[i])
                    return true;
            return false;
        }
        // True if a[i] & b[i] is nonzero for any word.
        inline bool rowOverlap(uint64_t const* a, uint64_t const* b, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                if(a[i] & b[i])
                    return true;
            return false;
        }
        // Call fn(denseIndex) for every set bit in a fixed-size Bitset.
        template <typename Fn>
        inline void forEachBit(Bitset const& bs, Fn&& fn)
        {
            for(size_t w = 0; w < bs.size(); ++w)
            {
                uint64_t bits = bs[w];
                while(bits)
                {
                    int bit = std::countr_zero(bits);
                    fn(static_cast<int>(w * 64 + bit));
                    bits &= bits - 1;
                }
            }
        }
        // Call fn(denseIndex) for every set bit in a raw row pointer.
        template <typename Fn>
        inline void forEachBit(uint64_t const* row, size_t words, Fn&& fn)
        {
            for(size_t w = 0; w < words; ++w)
            {
                uint64_t bits = row[w];
                while(bits)
                {
                    int bit = std::countr_zero(bits);
                    fn(static_cast<int>(w * 64 + bit));
                    bits &= bits - 1;
                }
            }
        }
        // Set/clear a single bit in a fixed-size Bitset by dense index.
        inline void dSet(Bitset& bs, int d)
        {
            bs[static_cast<size_t>(d) >> 6] |= uint64_t(1) << (d & 63);
        }
        inline void dClear(Bitset& bs, int d)
        {
            bs[static_cast<size_t>(d) >> 6] &= ~(uint64_t(1) << (d & 63));
        }
        inline void dOr(Bitset& dst, Bitset const& src)
        {
            for(size_t i = 0; i < dst.size(); ++i)
                dst[i] |= src[i];
        }
        // dst &= ~mask (clear bits present in mask).
        inline void dAndNot(Bitset& dst, Bitset const& mask)
        {
            for(size_t i = 0; i < dst.size(); ++i)
                dst[i] &= ~mask[i];
        }
    } // anonymous namespace
    //=========================================================================
    // denseOfTag / selectCacheMatrix
    //=========================================================================
    int ControlGraph::denseOfTag(int tag) const
    {
        auto it = m_cacheTagToDense.find(tag);
        if(it == m_cacheTagToDense.end())
            return -1;
        return it->second;
    }
    std::vector<uint64_t>& ControlGraph::selectCacheMatrix(NodeOrdering order) const
    {
        switch(order)
        {
        case NodeOrdering::LeftFirst:
            return m_cacheAfter;
        case NodeOrdering::RightFirst:
            return m_cacheBefore;
        case NodeOrdering::RightInBodyOfLeft:
            return m_cacheInBody;
        case NodeOrdering::LeftInBodyOfRight:
            return m_cacheContaining;
        default:
            break;
        }
        AssertFatal(false, "Invalid NodeOrdering", ShowValue(order));
        return m_cacheAfter;
    }
    //=========================================================================
    // nodeOrderTable / nodeOrderTableString
    //=========================================================================
    std::unordered_map<int, std::unordered_map<int, NodeOrdering>>
        ControlGraph::nodeOrderTable() const
    {
        populateOrderCache();
        std::unordered_map<int, std::unordered_map<int, NodeOrdering>> table;
        int const n = static_cast<int>(m_cacheDenseToTag.size());
        for(int aDense = 0; aDense < n; ++aDense)
        {
            int const aTag = m_cacheDenseToTag[aDense];
            forEachBit(matRow(m_cacheAfter, m_cacheWords, aDense), m_cacheWords, [&](int bDense) {
                if(bDense >= n)
                    return;
                int const bTag = m_cacheDenseToTag[bDense];
                if(aTag < bTag)
                    table[aTag][bTag] = NodeOrdering::LeftFirst;
            });
            forEachBit(matRow(m_cacheInBody, m_cacheWords, aDense), m_cacheWords, [&](int bDense) {
                if(bDense >= n)
                    return;
                int const bTag = m_cacheDenseToTag[bDense];
                if(aTag < bTag)
                    table[aTag][bTag] = NodeOrdering::RightInBodyOfLeft;
                else if(bTag < aTag)
                    table[bTag][aTag] = NodeOrdering::LeftInBodyOfRight;
            });
        }
        return table;
    }
    std::string ControlGraph::nodeOrderTableString(std::set<int> const& nodes) const
    {
        populateOrderCache();
        if(nodes.empty())
        {
            return "Empty order cache.\n";
        }
        std::ostringstream msg;
        int                width = std::ceil(std::log10(static_cast<float>(*nodes.rbegin())));
        width                    = std::max(width, 3);
        msg << std::setw(width) << " "
            << "\\";
        for(int n : nodes)
            msg << " " << std::setw(width) << n;
        for(int i : nodes)
        {
            msg << std::endl << std::setw(width) << i << "|";
            for(int j : nodes)
            {
                if(i == j)
                {
                    msg << " ";
                    auto oldFill = msg.fill('-');
                    msg << std::setw(width) << "-";
                    msg.fill(oldFill);
                }
                else
                {
                    msg << " " << std::setw(width) << abbrev(lookupOrder(CacheOnly, i, j));
                }
            }
            msg << " | " << std::setw(width) << i;
        }
        msg << std::endl
            << std::setw(width) << " "
            << "|";
        for(int n : nodes)
            msg << " " << std::setw(width) << n;
        msg << std::endl;
        return msg.str();
    }
    std::string ControlGraph::nodeOrderTableString() const
    {
        populateOrderCache();
        TIMER(t, "nodeOrderTable");
        std::set<int> nodes;
        int const     n = static_cast<int>(m_cacheDenseToTag.size());
        for(int d = 0; d < n; ++d)
        {
            int const   tag  = m_cacheDenseToTag[d];
            auto const* aRow = matRow(m_cacheAfter, m_cacheWords, d);
            auto const* bRow = matRow(m_cacheBefore, m_cacheWords, d);
            auto const* iRow = matRow(m_cacheInBody, m_cacheWords, d);
            auto const* cRow = matRow(m_cacheContaining, m_cacheWords, d);
            if(rowAny(aRow, m_cacheWords) || rowAny(bRow, m_cacheWords)
               || rowAny(iRow, m_cacheWords) || rowAny(cRow, m_cacheWords))
            {
                nodes.insert(tag);
            }
            auto addNodes = [&](uint64_t const* row) {
                forEachBit(row, m_cacheWords, [&](int otherDense) {
                    if(otherDense < n)
                        nodes.insert(m_cacheDenseToTag[otherDense]);
                });
            };
            addNodes(aRow);
            addNodes(bRow);
            addNodes(iRow);
            addNodes(cRow);
        }
        return nodeOrderTableString(nodes);
    }
    //=========================================================================
    // clearCache
    //=========================================================================
    void ControlGraph::clearCache(Graph::GraphModification modification)
    {
        Hypergraph<Operation, ControlEdge, false>::clearCache(modification);
        if(modification == Graph::GraphModification::AddElement
           && m_cacheStatus != CacheStatus::Invalid)
        {
            m_cacheStatus = CacheStatus::Partial;
        }
        else
        {
            m_cacheStatus = CacheStatus::Invalid;
        }
        m_cacheDenseToTag.clear();
        m_cacheTagToDense.clear();
        m_cacheWords = 0;
        m_cacheAfter.clear();
        m_cacheBefore.clear();
        m_cacheInBody.clear();
        m_cacheContaining.clear();
    }
    //=========================================================================
    // isModificationAllowed (unchanged)
    //=========================================================================
    bool ControlGraph::isModificationAllowed(int index) const
    {
        if(not Settings::getInstance()->get(Settings::EnforceGraphConstraints))
            return true;
        if(not m_changesRestricted)
            return true;
        auto const& el = getElement(index);
        if(std::holds_alternative<Operation>(el))
        {
            return std::visit(
                [](auto&& arg) {
                    using OpType = std::decay_t<decltype(arg)>;
                    return !(
                        std::is_same_v<OpType, ForLoopOp> or std::is_same_v<OpType, SetCoordinate>);
                },
                std::get<Operation>(el));
        }
        else
        {
            return true;
        }
    }
    //=========================================================================
    // validateOrderCache
    //=========================================================================
    void ControlGraph::validateOrderCache() const
    {
        int const n = static_cast<int>(m_cacheDenseToTag.size());
        for(int r = 0; r < n; ++r)
        {
            auto const* aRow = matRow(m_cacheAfter, m_cacheWords, r);
            auto const* bRow = matRow(m_cacheBefore, m_cacheWords, r);
            auto const* iRow = matRow(m_cacheInBody, m_cacheWords, r);
            auto const* cRow = matRow(m_cacheContaining, m_cacheWords, r);
            AssertFatal(!rowOverlap(aRow, bRow, m_cacheWords),
                        "Node has conflicting orders (after & before)");
            AssertFatal(!rowOverlap(aRow, iRow, m_cacheWords),
                        "Node has conflicting orders (after & inBody)");
            AssertFatal(!rowOverlap(aRow, cRow, m_cacheWords),
                        "Node has conflicting orders (after & containing)");
            AssertFatal(!rowOverlap(bRow, iRow, m_cacheWords),
                        "Node has conflicting orders (before & inBody)");
            AssertFatal(!rowOverlap(bRow, cRow, m_cacheWords),
                        "Node has conflicting orders (before & containing)");
            AssertFatal(!rowOverlap(iRow, cRow, m_cacheWords),
                        "Node has conflicting orders (inBody & containing)");
        }
    }
    //=========================================================================
    // populateOrderCache  (ancestor-upward algorithm -- my idea)
    //=========================================================================
    void ControlGraph::populateOrderCache() const
    {
        TIMER(t, "populateOrderCache");
        if(m_cacheStatus == CacheStatus::Valid)
            return;
        // ---- Reset all cache storage ----
        m_cacheDenseToTag.clear();
        m_cacheTagToDense.clear();
        m_cacheWords = 0;
        m_cacheAfter.clear();
        m_cacheBefore.clear();
        m_cacheInBody.clear();
        m_cacheContaining.clear();
        static_assert(std::variant_size_v<ControlEdge> == 5,
                      "Assumes Sequence(0), Initialize(1), ForLoopIncrement(2), Body(3), Else(4).");
        using GD = Graph::Direction;
        // Collect all node tags.
        auto nodeTags = getNodes().to<std::vector>();
        if(nodeTags.empty())
        {
            m_cacheStatus = CacheStatus::Valid;
            return;
        }
        int const nodeCount = static_cast<int>(nodeTags.size());
        // ---- Build dense <-> tag mapping ----
        // Dense indices are [0, nodeCount). All hot-path operations use dense indices.
        // The single unordered_map is only touched at boundary (build + query entry).
        m_cacheDenseToTag = nodeTags;
        m_cacheTagToDense.reserve(nodeCount * 2);
        for(int d = 0; d < nodeCount; ++d)
            m_cacheTagToDense.emplace(m_cacheDenseToTag[d], d);
        // ---- Allocate row-major bit matrices ----
        // Each matrix has nodeCount rows, each row has m_cacheWords uint64_t words.
        // Bit position is a dense node index, NOT a graph tag.
        m_cacheWords             = wordsForBits(static_cast<size_t>(nodeCount));
        size_t const matrixWords = static_cast<size_t>(nodeCount) * m_cacheWords;
        m_cacheAfter.assign(matrixWords, 0);
        m_cacheBefore.assign(matrixWords, 0);
        m_cacheInBody.assign(matrixWords, 0);
        m_cacheContaining.assign(matrixWords, 0);
        // ---- Build node-level adjacency in dense space ----
        // childrenByType[u][edgeType] = list of direct child dense indices via that edge type.
        // parents[v] = list of (parent_dense, edgeType) pairs.
        std::vector<std::array<std::vector<int>, 5>>      childrenByType(nodeCount);
        std::vector<std::vector<std::pair<int, uint8_t>>> parents(nodeCount);
        for(int u = 0; u < nodeCount; ++u)
        {
            int const parentTag = m_cacheDenseToTag[u];
            for(int edgeTag : getNeighbours<GD::Downstream>(parentTag))
            {
                uint8_t const edgeType = static_cast<uint8_t>(getEdge(edgeTag).index());
                for(int childTag : getNeighbours<GD::Downstream>(edgeTag))
                {
                    int const v = denseOfTag(childTag);
                    if(v < 0)
                        continue;
                    childrenByType[u][edgeType].push_back(v);
                    parents[v].emplace_back(u, edgeType);
                }
            }
        }
        // Dedup adjacency (a node can be reached via multiple edges of same type).
        std::vector<int> indegree(nodeCount, 0);
        for(int u = 0; u < nodeCount; ++u)
        {
            for(auto& bucket : childrenByType[u])
            {
                std::sort(bucket.begin(), bucket.end());
                bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
                for(int v : bucket)
                    ++indegree[v];
            }
            auto& p = parents[u];
            std::sort(p.begin(), p.end());
            p.erase(std::unique(p.begin(), p.end()), p.end());
        }
        // ---- Topological order (Kahn's algorithm) ----
        std::vector<int> topoQueue;
        topoQueue.reserve(nodeCount);
        for(int i = 0; i < nodeCount; ++i)
            if(indegree[i] == 0)
                topoQueue.push_back(i);
        std::vector<int> topo;
        topo.reserve(nodeCount);
        for(size_t head = 0; head < topoQueue.size(); ++head)
        {
            int const u = topoQueue[head];
            topo.push_back(u);
            for(auto const& bucket : childrenByType[u])
                for(int v : bucket)
                    if(--indegree[v] == 0)
                        topoQueue.push_back(v);
        }
        AssertFatal(static_cast<int>(topo.size()) == nodeCount,
                    "ControlGraph must be acyclic for order-cache construction.");
        // ---- Phase 1: Bottom-up DP to precompute "before" sets per ancestor ----
        //
        // For each node u we compute:
        //   allDesc[u]  = all descendants of u (dense bitset)
        //   beforeByArrival[u].viaBody = descendants reachable via Init edges from u
        //   beforeByArrival[u].viaElse = Init | Body descendants
        //   beforeByArrival[u].viaInc  = Init | Body | Else descendants
        //   beforeByArrival[u].viaSeq  = Init | Body | Else | Inc descendants
        //
        // These represent: if a target node arrives at ancestor u via a given
        // edge type, which of u's descendants are known to be before the target?
        //
        // Edge precedence: Initialize -> Body -> Else -> ForLoopIncrement -> Sequence.
        struct BeforeByArrival
        {
            Bitset viaBody; // descendants before if target arrives via Body
            Bitset viaElse; // descendants before if target arrives via Else
            Bitset viaInc; // descendants before if target arrives via Inc
            Bitset viaSeq; // descendants before if target arrives via Sequence
        };
        std::vector<BeforeByArrival> beforeByArrival(nodeCount);
        for(auto& s : beforeByArrival)
        {
            s.viaBody.assign(m_cacheWords, 0);
            s.viaElse.assign(m_cacheWords, 0);
            s.viaInc.assign(m_cacheWords, 0);
            s.viaSeq.assign(m_cacheWords, 0);
        }
        std::vector<Bitset> allDesc(nodeCount, Bitset(m_cacheWords, 0));
        // Reused temporaries (avoid per-node allocation).
        std::array<Bitset, 5> typed;
        for(auto& bs : typed)
            bs.assign(m_cacheWords, 0);
        Bitset seen(m_cacheWords, 0);
        // Process in reverse topological order so children are done before parents.
        for(auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            int const u = *it;
            // Zero out temporaries.
            for(auto& bs : typed)
                std::fill(bs.begin(), bs.end(), 0);
            std::fill(seen.begin(), seen.end(), 0);
            // typed[t] = descendants whose first edge from u is type t.
            for(uint8_t t = 0; t < 5; ++t)
            {
                for(int c : childrenByType[u][t])
                {
                    dSet(typed[t], c); // direct child
                    dOr(typed[t], allDesc[c]); // all of child's descendants
                }
            }
            // Make partitions disjoint by precedence to prevent A-before-B AND B-before-A.
            // If a descendant is reachable via multiple edge types, keep the earliest one.
            seen = typed[kInit];
            dAndNot(typed[kBody], seen);
            dOr(seen, typed[kBody]);
            dAndNot(typed[kElse], seen);
            dOr(seen, typed[kElse]);
            dAndNot(typed[kInc], seen);
            dOr(seen, typed[kInc]);
            dAndNot(typed[kSeq], seen);
            // Build cumulative "before" sets.
            // viaBody: if target arrives via Body, Init descendants are before it.
            beforeByArrival[u].viaBody = typed[kInit];
            // viaElse: Init | Body descendants are before it.
            beforeByArrival[u].viaElse = beforeByArrival[u].viaBody;
            dOr(beforeByArrival[u].viaElse, typed[kBody]);
            // viaInc: Init | Body | Else descendants are before it.
            beforeByArrival[u].viaInc = beforeByArrival[u].viaElse;
            dOr(beforeByArrival[u].viaInc, typed[kElse]);
            // viaSeq: Init | Body | Else | Inc descendants are before it.
            beforeByArrival[u].viaSeq = beforeByArrival[u].viaInc;
            dOr(beforeByArrival[u].viaSeq, typed[kInc]);
            // All descendants = viaSeq set | Sequence descendants.
            allDesc[u] = beforeByArrival[u].viaSeq;
            dOr(allDesc[u], typed[kSeq]);
            dClear(allDesc[u], u); // node is not its own descendant
        }
        // ---- Phase 2: Per-target upward ancestor traversal ----
        //
        // My idea:
        // For each target node, climb upward through all ancestors.
        // At each ancestor, based on the edge type connecting the ancestor
        // to the path towards the target:
        //   - If edge is Sequence: ancestor is before target (LeftFirst).
        //   - If edge is Init/Body/Else/Inc: target is in ancestor's body (RightInBodyOfLeft).
        //   - Also, ancestor's descendants that come before the target (from Phase 1)
        //     are all before the target (LeftFirst).
        //
        // We accumulate two dense bitsets per target:
        //   rowsAfter     = nodes that should be "before target" (target goes in their .after)
        //   rowsContaining = nodes that should "contain target" (target goes in their .inBody)
        //
        // Then we do two many-to-one writes per target, which is efficient:
        // one bit set per source row + one bulk OR on target's row.
        // Dedup state: stamp per node, plus the best (earliest precedence) edge type seen.
        // "Best" means the numerically smallest index in the precedence order
        // {Init=1, Body=3, Else=4, Inc=2, Seq=0}, but we remap to a precedence rank
        // so that comparison is just <.
        // Precedence rank: Init(0) < Body(1) < Else(2) < Inc(3) < Seq(4).
        auto precedenceRank = [](uint8_t edgeType) -> uint8_t {
            switch(edgeType)
            {
            case kInit:
                return 0;
            case kBody:
                return 1;
            case kElse:
                return 2;
            case kInc:
                return 3;
            case kSeq:
                return 4;
            default:
                return 5;
            }
        };
        std::vector<uint32_t> visitStamp(nodeCount, 0);
        std::vector<uint8_t>  bestEdgeType(nodeCount, 0xFF);
        uint32_t              stamp = 0;
        // Track which ancestors were reached for this target (to avoid scanning all nodes).
        std::vector<int> reachedAncestors;
        reachedAncestors.reserve(64);
        // DFS stack for upward traversal.
        std::vector<int> stack;
        stack.reserve(64);
        Bitset rowsAfter(m_cacheWords, 0);
        Bitset rowsContaining(m_cacheWords, 0);
        for(int target = 0; target < nodeCount; ++target)
        {
            // ---- Upward BFS/DFS from target to collect all ancestors ----
            // For each ancestor, record the "best" (earliest-precedence) edge type
            // through which the target is reachable from it.
            if(++stamp == 0)
            {
                // Handle extremely unlikely uint32 wrap.
                std::fill(visitStamp.begin(), visitStamp.end(), 0);
                stamp = 1;
            }
            reachedAncestors.clear();
            stack.clear();
            // Seed: target's direct parents.
            for(auto const& [p, eType] : parents[target])
            {
                if(visitStamp[p] != stamp)
                {
                    visitStamp[p]   = stamp;
                    bestEdgeType[p] = eType;
                    reachedAncestors.push_back(p);
                    stack.push_back(p);
                }
                else if(precedenceRank(eType) < precedenceRank(bestEdgeType[p]))
                {
                    bestEdgeType[p] = eType;
                }
            }
            // Climb upward: each ancestor's parents inherit "best reachable edge type"
            // propagated through the path. An ancestor's own edge type toward its child
            // is what matters (not the child->grandchild type), because the ordering rule
            // depends on which bucket of the ancestor the target falls into.
            while(!stack.empty())
            {
                int const node = stack.back();
                stack.pop_back();
                for(auto const& [p, eType] : parents[node])
                {
                    if(visitStamp[p] != stamp)
                    {
                        visitStamp[p]   = stamp;
                        bestEdgeType[p] = eType;
                        reachedAncestors.push_back(p);
                        stack.push_back(p);
                    }
                    else if(precedenceRank(eType) < precedenceRank(bestEdgeType[p]))
                    {
                        bestEdgeType[p] = eType;
                    }
                }
            }
            // ---- Gather relation sets from ancestors ----
            std::fill(rowsAfter.begin(), rowsAfter.end(), 0);
            std::fill(rowsContaining.begin(), rowsContaining.end(), 0);
            for(int anc : reachedAncestors)
            {
                uint8_t const viaType = bestEdgeType[anc];
                // Direct ancestor relation.
                if(viaType == kSeq)
                    dSet(rowsAfter, anc); // ancestor is before target
                else
                    dSet(rowsContaining, anc); // target is in ancestor's body
                // Descendants of ancestor that are before target by edge precedence.
                switch(viaType)
                {
                case kInit:
                    // Nothing is before Init descendants.
                    break;
                case kBody:
                    dOr(rowsAfter, beforeByArrival[anc].viaBody);
                    break;
                case kElse:
                    dOr(rowsAfter, beforeByArrival[anc].viaElse);
                    break;
                case kInc:
                    dOr(rowsAfter, beforeByArrival[anc].viaInc);
                    break;
                case kSeq:
                    dOr(rowsAfter, beforeByArrival[anc].viaSeq);
                    break;
                default:
                    AssertFatal(false, "Unexpected edge type", ShowValue(viaType));
                }
            }
            // Target should not be before/contain itself.
            dClear(rowsAfter, target);
            dClear(rowsContaining, target);
            // ---- Write into cache matrices ----
            // Many-to-one pattern:
            //   For each source in rowsAfter: set bit `target` in source's After row.
            //   Then OR rowsAfter into target's Before row (the opposite relation).
            //   Same pattern for rowsContaining -> InBody/Containing.
            size_t const   colWord = static_cast<size_t>(target) >> 6;
            uint64_t const colMask = uint64_t(1) << (target & 63);
            // LeftFirst: source.after[target]=1, target.before |= sources
            forEachBit(rowsAfter, [&](int srcDense) {
                matRow(m_cacheAfter, m_cacheWords, srcDense)[colWord] |= colMask;
            });
            rowOr(matRow(m_cacheBefore, m_cacheWords, target), rowsAfter.data(), m_cacheWords);
            // RightInBodyOfLeft: source.inBody[target]=1, target.containing |= sources
            forEachBit(rowsContaining, [&](int srcDense) {
                matRow(m_cacheInBody, m_cacheWords, srcDense)[colWord] |= colMask;
            });
            rowOr(matRow(m_cacheContaining, m_cacheWords, target),
                  rowsContaining.data(),
                  m_cacheWords);
        }
        validateOrderCache();
        m_cacheStatus = CacheStatus::Valid;
    }
    //=========================================================================
    // lookupOrder(IgnoreCache) -- unchanged algorithm, no cache involved
    //=========================================================================
    NodeOrdering ControlGraph::lookupOrder(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        TIMER(t, "ControlGraph::lookupOrder");
        using GD = Graph::Direction;
        std::unordered_set<int> visited_nodes;
        auto const              getOrderIfParent = [&](int edge) {
            return std::holds_alternative<Sequence>(getEdge(edge))
                                    ? NodeOrdering::LeftFirst
                                    : NodeOrdering::RightInBodyOfLeft;
        };
        auto const getOrderOfDescendants = [&](int edgeA, int edgeB) {
            auto edgeAElem = getEdge(edgeA);
            auto edgeBElem = getEdge(edgeB);
            AssertFatal(edgeAElem.index() != edgeBElem.index(),
                        "edgeA and edgeB types should not be the same");
            return std::visit(
                rocRoller::overloaded{
                    [&](auto const&) {
                        AssertFatal(false, "Unhandled edge type in getOrderOfDescendants");
                        return NodeOrdering::Undefined;
                    },
                    [&](Body const&) {
                        return std::holds_alternative<Initialize>(edgeBElem)
                                   ? opposite(NodeOrdering::LeftFirst)
                                   : NodeOrdering::LeftFirst;
                    },
                    [&](Sequence const&) { return opposite(NodeOrdering::LeftFirst); },
                    [&](Initialize const&) { return NodeOrdering::LeftFirst; },
                    [&](ForLoopIncrement const&) {
                        return std::holds_alternative<Sequence>(edgeBElem)
                                   ? NodeOrdering::LeftFirst
                                   : opposite(NodeOrdering::LeftFirst);
                    },
                    [&](Else const&) {
                        return std::holds_alternative<Initialize>(edgeBElem)
                                       || std::holds_alternative<Body>(edgeBElem)
                                   ? opposite(NodeOrdering::LeftFirst)
                                   : NodeOrdering::LeftFirst;
                    }},
                edgeAElem);
        };
        std::unordered_map<int, int> A_ancestors;
        std::vector<int>             stk{nodeA};
        while(!stk.empty())
        {
            auto node = stk.back();
            stk.pop_back();
            for(auto edge : getNeighbours<GD::Upstream>(node))
            {
                for(auto parent : getNeighbours<GD::Upstream>(edge))
                {
                    if(parent == nodeB)
                        return opposite(getOrderIfParent(edge));
                    A_ancestors.insert({parent, edge});
                    if(!visited_nodes.contains(parent))
                    {
                        visited_nodes.insert(parent);
                        stk.push_back(parent);
                    }
                }
            }
        }
        AssertFatal(stk.empty());
        stk.push_back(nodeB);
        visited_nodes.clear();
        while(!stk.empty())
        {
            auto node = stk.back();
            stk.pop_back();
            for(auto edge : getNeighbours<GD::Upstream>(node))
            {
                for(auto parent : getNeighbours<GD::Upstream>(edge))
                {
                    if(parent == nodeA)
                        return getOrderIfParent(edge);
                    if(A_ancestors.contains(parent))
                    {
                        if(getEdge(A_ancestors.at(parent)).index() != getEdge(edge).index())
                        {
                            return getOrderOfDescendants(A_ancestors.at(parent), edge);
                        }
                    }
                    if(!visited_nodes.contains(parent))
                    {
                        visited_nodes.insert(parent);
                        stk.push_back(parent);
                    }
                }
            }
        }
        return NodeOrdering::Undefined;
    }
    //=========================================================================
    // compareNodes variants
    //=========================================================================
    static void validateNodes(ControlGraph const& control, int nodeA, int nodeB)
    {
        AssertFatal(nodeA != nodeB, ShowValue(nodeA));
        AssertFatal(control.getElementType(nodeA) == Graph::ElementType::Node
                        && control.getElementType(nodeB) == Graph::ElementType::Node,
                    ShowValue(control.getElementType(nodeA)),
                    ShowValue(control.getElementType(nodeB)));
    }
    NodeOrdering ControlGraph::compareNodes(CacheOnlyPolicy const, int nodeA, int nodeB) const
    {
        AssertFatal(m_cacheStatus == CacheStatus::Valid);
        validateNodes(*this, nodeA, nodeB);
        return lookupOrder(CacheOnly, nodeA, nodeB);
    }
    NodeOrdering ControlGraph::compareNodes(UpdateCachePolicy const, int nodeA, int nodeB) const
    {
        if(m_cacheStatus != CacheStatus::Valid)
            populateOrderCache();
        return compareNodes(CacheOnly, nodeA, nodeB);
    }
    NodeOrdering
        ControlGraph::compareNodes(UseCacheIfAvailablePolicy const, int nodeA, int nodeB) const
    {
        validateNodes(*this, nodeA, nodeB);
        return m_cacheStatus == CacheStatus::Valid ? compareNodes(CacheOnly, nodeA, nodeB)
                                                   : compareNodes(IgnoreCache, nodeA, nodeB);
    }
    NodeOrdering ControlGraph::compareNodes(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        validateNodes(*this, nodeA, nodeB);
        return lookupOrder(IgnoreCache, nodeA, nodeB);
    }
}
