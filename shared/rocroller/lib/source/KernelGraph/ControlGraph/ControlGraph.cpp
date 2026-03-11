// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <bitset>
#include <cmath>
#include <iomanip>

namespace rocRoller::KernelGraph::ControlGraph
{
    std::unordered_map<int, std::unordered_map<int, NodeOrdering>>
        ControlGraph::nodeOrderTable() const
    {
        populateOrderCache();
        std::unordered_map<int, std::unordered_map<int, NodeOrdering>> table;
        for(auto const& [node, orders] : m_orderCache)
        {
            for(int other : bitsetToSortedVec(orders.after))
            {
                if(node < other)
                    table[node][other] = NodeOrdering::LeftFirst;
            }
            for(int other : bitsetToSortedVec(orders.inBody))
            {
                if(node < other)
                    table[node][other] = NodeOrdering::RightInBodyOfLeft;
                else
                    table[other][node] = NodeOrdering::LeftInBodyOfRight;
            }
        }
        return table;

        //populateOrderCache();

        //std::unordered_map<int, std::unordered_map<int, NodeOrdering>> table;
        //for(auto const& [node, orders] : m_orderCache)
        //{
        //    for(int other : orders.after)
        //    {
        //        if(node < other)
        //            table[node][other] = NodeOrdering::LeftFirst;
        //    }
        //    for(int other : orders.inBody)
        //    {
        //        if(node < other)
        //            table[node][other] = NodeOrdering::RightInBodyOfLeft;
        //        else
        //            table[other][node] = NodeOrdering::LeftInBodyOfRight;
        //    }
        //}
        //return table;
    }

    std::string ControlGraph::nodeOrderTableString(std::set<int> const& nodes) const
    {
        populateOrderCache();
        if(nodes.empty())
        {
            return "Empty order cache.\n";
        }

        std::ostringstream msg;

        int width = std::ceil(std::log10(static_cast<float>(*nodes.rbegin())));
        width     = std::max(width, 3);

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
        for(auto const& [node, orders] : m_orderCache)
        {
            if(bitsetAny(orders.after) || bitsetAny(orders.before) || bitsetAny(orders.inBody)
               || bitsetAny(orders.containing))
            {
                nodes.insert(node);
            }
            for(int n : bitsetToSortedVec(orders.after))
                nodes.insert(n);
            for(int n : bitsetToSortedVec(orders.before))
                nodes.insert(n);
            for(int n : bitsetToSortedVec(orders.inBody))
                nodes.insert(n);
            for(int n : bitsetToSortedVec(orders.containing))
                nodes.insert(n);
        }
        return nodeOrderTableString(nodes);

        //populateOrderCache();

        //TIMER(t, "nodeOrderTable");

        //std::set<int> nodes;
        //for(auto const& [node, orders] : m_orderCache)
        //{
        //    if(!orders.after.empty() or !orders.before.empty() or !orders.inBody.empty()
        //       or !orders.containing.empty())
        //    {
        //        nodes.insert(node);
        //    }
        //    for(int n : orders.after)
        //        nodes.insert(n);
        //    for(int n : orders.before)
        //        nodes.insert(n);
        //    for(int n : orders.inBody)
        //        nodes.insert(n);
        //    for(int n : orders.containing)
        //        nodes.insert(n);
        //}
        //return nodeOrderTableString(nodes);
    }

    void ControlGraph::clearCache(Graph::GraphModification modification)
    {
        Hypergraph<Operation, ControlEdge, false>::clearCache(modification);

        if(modification == Graph::GraphModification::AddElement
           && m_cacheStatus != CacheStatus::Invalid)
        {
            // If adding a new element and order is non-empty (partial or valid)
            m_cacheStatus = CacheStatus::Partial;
        }
        else
        {
            m_orderCache.clear();
            m_cacheStatus = CacheStatus::Invalid;
        }

        m_descendentCache.clear();
    }

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
            //
            // Theoretically, add/delete Body edge should be disallowed. But sometimes
            // delete Body edges is OK (e.g., Simplify), and currently there is no way
            // to know if this is called in a valid or invalid use case.
            //
            return true;
        }
    }

    void ControlGraph::validateOrderCache() const
    {
        for(auto const& [node, orders] : m_orderCache)
        {
            AssertFatal(!bitsetOverlaps(orders.after, orders.before),
                        "Node has conflicting orders (after & before)");
            AssertFatal(!bitsetOverlaps(orders.after, orders.inBody),
                        "Node has conflicting orders (after & inBody)");
            AssertFatal(!bitsetOverlaps(orders.after, orders.containing),
                        "Node has conflicting orders (after & containing)");
            AssertFatal(!bitsetOverlaps(orders.before, orders.inBody),
                        "Node has conflicting orders (before & inBody)");
            AssertFatal(!bitsetOverlaps(orders.before, orders.containing),
                        "Node has conflicting orders (before & containing)");
            AssertFatal(!bitsetOverlaps(orders.inBody, orders.containing),
                        "Node has conflicting orders (inBody & containing)");
        }

        //int const                    maxId = std::ranges::max(m_orderCache | std::views::keys);
        //std::vector<std::bitset<64>> bits((maxId + 64) / 64);

        //auto check = [&](const auto&... vec) {
        //    (
        //        [&]() {
        //            for(auto v : vec)
        //            {
        //                int const id     = v / 64;
        //                int const remain = v & 63;
        //                AssertFatal(!bits[id][remain],
        //                            "A node has two orders",
        //                            ShowValue(id),
        //                            ShowValue(remain),
        //                            ShowValue(v));
        //                bits[id].set(remain);
        //            }
        //        }(),
        //        ...);
        //    std::ranges::fill(bits, std::bitset<64>{});
        //};

        //for(auto& [node, orders] : m_orderCache)
        //{
        //    std::ranges::sort(orders.after);
        //    std::ranges::sort(orders.before);
        //    std::ranges::sort(orders.inBody);
        //    std::ranges::sort(orders.containing);
        //    check(orders.after, orders.before, orders.inBody, orders.containing);
        //}
    }

    using NodeOrderField = Bitset NodeOrders::*;
    // Maps ordering enum to the exact bitset field in NodeOrders.
    // This lets writeOrderCache compute the field once per call,
    // not once per element in hot loops.
    inline NodeOrderField orderField(NodeOrdering order)
    {
        switch(order)
        {
        case NodeOrdering::LeftFirst:
            return &NodeOrders::after;
        case NodeOrdering::RightFirst:
            return &NodeOrders::before;
        case NodeOrdering::RightInBodyOfLeft:
            return &NodeOrders::inBody;
        case NodeOrdering::LeftInBodyOfRight:
            return &NodeOrders::containing;
        default:
            break;
        }
        AssertFatal(false, "Invalid NodeOrdering", ShowValue(order));
        return &NodeOrders::after;
    }
    inline void clearBit(Bitset& bs, int id)
    {
        size_t const word = static_cast<size_t>(id) >> 6;
        if(word < bs.size())
            bs[word] &= ~(uint64_t(1) << (id & 63));
    }

    void ControlGraph::populateOrderCache() const
    {
        TIMER(t, "populateOrderCache");
        if(m_cacheStatus == CacheStatus::Valid)
            return;
        m_orderCache.clear();
        m_descendentCache.clear();
        static_assert(std::variant_size_v<ControlEdge> == 5,
                      "Assumes edge indices: Sequence(0), Initialize(1), "
                      "ForLoopIncrement(2), Body(3), Else(4).");
        constexpr uint8_t kSequence      = 0;
        constexpr uint8_t kInit          = 1;
        constexpr uint8_t kInc           = 2;
        constexpr uint8_t kBody          = 3;
        constexpr uint8_t kElse          = 4;
        constexpr size_t  kEdgeTypeCount = std::variant_size_v<ControlEdge>;
        using GD                         = Graph::Direction;
        auto nodeTags                    = getNodes().to<std::vector>();
        if(nodeTags.empty())
        {
            m_cacheStatus = CacheStatus::Valid;
            return;
        }
        int const nodeCount  = static_cast<int>(nodeTags.size());
        int const maxNodeTag = *std::max_element(nodeTags.begin(), nodeTags.end());
        // Dense lookup: graph tag -> dense node index [0, nodeCount).
        std::vector<int> tagToDense(static_cast<size_t>(maxNodeTag) + 1, -1);
        for(int i = 0; i < nodeCount; ++i)
            tagToDense[static_cast<size_t>(nodeTags[i])] = i;
        struct NodeAdj
        {
            // child dense node indices grouped by outgoing edge type.
            std::array<std::vector<int>, kEdgeTypeCount> childrenByType;
        };
        std::vector<NodeAdj> adj(nodeCount);
        std::vector<int>     indegree(nodeCount, 0);
        // Build node-level adjacency once.
        for(int parentDense = 0; parentDense < nodeCount; ++parentDense)
        {
            int const parentTag = nodeTags[parentDense];
            for(int edgeTag : getNeighbours<GD::Downstream>(parentTag))
            {
                uint8_t const edgeType  = static_cast<uint8_t>(getEdge(edgeTag).index());
                auto&         bucket    = adj[parentDense].childrenByType[edgeType];
                auto          childTags = getNeighbours<GD::Downstream>(edgeTag);
                bucket.reserve(bucket.size() + childTags.size());
                for(int childTag : childTags)
                {
                    if(childTag < 0 || childTag > maxNodeTag)
                        continue;
                    int const childDense = tagToDense[static_cast<size_t>(childTag)];
                    if(childDense < 0)
                        continue;
                    bucket.push_back(childDense);
                    ++indegree[childDense];
                }
            }
        }
        // Topological order on node DAG.
        std::vector<int> zeroIn;
        zeroIn.reserve(nodeCount);
        for(int i = 0; i < nodeCount; ++i)
            if(indegree[i] == 0)
                zeroIn.push_back(i);
        std::vector<int> topoOrder;
        topoOrder.reserve(nodeCount);
        for(size_t head = 0; head < zeroIn.size(); ++head)
        {
            int const u = zeroIn[head];
            topoOrder.push_back(u);
            for(auto const& byType : adj[u].childrenByType)
            {
                for(int v : byType)
                {
                    if(--indegree[v] == 0)
                        zeroIn.push_back(v);
                }
            }
        }
        AssertFatal(static_cast<int>(topoOrder.size()) == nodeCount,
                    "ControlGraph must be acyclic for order-cache construction.");
        // Dense build storage; avoids unordered_map lookups in hot loops.
        std::vector<NodeOrders> denseOrders(nodeCount);
        // allDesc[u] = all descendants of u across all edge types.
        std::vector<Bitset> allDesc(nodeCount);
        auto writeNodeDense = [&](int nodeDense, Bitset const& nodesB, NodeOrdering order) {
            if(!bitsetAny(nodesB))
                return;
            auto const fieldAB = orderField(order);
            auto const fieldBA = orderField(opposite(order));
            int const  nodeTag = nodeTags[nodeDense];
            bitsetOr(denseOrders[nodeDense].*fieldAB, nodesB);
            for(size_t w = 0; w < nodesB.size(); ++w)
            {
                uint64_t bits = nodesB[w];
                while(bits)
                {
                    int const bit  = std::countr_zero(bits);
                    int const bTag = static_cast<int>(w * 64 + bit);
                    AssertFatal(
                        bTag >= 0 && bTag <= maxNodeTag, ShowValue(bTag), ShowValue(maxNodeTag));
                    int const bDense = tagToDense[static_cast<size_t>(bTag)];
                    AssertFatal(bDense >= 0, ShowValue(bTag));
                    bitsetSet(denseOrders[bDense].*fieldBA, nodeTag);
                    bits &= bits - 1;
                }
            }
        };
        auto writeDense = [&](Bitset const& nodesA, Bitset const& nodesB, NodeOrdering order) {
            if(!bitsetAny(nodesA) || !bitsetAny(nodesB))
                return;
            auto const fieldAB = orderField(order);
            auto const fieldBA = orderField(opposite(order));
            for(size_t w = 0; w < nodesA.size(); ++w)
            {
                uint64_t bits = nodesA[w];
                while(bits)
                {
                    int const bit  = std::countr_zero(bits);
                    int const aTag = static_cast<int>(w * 64 + bit);
                    AssertFatal(
                        aTag >= 0 && aTag <= maxNodeTag, ShowValue(aTag), ShowValue(maxNodeTag));
                    int const aDense = tagToDense[static_cast<size_t>(aTag)];
                    AssertFatal(aDense >= 0, ShowValue(aTag));
                    bitsetOr(denseOrders[aDense].*fieldAB, nodesB);
                    bits &= bits - 1;
                }
            }
            for(size_t w = 0; w < nodesB.size(); ++w)
            {
                uint64_t bits = nodesB[w];
                while(bits)
                {
                    int const bit  = std::countr_zero(bits);
                    int const bTag = static_cast<int>(w * 64 + bit);
                    AssertFatal(
                        bTag >= 0 && bTag <= maxNodeTag, ShowValue(bTag), ShowValue(maxNodeTag));
                    int const bDense = tagToDense[static_cast<size_t>(bTag)];
                    AssertFatal(bDense >= 0, ShowValue(bTag));
                    bitsetOr(denseOrders[bDense].*fieldBA, nodesA);
                    bits &= bits - 1;
                }
            }
        };
        auto mergeChildrenDesc = [&](Bitset& dst, std::vector<int> const& childDenseList) {
            for(int childDense : childDenseList)
            {
                int const childTag = nodeTags[childDense];
                bitsetSet(dst, childTag);
                bitsetOr(dst, allDesc[childDense]);
            }
        };
        std::array<Bitset, kEdgeTypeCount> typedDesc;
        Bitset                             bodyLike;
        Bitset                             afterInit;
        Bitset                             afterBody;
        Bitset                             afterElse;
        // Reverse topo: children already have allDesc computed.
        for(auto it = topoOrder.rbegin(); it != topoOrder.rend(); ++it)
        {
            int const uDense = *it;
            int const uTag   = nodeTags[uDense];
            for(auto& bs : typedDesc)
                bs.clear();
            bodyLike.clear();
            afterInit.clear();
            afterBody.clear();
            afterElse.clear();
            mergeChildrenDesc(typedDesc[kInit], adj[uDense].childrenByType[kInit]);
            mergeChildrenDesc(typedDesc[kBody], adj[uDense].childrenByType[kBody]);
            mergeChildrenDesc(typedDesc[kElse], adj[uDense].childrenByType[kElse]);
            mergeChildrenDesc(typedDesc[kInc], adj[uDense].childrenByType[kInc]);
            mergeChildrenDesc(typedDesc[kSequence], adj[uDense].childrenByType[kSequence]);
            auto& initNodes = typedDesc[kInit];
            auto& bodyNodes = typedDesc[kBody];
            auto& elseNodes = typedDesc[kElse];
            auto& incNodes  = typedDesc[kInc];
            auto& seqNodes  = typedDesc[kSequence];
            // Defensive only: keep self out of all descendant partitions.
            clearBit(initNodes, uTag);
            clearBit(bodyNodes, uTag);
            clearBit(elseNodes, uTag);
            clearBit(incNodes, uTag);
            clearBit(seqNodes, uTag);
            // bodyLike = init | body | else | inc
            bodyLike = initNodes;
            bitsetOr(bodyLike, bodyNodes);
            bitsetOr(bodyLike, elseNodes);
            bitsetOr(bodyLike, incNodes);
            // Parent-to-descendant relations.
            writeNodeDense(uDense, bodyLike, NodeOrdering::RightInBodyOfLeft);
            writeNodeDense(uDense, seqNodes, NodeOrdering::LeftFirst);
            // Cross-partition ordering with strict precedence:
            // Initialize -> Body -> Else -> ForLoopIncrement -> Sequence
            //
            // Critical correctness fix:
            // afterInit excludes initNodes itself.
            afterInit = bodyNodes;
            bitsetOr(afterInit, elseNodes);
            bitsetOr(afterInit, incNodes);
            bitsetOr(afterInit, seqNodes);
            writeDense(initNodes, afterInit, NodeOrdering::LeftFirst);
            afterBody = elseNodes;
            bitsetOr(afterBody, incNodes);
            bitsetOr(afterBody, seqNodes);
            writeDense(bodyNodes, afterBody, NodeOrdering::LeftFirst);
            afterElse = incNodes;
            bitsetOr(afterElse, seqNodes);
            writeDense(elseNodes, afterElse, NodeOrdering::LeftFirst);
            writeDense(incNodes, seqNodes, NodeOrdering::LeftFirst);
            allDesc[uDense] = bodyLike;
            bitsetOr(allDesc[uDense], seqNodes);
        }
        // Move dense result into public cache structure.
        m_orderCache.reserve(nodeCount);
        for(int i = 0; i < nodeCount; ++i)
        {
            auto& orders = denseOrders[i];
            if(bitsetAny(orders.after) || bitsetAny(orders.before) || bitsetAny(orders.inBody)
               || bitsetAny(orders.containing))
            {
                m_orderCache.emplace(nodeTags[i], std::move(orders));
            }
        }
        validateOrderCache();
        m_cacheStatus = CacheStatus::Valid;
        m_descendentCache.clear();
    }

    /*
void ControlGraph::populateOrderCache() const
  {
      TIMER(t, "populateOrderCache");
      if(m_cacheStatus == CacheStatus::Valid)
          return;
      m_orderCache.clear();
      m_descendentCache.clear();
      static_assert(std::variant_size_v<ControlEdge> == 5,
                    "Edge types assumed: Sequence(0), Initialize(1), ForLoopIncrement(2), "
                    "Body(3), Else(4). Update populateOrderCache if this changes.");
      constexpr int kEdgeTypeCount = std::variant_size_v<ControlEdge>;
      // Variant indices in ControlEdge.
      constexpr uint8_t kSequence = 0;
      constexpr uint8_t kInit     = 1;
      constexpr uint8_t kInc      = 2;
      constexpr uint8_t kBody     = 3;
      constexpr uint8_t kElse     = 4;
      // Per-node topology in dense-node-index space (not graph tag space).
      // We keep dense indexing to avoid hash lookups in hot loops.
      struct NodeTopology
      {
          // Child node indices grouped by outgoing edge type.
          std::array<std::vector<int>, kEdgeTypeCount> childrenByTypeIdx;
          // Immediate parents of this node: {parentNodeIndex, edgeType(parent->this)}.
          std::vector<std::pair<int, uint8_t>> parents;
      };
      // Memoized descendants grouped by "first edge type" from the source node.
      struct DescInfo
      {
          std::array<Bitset, kEdgeTypeCount> byFirstEdge;
          Bitset                             all; // union of all byFirstEdge sets
          uint8_t                            dfsState = 0; // 0=unvisited,1=visiting,2=done
      };
      using GD = Graph::Direction;
      auto nodeTags = getNodes().to<std::vector>();
      if(nodeTags.empty())
      {
          m_cacheStatus = CacheStatus::Valid;
          return;
      }
      // Optional micro-optimization:
      // Process larger tags first so "set one bit in many rows" tends to grow bitsets once.
      std::sort(nodeTags.begin(), nodeTags.end(), std::greater<int>{});
      int const nodeCount = static_cast<int>(nodeTags.size());
      // Tag -> dense index.
      std::unordered_map<int, int> tagToIdx;
      tagToIdx.reserve(nodeTags.size());
      for(int i = 0; i < nodeCount; ++i)
          tagToIdx.emplace(nodeTags[i], i);
      std::vector<NodeTopology> topo(nodeCount);
      // Build parent and childrenByType adjacency once from graph edges.
      // This avoids repeated getNeighbours() calls inside per-node ancestor traversals.
      for(int edge : getEdges().to<std::vector>())
      {
          uint8_t const edgeType = static_cast<uint8_t>(getEdge(edge).index());
          auto const parents = getNeighbours<GD::Upstream>(edge);
          auto const kids    = getNeighbours<GD::Downstream>(edge);
          for(int pTag : parents)
          {
              auto pIt = tagToIdx.find(pTag);
              if(pIt == tagToIdx.end())
                  continue;
              int const pIdx = pIt->second;
              auto&     bucket = topo[pIdx].childrenByTypeIdx[edgeType];
              for(int cTag : kids)
              {
                  auto cIt = tagToIdx.find(cTag);
                  if(cIt == tagToIdx.end())
                      continue;
                  int const cIdx = cIt->second;
                  bucket.push_back(cIdx);
                  topo[cIdx].parents.emplace_back(pIdx, edgeType);
              }
          }
      }
      // Descendant cache by first-edge-type, computed once per node.
      std::vector<DescInfo> desc(nodeCount);
      auto buildDescByFirstEdge = [&](auto&& self, int nodeIdx) -> DescInfo const& {
          auto& info = desc[nodeIdx];
          if(info.dfsState == 2)
              return info;
          // If this fires, graph has a cycle in node-level reachability.
          // Current cache logic (old and new) assumes acyclic control structure.
          AssertFatal(info.dfsState != 1,
                      "ControlGraph must be acyclic while building order cache.",
                      ShowValue(nodeTags[nodeIdx]));
          info.dfsState = 1;
          for(uint8_t edgeType = 0; edgeType < kEdgeTypeCount; ++edgeType)
          {
              for(int childIdx : topo[nodeIdx].childrenByTypeIdx[edgeType])
              {
                  int const childTag = nodeTags[childIdx];
                  // Direct child belongs to this first-edge-type bucket.
                  bitsetSet(info.byFirstEdge[edgeType], childTag);
                  // Everything reachable from that child also belongs to same first-edge bucket.
                  auto const& childInfo = self(self, childIdx);
                  bitsetOr(info.byFirstEdge[edgeType], childInfo.all);
              }
              bitsetOr(info.all, info.byFirstEdge[edgeType]);
          }
          info.dfsState = 2;
          return info;
      };
      for(int i = 0; i < nodeCount; ++i)
          (void)buildDescByFirstEdge(buildDescByFirstEdge, i);
      // Select which NodeOrders bitset a NodeOrdering writes to.
      auto selectField = [](NodeOrdering order) -> Bitset NodeOrders::* {
          switch(order)
          {
          case NodeOrdering::LeftFirst:
              return &NodeOrders::after;
          case NodeOrdering::RightFirst:
              return &NodeOrders::before;
          case NodeOrdering::RightInBodyOfLeft:
              return &NodeOrders::inBody;
          case NodeOrdering::LeftInBodyOfRight:
              return &NodeOrders::containing;
          default:
              break;
          }
          AssertFatal(false, "Invalid order: ", ShowValue(order));
          return &NodeOrders::after;
      };
      // Writes A -> nodeB (many-to-one) efficiently:
      // 1) set one bit in each left row
      // 2) one bulk OR into right row's opposite set
      auto writeManyToOne = [&](Bitset const& leftNodes, int nodeB, NodeOrdering order) {
          auto const lhsField = selectField(order);
          auto const rhsField = selectField(opposite(order));
          bool any = false;
          for(size_t w = 0; w < leftNodes.size(); ++w)
          {
              uint64_t bits = leftNodes[w];
              while(bits)
              {
                  int const bit     = std::countr_zero(bits);
                  int const leftTag = static_cast<int>(w * 64 + bit);
                  bitsetSet((m_orderCache[leftTag].*lhsField), nodeB);
                  bits &= bits - 1;
                  any = true;
              }
          }
          if(any)
              bitsetOr((m_orderCache[nodeB].*rhsField), leftNodes);
      };
      auto clearBit = [](Bitset& bs, int id) {
          size_t const word = static_cast<size_t>(id) >> 6;
          if(word < bs.size())
              bs[word] &= ~(uint64_t(1) << (id & 63));
      };
      m_orderCache.reserve(nodeTags.size());
      // Stamp+mask visitation:
      // - stamp tells if ancestor was visited for current target node
      // - mask dedups by (ancestor, viaEdgeType), not just ancestor
      // This is important because same ancestor can be reached through different first-edge types.
      std::vector<uint32_t> seenStamp(nodeCount, 0);
      std::vector<uint8_t>  seenMask(nodeCount, 0);
      uint32_t              stamp = 0;
      std::vector<std::pair<int, uint8_t>> stack;
      stack.reserve(64);
      // Main algorithm (your idea):
      // For each target node:
      //   - traverse ancestors upward
      //   - add direct ancestor relation based on incoming edge type
      //   - add all descendants of ancestor that are known to execute before target
      for(int nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx)
      {
          int const nodeTag = nodeTags[nodeIdx];
          ++stamp;
          Bitset before;     // nodes definitely before nodeTag
          Bitset containing; // nodes that contain nodeTag in body
          stack.clear();
          for(auto const& p : topo[nodeIdx].parents)
              stack.push_back(p);
          while(!stack.empty())
          {
              auto const [ancIdx, viaEdgeType] = stack.back();
              stack.pop_back();
              if(seenStamp[ancIdx] != stamp)
              {
                  seenStamp[ancIdx] = stamp;
                  seenMask[ancIdx]  = 0;
              }
              uint8_t const edgeMask = static_cast<uint8_t>(1u << viaEdgeType);
              if(seenMask[ancIdx] & edgeMask)
                  continue;
              seenMask[ancIdx] |= edgeMask;
              int const ancTag = nodeTags[ancIdx];
              // Ancestor relation.
              if(viaEdgeType == kSequence)
                  bitsetSet(before, ancTag); // ancestor is before target
              else
                  bitsetSet(containing, ancTag); // target is in ancestor's body
              // Descendants of ancestor that are before this target, based on edge precedence:
              // Initialize < Body < Else < ForLoopIncrement < Sequence
              auto const& byType = desc[ancIdx].byFirstEdge;
              switch(viaEdgeType)
              {
              case kInit:
                  // Nothing before Initialize bucket.
                  break;
              case kBody:
                  bitsetOr(before, byType[kInit]);
                  break;
              case kElse:
                  bitsetOr(before, byType[kInit]);
                  bitsetOr(before, byType[kBody]);
                  break;
              case kInc:
                  bitsetOr(before, byType[kInit]);
                  bitsetOr(before, byType[kBody]);
                  bitsetOr(before, byType[kElse]);
                  break;
              case kSequence:
                  bitsetOr(before, byType[kInit]);
                  bitsetOr(before, byType[kBody]);
                  bitsetOr(before, byType[kElse]);
                  bitsetOr(before, byType[kInc]);
                  break;
              default:
                  AssertFatal(false, "Unhandled edge type index", ShowValue(viaEdgeType));
              }
              // Continue climbing ancestors.
              for(auto const& p : topo[ancIdx].parents)
                  stack.push_back(p);
          }
          // Defensive: avoid self relation if malformed graph introduces it.
          clearBit(before, nodeTag);
          clearBit(containing, nodeTag);
          // Materialize two relation families.
          writeManyToOne(before, nodeTag, NodeOrdering::LeftFirst);
          writeManyToOne(containing, nodeTag, NodeOrdering::RightInBodyOfLeft);
      }
      validateOrderCache();
      m_cacheStatus = CacheStatus::Valid;
      m_descendentCache.clear();
  }
*/

    //void ControlGraph::populateOrderCache() const
    //{
    //    TIMER(t, "populateOrderCache");

    //    if(m_cacheStatus == CacheStatus::Valid)
    //        return;

    //    m_orderCache.clear();

    //    auto rootNodes = roots().to<std::vector>();

    //    populateOrderCache(rootNodes);
    //    validateOrderCache();
    //    //sortOrderCache();

    //    m_cacheStatus = CacheStatus::Valid;
    //    //
    //    // m_descendentCache is only used to help build m_orderCache,
    //    // and it must be cleared after finish building m_orderCache
    //    // to ensure no stale data being used when building m_orderCache
    //    // next time.
    //    //
    //    m_descendentCache.clear();
    //}

    template <CForwardRangeOf<int> Range>
    //std::vector<int> ControlGraph::populateOrderCache(Range const& startingNodes) const
    Bitset ControlGraph::populateOrderCache(Range const& startingNodes) const
    {
        Bitset rv;
        for(auto it = startingNodes.begin(); it != startingNodes.end(); ++it)
        {
            auto nodes = populateOrderCache(*it);
            bitsetOr(rv, nodes);
        }
        return rv;

        //std::vector<int> rv;
        //for(auto it = startingNodes.begin(); it != startingNodes.end(); ++it)
        //{
        //    auto nodes = populateOrderCache(*it);
        //    rv.insert(rv.end(), nodes.begin(), nodes.end());
        //}
        //return rv;
    }

    //std::vector<int> ControlGraph::populateOrderCache(int startingNode) const
    Bitset ControlGraph::populateOrderCache(int startingNode) const
    {

        auto ccEntry = m_descendentCache.find(startingNode);
        if(ccEntry != m_descendentCache.end())
            return ccEntry->second;
        static_assert(std::variant_size_v<ControlEdge> == 5,
                      "Currently the available edge types are Sequence(0), Initialize(1), "
                      "ForLoopIncrement(2), Body(3) and Else(4)."
                      "If more edge types are added, this function has to be updated.");
        using GD = Graph::Direction;
        std::array<std::vector<int>, std::variant_size_v<ControlEdge>> directChildren;
        for(auto edge : getNeighbours<GD::Downstream>(startingNode))
        {
            auto edgeTypeIndex = getEdge(edge).index();
            for(auto child : getNeighbours<GD::Downstream>(edge))
                directChildren[edgeTypeIndex].push_back(child);
        }
        auto addDescendents = [this](std::vector<int> const& children) -> Bitset {
            Bitset descendents = populateOrderCache(children);
            Bitset result      = descendents;
            for(int c : children)
                bitsetSet(result, c);
            return result;
        };
        auto initNodes = addDescendents(directChildren[1]);
        auto bodyNodes = addDescendents(directChildren[3]);
        auto elseNodes = addDescendents(directChildren[4]);
        auto incNodes  = addDescendents(directChildren[2]);
        auto seqNodes  = addDescendents(directChildren[0]);
        writeOrderCache(startingNode, initNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache(startingNode, bodyNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache(startingNode, elseNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache(startingNode, incNodes, NodeOrdering::RightInBodyOfLeft);

        writeOrderCache(startingNode, seqNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, bodyNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, elseNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, seqNodes, NodeOrdering::LeftFirst);
        writeOrderCache(bodyNodes, elseNodes, NodeOrdering::LeftFirst);
        writeOrderCache(bodyNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(bodyNodes, seqNodes, NodeOrdering::LeftFirst);
        writeOrderCache(elseNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(elseNodes, seqNodes, NodeOrdering::LeftFirst);
        writeOrderCache(incNodes, seqNodes, NodeOrdering::LeftFirst);
        Bitset allNodes;
        bitsetOr(allNodes, initNodes);
        bitsetOr(allNodes, bodyNodes);
        bitsetOr(allNodes, elseNodes);
        bitsetOr(allNodes, incNodes);
        bitsetOr(allNodes, seqNodes);
        m_descendentCache[startingNode] = allNodes;
        return allNodes;

        //auto ccEntry = m_descendentCache.find(startingNode);
        //if(ccEntry != m_descendentCache.end())
        //    return ccEntry->second;

        //static_assert(std::variant_size_v<ControlEdge> == 5,
        //              "Currently the available edge types are Sequence(0), Initialize(1), "
        //              "ForLoopIncrement(2), Body(3) and Else(4)."
        //              "If more edge types are added, this function has to be updated.");

        //using GD = Graph::Direction;
        //// Edge variant indices: Sequence(0), Initialize(1), ForLoopIncrement(2), Body(3), Else(4)
        //std::array<std::vector<int>, std::variant_size_v<ControlEdge>> directChildren;
        //for(auto edge : getNeighbours<GD::Downstream>(startingNode))
        //{
        //    auto edgeTypeIndex = getEdge(edge).index();
        //    for(auto child : getNeighbours<GD::Downstream>(edge))
        //        directChildren[edgeTypeIndex].push_back(child);
        //}

        //auto addDescendents = [this](std::vector<int> const& children) -> std::vector<int> {
        //    auto             descendents = populateOrderCache(children);
        //    std::vector<int> result;
        //    result.reserve(children.size() + descendents.size());
        //    result.insert(result.end(), children.begin(), children.end());
        //    result.insert(result.end(), descendents.begin(), descendents.end());

        //    std::ranges::sort(result);
        //    result.erase(std::unique(result.begin(), result.end()), result.end());
        //    return result;
        //};

        //// Index: Initialize(1), Body(3), Else(4), ForLoopIncrement(2), Sequence(0)
        //auto initNodes = addDescendents(directChildren[1]);
        //auto bodyNodes = addDescendents(directChildren[3]);
        //auto elseNodes = addDescendents(directChildren[4]);
        //auto incNodes  = addDescendents(directChildren[2]);
        //auto seqNodes  = addDescendents(directChildren[0]);

        //// {init, body, else, inc} nodes are in the body of the current node
        //writeOrderCache({startingNode}, initNodes, NodeOrdering::RightInBodyOfLeft);
        //writeOrderCache({startingNode}, bodyNodes, NodeOrdering::RightInBodyOfLeft);
        //writeOrderCache({startingNode}, elseNodes, NodeOrdering::RightInBodyOfLeft);
        //writeOrderCache({startingNode}, incNodes, NodeOrdering::RightInBodyOfLeft);

        //// Sequence connected nodes are after the current node
        //writeOrderCache({startingNode}, seqNodes, NodeOrdering::LeftFirst);

        //// {body, else, inc, sequence} are after init nodes
        //writeOrderCache(initNodes, bodyNodes, NodeOrdering::LeftFirst);
        //writeOrderCache(initNodes, elseNodes, NodeOrdering::LeftFirst);
        //writeOrderCache(initNodes, incNodes, NodeOrdering::LeftFirst);
        //writeOrderCache(initNodes, seqNodes, NodeOrdering::LeftFirst);

        //// {else, inc, sequence} are after body nodes
        //writeOrderCache(bodyNodes, elseNodes, NodeOrdering::LeftFirst);
        //writeOrderCache(bodyNodes, incNodes, NodeOrdering::LeftFirst);
        //writeOrderCache(bodyNodes, seqNodes, NodeOrdering::LeftFirst);

        //// {inc, sequence} are after else nodes
        //writeOrderCache(elseNodes, incNodes, NodeOrdering::LeftFirst);
        //writeOrderCache(elseNodes, seqNodes, NodeOrdering::LeftFirst);

        //// sequence are after inc nodes
        //writeOrderCache(incNodes, seqNodes, NodeOrdering::LeftFirst);

        //std::vector<int> allNodes;
        //allNodes.reserve(initNodes.size() + bodyNodes.size() + elseNodes.size() + incNodes.size()
        //                 + seqNodes.size());

        //allNodes.insert(allNodes.end(), initNodes.begin(), initNodes.end());
        //allNodes.insert(allNodes.end(), bodyNodes.begin(), bodyNodes.end());
        //allNodes.insert(allNodes.end(), elseNodes.begin(), elseNodes.end());
        //allNodes.insert(allNodes.end(), incNodes.begin(), incNodes.end());
        //allNodes.insert(allNodes.end(), seqNodes.begin(), seqNodes.end());

        //std::ranges::sort(allNodes);
        //allNodes.erase(std::unique(allNodes.begin(), allNodes.end()), allNodes.end());
        //m_descendentCache[startingNode] = allNodes;

        //return allNodes;
    }

    void ControlGraph::writeOrderCache(Bitset const& nodesA,
                                       Bitset const& nodesB,
                                       NodeOrdering  order) const
    {
        if(!bitsetAny(nodesA) || !bitsetAny(nodesB))
            return;
        auto selectBitset = [](NodeOrders& orders, NodeOrdering order) -> Bitset& {
            switch(order)
            {
            case NodeOrdering::LeftFirst:
                return orders.after;
            case NodeOrdering::RightFirst:
                return orders.before;
            case NodeOrdering::RightInBodyOfLeft:
                return orders.inBody;
            case NodeOrdering::LeftInBodyOfRight:
                return orders.containing;
            default:
                break;
            }
            AssertFatal(false, "Invalid order: ", ShowValue(order));
            return orders.after;
        };
        for(size_t w = 0; w < nodesA.size(); ++w)
        {
            uint64_t bits = nodesA[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int a   = static_cast<int>(w * 64 + bit);
                bitsetOr(selectBitset(m_orderCache[a], order), nodesB);
                bits &= bits - 1;
            }
        }
        auto oppositeOrder = opposite(order);
        for(size_t w = 0; w < nodesB.size(); ++w)
        {
            uint64_t bits = nodesB[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int b   = static_cast<int>(w * 64 + bit);
                bitsetOr(selectBitset(m_orderCache[b], oppositeOrder), nodesA);
                bits &= bits - 1;
            }
        }
    }

    //template <CForwardRangeOf<int> ARange, CForwardRangeOf<int> BRange>
    //void ControlGraph::writeOrderCache(ARange const& nodesA,
    //                                   BRange const& nodesB,
    //                                   NodeOrdering  order) const
    //{
    //    if(nodesA.size() == 0 or nodesB.size() == 0)
    //        return;

    //    auto selectVec = [](NodeOrders& orders, NodeOrdering order) -> std::vector<int>& {
    //        switch(order)
    //        {
    //        case NodeOrdering::LeftFirst:
    //            return orders.after;
    //        case NodeOrdering::RightFirst:
    //            return orders.before;
    //        case NodeOrdering::RightInBodyOfLeft:
    //            return orders.inBody;
    //        case NodeOrdering::LeftInBodyOfRight:
    //            return orders.containing;
    //        default:
    //            break;
    //        }
    //        AssertFatal(false, "Invalid order: ", ShowValue(order));
    //        return orders.after; // this statement should never be reached
    //    };
    //    for(int a : nodesA)
    //    {
    //        auto& vec = selectVec(m_orderCache[a], order);
    //        vec.insert(vec.end(), nodesB.begin(), nodesB.end());
    //    }

    //    auto oppositeOrder = opposite(order);
    //    for(int b : nodesB)
    //    {
    //        auto& vec = selectVec(m_orderCache[b], oppositeOrder);
    //        vec.insert(vec.end(), nodesA.begin(), nodesA.end());
    //    }
    //}

    void ControlGraph::writeOrderCache(int nodeA, Bitset const& nodesB, NodeOrdering order) const
    {
        if(!bitsetAny(nodesB))
            return;
        auto selectBitset = [](NodeOrders& orders, NodeOrdering order) -> Bitset& {
            switch(order)
            {
            case NodeOrdering::LeftFirst:
                return orders.after;
            case NodeOrdering::RightFirst:
                return orders.before;
            case NodeOrdering::RightInBodyOfLeft:
                return orders.inBody;
            case NodeOrdering::LeftInBodyOfRight:
                return orders.containing;
            default:
                break;
            }
            AssertFatal(false, "Invalid order: ", ShowValue(order));
            return orders.after;
        };
        bitsetOr(selectBitset(m_orderCache[nodeA], order), nodesB);
        auto oppositeOrder = opposite(order);
        for(size_t w = 0; w < nodesB.size(); ++w)
        {
            uint64_t bits = nodesB[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int b   = static_cast<int>(w * 64 + bit);
                bitsetSet(selectBitset(m_orderCache[b], oppositeOrder), nodeA);
                bits &= bits - 1;
            }
        }
    }

    //void ControlGraph::writeOrderCache(int nodeA, int nodeB, NodeOrdering order) const
    //{
    //    switch(order)
    //    {
    //    case NodeOrdering::LeftFirst:
    //        m_orderCache[nodeA].after.push_back(nodeB);
    //        m_orderCache[nodeB].before.push_back(nodeA);
    //        break;
    //    case NodeOrdering::RightFirst:
    //        m_orderCache[nodeA].before.push_back(nodeB);
    //        m_orderCache[nodeB].after.push_back(nodeA);
    //        break;
    //    case NodeOrdering::RightInBodyOfLeft:
    //        m_orderCache[nodeA].inBody.push_back(nodeB);
    //        m_orderCache[nodeB].containing.push_back(nodeA);
    //        break;
    //    case NodeOrdering::LeftInBodyOfRight:
    //        m_orderCache[nodeA].containing.push_back(nodeB);
    //        m_orderCache[nodeB].inBody.push_back(nodeA);
    //        break;
    //    default:
    //        break;
    //    }
    //}

    void ControlGraph::writeOrderCache(int nodeA, int nodeB, NodeOrdering order) const
    {
        switch(order)
        {
        case NodeOrdering::LeftFirst:
            bitsetSet(m_orderCache[nodeA].after, nodeB);
            bitsetSet(m_orderCache[nodeB].before, nodeA);
            break;
        case NodeOrdering::RightFirst:
            bitsetSet(m_orderCache[nodeA].before, nodeB);
            bitsetSet(m_orderCache[nodeB].after, nodeA);
            break;
        case NodeOrdering::RightInBodyOfLeft:
            bitsetSet(m_orderCache[nodeA].inBody, nodeB);
            bitsetSet(m_orderCache[nodeB].containing, nodeA);
            break;
        case NodeOrdering::LeftInBodyOfRight:
            bitsetSet(m_orderCache[nodeA].containing, nodeB);
            bitsetSet(m_orderCache[nodeB].inBody, nodeA);
            break;
        default:
            break;
        }
    }

    NodeOrdering ControlGraph::lookupOrder(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        TIMER(t, "ControlGraph::lookupOrder");

        using GD = Graph::Direction;
        std::unordered_set<int> visited_nodes;

        // Decide order of A and B when A is parent of B
        auto const getOrderIfParent = [&](int edge) {
            return std::holds_alternative<Sequence>(getEdge(edge))
                       ? NodeOrdering::LeftFirst
                       : NodeOrdering::RightInBodyOfLeft;
        };

        // Decide order of A and B when A and B are descendants of a node
        // And order of edge type :
        // Initialize -> Body -> Else -> ForLoopIncrement -> Sequence.
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

        // {key, value} = {node index, edge index}
        std::unordered_map<int, int> A_ancestors;
        // stack
        std::vector<int> stk{nodeA};

        // Traverse upstream from A to collect all ancestors of A
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

        // Traverse upstream from B to find common ancestors to decide order
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
                        // If this is a common ancestor, compare the types of both edges to
                        // know the order
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
