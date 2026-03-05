#pragma once
#include <mint/core/index_t.h>
#include <mint/core/nd_index.h>
#include <mint/core/tuple.h>

namespace mint {

template <index_t kMaxNumNode, index_t kMaxNumEdge>
struct fixed_capacity_csr_graph {
  constexpr fixed_capacity_csr_graph() = default;

  constexpr fixed_capacity_csr_graph(
      index_t num_node,
      index_t num_edge,
      const nd_index<kMaxNumEdge>& edge_targets,
      const nd_index<kMaxNumNode + 1>& node_edge_start_indices)
      : num_node_{num_node},
        num_edge_{num_edge},
        edge_targets_{edge_targets},
        node_edge_start_indices_{node_edge_start_indices} {}

  index_t num_node_;
  index_t num_edge_;
  nd_index<kMaxNumEdge> edge_targets_;
  nd_index<kMaxNumNode + 1> node_edge_start_indices_;
};

template <index_t kMaxNumNode, index_t kMaxNumEdge>
constexpr auto try_sort_graph(
    const fixed_capacity_csr_graph<kMaxNumNode, kMaxNumEdge>& g)
    -> mint::tuple<nd_index<kMaxNumNode>, bool> {
  nd_index<kMaxNumNode> sorted_nodes;
  sorted_nodes.fill(0);

  nd_index<kMaxNumNode>
      is_visited; // 0 = not visited, 1 = visiting, 2 = visited
  is_visited.fill(0);
  nd_index<kMaxNumNode> stack;

  index_t num_sorted_node = 0;

  // sort with DFS
  for (index_t i = 0; i < g.num_node_; i++) {
    if (is_visited(i) == 0) {
      stack[0] = i;
      is_visited(i) = 1;
      index_t nstack = 1;

      while (nstack > 0) {
        bool continue_while_loop = false;
        index_t node = stack[nstack - 1];
        for (index_t j = g.node_edge_start_indices_[node];
             j < g.node_edge_start_indices_[node + 1];
             j++) {
          index_t dep_node = g.edge_targets_[j];
          // is_visited = 0, push to stack
          // is_visited = 1, fail due to circular dependency
          // is_visited = 2, do nothing
          if (is_visited(dep_node) == 0) {
            stack[nstack++] = dep_node;
            continue_while_loop = true;
            break;
          } else if (is_visited(dep_node) == 1) {
            return {sorted_nodes, false};
          }
        }

        if (continue_while_loop)
          continue;

        // all dep_node has been visited
        is_visited(node) = 2;
        sorted_nodes(num_sorted_node++) = node;
        nstack--;
      }
    }
  }

  // reverse sorted node list
  for (index_t i = 0; i < g.num_node_ / 2; i++) {
    index_t tmp = sorted_nodes[i];
    sorted_nodes[i] = sorted_nodes[g.num_node_ - i - 1];
    sorted_nodes[g.num_node_ - i - 1] = tmp;
  }

  return {sorted_nodes, true};
}

template <index_t kMaxNumNode, index_t kMaxNumEdge>
constexpr bool is_acyclic_graph(
    const fixed_capacity_csr_graph<kMaxNumNode, kMaxNumEdge>& g) {
  const auto [sorted_nodes, success] = try_sort_graph(g);
  return success;
}

template <index_t kMaxNumNode, index_t kMaxNumEdge>
constexpr auto sort_graph(
    const fixed_capacity_csr_graph<kMaxNumNode, kMaxNumEdge>& g)
    -> nd_index<kMaxNumNode> {
  const auto [sorted_nodes, success] = try_sort_graph(g);
  return sorted_nodes;
}

template <index_t kMaxNumNode, index_t kMaxNumEdge>
constexpr bool is_sorted_nodes(
    const nd_index<kMaxNumNode>& nodes,
    const fixed_capacity_csr_graph<kMaxNumNode, kMaxNumEdge>& g) {
  nd_index<kMaxNumNode> node_ranks;

  for (index_t i = 0; i < g.num_node_; i++)
    node_ranks[nodes[i]] = i;

  for (index_t node = 0; node < g.num_node_; node++) {
    for (index_t j = g.node_edge_start_indices_[node];
         j < g.node_edge_start_indices_[node + 1];
         j++) {
      index_t dep_node = g.edge_targets_[j];
      if (node_ranks[node] >= node_ranks[dep_node])
        return false;
    }
  }

  return true;
}

} // namespace mint
