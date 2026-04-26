#pragma once

#include <continuum/ir/node.hpp>

#include <optional>
#include <unordered_map>
#include <vector>

namespace continuum::ir {

class Graph {
 public:
  Graph() = default;
  ~Graph() = default;

  NodeId add_node(Node n);
  const Node& get(NodeId id) const;
  Node& get_mut(NodeId id);
  const std::vector<NodeId>& topo_order() const;

  Graph extract_subgraph(const std::vector<NodeId>& roots) const;
  std::uint64_t structural_hash() const;
  std::vector<std::uint8_t> serialize() const;
  static Graph deserialize(const std::uint8_t* data, std::size_t len);

 private:
  std::unordered_map<NodeId, Node> nodes_;
  NodeId next_id_ = 1;
  mutable std::optional<std::vector<NodeId>> topo_cache_;
};

}  // namespace continuum::ir
