#pragma once

#include <continuum/ir/graph.hpp>
#include <continuum/ir/value.hpp>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace continuum::runtime {

struct Checkpoint {
  std::vector<std::uint8_t> serialized_graph;
  std::uint64_t current_node_index = 0;
  std::unordered_map<ir::NodeId, continuum::Value> value_map;
};

std::vector<std::uint8_t> serialize_value(const continuum::Value& value);
continuum::Value deserialize_value(const std::uint8_t* data, std::size_t len);

std::vector<std::uint8_t> serialize_checkpoint(const Checkpoint& checkpoint);
Checkpoint deserialize_checkpoint(const std::vector<std::uint8_t>& bytes);

std::vector<std::uint8_t> checkpoint_graph(const ir::Graph& graph);
ir::Graph restore_graph(const std::vector<std::uint8_t>& bytes);

}  // namespace continuum::runtime
