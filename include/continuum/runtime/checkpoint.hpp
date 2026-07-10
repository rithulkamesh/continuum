#pragma once

#include <continuum/ir/graph.hpp>
#include <continuum/ir/value.hpp>
#include <continuum/runtime/cache.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace continuum::runtime {

// One portable KV cache entry: the token path it covers plus the backend
// state exported to bytes. Entries whose backend cannot export state are
// simply not checkpointed.
struct CheckpointCacheEntry {
  std::string model_id;
  DecodeParams decode{};
  std::int32_t prefix_len = 0;
  std::vector<std::int32_t> tokens;
  std::vector<std::uint8_t> state_bytes;
};

struct Checkpoint {
  std::vector<std::uint8_t> serialized_graph;
  std::uint64_t current_node_index = 0;
  std::unordered_map<ir::NodeId, continuum::Value> value_map;
  std::vector<CheckpointCacheEntry> cache_snapshot;
};

std::vector<std::uint8_t> serialize_value(const continuum::Value& value);
continuum::Value deserialize_value(const std::uint8_t* data, std::size_t len);

std::vector<std::uint8_t> serialize_checkpoint(const Checkpoint& checkpoint);
Checkpoint deserialize_checkpoint(const std::vector<std::uint8_t>& bytes);

std::vector<std::uint8_t> checkpoint_graph(const ir::Graph& graph);
ir::Graph restore_graph(const std::vector<std::uint8_t>& bytes);

}  // namespace continuum::runtime
