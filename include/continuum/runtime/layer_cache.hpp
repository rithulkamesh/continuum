#pragma once

#include <continuum/backend/backend.hpp>
#include <continuum/ir/node.hpp>

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace continuum::runtime {

struct LayerCheckpointKey {
  std::string model_id;
  std::string decode_hash;
  std::int32_t prefix_len = 0;
  std::int32_t layer_id = 0;
  std::uint64_t arch_version = 0;
  std::uint64_t token_hash = 0;

  bool operator==(const LayerCheckpointKey& o) const {
    return model_id == o.model_id && decode_hash == o.decode_hash &&
           prefix_len == o.prefix_len && layer_id == o.layer_id &&
           arch_version == o.arch_version && token_hash == o.token_hash;
  }
};

struct LayerCheckpointKeyHash {
  std::size_t operator()(const LayerCheckpointKey& k) const {
    auto h = std::hash<std::string>{}(k.model_id);
    h ^= std::hash<std::string>{}(k.decode_hash) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<std::int32_t>{}(k.prefix_len) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<std::int32_t>{}(k.layer_id) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<std::uint64_t>{}(k.arch_version) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<std::uint64_t>{}(k.token_hash) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

struct LayerCheckpoint {
  backend::BackendState state;
  // Tokens the cached state was computed over; reuse requires these to be a
  // prefix of the query tokens, otherwise the state belongs to another prompt.
  std::vector<std::int32_t> tokens;
  std::string model_id;
  std::string decode_hash;
  std::int32_t layer_id = 0;
  std::int32_t prefix_len = 0;
  std::uint64_t arch_version = 0;
  std::int64_t last_access_ns = 0;
  std::size_t estimated_bytes = 0;
};

class LayerKVCacheIndex {
 public:
  explicit LayerKVCacheIndex(std::size_t max_entries = 4096,
                             std::size_t max_bytes = 256 * 1024 * 1024);

  struct LookupResult {
    backend::BackendState state;
    std::int32_t layer_id = 0;
    std::int32_t prefix_len = 0;
    bool found = false;
  };

  LookupResult find_deepest(const std::string& model_id,
                              const std::string& decode_hash,
                              const std::vector<std::int32_t>& tokens,
                              std::int32_t total_layers,
                              std::uint64_t arch_version) const;

  void insert(LayerCheckpoint checkpoint);
  void invalidate_model(const std::string& model_id);
  void invalidate_arch(std::uint64_t arch_version);
  void clear();

  std::size_t size() const;
  std::size_t estimated_bytes() const;

 private:
  void evict_if_needed();

  mutable std::mutex mu_;
  std::unordered_map<LayerCheckpointKey, LayerCheckpoint, LayerCheckpointKeyHash> entries_;
  std::size_t max_entries_;
  std::size_t max_bytes_;
  std::size_t current_bytes_ = 0;
  std::uint64_t clock_ = 0;
};

}  // namespace continuum::runtime
