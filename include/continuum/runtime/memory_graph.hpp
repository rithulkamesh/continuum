#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace continuum::runtime {

enum class MemoryNodeType : std::uint8_t {
  Prompt = 0,
  Completion = 1,
  Entity = 2,
};

struct MemoryNode {
  std::uint64_t id = 0;
  MemoryNodeType type = MemoryNodeType::Prompt;
  std::string content;
  std::vector<float> embedding;
  std::string session_id;
  std::int64_t created_ns = 0;
};

/// Memory-graph recall tier: an append-only log of prompt / completion nodes
/// retrieved by embedding similarity.
///
/// Eviction: first-in-first-out by insertion order. Recall is a scan, so a
/// read does not refresh a node; when `size()` reaches `max_nodes` the oldest
/// node is dropped before the new one is added.
class MemoryGraphStore {
 public:
  explicit MemoryGraphStore(std::size_t max_nodes = 8192);

  std::uint64_t add_node(MemoryNode node);
  std::optional<MemoryNode> get_node(std::uint64_t id) const;

  struct RetrievalResult {
    MemoryNode node;
    float similarity = 0.0f;
  };

  std::vector<RetrievalResult> retrieve_similar(
      const std::vector<float>& query_embedding,
      std::size_t max_results = 5,
      float min_similarity = 0.7f,
      const std::string& cache_namespace = {}) const;

  void clear();
  std::size_t size() const;
  /// Capacity in nodes passed at construction.
  std::size_t max_nodes() const { return max_nodes_; }
  /// Approximate resident bytes: content, embeddings, ids, and per-node overhead.
  std::size_t estimated_bytes() const;

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::uint64_t, MemoryNode> nodes_;
  std::uint64_t next_id_ = 1;
  std::size_t max_nodes_;
};

}  // namespace continuum::runtime
