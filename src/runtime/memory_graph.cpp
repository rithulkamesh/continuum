#include "continuum/runtime/memory_graph.hpp"

#include <algorithm>
#include <chrono>

#include "continuum/runtime/semantic_cache.hpp"

namespace continuum::runtime {

MemoryGraphStore::MemoryGraphStore(std::size_t max_nodes)
    : max_nodes_(max_nodes) {}

std::uint64_t MemoryGraphStore::add_node(MemoryNode node) {
  std::lock_guard<std::mutex> lock(mu_);

  node.id = next_id_++;
  if (max_nodes_ == 0) {
    return node.id;
  }
  if (nodes_.size() >= max_nodes_) {
    // Ids are handed out monotonically, so the smallest id is the oldest node.
    auto oldest = nodes_.begin();
    for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
      if (it->first < oldest->first) {
        oldest = it;
      }
    }
    nodes_.erase(oldest);
  }

  node.created_ns = std::chrono::steady_clock::now().time_since_epoch().count();
  nodes_[node.id] = node;
  return node.id;
}

std::optional<MemoryNode> MemoryGraphStore::get_node(std::uint64_t id) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = nodes_.find(id);
  if (it == nodes_.end()) return std::nullopt;
  return it->second;
}

std::vector<MemoryGraphStore::RetrievalResult>
MemoryGraphStore::retrieve_similar(const std::vector<float>& query_embedding,
                                   std::size_t max_results,
                                   float min_similarity,
                                   const std::string& cache_namespace) const {
  std::lock_guard<std::mutex> lock(mu_);

  std::vector<RetrievalResult> scored;
  for (const auto& [id, node] : nodes_) {
    if (node.embedding.empty()) continue;
    if (node.session_id != cache_namespace) continue;
    float sim = SemanticCacheIndex::cosine_similarity(query_embedding, node.embedding);
    if (sim >= min_similarity) {
      scored.push_back({node, sim});
    }
  }

  std::sort(scored.begin(), scored.end(),
            [](const RetrievalResult& a, const RetrievalResult& b) {
              return a.similarity > b.similarity;
            });

  if (scored.size() > max_results) {
    scored.resize(max_results);
  }
  return scored;
}

void MemoryGraphStore::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  nodes_.clear();
  next_id_ = 1;
}

std::size_t MemoryGraphStore::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return nodes_.size();
}

std::size_t MemoryGraphStore::estimated_bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::size_t total = 0;
  for (const auto& [id, node] : nodes_) {
    total += sizeof(MemoryNode);
    total += node.content.size() + node.session_id.size() + node.embedding.size() * sizeof(float);
  }
  return total;
}

}  // namespace continuum::runtime
