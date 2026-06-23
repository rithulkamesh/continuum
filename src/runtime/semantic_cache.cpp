#include "continuum/runtime/semantic_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace continuum::runtime {

SemanticCacheIndex::SemanticCacheIndex(std::size_t max_entries,
                                       float similarity_threshold)
    : max_entries_(max_entries), similarity_threshold_(similarity_threshold) {}

SemanticCacheIndex::LookupResult SemanticCacheIndex::lookup(
    const std::vector<float>& query_embedding,
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mu_);

  LookupResult best;
  for (const auto& entry : entries_) {
    if (entry.model_id != model_id) continue;
    if (entry.embedding.size() != query_embedding.size()) continue;

    float sim = cosine_similarity(query_embedding, entry.embedding);
    if (sim > best.similarity) {
      best.similarity = sim;
      if (sim >= similarity_threshold_) {
        best.above_threshold = true;
        best.output = entry.cached_output;
      }
    }
  }
  return best;
}

void SemanticCacheIndex::insert(const std::vector<float>& embedding,
                                const std::string& model_id,
                                std::vector<std::uint8_t> output) {
  std::lock_guard<std::mutex> lock(mu_);

  if (entries_.size() >= max_entries_ && !entries_.empty()) {
    auto it = std::min_element(
        entries_.begin(), entries_.end(),
        [](const SemanticCacheEntry& a, const SemanticCacheEntry& b) {
          return a.last_access_ns < b.last_access_ns;
        });
    entries_.erase(it);
  }

  SemanticCacheEntry entry;
  entry.embedding = embedding;
  entry.model_id = model_id;
  entry.cached_output = std::move(output);
  entry.last_access_ns = ++clock_;
  entries_.push_back(std::move(entry));
}

void SemanticCacheIndex::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.clear();
  clock_ = 0;
}

std::size_t SemanticCacheIndex::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.size();
}

float SemanticCacheIndex::similarity_threshold() const {
  return similarity_threshold_;
}

void SemanticCacheIndex::set_similarity_threshold(float t) {
  similarity_threshold_ = t;
}

float SemanticCacheIndex::cosine_similarity(const std::vector<float>& a,
                                            const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0f;

  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  norm_a = std::sqrt(norm_a);
  norm_b = std::sqrt(norm_b);
  if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;

  return dot / (norm_a * norm_b);
}

// ---

BruteForceEmbeddingProvider::BruteForceEmbeddingProvider(std::size_t dim)
    : dim_(dim) {}

std::vector<float> BruteForceEmbeddingProvider::embed(
    const std::string& text) const {
  std::vector<float> result(dim_, 0.0f);

  for (std::size_t i = 0; i < text.size(); ++i) {
    std::size_t max_n = std::min(static_cast<std::size_t>(3), text.size() - i);
    for (std::size_t j = 0; j < max_n; ++j) {
      std::size_t h = 0;
      for (std::size_t k = 0; k <= j; ++k) {
        h = h * 31 + static_cast<unsigned char>(text[i + k]);
      }
      result[h % dim_] += 1.0f / static_cast<float>(j + 1);
    }
  }

  float norm = 0.0f;
  for (float v : result) norm += v * v;
  norm = std::sqrt(norm);

  if (norm > 0.0f) {
    for (auto& v : result) v /= norm;
  }
  return result;
}

std::size_t BruteForceEmbeddingProvider::dimension() const { return dim_; }

}  // namespace continuum::runtime
