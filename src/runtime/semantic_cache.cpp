#include "continuum/runtime/semantic_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>

namespace continuum::runtime {

SemanticCacheIndex::SemanticCacheIndex(std::size_t max_entries,
                                       float similarity_threshold)
    : max_entries_(max_entries), similarity_threshold_(similarity_threshold) {}

SemanticCacheIndex::LookupResult SemanticCacheIndex::lookup(
    const std::vector<float>& query_embedding,
    const std::string& model_id,
    const std::string& cache_namespace,
    const std::string& embedder_id,
    const std::string& query_prompt) const {
  struct Candidate {
    std::size_t index;
    float similarity;
    std::uint64_t stamp;
    std::string prompt;
  };
  LookupResult best;
  std::vector<Candidate> candidates;
  std::shared_ptr<const HitVerifier> verifier;
  {
    std::lock_guard<std::mutex> lock(mu_);
    verifier = verifier_;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      const auto& entry = entries_[i];
      if (entry.model_id != model_id) continue;
      if (entry.cache_namespace != cache_namespace) continue;
      if (entry.embedder_id != embedder_id) continue;
      if (entry.embedding.size() != query_embedding.size()) continue;
      const float sim = cosine_similarity(query_embedding, entry.embedding);
      best.similarity = std::max(best.similarity, sim);
      if (sim >= similarity_threshold_) {
        candidates.push_back({i, sim, static_cast<std::uint64_t>(entry.last_access_ns), entry.prompt});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.similarity > b.similarity; });
  // Verify outside the lock: a verifier may be slow (an LLM judge).
  for (const auto& cand : candidates) {
    const bool checkable = verifier != nullptr && !query_prompt.empty() && !cand.prompt.empty();
    if (checkable) {
      const std::string key = verifier->name() + '\x1f' + cand.prompt + '\x1f' + query_prompt;
      std::optional<bool> verdict;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = verdicts_.find(key);
        if (it != verdicts_.end()) verdict = it->second;
      }
      if (!verdict.has_value()) {
        verdict = verifier->verify(cand.prompt, query_prompt, cand.similarity);
        std::lock_guard<std::mutex> lock(mu_);
        if (verdicts_.size() >= kMaxVerdicts) verdicts_.clear();
        verdicts_[key] = *verdict;
      }
      if (!*verdict) {
        ++best.verifier_rejections;
        continue;
      }
    }
    std::lock_guard<std::mutex> lock(mu_);
    // The entry may have been evicted meanwhile; only serve it if unchanged.
    if (cand.index < entries_.size() &&
        static_cast<std::uint64_t>(entries_[cand.index].last_access_ns) == cand.stamp) {
      auto& entry = entries_[cand.index];
      best.above_threshold = true;
      best.similarity = cand.similarity;
      best.output = entry.cached_output;
      best.prompt = entry.prompt;
      entry.last_access_ns = static_cast<std::int64_t>(++clock_);
    }
    break;
  }
  if (best.verifier_rejections > 0) {
    std::lock_guard<std::mutex> lock(mu_);
    verifier_rejections_ += best.verifier_rejections;
  }
  return best;
}

void SemanticCacheIndex::set_verifier(std::shared_ptr<const HitVerifier> verifier) {
  std::lock_guard<std::mutex> lock(mu_);
  verifier_ = std::move(verifier);
  verdicts_.clear();
}

std::shared_ptr<const HitVerifier> SemanticCacheIndex::verifier() const {
  std::lock_guard<std::mutex> lock(mu_);
  return verifier_;
}

std::int64_t SemanticCacheIndex::verifier_rejections() const {
  std::lock_guard<std::mutex> lock(mu_);
  return verifier_rejections_;
}

void SemanticCacheIndex::insert(const std::vector<float>& embedding,
                                const std::string& model_id,
                                std::vector<std::uint8_t> output,
                                const std::string& cache_namespace,
                                const std::string& embedder_id,
                                const std::string& prompt) {
  std::lock_guard<std::mutex> lock(mu_);
  if (max_entries_ == 0) {
    return;
  }

  if (entries_.size() >= max_entries_) {
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
  entry.last_access_ns = static_cast<std::int64_t>(++clock_);
  entry.cache_namespace = cache_namespace;
  entry.embedder_id = embedder_id;
  entry.prompt = prompt;
  entries_.push_back(std::move(entry));
}

void SemanticCacheIndex::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.clear();
  clock_ = 0;
  verifier_rejections_ = 0;
  verdicts_.clear();
}

std::size_t SemanticCacheIndex::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.size();
}

std::size_t SemanticCacheIndex::estimated_bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::size_t total = 0;
  for (const auto& entry : entries_) {
    total += sizeof(SemanticCacheEntry);
    total += entry.embedding.size() * sizeof(float);
    total += entry.cached_output.size() + entry.model_id.size() + entry.cache_namespace.size() +
             entry.embedder_id.size() + entry.prompt.size();
  }
  return total;
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

std::string BruteForceEmbeddingProvider::identity() const {
  return "continuum/char-ngram-v1:" + std::to_string(dim_);
}

}  // namespace continuum::runtime
