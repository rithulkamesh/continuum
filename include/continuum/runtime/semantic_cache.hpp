#pragma once

#include <continuum/runtime/hit_verifier.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace continuum::runtime {

/// Source of the vectors the semantic tier compares.
///
/// Implement it in C++ or subclass `continuum._native.EmbeddingProvider` in
/// Python (a local model, a hosted endpoint, precomputed vectors, ...).
struct EmbeddingProvider {
  virtual ~EmbeddingProvider() = default;
  /// Embed \p text; the result must have `dimension()` elements.
  virtual std::vector<float> embed(const std::string& text) const = 0;
  virtual std::size_t dimension() const = 0;
  /// Stable identity of the embedding space (model name + version + any
  /// setting that changes vectors). It is part of the semantic cache key, so
  /// vectors from different embedders never compare against each other.
  virtual std::string identity() const = 0;
};

struct SemanticCacheEntry {
  std::vector<float> embedding;
  std::vector<std::uint8_t> cached_output;
  std::string model_id;
  std::int64_t last_access_ns = 0;
  std::string cache_namespace;
  /// `EmbeddingProvider::identity()` of the embedder that produced `embedding`.
  std::string embedder_id;
  /// Prompt text the entry was cached for, checked by the HitVerifier.
  std::string prompt;
};

/// Paraphrase-tolerant tier keyed by embedding similarity.
///
/// Eviction: least-recently-used. `insert` stamps an entry and a lookup that
/// clears the threshold refreshes the matched entry; when `size()` reaches
/// `max_entries` the least recently used entry is dropped before inserting.
///
/// Verification: candidates that clear the threshold are checked, best
/// first, by a HitVerifier (default: LexicalNearMissVerifier) against the
/// prompt they were cached for; the first one it accepts is served. The check
/// needs both prompt texts, so it applies when `insert` and `lookup` are given
/// them (the interpreter always does). `set_verifier(nullptr)` disables it.
class SemanticCacheIndex {
 public:
  explicit SemanticCacheIndex(std::size_t max_entries = 2048,
                              float similarity_threshold = 0.85f);

  struct LookupResult {
    std::vector<std::uint8_t> output;
    float similarity = 0.0f;
    bool above_threshold = false;  ///< true when an answer is served
    std::string prompt;            ///< prompt of the served entry
    std::int32_t verifier_rejections = 0;  ///< candidates the verifier turned down
  };

  /// Best entry for the same model, namespace, and embedder identity.
  LookupResult lookup(const std::vector<float>& query_embedding,
                      const std::string& model_id,
                      const std::string& cache_namespace = {},
                      const std::string& embedder_id = {},
                      const std::string& query_prompt = {}) const;

  void insert(const std::vector<float>& embedding,
              const std::string& model_id,
              std::vector<std::uint8_t> output,
              const std::string& cache_namespace = {},
              const std::string& embedder_id = {},
              const std::string& prompt = {});

  /// Replace the hit verifier; nullptr serves any candidate above threshold.
  void set_verifier(std::shared_ptr<const HitVerifier> verifier);
  std::shared_ptr<const HitVerifier> verifier() const;
  /// Candidates turned down by the verifier since construction / clear().
  std::int64_t verifier_rejections() const;

  void clear();
  std::size_t size() const;
  /// Capacity in entries passed at construction.
  std::size_t max_entries() const { return max_entries_; }
  /// Approximate resident bytes: embeddings, outputs, ids, and per-entry overhead.
  std::size_t estimated_bytes() const;
  float similarity_threshold() const;
  void set_similarity_threshold(float t);

  static float cosine_similarity(const std::vector<float>& a,
                                  const std::vector<float>& b);

 private:
  mutable std::mutex mu_;
  // Mutable so a const lookup can refresh LRU recency on a hit.
  mutable std::vector<SemanticCacheEntry> entries_;
  std::size_t max_entries_;
  float similarity_threshold_;
  mutable std::uint64_t clock_ = 0;
  std::shared_ptr<const HitVerifier> verifier_ = std::make_shared<LexicalNearMissVerifier>();
  mutable std::int64_t verifier_rejections_ = 0;
  // Verdicts per (cached prompt, query prompt): a Session's metrics pass and
  // its interpreter look up the same prompt, and a verifier may be costly.
  mutable std::unordered_map<std::string, bool> verdicts_;
  static constexpr std::size_t kMaxVerdicts = 4096;
};

class BruteForceEmbeddingProvider : public EmbeddingProvider {
 public:
  explicit BruteForceEmbeddingProvider(std::size_t dim = 64);

  std::vector<float> embed(const std::string& text) const override;
  std::size_t dimension() const override;
  /// `"continuum/char-ngram-v1:<dim>"`.
  std::string identity() const override;

 private:
  std::size_t dim_;
};

}  // namespace continuum::runtime
