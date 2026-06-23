#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace continuum::runtime {

struct EmbeddingProvider {
  virtual ~EmbeddingProvider() = default;
  virtual std::vector<float> embed(const std::string& text) const = 0;
  virtual std::size_t dimension() const = 0;
};

struct SemanticCacheEntry {
  std::vector<float> embedding;
  std::vector<std::uint8_t> cached_output;
  std::string model_id;
  std::int64_t last_access_ns = 0;
};

class SemanticCacheIndex {
 public:
  explicit SemanticCacheIndex(std::size_t max_entries = 2048,
                              float similarity_threshold = 0.85f);

  struct LookupResult {
    std::vector<std::uint8_t> output;
    float similarity = 0.0f;
    bool above_threshold = false;
  };

  LookupResult lookup(const std::vector<float>& query_embedding,
                      const std::string& model_id) const;

  void insert(const std::vector<float>& embedding,
              const std::string& model_id,
              std::vector<std::uint8_t> output);

  void clear();
  std::size_t size() const;
  float similarity_threshold() const;
  void set_similarity_threshold(float t);

  static float cosine_similarity(const std::vector<float>& a,
                                  const std::vector<float>& b);

 private:
  mutable std::mutex mu_;
  std::vector<SemanticCacheEntry> entries_;
  std::size_t max_entries_;
  float similarity_threshold_;
  std::uint64_t clock_ = 0;
};

class BruteForceEmbeddingProvider : public EmbeddingProvider {
 public:
  explicit BruteForceEmbeddingProvider(std::size_t dim = 64);

  std::vector<float> embed(const std::string& text) const override;
  std::size_t dimension() const override;

 private:
  std::size_t dim_;
};

}  // namespace continuum::runtime
