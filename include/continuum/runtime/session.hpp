#pragma once

#include <continuum/backend/backend.hpp>
#include <continuum/ir/graph.hpp>
#include <continuum/runtime/cache.hpp>
#include <continuum/runtime/interpreter.hpp>
#include <continuum/runtime/scheduler.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace continuum::runtime {

enum class ReusePolicyKind {
  Always,
  Never,
  ThresholdPrefixLen,
};

struct ReusePolicy {
  ReusePolicyKind kind = ReusePolicyKind::Always;
  std::int32_t min_prefix_len = 0;

  static ReusePolicy always() { return {ReusePolicyKind::Always, 0}; }
  static ReusePolicy never() { return {ReusePolicyKind::Never, 0}; }
  static ReusePolicy threshold(std::int32_t min_len) {
    return {ReusePolicyKind::ThresholdPrefixLen, min_len};
  }

  bool should_attempt(const std::vector<std::int32_t>& /*input_tokens*/,
                      std::int32_t potential_prefix_len) const {
    if (kind == ReusePolicyKind::Never) return false;
    if (kind == ReusePolicyKind::ThresholdPrefixLen) {
      return potential_prefix_len >= min_prefix_len;
    }
    return true;
  }
};

struct ReuseStepRecord {
  std::string node_name;
  bool cache_hit = false;
  std::int32_t prefix_hit_len = 0;
  std::int32_t total_tokens = 0;
  std::int32_t tokens_saved = 0;
  std::int32_t tokens_sent = 0;
  std::int32_t compute_steps = 0;
};

struct ReuseMetrics {
  std::string session_id;
  std::vector<ReuseStepRecord> steps;
  std::int64_t total_lookups = 0;
  std::int64_t total_hits = 0;
  std::int64_t total_tokens_saved = 0;
  std::int64_t total_tokens_processed = 0;
  std::int64_t run_count = 0;

  double hit_rate() const {
    return total_lookups > 0 ? static_cast<double>(total_hits) / static_cast<double>(total_lookups) : 0.0;
  }
  double token_reduction_ratio() const {
    const auto total = total_tokens_saved + total_tokens_processed;
    return total > 0 ? static_cast<double>(total_tokens_saved) / static_cast<double>(total) : 0.0;
  }
  void reset() {
    steps.clear();
    total_lookups = 0;
    total_hits = 0;
    total_tokens_saved = 0;
    total_tokens_processed = 0;
    run_count = 0;
  }
};

class Session {
 public:
  explicit Session(const std::string& id, backend::BackendRegistry& backends,
                   std::size_t max_cache_entries = 8192);
  explicit Session(const std::string& id, backend::BackendRegistry& backends,
                   KVCacheIndex& external_cache);

  std::vector<continuum::Value> run(
      const ir::Graph& g, const std::unordered_map<ir::NodeId, continuum::Value>& inputs);

  const ReusePolicy& policy() const { return policy_; }
  void set_policy(const ReusePolicy& policy) { policy_ = policy; }

  const ReuseMetrics& metrics() const { return metrics_; }
  void reset_metrics() { metrics_.reset(); }

  KVCacheIndex& cache() { return cache_; }
  const KVCacheIndex& cache() const { return cache_; }

  const std::string& id() const { return session_id_; }
  std::int64_t run_count() const { return metrics_.run_count; }
  std::size_t cache_size() const { return cache_.size(); }

  bool save_cache_metadata(const std::string& path) const;
  bool load_cache_metadata(const std::string& path);

 private:
  std::string session_id_;
  backend::BackendRegistry& backends_;
  std::unique_ptr<KVCacheIndex> owned_cache_;
  KVCacheIndex& cache_;
  ReusePolicy policy_;
  ReuseMetrics metrics_;
};

}  // namespace continuum::runtime
