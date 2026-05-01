#pragma once

#include <cstdint>
#include <continuum/backend/backend.hpp>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace continuum::runtime {

struct DecodeParams {
  std::string op_name;
  float temperature = 1.0f;
  std::int32_t max_tokens = 0;
};

struct CacheEntry {
  std::uint64_t prefix_hash = 0;
  std::string model_id;
  DecodeParams decode{};
  std::int32_t prefix_len = 0;
  continuum::backend::BackendState backend_state{};
  std::int64_t last_used_ns = 0;
};

class KVCacheIndex {
 public:
  struct TrieNode {
    std::int32_t token = -1;
    std::vector<CacheEntry> entries;
    std::vector<TrieNode> children;
  };

  explicit KVCacheIndex(std::size_t max_entries = 8192);
  std::optional<std::pair<CacheEntry, std::int32_t>> longest_prefix(
      const std::string& model_id, const DecodeParams& decode, const std::vector<std::int32_t>& tokens) const;
  void insert(CacheEntry entry, const std::vector<std::int32_t>& token_prefix);
  void insert_unlocked(CacheEntry entry, const std::vector<std::int32_t>& token_prefix);
  void invalidate(void* backend_handle);
  void clear();
  std::size_t size() const;

  bool save_metadata(const std::string& path) const;
  bool load_metadata(const std::string& path);

 private:
  mutable std::mutex mu_;
  TrieNode root_;
  std::uint64_t logical_clock_ = 0;
  std::size_t size_ = 0;
  std::size_t max_entries_;
};

}  // namespace continuum::runtime
