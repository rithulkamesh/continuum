#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace continuum::runtime {

struct PrefetchEntry {
  std::string key;
  std::vector<std::uint8_t> output;
  bool ready = false;
  bool valid = false;
  std::chrono::steady_clock::time_point created_at;
  std::int64_t compute_budget_ms = 0;
};

class FutureCache {
 public:
  explicit FutureCache(std::size_t max_entries = 256,
                       std::chrono::milliseconds ttl = std::chrono::milliseconds(30000));

  std::optional<std::vector<std::uint8_t>> get(const std::string& key) const;
  void put(const std::string& key, std::vector<std::uint8_t> output);
  bool has(const std::string& key) const;
  void invalidate(const std::string& key);
  void clear();
  std::size_t size() const;

 private:
  void evict_expired();

  mutable std::mutex mu_;
  std::unordered_map<std::string, PrefetchEntry> entries_;
  std::size_t max_entries_;
  std::chrono::milliseconds ttl_;
};

}  // namespace continuum::runtime
