#include <continuum/runtime/prefetch.hpp>

namespace continuum::runtime {

FutureCache::FutureCache(std::size_t max_entries, std::chrono::milliseconds ttl)
    : max_entries_(max_entries), ttl_(ttl) {}

std::optional<std::vector<std::uint8_t>> FutureCache::get(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end() || !it->second.ready || !it->second.valid) {
    return std::nullopt;
  }
  auto now = std::chrono::steady_clock::now();
  auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.created_at);
  if (age > ttl_) {
    return std::nullopt;
  }
  return it->second.output;
}

void FutureCache::put(const std::string& key, std::vector<std::uint8_t> output) {
  std::lock_guard<std::mutex> lock(mu_);
  auto& entry = entries_[key];
  entry.key = key;
  entry.output = std::move(output);
  entry.ready = true;
  entry.valid = true;
  entry.created_at = std::chrono::steady_clock::now();
  evict_expired();
  if (entries_.size() > max_entries_) {
    auto oldest = entries_.begin();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->second.created_at < oldest->second.created_at) {
        oldest = it;
      }
    }
    entries_.erase(oldest);
  }
}

bool FutureCache::has(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return false;
  }
  if (!it->second.ready || !it->second.valid) {
    return false;
  }
  auto now = std::chrono::steady_clock::now();
  auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.created_at);
  return age <= ttl_;
}

void FutureCache::invalidate(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.erase(key);
}

void FutureCache::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.clear();
}

std::size_t FutureCache::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.size();
}

void FutureCache::evict_expired() {
  auto now = std::chrono::steady_clock::now();
  for (auto it = entries_.begin(); it != entries_.end();) {
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.created_at);
    if (age > ttl_) {
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace continuum::runtime
