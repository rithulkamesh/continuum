#include <continuum/runtime/layer_cache.hpp>

#include <algorithm>

namespace continuum::runtime {
namespace {

std::uint64_t HashTokens(const std::vector<std::int32_t>& tokens) {
  std::uint64_t h = 1469598103934665603ULL;  // FNV-1a
  for (std::int32_t t : tokens) {
    h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(t));
    h *= 1099511628211ULL;
  }
  return h;
}

}  // namespace

LayerKVCacheIndex::LayerKVCacheIndex(std::size_t max_entries,
                                     std::size_t max_bytes)
    : max_entries_(max_entries), max_bytes_(max_bytes) {}

LayerKVCacheIndex::LookupResult LayerKVCacheIndex::find_deepest(
    const std::string& model_id,
    const std::string& decode_hash,
    const std::vector<std::int32_t>& tokens,
    std::int32_t total_layers,
    std::uint64_t arch_version) const {
  std::lock_guard<std::mutex> lock(mu_);

  LookupResult best;
  for (const auto& [key, cp] : entries_) {
    if (key.model_id != model_id || key.decode_hash != decode_hash ||
        key.arch_version != arch_version) {
      continue;
    }
    if (key.prefix_len > static_cast<std::int32_t>(tokens.size())) {
      continue;
    }
    if (key.layer_id > total_layers) {
      continue;
    }
    // A cached state is only valid if it was computed over a prefix of the
    // query tokens; matching on length alone would reuse another prompt's state.
    if (!std::equal(cp.tokens.begin(), cp.tokens.end(), tokens.begin())) {
      continue;
    }
    if (cp.layer_id > best.layer_id ||
        (cp.layer_id == best.layer_id && cp.prefix_len > best.prefix_len)) {
      best.state = cp.state;
      best.layer_id = cp.layer_id;
      best.prefix_len = cp.prefix_len;
      best.found = true;
    }
  }
  return best;
}

void LayerKVCacheIndex::insert(LayerCheckpoint checkpoint) {
  std::lock_guard<std::mutex> lock(mu_);

  LayerCheckpointKey key;
  key.model_id = checkpoint.model_id;
  key.decode_hash = checkpoint.decode_hash;
  key.layer_id = checkpoint.layer_id;
  key.prefix_len = checkpoint.prefix_len;
  key.arch_version = checkpoint.arch_version;
  key.token_hash = HashTokens(checkpoint.tokens);

  checkpoint.last_access_ns = ++clock_;

  auto it = entries_.find(key);
  if (it != entries_.end()) {
    current_bytes_ -= it->second.estimated_bytes;
  }
  current_bytes_ += checkpoint.estimated_bytes;

  entries_[key] = std::move(checkpoint);

  evict_if_needed();
}

void LayerKVCacheIndex::invalidate_model(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mu_);

  for (auto it = entries_.begin(); it != entries_.end();) {
    if (it->first.model_id == model_id) {
      current_bytes_ -= it->second.estimated_bytes;
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
}

void LayerKVCacheIndex::invalidate_arch(std::uint64_t arch_version) {
  std::lock_guard<std::mutex> lock(mu_);

  for (auto it = entries_.begin(); it != entries_.end();) {
    if (it->first.arch_version == arch_version) {
      current_bytes_ -= it->second.estimated_bytes;
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
}

void LayerKVCacheIndex::clear() {
  std::lock_guard<std::mutex> lock(mu_);

  entries_.clear();
  current_bytes_ = 0;
  clock_ = 0;
}

std::size_t LayerKVCacheIndex::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.size();
}

std::size_t LayerKVCacheIndex::estimated_bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  return current_bytes_;
}

void LayerKVCacheIndex::evict_if_needed() {
  while (entries_.size() > max_entries_ || current_bytes_ > max_bytes_) {
    if (entries_.empty()) {
      break;
    }

    auto victim = entries_.begin();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->second.last_access_ns < victim->second.last_access_ns) {
        victim = it;
      }
    }

    current_bytes_ -= victim->second.estimated_bytes;
    entries_.erase(victim);
  }
}

}  // namespace continuum::runtime
