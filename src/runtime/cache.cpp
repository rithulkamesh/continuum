#include <continuum/runtime/cache.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>

namespace continuum::runtime {

namespace {
std::uint64_t hash_prefix(const std::string& model_id, const DecodeParams& decode, const std::vector<std::int32_t>& tokens) {
  std::uint64_t h = std::hash<std::string>{}(model_id);
  h ^= std::hash<std::string>{}(decode.op_name) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= std::hash<std::int32_t>{}(decode.max_tokens) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= std::hash<int>{}(static_cast<int>(decode.temperature * 1000.0f)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  for (const auto t : tokens) {
    h ^= static_cast<std::uint64_t>(t) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  }
  return h;
}

bool SameDecode(const DecodeParams& a, const DecodeParams& b) {
  return a.op_name == b.op_name && a.max_tokens == b.max_tokens &&
         static_cast<int>(a.temperature * 1000.0f) == static_cast<int>(b.temperature * 1000.0f);
}

KVCacheIndex::TrieNode* FindChild(KVCacheIndex::TrieNode* node, std::int32_t token) {
  for (auto& child : node->children) {
    if (child.token == token) {
      return &child;
    }
  }
  return nullptr;
}

const KVCacheIndex::TrieNode* FindChild(const KVCacheIndex::TrieNode* node, std::int32_t token) {
  for (const auto& child : node->children) {
    if (child.token == token) {
      return &child;
    }
  }
  return nullptr;
}

std::vector<CacheEntry*> CollectEntries(KVCacheIndex::TrieNode* node) {
  std::vector<CacheEntry*> out;
  std::function<void(KVCacheIndex::TrieNode*)> visit = [&](KVCacheIndex::TrieNode* cur) {
    for (auto& entry : cur->entries) {
      out.push_back(&entry);
    }
    for (auto& child : cur->children) {
      visit(&child);
    }
  };
  visit(node);
  return out;
}

void CompactEmptyBranches(KVCacheIndex::TrieNode* node) {
  for (auto& child : node->children) {
    CompactEmptyBranches(&child);
  }
  node->children.erase(
      std::remove_if(
          node->children.begin(), node->children.end(),
          [](const KVCacheIndex::TrieNode& n) { return n.children.empty() && n.entries.empty(); }),
      node->children.end());
}
}  // namespace

KVCacheIndex::KVCacheIndex(std::size_t max_entries) : max_entries_(max_entries) {}

std::optional<std::pair<CacheEntry, std::int32_t>> KVCacheIndex::longest_prefix(
    const std::string& model_id, const DecodeParams& decode, const std::vector<std::int32_t>& tokens) const {
  std::lock_guard<std::mutex> lock(mu_);
  const TrieNode* cur = &root_;
  const CacheEntry* best = nullptr;
  std::int32_t best_len = 0;

  for (std::size_t i = 0; i < tokens.size(); ++i) {
    cur = FindChild(cur, tokens[i]);
    if (cur == nullptr) {
      break;
    }
    for (const auto& entry : cur->entries) {
      if (entry.model_id == model_id && SameDecode(entry.decode, decode) && entry.prefix_len > best_len) {
        best = &entry;
        best_len = entry.prefix_len;
      }
    }
  }
  if (best == nullptr) {
    return std::nullopt;
  }

  // Bump recency for LRU. longest_prefix is logically a read, but cache recency is metadata.
  KVCacheIndex* self = const_cast<KVCacheIndex*>(this);
  ++self->logical_clock_;
  TrieNode* mut = &self->root_;
  for (std::size_t i = 0; i < static_cast<std::size_t>(best_len) && i < tokens.size(); ++i) {
    mut = FindChild(mut, tokens[i]);
    if (mut == nullptr) {
      throw std::runtime_error("cache trie corrupted");
    }
  }
  for (auto& entry : mut->entries) {
    if (entry.model_id == model_id && SameDecode(entry.decode, decode) && entry.prefix_len == best_len &&
        entry.backend_state.handle == best->backend_state.handle) {
      entry.last_used_ns = static_cast<std::int64_t>(self->logical_clock_);
      return std::make_pair(entry, best_len);
    }
  }
  return std::make_pair(*best, best_len);
}

void KVCacheIndex::insert(CacheEntry entry, const std::vector<std::int32_t>& token_prefix) {
  std::lock_guard<std::mutex> lock(mu_);
  if (token_prefix.empty() || entry.prefix_len <= 0) {
    return;
  }
  std::vector<std::int32_t> prefix = token_prefix;
  entry.prefix_len = std::min<std::int32_t>(entry.prefix_len, static_cast<std::int32_t>(prefix.size()));
  prefix.resize(static_cast<std::size_t>(entry.prefix_len));
  entry.prefix_hash = hash_prefix(entry.model_id, entry.decode, prefix);
  entry.last_used_ns = static_cast<std::int64_t>(++logical_clock_);

  TrieNode* cur = &root_;
  for (const auto token : prefix) {
    TrieNode* next = FindChild(cur, token);
    if (next == nullptr) {
      cur->children.push_back(TrieNode{token, {}, {}});
      next = &cur->children.back();
    }
    cur = next;
  }

  auto existing = std::find_if(cur->entries.begin(), cur->entries.end(), [&](const CacheEntry& e) {
    return e.model_id == entry.model_id && SameDecode(e.decode, entry.decode) && e.prefix_hash == entry.prefix_hash;
  });
  if (existing != cur->entries.end()) {
    *existing = std::move(entry);
  } else {
    cur->entries.push_back(std::move(entry));
    ++size_;
  }

  while (size_ > max_entries_) {
    auto all = CollectEntries(&root_);
    if (all.empty()) {
      size_ = 0;
      break;
    }
    auto victim_it = std::min_element(all.begin(), all.end(), [](const CacheEntry* a, const CacheEntry* b) {
      return a->last_used_ns < b->last_used_ns;
    });
    if (victim_it == all.end()) {
      break;
    }
    const auto victim_hash = (*victim_it)->prefix_hash;
    const auto victim_model = (*victim_it)->model_id;

    std::function<bool(TrieNode*)> remove_once = [&](TrieNode* node) {
      auto e_it = std::find_if(node->entries.begin(), node->entries.end(), [&](const CacheEntry& e) {
        return e.prefix_hash == victim_hash && e.model_id == victim_model;
      });
      if (e_it != node->entries.end()) {
        node->entries.erase(e_it);
        --size_;
        return true;
      }
      for (auto& child : node->children) {
        if (remove_once(&child)) {
          return true;
        }
      }
      return false;
    };
    remove_once(&root_);
    CompactEmptyBranches(&root_);
  }
}

void KVCacheIndex::invalidate(void* backend_handle) {
  std::lock_guard<std::mutex> lock(mu_);
  std::function<void(TrieNode*)> erase_handle = [&](TrieNode* node) {
    const auto before = node->entries.size();
    node->entries.erase(
        std::remove_if(node->entries.begin(), node->entries.end(),
                       [&](const CacheEntry& entry) { return entry.backend_state.handle == backend_handle; }),
        node->entries.end());
    size_ -= (before - node->entries.size());
    for (auto& child : node->children) {
      erase_handle(&child);
    }
  };
  erase_handle(&root_);
  CompactEmptyBranches(&root_);
}

void KVCacheIndex::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  root_ = TrieNode{};
  logical_clock_ = 0;
  size_ = 0;
}

std::size_t KVCacheIndex::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return size_;
}

}  // namespace continuum::runtime
