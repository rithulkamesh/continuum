#include <continuum/runtime/cache.hpp>

#include <algorithm>
#include <cstring>
#include <functional>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

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
  insert_unlocked(std::move(entry), token_prefix);
}

void KVCacheIndex::insert_unlocked(CacheEntry entry, const std::vector<std::int32_t>& token_prefix) {
  if (token_prefix.empty() || entry.prefix_len <= 0) {
    return;
  }
  std::vector<std::int32_t> prefix = token_prefix;
  entry.prefix_len = std::min<std::int32_t>(entry.prefix_len, static_cast<std::int32_t>(prefix.size()));
  prefix.resize(static_cast<std::size_t>(entry.prefix_len));
  entry.prefix_hash = hash_prefix(entry.model_id, entry.decode, prefix);
  entry.last_used_ns = static_cast<std::int64_t>(++logical_clock_);

  TrieNode* cur = &root_;
  std::size_t depth = 0;
  for (const auto token : prefix) {
    TrieNode* next = FindChild(cur, token);
    if (next == nullptr) {
      cur->children.push_back(TrieNode{token, {}, {}});
      next = &cur->children.back();
    }
    cur = next;
    ++depth;
    CacheEntry partial = entry;
    partial.prefix_len = static_cast<std::int32_t>(depth);
    partial.prefix_hash = hash_prefix(partial.model_id, partial.decode,
        std::vector<std::int32_t>(prefix.begin(), prefix.begin() + depth));
    bool already = false;
    for (const auto& e : cur->entries) {
      if (e.model_id == partial.model_id && SameDecode(e.decode, partial.decode) &&
          e.prefix_hash == partial.prefix_hash && e.prefix_len == partial.prefix_len) {
        already = true;
        break;
      }
    }
    if (!already) {
      cur->entries.push_back(partial);
      ++size_;
    }
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

namespace {

constexpr const char* kMetadataMagic = "CPKV";
constexpr std::uint16_t kMetadataVersion = 1;

struct PersistedEntry {
  std::string model_id;
  DecodeParams decode;
  std::int32_t prefix_len = 0;
  std::uint64_t prefix_hash = 0;
  std::vector<std::int32_t> prefix_tokens;
  std::int64_t last_used_ns = 0;
};

void WriteU16(std::ostream& os, std::uint16_t v) {
  std::uint8_t buf[2];
  buf[0] = static_cast<std::uint8_t>(v & 0xFF);
  buf[1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
  os.write(reinterpret_cast<const char*>(buf), 2);
}

void WriteU32(std::ostream& os, std::uint32_t v) {
  std::uint8_t buf[4];
  for (int i = 0; i < 4; ++i) buf[i] = static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF);
  os.write(reinterpret_cast<const char*>(buf), 4);
}

void WriteU64(std::ostream& os, std::uint64_t v) {
  std::uint8_t buf[8];
  for (int i = 0; i < 8; ++i) buf[i] = static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF);
  os.write(reinterpret_cast<const char*>(buf), 8);
}

void WriteI32(std::ostream& os, std::int32_t v) {
  WriteU32(os, static_cast<std::uint32_t>(v));
}

void WriteI64(std::ostream& os, std::int64_t v) {
  WriteU64(os, static_cast<std::uint64_t>(v));
}

void WriteFloat(std::ostream& os, float v) {
  static_assert(sizeof(float) == 4, "expected 4-byte float");
  std::uint32_t bits;
  std::memcpy(&bits, &v, 4);
  WriteU32(os, bits);
}

void WriteStr(std::ostream& os, const std::string& s) {
  WriteU16(os, static_cast<std::uint16_t>(s.size()));
  if (!s.empty()) os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

bool ReadU16(std::istream& is, std::uint16_t& v) {
  std::uint8_t buf[2] = {};
  if (!is.read(reinterpret_cast<char*>(buf), 2)) return false;
  v = static_cast<std::uint16_t>(buf[0]) | (static_cast<std::uint16_t>(buf[1]) << 8);
  return true;
}

bool ReadU32(std::istream& is, std::uint32_t& v) {
  std::uint8_t buf[4] = {};
  if (!is.read(reinterpret_cast<char*>(buf), 4)) return false;
  v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(buf[i]) << (i * 8);
  return true;
}

bool ReadU64(std::istream& is, std::uint64_t& v) {
  std::uint8_t buf[8] = {};
  if (!is.read(reinterpret_cast<char*>(buf), 8)) return false;
  v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(buf[i]) << (i * 8);
  return true;
}

bool ReadI32(std::istream& is, std::int32_t& v) {
  std::uint32_t u;
  if (!ReadU32(is, u)) return false;
  v = static_cast<std::int32_t>(u);
  return true;
}

bool ReadI64(std::istream& is, std::int64_t& v) {
  std::uint64_t u;
  if (!ReadU64(is, u)) return false;
  v = static_cast<std::int64_t>(u);
  return true;
}

bool ReadFloat(std::istream& is, float& v) {
  std::uint32_t bits;
  if (!ReadU32(is, bits)) return false;
  std::memcpy(&v, &bits, 4);
  return true;
}

bool ReadStr(std::istream& is, std::string& s) {
  std::uint16_t len;
  if (!ReadU16(is, len)) return false;
  s.resize(len);
  if (len > 0 && !is.read(&s[0], len)) return false;
  return true;
}

void CollectPersistedEntries(const KVCacheIndex::TrieNode& node,
                             const std::vector<std::int32_t>& path_tokens,
                             std::vector<PersistedEntry>& out) {
  std::vector<std::int32_t> current_path = path_tokens;
  for (const auto& entry : node.entries) {
    PersistedEntry pe;
    pe.model_id = entry.model_id;
    pe.decode = entry.decode;
    pe.prefix_len = entry.prefix_len;
    pe.prefix_hash = entry.prefix_hash;
    pe.last_used_ns = entry.last_used_ns;
    pe.prefix_tokens = current_path;
    out.push_back(std::move(pe));
  }
  for (const auto& child : node.children) {
    current_path.push_back(child.token);
    CollectPersistedEntries(child, current_path, out);
    current_path.pop_back();
  }
}

}  // namespace

bool KVCacheIndex::save_metadata(const std::string& path) const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<PersistedEntry> entries;
  CollectPersistedEntries(root_, {}, entries);

  std::ofstream os(path, std::ios::binary);
  if (!os) return false;

  os.write(kMetadataMagic, 4);
  WriteU16(os, kMetadataVersion);
  WriteU32(os, static_cast<std::uint32_t>(entries.size()));

  for (const auto& e : entries) {
    WriteStr(os, e.model_id);
    WriteStr(os, e.decode.op_name);
    WriteFloat(os, e.decode.temperature);
    WriteI32(os, e.decode.max_tokens);
    WriteI32(os, e.prefix_len);
    WriteU64(os, e.prefix_hash);
    WriteI64(os, e.last_used_ns);
    WriteU32(os, static_cast<std::uint32_t>(e.prefix_tokens.size()));
    for (const auto t : e.prefix_tokens) {
      WriteI32(os, t);
    }
  }

  return os.good();
}

bool KVCacheIndex::load_metadata(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) return false;

  char magic[4] = {};
  if (!is.read(magic, 4)) return false;
  if (std::memcmp(magic, kMetadataMagic, 4) != 0) return false;

  std::uint16_t version;
  if (!ReadU16(is, version) || version != kMetadataVersion) return false;

  std::uint32_t count;
  if (!ReadU32(is, count)) return false;

  std::vector<PersistedEntry> loaded;
  loaded.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    PersistedEntry pe;
    if (!ReadStr(is, pe.model_id)) return false;
    if (!ReadStr(is, pe.decode.op_name)) return false;
    if (!ReadFloat(is, pe.decode.temperature)) return false;
    if (!ReadI32(is, pe.decode.max_tokens)) return false;
    if (!ReadI32(is, pe.prefix_len)) return false;
    if (!ReadU64(is, pe.prefix_hash)) return false;
    if (!ReadI64(is, pe.last_used_ns)) return false;

    std::uint32_t token_count;
    if (!ReadU32(is, token_count)) return false;
    pe.prefix_tokens.resize(token_count);
    for (std::uint32_t j = 0; j < token_count; ++j) {
      if (!ReadI32(is, pe.prefix_tokens[j])) return false;
    }

    loaded.push_back(std::move(pe));
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& pe : loaded) {
      CacheEntry entry;
      entry.model_id = pe.model_id;
      entry.decode = pe.decode;
      entry.prefix_len = pe.prefix_len;
      entry.prefix_hash = pe.prefix_hash;
      entry.last_used_ns = pe.last_used_ns;
      entry.backend_state = {};

      insert_unlocked(std::move(entry), pe.prefix_tokens);
    }
  }

  return true;
}

}  // namespace continuum::runtime
