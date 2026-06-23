#pragma once

#include <continuum/ir/node.hpp>
#include <continuum/ir/value.hpp>

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace continuum::runtime {

struct MemoKey {
  std::string node_kind_str;
  std::string payload_hash;
  std::vector<std::uint8_t> inputs_hash;

  bool operator==(const MemoKey& o) const {
    return node_kind_str == o.node_kind_str &&
           payload_hash == o.payload_hash &&
           inputs_hash == o.inputs_hash;
  }
};

struct MemoEntry {
  std::vector<std::uint8_t> output_bytes;
  std::uint64_t version = 0;
  std::int64_t access_count = 0;
  std::int64_t last_access_ns = 0;
};

struct MemoKeyHash {
  std::size_t operator()(const MemoKey& k) const {
    std::size_t h = std::hash<std::string>{}(k.node_kind_str);
    h ^= std::hash<std::string>{}(k.payload_hash) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    for (auto b : k.inputs_hash) {
      h ^= static_cast<std::size_t>(b) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
  }
};

class MemoTable {
 public:
  explicit MemoTable(std::size_t max_entries = 4096,
                     std::size_t version = 0);

  std::optional<MemoEntry> lookup(const MemoKey& key) const;
  void insert(MemoKey key, MemoEntry entry);

  void invalidate_version(std::size_t version);
  void invalidate_node(const std::string& node_kind_str);
  void clear();

  std::size_t size() const;
  std::size_t version() const;
  void set_version(std::size_t v);

  MemoKey make_key(const ir::Node& node, const std::vector<continuum::Value>& inputs) const;

  static std::vector<std::uint8_t> serialize_value(const continuum::Value& v);
  static std::optional<continuum::Value> deserialize_value(const std::vector<std::uint8_t>& bytes);

 private:
  mutable std::mutex mu_;
  std::unordered_map<MemoKey, MemoEntry, MemoKeyHash> table_;
  std::size_t max_entries_;
  std::size_t version_;
  mutable std::uint64_t clock_ = 0;
};

}  // namespace continuum::runtime
