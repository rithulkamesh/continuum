#include <continuum/runtime/checkpoint.hpp>

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace continuum::runtime {
namespace {

constexpr std::uint32_t kCheckpointMagic = 0x31545043U;  // "CPT1"
constexpr std::uint32_t kDeltaMagic = 0x31445043U;       // "CPD1"
constexpr std::uint16_t kDeltaVersion = 1;
// v1: no cache snapshot. v2: cache snapshot without namespace. v3: per-entry namespace.
constexpr std::uint16_t kCheckpointVersion = 3;
constexpr std::uint16_t kCheckpointVersionNoNamespace = 2;
constexpr std::uint16_t kCheckpointVersionNoSnapshot = 1;
constexpr std::uint8_t kValueTagTensor = 0;
constexpr std::uint8_t kValueTagTokens = 1;
constexpr std::uint8_t kValueTagSchema = 2;
constexpr std::uint8_t kValueTagString = 3;
constexpr std::uint8_t kValueTagDouble = 4;
constexpr std::uint8_t kValueTagInt64 = 5;
constexpr std::uint8_t kValueTagMlxTensor = 6;

template <typename T>
void WritePrimitive(std::vector<std::uint8_t>& out, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "primitive write requires trivially copyable type");
  const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), begin, begin + sizeof(T));
}

template <typename T>
T ReadPrimitive(const std::uint8_t*& cur, const std::uint8_t* end) {
  if (static_cast<std::size_t>(end - cur) < sizeof(T)) {
    throw std::runtime_error("checkpoint deserialize: unexpected eof");
  }
  T out{};
  std::memcpy(&out, cur, sizeof(T));
  cur += sizeof(T);
  return out;
}

void WriteBlob(std::vector<std::uint8_t>& out, const std::uint8_t* data, std::size_t len) {
  out.insert(out.end(), data, data + len);
}

void WriteString(std::vector<std::uint8_t>& out, const std::string& s) {
  WritePrimitive(out, static_cast<std::uint64_t>(s.size()));
  WriteBlob(out, reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
}

std::string ReadString(const std::uint8_t*& cur, const std::uint8_t* end) {
  const auto len = ReadPrimitive<std::uint64_t>(cur, end);
  if (static_cast<std::size_t>(end - cur) < len) {
    throw std::runtime_error("checkpoint deserialize: string truncated");
  }
  std::string s(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(len));
  cur += len;
  return s;
}

void WriteCacheEntry(std::vector<std::uint8_t>& out, const CheckpointCacheEntry& e) {
  WriteString(out, e.model_id);
  WriteString(out, e.decode.op_name);
  WritePrimitive(out, e.decode.temperature);
  WritePrimitive(out, e.decode.max_tokens);
  WritePrimitive(out, e.prefix_len);
  WritePrimitive(out, static_cast<std::uint64_t>(e.tokens.size()));
  for (const auto t : e.tokens) WritePrimitive(out, t);
  WritePrimitive(out, static_cast<std::uint64_t>(e.state_bytes.size()));
  WriteBlob(out, e.state_bytes.data(), e.state_bytes.size());
  WriteString(out, e.cache_namespace);
}

CheckpointCacheEntry ReadCacheEntry(const std::uint8_t*& cur, const std::uint8_t* end, bool has_namespace) {
  CheckpointCacheEntry e;
  e.model_id = ReadString(cur, end);
  e.decode.op_name = ReadString(cur, end);
  e.decode.temperature = ReadPrimitive<float>(cur, end);
  e.decode.max_tokens = ReadPrimitive<std::int32_t>(cur, end);
  e.prefix_len = ReadPrimitive<std::int32_t>(cur, end);
  const auto token_count = ReadPrimitive<std::uint64_t>(cur, end);
  if (static_cast<std::uint64_t>(end - cur) / sizeof(std::int32_t) < token_count) {
    throw std::runtime_error("checkpoint deserialize: token list truncated");
  }
  e.tokens.reserve(static_cast<std::size_t>(token_count));
  for (std::uint64_t j = 0; j < token_count; ++j) {
    e.tokens.push_back(ReadPrimitive<std::int32_t>(cur, end));
  }
  const auto state_len = ReadPrimitive<std::uint64_t>(cur, end);
  if (static_cast<std::size_t>(end - cur) < state_len) {
    throw std::runtime_error("checkpoint deserialize: state payload truncated");
  }
  e.state_bytes.insert(e.state_bytes.end(), cur, cur + state_len);
  cur += state_len;
  if (has_namespace) {
    e.cache_namespace = ReadString(cur, end);
  }
  return e;
}

std::vector<std::uint8_t> CacheEntryBytes(const CheckpointCacheEntry& e) {
  std::vector<std::uint8_t> out;
  WriteCacheEntry(out, e);
  return out;
}

// Canonical order so equal checkpoints serialize to equal bytes.
std::vector<const CheckpointCacheEntry*> SortedSnapshot(const std::vector<CheckpointCacheEntry>& snap) {
  std::vector<const CheckpointCacheEntry*> out;
  out.reserve(snap.size());
  for (const auto& e : snap) out.push_back(&e);
  std::sort(out.begin(), out.end(), [](const CheckpointCacheEntry* a, const CheckpointCacheEntry* b) {
    return std::tie(a->cache_namespace, a->model_id, a->decode.op_name, a->decode.temperature,
                    a->decode.max_tokens, a->tokens, a->prefix_len, a->state_bytes) <
           std::tie(b->cache_namespace, b->model_id, b->decode.op_name, b->decode.temperature,
                    b->decode.max_tokens, b->tokens, b->prefix_len, b->state_bytes);
  });
  return out;
}

std::vector<ir::NodeId> SortedIds(const std::unordered_map<ir::NodeId, continuum::Value>& values) {
  std::vector<ir::NodeId> ids;
  ids.reserve(values.size());
  for (const auto& [id, v] : values) ids.push_back(id);
  std::sort(ids.begin(), ids.end());
  return ids;
}

}  // namespace

std::vector<std::uint8_t> checkpoint_graph(const ir::Graph& graph) {
  return graph.serialize();
}

ir::Graph restore_graph(const std::vector<std::uint8_t>& bytes) {
  if (bytes.empty()) {
    return ir::Graph();
  }
  return ir::Graph::deserialize(bytes.data(), bytes.size());
}

std::vector<std::uint8_t> serialize_value(const continuum::Value& value) {
  std::vector<std::uint8_t> out;
  const auto tag = std::visit(
      [](const auto& v) -> std::uint8_t {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, continuum::TensorValue>) {
          return kValueTagTensor;
        } else if constexpr (std::is_same_v<T, continuum::TokensValue>) {
          return kValueTagTokens;
        } else if constexpr (std::is_same_v<T, continuum::SchemaValue>) {
          return kValueTagSchema;
        } else if constexpr (std::is_same_v<T, std::string>) {
          return kValueTagString;
        } else if constexpr (std::is_same_v<T, double>) {
          return kValueTagDouble;
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return kValueTagInt64;
        } else if constexpr (std::is_same_v<T, continuum::MlxTensorValue>) {
          return kValueTagMlxTensor;
        } else {
          static_assert(!sizeof(T*), "unsupported value type");
        }
      },
      value);
  WritePrimitive(out, tag);
  std::visit(
      [&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, continuum::TensorValue>) {
          const auto backend_len = static_cast<std::uint64_t>(v.backend_type.size());
          WritePrimitive(out, backend_len);
          out.insert(out.end(), v.backend_type.begin(), v.backend_type.end());
          auto t = v.tensor.contiguous().cpu();
          const auto dtype = static_cast<std::int32_t>(t.scalar_type());
          const auto device = static_cast<std::int32_t>(t.device().type());
          WritePrimitive(out, dtype);
          WritePrimitive(out, device);
          const auto dim = static_cast<std::uint64_t>(t.dim());
          WritePrimitive(out, dim);
          for (std::int64_t i = 0; i < t.dim(); ++i) {
            const auto d = static_cast<std::int64_t>(t.size(i));
            WritePrimitive(out, d);
          }
          const auto nbytes = static_cast<std::uint64_t>(t.numel() * t.element_size());
          WritePrimitive(out, nbytes);
          const auto* ptr = reinterpret_cast<const std::uint8_t*>(t.data_ptr());
          out.insert(out.end(), ptr, ptr + nbytes);
        } else if constexpr (std::is_same_v<T, continuum::TokensValue>) {
          const auto n = static_cast<std::uint64_t>(v.ids.size());
          WritePrimitive(out, n);
          for (int id : v.ids) {
            WritePrimitive(out, static_cast<std::int32_t>(id));
          }
        } else if constexpr (std::is_same_v<T, continuum::SchemaValue>) {
          const auto n = static_cast<std::uint64_t>(v.json.size());
          WritePrimitive(out, n);
          out.insert(out.end(), v.json.begin(), v.json.end());
        } else if constexpr (std::is_same_v<T, std::string>) {
          const auto n = static_cast<std::uint64_t>(v.size());
          WritePrimitive(out, n);
          out.insert(out.end(), v.begin(), v.end());
        } else if constexpr (std::is_same_v<T, double>) {
          WritePrimitive(out, v);
        } else if constexpr (std::is_same_v<T, int64_t>) {
          WritePrimitive(out, v);
        } else if constexpr (std::is_same_v<T, continuum::MlxTensorValue>) {
          const auto backend_len = static_cast<std::uint64_t>(v.backend_type.size());
          WritePrimitive(out, backend_len);
          out.insert(out.end(), v.backend_type.begin(), v.backend_type.end());
          const auto dim = static_cast<std::uint64_t>(v.shape.size());
          WritePrimitive(out, dim);
          for (std::int64_t d : v.shape) {
            WritePrimitive(out, d);
          }
          const auto n = static_cast<std::uint64_t>(v.data.size());
          WritePrimitive(out, n);
          for (float x : v.data) {
            WritePrimitive(out, x);
          }
        }
      },
      value);
  return out;
}

continuum::Value deserialize_value(const std::uint8_t* data, std::size_t len) {
  if (data == nullptr || len == 0) {
    throw std::runtime_error("checkpoint deserialize value: empty buffer");
  }
  const std::uint8_t* cur = data;
  const std::uint8_t* end = data + len;
  const auto tag = ReadPrimitive<std::uint8_t>(cur, end);
  switch (tag) {
    case kValueTagTensor: {
      const auto backend_len = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < backend_len) {
        throw std::runtime_error("checkpoint deserialize value: tensor backend payload truncated");
      }
      std::string backend_type(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(backend_len));
      cur += backend_len;
      const auto dtype = static_cast<torch::ScalarType>(ReadPrimitive<std::int32_t>(cur, end));
      (void)ReadPrimitive<std::int32_t>(cur, end);  // backend/device state is not restored in v0.1
      const auto dim = ReadPrimitive<std::uint64_t>(cur, end);
      std::vector<std::int64_t> shape;
      shape.reserve(static_cast<std::size_t>(dim));
      for (std::uint64_t i = 0; i < dim; ++i) {
        shape.push_back(ReadPrimitive<std::int64_t>(cur, end));
      }
      const auto nbytes = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < nbytes) {
        throw std::runtime_error("checkpoint deserialize value: tensor payload truncated");
      }
      auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
      auto t = torch::empty(shape, options);
      std::memcpy(t.data_ptr(), cur, static_cast<std::size_t>(nbytes));
      cur += nbytes;
      return continuum::TensorValue{t, std::move(backend_type)};
    }
    case kValueTagTokens: {
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      continuum::TokensValue tv;
      tv.ids.reserve(static_cast<std::size_t>(n));
      for (std::uint64_t i = 0; i < n; ++i) {
        tv.ids.push_back(static_cast<int>(ReadPrimitive<std::int32_t>(cur, end)));
      }
      return tv;
    }
    case kValueTagSchema: {
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < n) {
        throw std::runtime_error("checkpoint deserialize value: schema payload truncated");
      }
      continuum::SchemaValue sv;
      sv.json.assign(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(n));
      cur += n;
      return sv;
    }
    case kValueTagString: {
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < n) {
        throw std::runtime_error("checkpoint deserialize value: string payload truncated");
      }
      std::string s(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(n));
      cur += n;
      return s;
    }
    case kValueTagDouble:
      return ReadPrimitive<double>(cur, end);
    case kValueTagInt64:
      return ReadPrimitive<int64_t>(cur, end);
    case kValueTagMlxTensor: {
      const auto backend_len = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < backend_len) {
        throw std::runtime_error("checkpoint deserialize value: mlx backend payload truncated");
      }
      std::string backend_type(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(backend_len));
      cur += backend_len;
      const auto dim = ReadPrimitive<std::uint64_t>(cur, end);
      continuum::MlxTensorValue out;
      out.shape.reserve(static_cast<std::size_t>(dim));
      for (std::uint64_t i = 0; i < dim; ++i) {
        out.shape.push_back(ReadPrimitive<std::int64_t>(cur, end));
      }
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      out.data.reserve(static_cast<std::size_t>(n));
      for (std::uint64_t i = 0; i < n; ++i) {
        out.data.push_back(ReadPrimitive<float>(cur, end));
      }
      out.backend_type = std::move(backend_type);
      return out;
    }
    default:
      throw std::runtime_error("checkpoint deserialize value: unknown tag");
  }
}

std::vector<std::uint8_t> serialize_checkpoint(const Checkpoint& checkpoint) {
  std::vector<std::uint8_t> out;
  WritePrimitive(out, kCheckpointMagic);
  WritePrimitive(out, kCheckpointVersion);
  WritePrimitive(out, static_cast<std::uint64_t>(checkpoint.serialized_graph.size()));
  out.insert(out.end(), checkpoint.serialized_graph.begin(), checkpoint.serialized_graph.end());
  WritePrimitive(out, checkpoint.current_node_index);
  WritePrimitive(out, static_cast<std::uint64_t>(checkpoint.value_map.size()));
  for (const auto id : SortedIds(checkpoint.value_map)) {
    WritePrimitive(out, id);
    auto vbytes = serialize_value(checkpoint.value_map.at(id));
    WritePrimitive(out, static_cast<std::uint64_t>(vbytes.size()));
    out.insert(out.end(), vbytes.begin(), vbytes.end());
  }
  WritePrimitive(out, static_cast<std::uint64_t>(checkpoint.cache_snapshot.size()));
  for (const auto* e : SortedSnapshot(checkpoint.cache_snapshot)) {
    WriteCacheEntry(out, *e);
  }
  return out;
}

Checkpoint deserialize_checkpoint(const std::vector<std::uint8_t>& bytes) {
  if (bytes.empty()) {
    throw std::runtime_error("checkpoint deserialize: empty buffer");
  }
  const std::uint8_t* cur = bytes.data();
  const std::uint8_t* end = cur + bytes.size();
  const auto magic = ReadPrimitive<std::uint32_t>(cur, end);
  const auto version = ReadPrimitive<std::uint16_t>(cur, end);
  if (magic != kCheckpointMagic) {
    throw std::runtime_error("checkpoint deserialize: unknown magic (expected CPT1)");
  }
  if (version != kCheckpointVersion && version != kCheckpointVersionNoNamespace &&
      version != kCheckpointVersionNoSnapshot) {
    throw std::runtime_error("checkpoint deserialize: unsupported version " + std::to_string(version) +
                             " (supported: 1-3)");
  }
  Checkpoint out;
  const auto graph_len = ReadPrimitive<std::uint64_t>(cur, end);
  if (static_cast<std::size_t>(end - cur) < graph_len) {
    throw std::runtime_error("checkpoint deserialize: graph payload truncated");
  }
  out.serialized_graph.insert(out.serialized_graph.end(), cur, cur + graph_len);
  cur += graph_len;
  out.current_node_index = ReadPrimitive<std::uint64_t>(cur, end);
  const auto value_count = ReadPrimitive<std::uint64_t>(cur, end);
  for (std::uint64_t i = 0; i < value_count; ++i) {
    const auto id = ReadPrimitive<ir::NodeId>(cur, end);
    const auto value_len = ReadPrimitive<std::uint64_t>(cur, end);
    if (static_cast<std::size_t>(end - cur) < value_len) {
      throw std::runtime_error("checkpoint deserialize: value payload truncated");
    }
    out.value_map[id] = deserialize_value(cur, static_cast<std::size_t>(value_len));
    cur += value_len;
  }
  if (version >= kCheckpointVersionNoNamespace) {
    const auto snap_count = ReadPrimitive<std::uint64_t>(cur, end);
    out.cache_snapshot.reserve(static_cast<std::size_t>(snap_count));
    for (std::uint64_t i = 0; i < snap_count; ++i) {
      out.cache_snapshot.push_back(ReadCacheEntry(cur, end, version >= kCheckpointVersion));
    }
  }
  return out;
}

std::vector<std::uint8_t> migrate_checkpoint(const std::vector<std::uint8_t>& bytes) {
  // Known versions deserialize then re-serialize at the current wire format.
  return serialize_checkpoint(deserialize_checkpoint(bytes));
}

// Delta wire format ("CPD1" v1):
//   magic u32, version u16
//   u8 graph_changed; if 1: u64 len + graph bytes
//   u64 current_node_index
//   u64 n_upserts; n x (NodeId, u64 len, value bytes)     values new or changed
//   u64 n_removed; n x NodeId                              values dropped
//   u64 n_cache_removed; n x u64                           indices into base's canonical snapshot order
//   u64 n_cache_added; n x cache entry                     entries not in base
std::vector<std::uint8_t> serialize_checkpoint_delta(const Checkpoint& base, const Checkpoint& next) {
  std::vector<std::uint8_t> out;
  WritePrimitive(out, kDeltaMagic);
  WritePrimitive(out, kDeltaVersion);
  const bool graph_changed = base.serialized_graph != next.serialized_graph;
  WritePrimitive(out, static_cast<std::uint8_t>(graph_changed ? 1 : 0));
  if (graph_changed) {
    WritePrimitive(out, static_cast<std::uint64_t>(next.serialized_graph.size()));
    WriteBlob(out, next.serialized_graph.data(), next.serialized_graph.size());
  }
  WritePrimitive(out, next.current_node_index);

  std::vector<std::pair<ir::NodeId, std::vector<std::uint8_t>>> upserts;
  for (const auto id : SortedIds(next.value_map)) {
    auto bytes = serialize_value(next.value_map.at(id));
    auto it = base.value_map.find(id);
    if (it == base.value_map.end() || serialize_value(it->second) != bytes) {
      upserts.emplace_back(id, std::move(bytes));
    }
  }
  WritePrimitive(out, static_cast<std::uint64_t>(upserts.size()));
  for (const auto& [id, bytes] : upserts) {
    WritePrimitive(out, id);
    WritePrimitive(out, static_cast<std::uint64_t>(bytes.size()));
    WriteBlob(out, bytes.data(), bytes.size());
  }
  std::vector<ir::NodeId> removed;
  for (const auto id : SortedIds(base.value_map)) {
    if (next.value_map.find(id) == next.value_map.end()) removed.push_back(id);
  }
  WritePrimitive(out, static_cast<std::uint64_t>(removed.size()));
  for (const auto id : removed) WritePrimitive(out, id);

  const auto base_snap = SortedSnapshot(base.cache_snapshot);
  const auto next_snap = SortedSnapshot(next.cache_snapshot);
  std::multiset<std::vector<std::uint8_t>> next_keys;
  for (const auto* e : next_snap) next_keys.insert(CacheEntryBytes(*e));
  std::multiset<std::vector<std::uint8_t>> kept;
  std::vector<std::uint64_t> cache_removed;
  for (std::size_t i = 0; i < base_snap.size(); ++i) {
    auto key = CacheEntryBytes(*base_snap[i]);
    auto it = next_keys.find(key);
    if (it == next_keys.end()) {
      cache_removed.push_back(i);
    } else {
      next_keys.erase(it);
      kept.insert(std::move(key));
    }
  }
  WritePrimitive(out, static_cast<std::uint64_t>(cache_removed.size()));
  for (const auto i : cache_removed) WritePrimitive(out, i);
  std::vector<const CheckpointCacheEntry*> added;
  for (const auto* e : next_snap) {
    auto it = kept.find(CacheEntryBytes(*e));
    if (it == kept.end()) {
      added.push_back(e);
    } else {
      kept.erase(it);
    }
  }
  WritePrimitive(out, static_cast<std::uint64_t>(added.size()));
  for (const auto* e : added) WriteCacheEntry(out, *e);
  return out;
}

bool is_checkpoint_delta(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() < sizeof(std::uint32_t)) return false;
  std::uint32_t magic = 0;
  std::memcpy(&magic, bytes.data(), sizeof(magic));
  return magic == kDeltaMagic;
}

Checkpoint apply_checkpoint_delta(const Checkpoint& base, const std::vector<std::uint8_t>& delta) {
  const std::uint8_t* cur = delta.data();
  const std::uint8_t* end = cur + delta.size();
  if (ReadPrimitive<std::uint32_t>(cur, end) != kDeltaMagic) {
    throw std::runtime_error("checkpoint delta: unknown magic (expected CPD1)");
  }
  const auto version = ReadPrimitive<std::uint16_t>(cur, end);
  if (version != kDeltaVersion) {
    throw std::runtime_error("checkpoint delta: unsupported version " + std::to_string(version));
  }
  Checkpoint out = base;
  if (ReadPrimitive<std::uint8_t>(cur, end) != 0) {
    const auto len = ReadPrimitive<std::uint64_t>(cur, end);
    if (static_cast<std::size_t>(end - cur) < len) {
      throw std::runtime_error("checkpoint delta: graph payload truncated");
    }
    out.serialized_graph.assign(cur, cur + len);
    cur += len;
  }
  out.current_node_index = ReadPrimitive<std::uint64_t>(cur, end);
  const auto n_upserts = ReadPrimitive<std::uint64_t>(cur, end);
  for (std::uint64_t i = 0; i < n_upserts; ++i) {
    const auto id = ReadPrimitive<ir::NodeId>(cur, end);
    const auto len = ReadPrimitive<std::uint64_t>(cur, end);
    if (static_cast<std::size_t>(end - cur) < len) {
      throw std::runtime_error("checkpoint delta: value payload truncated");
    }
    out.value_map[id] = deserialize_value(cur, static_cast<std::size_t>(len));
    cur += len;
  }
  const auto n_removed = ReadPrimitive<std::uint64_t>(cur, end);
  for (std::uint64_t i = 0; i < n_removed; ++i) {
    out.value_map.erase(ReadPrimitive<ir::NodeId>(cur, end));
  }
  const auto base_snap = SortedSnapshot(base.cache_snapshot);
  std::vector<bool> drop(base_snap.size(), false);
  const auto n_cache_removed = ReadPrimitive<std::uint64_t>(cur, end);
  for (std::uint64_t i = 0; i < n_cache_removed; ++i) {
    const auto idx = ReadPrimitive<std::uint64_t>(cur, end);
    if (idx >= base_snap.size()) {
      throw std::runtime_error("checkpoint delta: cache index out of range (wrong base?)");
    }
    drop[static_cast<std::size_t>(idx)] = true;
  }
  std::vector<CheckpointCacheEntry> snapshot;
  for (std::size_t i = 0; i < base_snap.size(); ++i) {
    if (!drop[i]) snapshot.push_back(*base_snap[i]);
  }
  const auto n_added = ReadPrimitive<std::uint64_t>(cur, end);
  for (std::uint64_t i = 0; i < n_added; ++i) {
    snapshot.push_back(ReadCacheEntry(cur, end, true));
  }
  if (cur != end) {
    throw std::runtime_error("checkpoint delta: trailing bytes");
  }
  out.cache_snapshot = std::move(snapshot);
  return out;
}

}  // namespace continuum::runtime
