#include <continuum/runtime/memo_table.hpp>

#include <cstring>
#include <type_traits>

namespace continuum::runtime {

namespace {

std::string node_kind_to_string(ir::NodeKind kind) {
  switch (kind) {
    case ir::NodeKind::TensorOp: return "TensorOp";
    case ir::NodeKind::TokenOp: return "TokenOp";
    case ir::NodeKind::PromptOp: return "PromptOp";
    case ir::NodeKind::ToolOp: return "ToolOp";
    case ir::NodeKind::ControlOp: return "ControlOp";
  }
  return "Unknown";
}

void hash_combine(std::size_t& h, std::size_t val) {
  h ^= val + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}

std::size_t hash_payload(const ir::Node& node) {
  std::size_t h = std::hash<std::string>{}(node.debug_name);
  std::visit([&h](const auto& payload) {
    using T = std::decay_t<decltype(payload)>;
    if constexpr (std::is_same_v<T, ir::ToolOpPayload>) {
      hash_combine(h, std::hash<std::string>{}(payload.tool_name));
      hash_combine(h, std::hash<std::string>{}(payload.input_schema.canonical_json));
      hash_combine(h, std::hash<std::string>{}(payload.output_schema.canonical_json));
    } else if constexpr (std::is_same_v<T, ir::TensorOpPayload>) {
      hash_combine(h, std::hash<std::string>{}(payload.op_name));
      for (auto a : payload.attrs) {
        hash_combine(h, std::hash<std::int64_t>{}(a));
      }
    } else if constexpr (std::is_same_v<T, ir::TokenOpPayload>) {
      hash_combine(h, std::hash<std::string>{}(payload.op_name));
      hash_combine(h, std::hash<std::string>{}(payload.model_id));
      hash_combine(h, std::hash<int>{}(static_cast<int>(payload.temperature * 1000.0f)));
      hash_combine(h, std::hash<std::int32_t>{}(payload.max_tokens));
    } else if constexpr (std::is_same_v<T, ir::PromptOpPayload>) {
      hash_combine(h, std::hash<std::string>{}(payload.template_id));
      for (auto s : payload.slot_inputs) {
        hash_combine(h, std::hash<ir::NodeId>{}(s));
      }
    } else if constexpr (std::is_same_v<T, ir::ControlOpPayload>) {
      hash_combine(h, std::hash<int>{}(static_cast<int>(payload.kind)));
      for (auto b : payload.branch_entries) {
        hash_combine(h, std::hash<ir::NodeId>{}(b));
      }
    }
  }, node.payload);
  return h;
}

void push_bytes(std::vector<std::uint8_t>& buf, const void* data, std::size_t len) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  buf.insert(buf.end(), p, p + len);
}

void push_u32(std::vector<std::uint8_t>& buf, std::uint32_t v) {
  push_bytes(buf, &v, sizeof(v));
}

void push_u64(std::vector<std::uint8_t>& buf, std::uint64_t v) {
  push_bytes(buf, &v, sizeof(v));
}

bool read_bytes(const std::vector<std::uint8_t>& buf, std::size_t& pos, void* out, std::size_t len) {
  if (pos + len > buf.size()) return false;
  std::memcpy(out, buf.data() + pos, len);
  pos += len;
  return true;
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& buf, std::size_t& pos) {
  std::uint32_t v = 0;
  read_bytes(buf, pos, &v, sizeof(v));
  return v;
}

std::size_t tensor_fingerprint(const torch::Tensor& t) {
  std::size_t h = std::hash<int>{}(static_cast<int>(t.dim()));
  for (int i = 0; i < t.dim(); ++i) {
    hash_combine(h, std::hash<std::int64_t>{}(t.size(i)));
  }
  hash_combine(h, std::hash<int>{}(static_cast<int>(t.scalar_type())));
  if (t.numel() > 0) {
    hash_combine(h, std::hash<std::int64_t>{}(t.nbytes()));
  }
  return h;
}

}  // namespace

MemoTable::MemoTable(std::size_t max_entries, std::size_t version)
    : max_entries_(max_entries), version_(version) {}

std::optional<MemoEntry> MemoTable::lookup(const MemoKey& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = table_.find(key);
  if (it != table_.end()) {
    if (it->second.version == version_) {
      ++const_cast<MemoEntry&>(it->second).access_count;
      const_cast<MemoEntry&>(it->second).last_access_ns = static_cast<std::int64_t>(++clock_);
      return it->second;
    }
    const_cast<std::unordered_map<MemoKey, MemoEntry, MemoKeyHash>&>(table_).erase(it);
  }
  return std::nullopt;
}

void MemoTable::insert(MemoKey key, MemoEntry entry) {
  std::lock_guard<std::mutex> lock(mu_);
  entry.version = version_;
  entry.last_access_ns = static_cast<std::int64_t>(++clock_);

  auto it = table_.find(key);
  if (it != table_.end()) {
    it->second = std::move(entry);
  } else {
    table_.emplace(std::move(key), std::move(entry));
  }

  while (table_.size() > max_entries_) {
    auto victim = table_.begin();
    for (auto eit = table_.begin(); eit != table_.end(); ++eit) {
      if (eit->second.last_access_ns < victim->second.last_access_ns) {
        victim = eit;
      }
    }
    table_.erase(victim);
  }
}

void MemoTable::invalidate_version(std::size_t version) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto it = table_.begin(); it != table_.end();) {
    if (it->second.version != version) {
      it = table_.erase(it);
    } else {
      ++it;
    }
  }
}

void MemoTable::invalidate_node(const std::string& node_kind_str) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto it = table_.begin(); it != table_.end();) {
    if (it->first.node_kind_str == node_kind_str) {
      it = table_.erase(it);
    } else {
      ++it;
    }
  }
}

void MemoTable::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  table_.clear();
  clock_ = 0;
}

std::size_t MemoTable::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return table_.size();
}

std::size_t MemoTable::version() const {
  return version_;
}

void MemoTable::set_version(std::size_t v) {
  std::lock_guard<std::mutex> lock(mu_);
  version_ = v;
}

MemoKey MemoTable::make_key(const ir::Node& node, const std::vector<Value>& inputs) const {
  MemoKey key;
  key.node_kind_str = node_kind_to_string(node.kind);
  key.payload_hash = std::to_string(hash_payload(node));

  for (const auto& v : inputs) {
    auto bytes = serialize_value(v);
    key.inputs_hash.insert(key.inputs_hash.end(), bytes.begin(), bytes.end());
  }
  return key;
}

std::vector<std::uint8_t> MemoTable::serialize_value(const Value& v) {
  std::vector<std::uint8_t> buf;
  std::visit([&buf](const auto& val) {
    using T = std::decay_t<decltype(val)>;
    if constexpr (std::is_same_v<T, std::string>) {
      buf.push_back(0);
      push_u32(buf, static_cast<std::uint32_t>(val.size()));
      push_bytes(buf, val.data(), val.size());
    } else if constexpr (std::is_same_v<T, double>) {
      buf.push_back(1);
      push_bytes(buf, &val, sizeof(val));
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
      buf.push_back(2);
      push_bytes(buf, &val, sizeof(val));
    } else if constexpr (std::is_same_v<T, TokensValue>) {
      buf.push_back(3);
      push_u32(buf, static_cast<std::uint32_t>(val.ids.size()));
      for (int id : val.ids) {
        std::int32_t iid = static_cast<std::int32_t>(id);
        push_bytes(buf, &iid, sizeof(iid));
      }
    } else if constexpr (std::is_same_v<T, TensorValue>) {
      buf.push_back(4);
      std::size_t fp = tensor_fingerprint(val.tensor);
      push_u64(buf, static_cast<std::uint64_t>(fp));
    } else if constexpr (std::is_same_v<T, MlxTensorValue>) {
      buf.push_back(5);
      std::size_t h = std::hash<std::size_t>{}(val.shape.size());
      for (auto s : val.shape) {
        hash_combine(h, std::hash<std::int64_t>{}(s));
      }
      push_u64(buf, static_cast<std::uint64_t>(h));
    } else if constexpr (std::is_same_v<T, SchemaValue>) {
      buf.push_back(6);
      push_u32(buf, static_cast<std::uint32_t>(val.json.size()));
      push_bytes(buf, val.json.data(), val.json.size());
    }
  }, v);
  return buf;
}

std::optional<Value> MemoTable::deserialize_value(const std::vector<std::uint8_t>& bytes) {
  if (bytes.empty()) return std::nullopt;
  std::size_t pos = 0;
  std::uint8_t tag = bytes[pos++];
  switch (tag) {
    case 0: {
      std::uint32_t len = read_u32(bytes, pos);
      if (pos + len > bytes.size()) return std::nullopt;
      return std::string(reinterpret_cast<const char*>(bytes.data() + pos), len);
    }
    case 1: {
      double val;
      if (!read_bytes(bytes, pos, &val, sizeof(val))) return std::nullopt;
      return val;
    }
    case 2: {
      std::int64_t val;
      if (!read_bytes(bytes, pos, &val, sizeof(val))) return std::nullopt;
      return val;
    }
    case 3: {
      std::uint32_t count = read_u32(bytes, pos);
      TokensValue tv;
      tv.ids.resize(count);
      for (std::uint32_t i = 0; i < count; ++i) {
        std::int32_t iid;
        if (!read_bytes(bytes, pos, &iid, sizeof(iid))) return std::nullopt;
        tv.ids[i] = iid;
      }
      return tv;
    }
    case 4:
    case 5:
      return std::nullopt;
    case 6: {
      std::uint32_t len = read_u32(bytes, pos);
      if (pos + len > bytes.size()) return std::nullopt;
      SchemaValue sv;
      sv.json.assign(reinterpret_cast<const char*>(bytes.data() + pos), len);
      return sv;
    }
    default:
      return std::nullopt;
  }
}

}  // namespace continuum::runtime
