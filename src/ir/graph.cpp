#include <continuum/ir/graph.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>

namespace continuum::ir {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::uint32_t kMagic = 0x31495243U;  // "CIR1"
constexpr std::uint16_t kVersion = 1;

template <typename T>
void HashPrimitive(std::uint64_t& h, const T& value) {
  static_assert(std::is_trivially_copyable<T>::value, "hash primitive must be trivially copyable");
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    h ^= bytes[i];
    h *= kFnvPrime;
  }
}

void HashString(std::uint64_t& h, const std::string& s) {
  HashPrimitive(h, static_cast<std::uint64_t>(s.size()));
  for (unsigned char c : s) {
    HashPrimitive(h, c);
  }
}

template <typename T>
void WritePrimitive(std::vector<std::uint8_t>& out, const T& value) {
  static_assert(std::is_trivially_copyable<T>::value, "primitive write requires trivially copyable type");
  const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), begin, begin + sizeof(T));
}

void WriteString(std::vector<std::uint8_t>& out, const std::string& s) {
  WritePrimitive(out, static_cast<std::uint64_t>(s.size()));
  out.insert(out.end(), s.begin(), s.end());
}

template <typename T>
T ReadPrimitive(const std::uint8_t*& cur, const std::uint8_t* end) {
  if (static_cast<std::size_t>(end - cur) < sizeof(T)) {
    throw std::runtime_error("graph deserialize: unexpected eof");
  }
  T out{};
  std::memcpy(&out, cur, sizeof(T));
  cur += sizeof(T);
  return out;
}

std::string ReadString(const std::uint8_t*& cur, const std::uint8_t* end) {
  const auto sz = ReadPrimitive<std::uint64_t>(cur, end);
  if (static_cast<std::size_t>(end - cur) < sz) {
    throw std::runtime_error("graph deserialize: unexpected eof in string");
  }
  std::string out(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(sz));
  cur += sz;
  return out;
}

void WriteType(std::vector<std::uint8_t>& out, const Type& t) {
  WritePrimitive(out, static_cast<std::uint8_t>(t.index()));
  std::visit(
      [&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, TensorType>) {
          WritePrimitive(out, static_cast<std::uint64_t>(v.shape.size()));
          for (auto d : v.shape) {
            WritePrimitive(out, d);
          }
          WritePrimitive(out, static_cast<std::uint8_t>(v.dtype));
          WritePrimitive(out, static_cast<std::uint8_t>(v.device));
        } else if constexpr (std::is_same_v<T, TokensType>) {
          WriteString(out, v.vocab_id);
          WritePrimitive(out, v.max_len);
          WriteString(out, v.model_family);
        } else if constexpr (std::is_same_v<T, SchemaType>) {
          WriteString(out, v.canonical_json);
          WritePrimitive(out, v.schema_hash);
        } else if constexpr (std::is_same_v<T, EffectType>) {
          WritePrimitive(out, v.bits);
        }
      },
      t);
}

Type ReadType(const std::uint8_t*& cur, const std::uint8_t* end) {
  const auto idx = ReadPrimitive<std::uint8_t>(cur, end);
  switch (idx) {
    case 0: {
      const auto ndim = ReadPrimitive<std::uint64_t>(cur, end);
      TensorType t;
      t.shape.reserve(static_cast<std::size_t>(ndim));
      for (std::uint64_t i = 0; i < ndim; ++i) {
        t.shape.push_back(ReadPrimitive<std::int64_t>(cur, end));
      }
      t.dtype = static_cast<DType>(ReadPrimitive<std::uint8_t>(cur, end));
      t.device = static_cast<Device>(ReadPrimitive<std::uint8_t>(cur, end));
      return t;
    }
    case 1:
      return TokensType{ReadString(cur, end), ReadPrimitive<std::int32_t>(cur, end), ReadString(cur, end)};
    case 2:
      return SchemaType{ReadString(cur, end), ReadPrimitive<std::uint64_t>(cur, end)};
    case 3:
      return EffectType{ReadPrimitive<std::uint8_t>(cur, end)};
    default:
      throw std::runtime_error("graph deserialize: unknown Type alternative");
  }
}

void WritePayload(std::vector<std::uint8_t>& out, const OpPayload& payload) {
  WritePrimitive(out, static_cast<std::uint8_t>(payload.index()));
  std::visit(
      [&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, TensorOpPayload>) {
          WriteString(out, v.op_name);
          WritePrimitive(out, static_cast<std::uint64_t>(v.attrs.size()));
          for (auto a : v.attrs) {
            WritePrimitive(out, a);
          }
        } else if constexpr (std::is_same_v<T, TokenOpPayload>) {
          WriteString(out, v.op_name);
          WriteString(out, v.model_id);
          WritePrimitive(out, v.temperature);
          WritePrimitive(out, v.max_tokens);
        } else if constexpr (std::is_same_v<T, PromptOpPayload>) {
          WriteString(out, v.template_id);
          WritePrimitive(out, static_cast<std::uint64_t>(v.slot_inputs.size()));
          for (auto id : v.slot_inputs) {
            WritePrimitive(out, id);
          }
        } else if constexpr (std::is_same_v<T, ToolOpPayload>) {
          WriteString(out, v.tool_name);
          WriteString(out, v.input_schema.canonical_json);
          WriteString(out, v.output_schema.canonical_json);
        } else if constexpr (std::is_same_v<T, ControlOpPayload>) {
          WritePrimitive(out, static_cast<std::uint8_t>(v.kind));
          WritePrimitive(out, static_cast<std::uint64_t>(v.branch_entries.size()));
          for (auto id : v.branch_entries) {
            WritePrimitive(out, id);
          }
        }
      },
      payload);
}

OpPayload ReadPayload(const std::uint8_t*& cur, const std::uint8_t* end) {
  const auto idx = ReadPrimitive<std::uint8_t>(cur, end);
  switch (idx) {
    case 0: {
      TensorOpPayload p;
      p.op_name = ReadString(cur, end);
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      p.attrs.reserve(static_cast<std::size_t>(n));
      for (std::uint64_t i = 0; i < n; ++i) {
        p.attrs.push_back(ReadPrimitive<std::int64_t>(cur, end));
      }
      return p;
    }
    case 1:
      return TokenOpPayload{
          ReadString(cur, end), ReadString(cur, end), ReadPrimitive<float>(cur, end), ReadPrimitive<std::int32_t>(cur, end)};
    case 2: {
      PromptOpPayload p;
      p.template_id = ReadString(cur, end);
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      p.slot_inputs.reserve(static_cast<std::size_t>(n));
      for (std::uint64_t i = 0; i < n; ++i) {
        p.slot_inputs.push_back(ReadPrimitive<NodeId>(cur, end));
      }
      return p;
    }
    case 3:
      return ToolOpPayload{ReadString(cur, end), Schema{ReadString(cur, end)}, Schema{ReadString(cur, end)}};
    case 4: {
      ControlOpPayload p;
      p.kind = static_cast<ControlOpPayload::Kind>(ReadPrimitive<std::uint8_t>(cur, end));
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      p.branch_entries.reserve(static_cast<std::size_t>(n));
      for (std::uint64_t i = 0; i < n; ++i) {
        p.branch_entries.push_back(ReadPrimitive<NodeId>(cur, end));
      }
      return p;
    }
    default:
      throw std::runtime_error("graph deserialize: unknown OpPayload alternative");
  }
}

void RemapPayloadNodeReferences(OpPayload& payload, const std::function<NodeId(NodeId)>& remap) {
  std::visit(
      [&](auto& p) {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, PromptOpPayload>) {
          for (auto& id : p.slot_inputs) {
            id = remap(id);
          }
        } else if constexpr (std::is_same_v<T, ControlOpPayload>) {
          for (auto& id : p.branch_entries) {
            id = remap(id);
          }
        }
      },
      payload);
}

}  // namespace

NodeId Graph::add_node(Node n) {
  n.id = next_id_++;
  nodes_.insert({n.id, std::move(n)});
  topo_cache_.reset();
  return next_id_ - 1;
}

const Node& Graph::get(NodeId id) const { return nodes_.at(id); }
Node& Graph::get_mut(NodeId id) { return nodes_.at(id); }

const std::vector<NodeId>& Graph::topo_order() const {
  if (topo_cache_.has_value()) {
    return topo_cache_.value();
  }

  std::unordered_map<NodeId, std::size_t> indegree;
  std::unordered_map<NodeId, std::vector<NodeId>> children;
  indegree.reserve(nodes_.size());
  children.reserve(nodes_.size());

  for (const auto& [id, _] : nodes_) {
    indegree[id] = 0;
  }

  for (const auto& [id, node] : nodes_) {
    for (NodeId parent : node.inputs) {
      if (nodes_.find(parent) == nodes_.end()) {
        throw std::runtime_error("graph topo_order: missing input node");
      }
      ++indegree[id];
      children[parent].push_back(id);
    }
  }

  std::priority_queue<NodeId, std::vector<NodeId>, std::greater<NodeId>> ready;
  for (const auto& [id, deg] : indegree) {
    if (deg == 0) {
      ready.push(id);
    }
  }

  std::vector<NodeId> order;
  order.reserve(nodes_.size());
  while (!ready.empty()) {
    const NodeId id = ready.top();
    ready.pop();
    order.push_back(id);
    auto it = children.find(id);
    if (it == children.end()) {
      continue;
    }
    for (NodeId child : it->second) {
      auto deg_it = indegree.find(child);
      if (--deg_it->second == 0) {
        ready.push(child);
      }
    }
  }

  if (order.size() != nodes_.size()) {
    throw std::runtime_error("graph topo_order: cycle detected");
  }
  topo_cache_ = std::move(order);
  return topo_cache_.value();
}

std::uint64_t Graph::structural_hash() const {
  std::uint64_t h = kFnvOffset;
  const auto& topo = topo_order();
  std::unordered_map<NodeId, std::uint64_t> ordinal;
  ordinal.reserve(topo.size());
  for (std::uint64_t i = 0; i < topo.size(); ++i) {
    ordinal[topo[static_cast<std::size_t>(i)]] = i;
  }

  for (NodeId id : topo) {
    const auto& n = get(id);
    HashPrimitive(h, static_cast<std::uint8_t>(n.kind));
    HashPrimitive(h, static_cast<std::uint8_t>(n.out_type.index()));
    HashPrimitive(h, n.effect.bits);
    HashString(h, n.debug_name);

    HashPrimitive(h, static_cast<std::uint64_t>(n.inputs.size()));
    for (NodeId parent : n.inputs) {
      HashPrimitive(h, ordinal.at(parent));
    }

    std::visit(
        [&](const auto& p) {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, TensorOpPayload>) {
            HashString(h, p.op_name);
            HashPrimitive(h, static_cast<std::uint64_t>(p.attrs.size()));
            for (auto a : p.attrs) {
              HashPrimitive(h, a);
            }
          } else if constexpr (std::is_same_v<T, TokenOpPayload>) {
            HashString(h, p.op_name);
            HashString(h, p.model_id);
            HashPrimitive(h, p.temperature);
            HashPrimitive(h, p.max_tokens);
          } else if constexpr (std::is_same_v<T, PromptOpPayload>) {
            HashString(h, p.template_id);
            HashPrimitive(h, static_cast<std::uint64_t>(p.slot_inputs.size()));
            for (NodeId input : p.slot_inputs) {
              const auto it = ordinal.find(input);
              HashPrimitive(h, it == ordinal.end() ? std::numeric_limits<std::uint64_t>::max() : it->second);
            }
          } else if constexpr (std::is_same_v<T, ToolOpPayload>) {
            HashString(h, p.tool_name);
            HashString(h, p.input_schema.canonical_json);
            HashString(h, p.output_schema.canonical_json);
          } else if constexpr (std::is_same_v<T, ControlOpPayload>) {
            HashPrimitive(h, static_cast<std::uint8_t>(p.kind));
            HashPrimitive(h, static_cast<std::uint64_t>(p.branch_entries.size()));
            for (NodeId entry : p.branch_entries) {
              const auto it = ordinal.find(entry);
              HashPrimitive(h, it == ordinal.end() ? std::numeric_limits<std::uint64_t>::max() : it->second);
            }
          }
        },
        n.payload);
  }
  return h;
}

Graph Graph::extract_subgraph(const std::vector<NodeId>& roots) const {
  std::unordered_set<NodeId> keep;
  std::vector<NodeId> stack = roots;
  while (!stack.empty()) {
    const NodeId id = stack.back();
    stack.pop_back();
    if (keep.find(id) != keep.end()) {
      continue;
    }
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
      continue;
    }
    keep.insert(id);
    for (NodeId parent : it->second.inputs) {
      stack.push_back(parent);
    }
    std::visit(
        [&](const auto& p) {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, PromptOpPayload>) {
            for (NodeId in : p.slot_inputs) {
              stack.push_back(in);
            }
          } else if constexpr (std::is_same_v<T, ControlOpPayload>) {
            for (NodeId in : p.branch_entries) {
              stack.push_back(in);
            }
          }
        },
        it->second.payload);
  }

  Graph out;
  std::unordered_map<NodeId, NodeId> remap;
  for (NodeId id : topo_order()) {
    if (keep.find(id) == keep.end()) {
      continue;
    }
    Node copy = get(id);
    for (auto& in : copy.inputs) {
      in = remap.at(in);
    }
    RemapPayloadNodeReferences(copy.payload, [&](NodeId old_id) { return remap.at(old_id); });
    const NodeId new_id = out.add_node(std::move(copy));
    remap[id] = new_id;
  }
  return out;
}

std::vector<std::uint8_t> Graph::serialize() const {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(64 + (nodes_.size() * 96));

  WritePrimitive(bytes, kMagic);
  WritePrimitive(bytes, kVersion);
  const auto& topo = topo_order();
  WritePrimitive(bytes, static_cast<std::uint64_t>(topo.size()));

  for (NodeId id : topo) {
    const Node& n = get(id);
    WritePrimitive(bytes, id);
    WritePrimitive(bytes, static_cast<std::uint8_t>(n.kind));
    WriteString(bytes, n.debug_name);
    WritePrimitive(bytes, static_cast<std::uint64_t>(n.inputs.size()));
    for (NodeId in : n.inputs) {
      WritePrimitive(bytes, in);
    }
    WriteType(bytes, n.out_type);
    WritePrimitive(bytes, n.effect.bits);
    WritePayload(bytes, n.payload);
  }
  return bytes;
}

Graph Graph::deserialize(const std::uint8_t* data, std::size_t len) {
  Graph g;
  if (data == nullptr || len < sizeof(std::uint32_t) + sizeof(std::uint16_t) + sizeof(std::uint64_t)) {
    return g;
  }

  const std::uint8_t* cur = data;
  const std::uint8_t* end = data + len;
  const auto magic = ReadPrimitive<std::uint32_t>(cur, end);
  const auto version = ReadPrimitive<std::uint16_t>(cur, end);
  if (magic != kMagic || version != kVersion) {
    throw std::runtime_error("graph deserialize: unsupported format");
  }

  const auto node_count = ReadPrimitive<std::uint64_t>(cur, end);
  std::unordered_map<NodeId, NodeId> remap;
  remap.reserve(static_cast<std::size_t>(node_count));

  struct DeferredNodeRefs {
    NodeId new_id = 0;
    std::vector<NodeId> old_inputs;
    std::vector<NodeId> old_prompt_slots;
    std::vector<NodeId> old_branch_entries;
  };
  std::vector<DeferredNodeRefs> deferred;
  deferred.reserve(static_cast<std::size_t>(node_count));

  for (std::uint64_t i = 0; i < node_count; ++i) {
    const NodeId old_id = ReadPrimitive<NodeId>(cur, end);
    Node n;
    n.kind = static_cast<NodeKind>(ReadPrimitive<std::uint8_t>(cur, end));
    n.debug_name = ReadString(cur, end);

    const auto input_count = ReadPrimitive<std::uint64_t>(cur, end);
    DeferredNodeRefs refs;
    refs.old_inputs.reserve(static_cast<std::size_t>(input_count));
    for (std::uint64_t j = 0; j < input_count; ++j) {
      refs.old_inputs.push_back(ReadPrimitive<NodeId>(cur, end));
    }

    n.out_type = ReadType(cur, end);
    n.effect.bits = ReadPrimitive<std::uint8_t>(cur, end);
    n.payload = ReadPayload(cur, end);

    std::visit(
        [&](const auto& p) {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, PromptOpPayload>) {
            refs.old_prompt_slots = p.slot_inputs;
          } else if constexpr (std::is_same_v<T, ControlOpPayload>) {
            refs.old_branch_entries = p.branch_entries;
          }
        },
        n.payload);

    const NodeId new_id = g.add_node(std::move(n));
    refs.new_id = new_id;
    remap[old_id] = new_id;
    deferred.push_back(std::move(refs));
  }

  for (const auto& refs : deferred) {
    Node& n = g.get_mut(refs.new_id);
    n.inputs.clear();
    n.inputs.reserve(refs.old_inputs.size());
    for (NodeId old_in : refs.old_inputs) {
      auto it = remap.find(old_in);
      if (it == remap.end()) {
        throw std::runtime_error("graph deserialize: dangling input id");
      }
      n.inputs.push_back(it->second);
    }

    std::visit(
        [&](auto& p) {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, PromptOpPayload>) {
            p.slot_inputs.clear();
            p.slot_inputs.reserve(refs.old_prompt_slots.size());
            for (NodeId old_id : refs.old_prompt_slots) {
              p.slot_inputs.push_back(remap.at(old_id));
            }
          } else if constexpr (std::is_same_v<T, ControlOpPayload>) {
            p.branch_entries.clear();
            p.branch_entries.reserve(refs.old_branch_entries.size());
            for (NodeId old_id : refs.old_branch_entries) {
              p.branch_entries.push_back(remap.at(old_id));
            }
          }
        },
        n.payload);
  }

  return g;
}

}  // namespace continuum::ir
