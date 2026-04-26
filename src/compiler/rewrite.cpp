#include <continuum/compiler/rewrite.hpp>

#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace continuum::compiler {
namespace {

void RewriteAllUses(ir::Graph& g, ir::NodeId from, ir::NodeId to) {
  if (from == to) {
    return;
  }
  for (auto node_id : g.topo_order()) {
    auto& n = g.get_mut(node_id);
    for (auto& in : n.inputs) {
      if (in == from) {
        in = to;
      }
    }
    std::visit(
        [&](auto& payload) {
          using P = std::decay_t<decltype(payload)>;
          if constexpr (std::is_same_v<P, ir::PromptOpPayload>) {
            for (auto& in : payload.slot_inputs) {
              if (in == from) {
                in = to;
              }
            }
          } else if constexpr (std::is_same_v<P, ir::ControlOpPayload>) {
            for (auto& entry : payload.branch_entries) {
              if (entry == from) {
                entry = to;
              }
            }
          }
        },
        n.payload);
  }
}

std::string TensorInputKey(const std::vector<ir::NodeId>& inputs) {
  std::ostringstream oss;
  for (auto in : inputs) {
    oss << in << ",";
  }
  return oss.str();
}

std::string CanonicalizeText(const std::string& raw) {
  std::string out;
  out.reserve(raw.size());
  bool prev_space = false;
  for (char c : raw) {
    const bool is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
    if (is_space) {
      if (!prev_space) out.push_back(' ');
      prev_space = true;
    } else {
      out.push_back(c);
      prev_space = false;
    }
  }
  while (!out.empty() && out.front() == ' ') out.erase(out.begin());
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

std::vector<std::int32_t> ToTokens(const std::string& text) {
  std::vector<std::int32_t> tokens;
  tokens.reserve(text.size());
  for (unsigned char c : text) tokens.push_back(static_cast<std::int32_t>(c));
  return tokens;
}

std::size_t CommonPrefix(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

}  // namespace

void hoist_common_token_prefixes(ir::Graph& g) {
  struct TokenSig {
    ir::NodeId id = 0;
    std::vector<std::int32_t> prompt_tokens;
    std::string group;
  };
  std::vector<TokenSig> sigs;
  sigs.reserve(g.topo_order().size());

  for (auto node_id : g.topo_order()) {
    auto& n = g.get_mut(node_id);
    if (n.kind != ir::NodeKind::TokenOp) {
      continue;
    }
    const auto* payload = std::get_if<ir::TokenOpPayload>(&n.payload);
    if (payload == nullptr) {
      continue;
    }
    std::ostringstream group_key;
    group_key << payload->op_name << "|" << payload->model_id << "|" << payload->temperature << "|" << payload->max_tokens;
    const std::string prompt_basis = CanonicalizeText(n.debug_name);
    sigs.push_back(TokenSig{node_id, ToTokens(prompt_basis), group_key.str()});
  }

  std::unordered_set<ir::NodeId> redirected;
  for (std::size_t i = 0; i < sigs.size(); ++i) {
    for (std::size_t j = i + 1; j < sigs.size(); ++j) {
      if (sigs[i].group != sigs[j].group) continue;
      const auto common = CommonPrefix(sigs[i].prompt_tokens, sigs[j].prompt_tokens);
      if (common == 0) continue;

      if (sigs[i].prompt_tokens == sigs[j].prompt_tokens) {
        RewriteAllUses(g, sigs[j].id, sigs[i].id);
        redirected.insert(sigs[j].id);
        continue;
      }

      // Token-based prefix sharing: downstream token op depends on upstream prefix op.
      auto& downstream = g.get_mut(sigs[j].id);
      if (!downstream.inputs.empty() && redirected.find(sigs[j].id) == redirected.end()) {
        downstream.inputs[0] = sigs[i].id;
        redirected.insert(sigs[j].id);
      }
    }
  }
}

void memoize_pure_tool_ops(ir::Graph& g) {
  std::unordered_map<std::string, ir::NodeId> first_by_key;
  for (auto node_id : g.topo_order()) {
    auto& n = g.get_mut(node_id);
    if (n.kind != ir::NodeKind::ToolOp) {
      continue;
    }
    const auto pure = static_cast<std::uint8_t>(ir::EffectBit::Pure);
    const auto idem = static_cast<std::uint8_t>(ir::EffectBit::Idem);
    const auto effect_bits = n.effect.bits;
    if ((effect_bits & pure) == 0 || (effect_bits & idem) == 0) {
      continue;
    }
    const auto* payload = std::get_if<ir::ToolOpPayload>(&n.payload);
    if (payload == nullptr) {
      continue;
    }
    std::ostringstream key;
    key << payload->tool_name << "|" << payload->input_schema.canonical_json << "|"
        << payload->output_schema.canonical_json << "|" << TensorInputKey(n.inputs);
    const auto key_str = key.str();
    auto it = first_by_key.find(key_str);
    if (it == first_by_key.end()) {
      first_by_key[key_str] = node_id;
      continue;
    }
    RewriteAllUses(g, node_id, it->second);
  }
}

void specialize_structured_outputs(ir::Graph& g) {
  for (auto node_id : g.topo_order()) {
    auto& n = g.get_mut(node_id);
    if (n.kind != ir::NodeKind::TokenOp || !std::holds_alternative<ir::SchemaType>(n.out_type)) {
      continue;
    }
    auto* payload = std::get_if<ir::TokenOpPayload>(&n.payload);
    if (payload == nullptr) {
      continue;
    }
    if (payload->op_name.find(":structured") == std::string::npos) {
      payload->op_name += ":structured";
    }
  }
}

void run_tier0(ir::Graph& g) {
  hoist_common_token_prefixes(g);
  memoize_pure_tool_ops(g);
  specialize_structured_outputs(g);
}

}  // namespace continuum::compiler
