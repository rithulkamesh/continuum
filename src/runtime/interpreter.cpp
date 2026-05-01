#include <continuum/runtime/interpreter.hpp>
#include <continuum/runtime/session.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <torch/torch.h>

#include <continuum/utils/logging.hpp>

namespace continuum::runtime {
namespace {

std::string CanonicalizeText(const std::string& raw) {
  std::string out;
  out.reserve(raw.size());
  bool prev_space = false;
  for (char c : raw) {
    const bool is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
    if (is_space) {
      if (!prev_space) {
        out.push_back(' ');
      }
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

std::vector<std::int32_t> TokenizeCanonicalText(const std::string& text) {
  std::vector<std::int32_t> tokens;
  tokens.reserve(text.size());
  for (unsigned char c : text) {
    tokens.push_back(static_cast<std::int32_t>(c));
  }
  return tokens;
}

std::vector<std::int32_t> CanonicalizeInputTokens(const std::vector<continuum::Value>& input_values) {
  if (!input_values.empty()) {
    if (const auto* tv = std::get_if<continuum::TokensValue>(&input_values.front())) {
      std::vector<std::int32_t> tokens;
      tokens.reserve(tv->ids.size());
      for (int id : tv->ids) {
        tokens.push_back(static_cast<std::int32_t>(id));
      }
      return tokens;
    }
  }
  std::ostringstream joined;
  for (std::size_t i = 0; i < input_values.size(); ++i) {
    if (i != 0) joined << " | ";
    const auto& v = input_values[i];
    if (const auto* s = std::get_if<std::string>(&v)) {
      joined << CanonicalizeText(*s);
    } else if (const auto* sch = std::get_if<continuum::SchemaValue>(&v)) {
      joined << CanonicalizeText(sch->json);
    } else if (const auto* d = std::get_if<double>(&v)) {
      joined << *d;
    } else if (const auto* n = std::get_if<int64_t>(&v)) {
      joined << *n;
    }
  }
  return TokenizeCanonicalText(CanonicalizeText(joined.str()));
}

bool IsTensorLike(const continuum::Value& value) {
  return std::holds_alternative<continuum::TensorValue>(value) || std::holds_alternative<continuum::MlxTensorValue>(value);
}

std::string TensorBackendOf(const continuum::Value& value) {
  if (const auto* t = std::get_if<continuum::TensorValue>(&value)) {
    return t->backend_type;
  }
  if (const auto* t = std::get_if<continuum::MlxTensorValue>(&value)) {
    return t->backend_type;
  }
  return {};
}

std::vector<continuum::Value> NormalizeTensorInputs(
    const std::vector<continuum::Value>& input_values, const std::string& target_backend) {
  std::vector<continuum::Value> out;
  out.reserve(input_values.size());
  for (const auto& value : input_values) {
    if (!IsTensorLike(value)) {
      out.push_back(value);
      continue;
    }
    if (TensorBackendOf(value) == target_backend) {
      out.push_back(value);
      continue;
    }
    out.push_back(convert_value_for_backend(value, target_backend));
  }
  return out;
}

}  // namespace

Interpreter::Interpreter(backend::BackendRegistry& backends, KVCacheIndex& cache,
                         const ReusePolicy* policy)
    : backends_(backends), cache_(cache), policy_(policy) {}

continuum::Value convert_value_for_backend(const continuum::Value& value, const std::string& target_backend) {
  if (const auto* t = std::get_if<continuum::TensorValue>(&value)) {
    if (target_backend == "libtorch") {
      return value;
    }
    if (target_backend == "mlx") {
      auto flat = t->tensor.flatten().to(torch::kFloat32).contiguous();
      std::vector<float> data(static_cast<std::size_t>(flat.numel()));
      std::memcpy(data.data(), flat.data_ptr<float>(), data.size() * sizeof(float));
      std::vector<std::int64_t> shape;
      shape.reserve(static_cast<std::size_t>(t->tensor.dim()));
      for (std::int64_t i = 0; i < t->tensor.dim(); ++i) {
        shape.push_back(t->tensor.size(i));
      }
      return continuum::MlxTensorValue{std::move(shape), std::move(data), "mlx"};
    }
    throw std::runtime_error("tensor conversion not implemented: libtorch -> " + target_backend);
  }
  if (const auto* t = std::get_if<continuum::MlxTensorValue>(&value)) {
    if (target_backend == "mlx") {
      return value;
    }
    if (target_backend == "libtorch") {
      auto out = torch::empty(
          t->shape.empty() ? std::vector<std::int64_t>{1} : t->shape,
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      if (!t->data.empty()) {
        std::memcpy(out.data_ptr<float>(), t->data.data(), t->data.size() * sizeof(float));
      }
      return continuum::TensorValue{std::move(out), "libtorch"};
    }
    throw std::runtime_error("tensor conversion not implemented: mlx -> " + target_backend);
  }
  return value;
}

std::vector<continuum::Value> Interpreter::run(
    const ir::Graph& g, const std::unordered_map<ir::NodeId, continuum::Value>& inputs) {
  begin(g, inputs);
  return run_to_end();
}

void Interpreter::begin(const ir::Graph& g, const std::unordered_map<ir::NodeId, continuum::Value>& inputs) {
  ActiveExecution state;
  state.graph = g;
  state.plan = scheduler_.schedule(state.graph);
  state.values = inputs;
  state.out.reserve(state.graph.topo_order().size());
  state.next_group_index = 0;
  state.next_node_in_group = 0;
  state.executed_nodes = 0;
  active_ = std::move(state);
}

Checkpoint Interpreter::run_until(ir::NodeId node_id) {
  if (!active_.has_value()) {
    throw std::runtime_error("interpreter run_until: no active graph");
  }
  auto& st = active_.value();
  while (st.next_group_index < st.plan.size()) {
    const auto id = next_planned_node(st);
    auto it = st.values.find(id);
    if (it != st.values.end()) {
      st.out.push_back(it->second);
      ++st.executed_nodes;
      advance_plan_cursor(st);
    } else {
      const auto& node = st.graph.get(id);
      std::vector<continuum::Value> in_vals;
      in_vals.reserve(node.inputs.size());
      for (const auto input_id : node.inputs) {
        auto in_it = st.values.find(input_id);
        if (in_it == st.values.end()) {
          throw std::runtime_error("interpreter run_until: missing input node value");
        }
        in_vals.push_back(in_it->second);
      }
      auto value = step(node, in_vals);
      st.values[id] = value;
      st.out.push_back(std::move(value));
      ++st.executed_nodes;
      advance_plan_cursor(st);
    }
    if (id == node_id) {
      break;
    }
  }
  Checkpoint cp;
  cp.serialized_graph = st.graph.serialize();
  cp.current_node_index = static_cast<std::uint64_t>(st.executed_nodes);
  cp.value_map = st.values;
  return cp;
}

std::vector<continuum::Value> Interpreter::resume(const Checkpoint& checkpoint) {
  ActiveExecution state;
  state.graph = ir::Graph::deserialize(checkpoint.serialized_graph.data(), checkpoint.serialized_graph.size());
  state.plan = scheduler_.schedule(state.graph);
  state.values = checkpoint.value_map;
  state.out.reserve(state.graph.topo_order().size());
  const auto target_executed = static_cast<std::size_t>(checkpoint.current_node_index);
  if (target_executed > state.graph.topo_order().size()) {
    throw std::runtime_error("interpreter resume: checkpoint index out of range");
  }
  state.next_group_index = 0;
  state.next_node_in_group = 0;
  state.executed_nodes = 0;
  for (std::size_t i = 0; i < target_executed; ++i) {
    const auto id = next_planned_node(state);
    auto it = state.values.find(id);
    if (it == state.values.end()) {
      throw std::runtime_error("interpreter resume: checkpoint missing computed value");
    }
    state.out.push_back(it->second);
    ++state.executed_nodes;
    advance_plan_cursor(state);
  }
  // v0.1 scope: backend state handles are not serialized; invalidate token cache on resume.
  cache_.clear();
  active_ = std::move(state);
  return run_to_end();
}

std::vector<continuum::Value> Interpreter::run_to_end() {
  if (!active_.has_value()) {
    throw std::runtime_error("interpreter run_to_end: no active graph");
  }
  auto& st = active_.value();
  while (st.next_group_index < st.plan.size()) {
    const auto id = next_planned_node(st);
    auto it = st.values.find(id);
    if (it != st.values.end()) {
      st.out.push_back(it->second);
      ++st.executed_nodes;
      advance_plan_cursor(st);
      continue;
    }
    const auto& node = st.graph.get(id);
    std::vector<continuum::Value> in_vals;
    in_vals.reserve(node.inputs.size());
    for (const auto input_id : node.inputs) {
      auto in_it = st.values.find(input_id);
      if (in_it == st.values.end()) {
        throw std::runtime_error("interpreter run: missing input node value");
      }
      in_vals.push_back(in_it->second);
    }
    auto value = step(node, in_vals);
    st.values[id] = value;
    st.out.push_back(std::move(value));
    ++st.executed_nodes;
    advance_plan_cursor(st);
  }
  return st.out;
}

ir::NodeId Interpreter::next_planned_node(const ActiveExecution& state) const {
  if (state.next_group_index >= state.plan.size()) {
    throw std::runtime_error("interpreter plan cursor out of range");
  }
  const auto& group = state.plan[state.next_group_index];
  if (state.next_node_in_group >= group.size()) {
    throw std::runtime_error("interpreter group cursor out of range");
  }
  return group[state.next_node_in_group];
}

void Interpreter::advance_plan_cursor(ActiveExecution& state) {
  const auto& group = state.plan[state.next_group_index];
  ++state.next_node_in_group;
  if (state.next_node_in_group >= group.size()) {
    ++state.next_group_index;
    state.next_node_in_group = 0;
  }
}

continuum::Value Interpreter::step(const ir::Node& n, const std::vector<continuum::Value>& input_values) {
  if (n.kind == ir::NodeKind::TensorOp) {
    auto selected = backends_.select_backend(n);
    const auto target_backend =
        selected.backend->tensor_backend_type().empty() ? selected.name : selected.backend->tensor_backend_type();
    auto normalized_inputs = NormalizeTensorInputs(input_values, target_backend);
    auto run_result = selected.backend->run_with_cache(n, normalized_inputs, std::nullopt, 0);
    return run_result.output;
  }
  if (n.kind == ir::NodeKind::TokenOp) {
    const auto* payload = std::get_if<ir::TokenOpPayload>(&n.payload);
    std::vector<std::int32_t> input_tokens = CanonicalizeInputTokens(input_values);
    const auto target_tokens = payload == nullptr ? 0 : payload->max_tokens;
    const DecodeParams decode_params{
        payload == nullptr ? std::string{} : payload->op_name,
        payload == nullptr ? 1.0f : payload->temperature,
        payload == nullptr ? 0 : payload->max_tokens};
    std::optional<continuum::runtime::CacheEntry> cache_hit;
    std::int32_t prefix_len = 0;
    const bool policy_allows_cache = (policy_ == nullptr || policy_->kind != ReusePolicyKind::Never);
    if (policy_allows_cache && payload != nullptr && !input_tokens.empty()) {
      auto hit = cache_.longest_prefix(payload->model_id, decode_params, input_tokens);
      if (hit.has_value()) {
        const std::int32_t hit_len = std::min<std::int32_t>(hit->second, static_cast<std::int32_t>(input_tokens.size()));
        if (policy_ == nullptr || policy_->should_attempt(input_tokens, hit_len)) {
          cache_hit = hit->first;
          prefix_len = hit_len;
        }
      }
    }
    const std::int32_t remaining_tokens =
        std::max<std::int32_t>(0, static_cast<std::int32_t>(input_tokens.size()) - prefix_len);
    const int skipped_tokens = prefix_len;
    const auto selected = backends_.select_backend(n);
    const auto& key = selected.name;
    if (cache_hit.has_value()) {
      LOG_INFO(
          runtime,
          "cache_hit backend={} model={} prefix_len={} skipped_tokens={} remaining_tokens={}",
          key,
          payload->model_id,
          prefix_len,
          skipped_tokens,
          remaining_tokens);
    } else {
      LOG_INFO(
          runtime,
          "cache_miss backend={} model={} skipped_tokens=0 remaining_tokens={}",
          key,
          (payload == nullptr ? "" : payload->model_id),
          remaining_tokens);
    }
    try {
      const std::optional<backend::BackendState> prefix_state =
          cache_hit.has_value() ? std::optional<backend::BackendState>{cache_hit->backend_state} : std::nullopt;
      auto run_result = selected.backend->run_with_cache(n, input_values, prefix_state, remaining_tokens);
      auto out = run_result.output;
      if (payload != nullptr) {
        // Invariant: cache_prefix_len must align with canonical input token count, or later prefix hits become invalid.
        const std::int32_t cache_prefix_len = run_result.reused_prefix_len > 0
                                                  ? std::min<std::int32_t>(run_result.reused_prefix_len, static_cast<std::int32_t>(input_tokens.size()))
                                                  : static_cast<std::int32_t>(input_tokens.size());
        cache_.insert(
            continuum::runtime::CacheEntry{
                0,
                payload->model_id,
                decode_params,
                cache_prefix_len,
                run_result.resulting_state,
                0},
            input_tokens);
        LOG_INFO(
            runtime,
            "backend_run backend={} model={} compute_steps={} used_cached_state={} reused_prefix_len={} target_tokens={} tokens_sent={} tokens_saved={}",
            key,
            payload->model_id,
            run_result.compute_steps,
            run_result.used_cached_state ? 1 : 0,
            run_result.reused_prefix_len,
            target_tokens,
            run_result.tokens_sent,
            run_result.tokens_saved);
      }
      return out;
    } catch (const std::runtime_error&) {
      throw;
    }
  }
  if (n.kind == ir::NodeKind::PromptOp) {
    if (!input_values.empty()) {
      return input_values.front();
    }
    return std::string{};
  }
  if (n.kind == ir::NodeKind::ToolOp) {
    if (backends_.has("default")) {
      return backends_.get("default")->run_with_cache(n, input_values, std::nullopt, 0).output;
    }
    return input_values.empty() ? continuum::Value{std::string{}} : input_values.back();
  }
  if (!input_values.empty()) {  // ControlOp fallback
    return input_values.back();
  }
  return std::string{};
}

}  // namespace continuum::runtime
