#include <continuum/backend/backend.hpp>
#include <continuum/backend/backend_abi.h>
#include <continuum/ir/node.hpp>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace continuum::backend {
namespace {

BackendCapabilities FromAbiCaps(const continuum_backend_caps_t& caps) {
  return BackendCapabilities{
      static_cast<bool>(caps.supports_tensor),
      static_cast<bool>(caps.supports_token),
      static_cast<bool>(caps.supports_cache)};
}

continuum_backend_value_t ToAbiValue(const continuum::Value& value) {
  if (const auto* s = std::get_if<std::string>(&value)) {
    return continuum_backend_value_t{CONTINUUM_BACKEND_VALUE_STRING, s->c_str(), nullptr, 0};
  }
  if (const auto* t = std::get_if<continuum::TokensValue>(&value)) {
    return continuum_backend_value_t{
        CONTINUUM_BACKEND_VALUE_TOKENS, nullptr, reinterpret_cast<const int32_t*>(t->ids.data()), t->ids.size()};
  }
  return continuum_backend_value_t{CONTINUUM_BACKEND_VALUE_NONE, nullptr, nullptr, 0};
}

continuum::Value FromAbiValue(const continuum_backend_value_t& value) {
  if (value.kind == CONTINUUM_BACKEND_VALUE_STRING) {
    return value.string_data == nullptr ? std::string{} : std::string(value.string_data);
  }
  if (value.kind == CONTINUUM_BACKEND_VALUE_TOKENS) {
    continuum::TokensValue out;
    out.ids.reserve(value.token_count);
    for (size_t i = 0; i < value.token_count; ++i) {
      out.ids.push_back(static_cast<int>(value.token_ids[i]));
    }
    return out;
  }
  return std::string{};
}

continuum_backend_node_meta_t ToAbiNodeMeta(const ir::Node& node) {
  continuum_backend_node_meta_t out{};
  out.node_kind = static_cast<uint8_t>(node.kind);
  if (const auto* payload = std::get_if<ir::TensorOpPayload>(&node.payload)) {
    out.op_name = payload->op_name.c_str();
    out.attrs = payload->attrs.data();
    out.attr_count = payload->attrs.size();
  } else if (const auto* payload = std::get_if<ir::TokenOpPayload>(&node.payload)) {
    out.op_name = payload->op_name.c_str();
    out.model_id = payload->model_id.c_str();
  }
  return out;
}

}  // namespace

class BackendAbiAdapter final : public Backend {
 public:
  explicit BackendAbiAdapter(continuum_backend_vtable_t vtable) : vtable_(vtable) {
    if (vtable_.abi_version != CONTINUUM_BACKEND_ABI_VERSION) {
      throw std::runtime_error("backend ABI version mismatch");
    }
  }

  BackendCapabilities capabilities() const override {
    if (vtable_.capabilities == nullptr) {
      return BackendCapabilities{};
    }
    return FromAbiCaps(vtable_.capabilities(vtable_.instance));
  }

  std::string tensor_backend_type() const override {
    if (vtable_.tensor_backend_type == nullptr) {
      return "";
    }
    const char* name = vtable_.tensor_backend_type(vtable_.instance);
    return name == nullptr ? std::string{} : std::string(name);
  }

  BackendRunResult run_with_cache(
      const ir::Node& node,
      const std::vector<continuum::Value>& inputs,
      const std::optional<BackendState>& prefix_state,
      std::int32_t remaining_tokens) override {
    if (vtable_.run_with_cache == nullptr) {
      throw std::runtime_error("backend ABI vtable missing run_with_cache");
    }
    std::vector<continuum_backend_value_t> abi_inputs;
    abi_inputs.reserve(inputs.size());
    for (const auto& v : inputs) {
      abi_inputs.push_back(ToAbiValue(v));
    }
    const auto abi_node = ToAbiNodeMeta(node);
    const continuum_backend_state_t* abi_prefix = nullptr;
    continuum_backend_state_t abi_prefix_value{};
    if (prefix_state.has_value()) {
      abi_prefix_value.handle = prefix_state->handle;
      abi_prefix = &abi_prefix_value;
    }
    const auto abi_out = vtable_.run_with_cache(
        vtable_.instance,
        abi_node,
        abi_inputs.empty() ? nullptr : abi_inputs.data(),
        abi_inputs.size(),
        abi_prefix,
        remaining_tokens);
    BackendRunResult out;
    out.output = FromAbiValue(abi_out.output);
    out.resulting_state = BackendState{abi_out.resulting_state.handle};
    out.reused_prefix_len = abi_out.reused_prefix_len;
    out.compute_steps = abi_out.compute_steps;
    out.tokens_sent = abi_out.tokens_sent;
    out.tokens_saved = abi_out.tokens_saved;
    out.used_cached_state = static_cast<bool>(abi_out.used_cached_state);
    return out;
  }

 private:
  continuum_backend_vtable_t vtable_{};
};

std::shared_ptr<Backend> MakeBackendFromAbi(continuum_backend_vtable_t vtable) {
  return std::make_shared<BackendAbiAdapter>(vtable);
}

}  // namespace continuum::backend
