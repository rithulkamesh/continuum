#include <continuum/backend/anthropic.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace continuum::backend {
namespace {

std::vector<std::int32_t> ExtractPromptTokens(const std::vector<continuum::Value>& inputs, int salt) {
  std::string prompt;
  if (!inputs.empty()) {
    if (const auto* s = std::get_if<std::string>(&inputs[0])) {
      prompt = *s;
    } else if (const auto* t = std::get_if<continuum::TokensValue>(&inputs[0])) {
      std::vector<std::int32_t> out;
      out.reserve(t->ids.size());
      for (int id : t->ids) {
        out.push_back(id);
      }
      return out;
    }
  }
  if (prompt.empty()) {
    prompt = " ";
  }
  std::vector<std::int32_t> out;
  out.reserve(prompt.size());
  for (unsigned char c : prompt) {
    out.push_back((static_cast<int>(c) + salt) % 10009);
  }
  return out;
}

continuum::TokensValue GenerateDeterministicOutput(
    const std::vector<std::int32_t>& prompt_tokens, std::int32_t max_tokens, int salt) {
  continuum::TokensValue out;
  out.ids.reserve(static_cast<std::size_t>(std::max<std::int32_t>(0, max_tokens)));
  for (std::int32_t i = 0; i < max_tokens; ++i) {
    const auto base = prompt_tokens.empty() ? 0 : prompt_tokens[static_cast<std::size_t>(i) % prompt_tokens.size()];
    out.ids.push_back((base + salt + i) % 10009);
  }
  return out;
}

}  // namespace

BackendCapabilities AnthropicBackend::capabilities() const {
  return BackendCapabilities{false, true, true};
}

BackendRunResult AnthropicBackend::run_with_cache(
    const ir::Node& node,
    const std::vector<continuum::Value>& inputs,
    const std::optional<BackendState>& prefix_state,
    std::int32_t remaining_tokens) {
  const auto prompt_tokens = ExtractPromptTokens(inputs, 29);
  const auto total_prompt = static_cast<std::int32_t>(prompt_tokens.size());
  const auto reused_prefix = std::max<std::int32_t>(0, total_prompt - std::max<std::int32_t>(0, remaining_tokens));
  const auto* payload = std::get_if<ir::TokenOpPayload>(&node.payload);
  const auto max_tokens = payload == nullptr ? 0 : payload->max_tokens;

  static std::uint64_t next_state = 1;
  static std::unordered_map<std::uint64_t, std::int32_t> prefix_by_state;
  const auto new_state_id = next_state++;
  prefix_by_state[new_state_id] = total_prompt;

  BackendRunResult r;
  r.output = GenerateDeterministicOutput(prompt_tokens, max_tokens, 29);
  r.resulting_state.handle = reinterpret_cast<void*>(new_state_id);
  r.reused_prefix_len = (prefix_state.has_value() && prefix_state->handle != nullptr) ? reused_prefix : 0;
  r.compute_steps = std::max<std::int32_t>(0, remaining_tokens) + std::max<std::int32_t>(0, max_tokens);
  r.used_cached_state = prefix_state.has_value() && prefix_state->handle != nullptr;
  return r;
}

}  // namespace continuum::backend
