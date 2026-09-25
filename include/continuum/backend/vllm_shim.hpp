#pragma once
#include <continuum/backend/backend.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace continuum::backend {

/// Token backend for any server speaking the OpenAI `/v1/completions` wire
/// format: vLLM, Ollama, llama.cpp server, and similar.
///
/// With `VLLM_BASE_URL` set, each call POSTs the full prompt to
/// `$VLLM_BASE_URL/v1/completions` and returns the completion text as a
/// string value. The model name is the node's `model_id` with a leading
/// `vllm/` stripped (`vllm/gemma4` -> `gemma4`), else `VLLM_MODEL`. Without
/// `VLLM_BASE_URL` it runs offline and returns deterministic token ids.
///
/// Prefix reuse: the server keeps the KV blocks (vLLM automatic prefix
/// caching, Ollama / llama.cpp prompt cache) and skips recomputing a shared
/// prefix; its `usage.prompt_tokens_details.cached_tokens` is reported as
/// `tokens_saved`. Continuum's state handle records which prefix the server
/// holds warm. It is portable (export_state / import_state), so it survives a
/// checkpoint / resume, and with `VLLM_REWARM_ON_IMPORT=1` an imported state
/// re-warms the server with a 1-token request over the prefix.
class VllmShimBackend : public Backend {
 public:
  /// Prefix text and model a state handle refers to.
  struct PrefixState {
    std::string prefix_text;
    std::string model;
  };

  std::vector<std::uint8_t> export_state(const BackendState& state) const override;
  std::optional<BackendState> import_state(const std::vector<std::uint8_t>& bytes) override;
  /// Live state handles (bounded; oldest dropped first).
  std::size_t state_count() const;
  /// Re-warm requests sent by import_state.
  std::size_t rewarm_count() const { return rewarm_count_.load(); }

  /// Decode the first JSON string value stored under \p key in \p body,
  /// resolving escapes (including `\uXXXX` and surrogate pairs) to UTF-8.
  /// Returns an empty string when the key is absent.
  static std::string ExtractJsonString(const std::string& body, const std::string& key);
  /// First non-negative integer stored under \p key in \p body, or 0.
  static std::int32_t ExtractJsonInt(const std::string& body, const std::string& key);

  BackendCapabilities capabilities() const override;
  BackendRunResult run_with_cache(
      const ir::Node& node,
      const std::vector<continuum::Value>& inputs,
      const std::optional<BackendState>& prefix_state,
      std::int32_t remaining_tokens) override;

 private:
  static constexpr std::size_t kMaxStates = 4096;
  BackendState remember(std::string prefix_text, std::string model);
  std::optional<PrefixState> lookup(const BackendState& state) const;

  mutable std::mutex mu_;
  std::map<std::uint64_t, PrefixState> states_;  // ordered: begin() is the oldest
  std::uint64_t next_state_id_ = 1;
  std::atomic<std::size_t> rewarm_count_{0};
};

}  // namespace continuum::backend
