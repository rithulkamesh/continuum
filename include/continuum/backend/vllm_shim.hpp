#pragma once
#include <continuum/backend/backend.hpp>

#include <cstdint>
#include <string>

namespace continuum::backend {

/// Token backend for any server speaking the OpenAI `/v1/completions` wire
/// format: vLLM, Ollama, llama.cpp server, and similar.
///
/// With `VLLM_BASE_URL` set, each call POSTs the full prompt to
/// `$VLLM_BASE_URL/v1/completions` and returns the completion text as a
/// string value. The model name is the node's `model_id` with a leading
/// `vllm/` stripped (`vllm/gemma4` -> `gemma4`), else `VLLM_MODEL`. Without
/// `VLLM_BASE_URL` it runs offline and returns deterministic token ids.
class VllmShimBackend : public Backend {
 public:
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
};

}  // namespace continuum::backend
