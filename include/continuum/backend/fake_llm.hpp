#pragma once

#include <continuum/backend/backend.hpp>

namespace continuum::backend {

class FakeLLMBackend : public Backend {
 public:
  BackendCapabilities capabilities() const override;
  std::vector<std::uint8_t> export_state(const BackendState& state) const override;
  std::optional<BackendState> import_state(const std::vector<std::uint8_t>& bytes) override;
  BackendRunResult run_with_cache(
      const ir::Node& node,
      const std::vector<continuum::Value>& inputs,
      const std::optional<BackendState>& prefix_state,
      std::int32_t remaining_tokens) override;
};

}  // namespace continuum::backend
