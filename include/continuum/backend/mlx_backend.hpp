#pragma once

#include <continuum/backend/backend.hpp>

namespace continuum::backend {

class MLXBackend : public Backend {
 public:
  BackendCapabilities capabilities() const override;
  std::string tensor_backend_type() const override { return "mlx"; }
  BackendRunResult run_with_cache(
      const ir::Node& node,
      const std::vector<continuum::Value>& inputs,
      const std::optional<BackendState>& prefix_state,
      std::int32_t remaining_tokens) override;
};

}  // namespace continuum::backend
