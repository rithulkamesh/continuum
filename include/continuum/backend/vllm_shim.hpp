#pragma once
#include <continuum/backend/backend.hpp>

namespace continuum::backend {

class VllmShimBackend : public Backend {
 public:
  BackendCapabilities capabilities() const override;
  BackendRunResult run_with_cache(
      const ir::Node& node,
      const std::vector<continuum::Value>& inputs,
      const std::optional<BackendState>& prefix_state,
      std::int32_t remaining_tokens) override;
};

}  // namespace continuum::backend
