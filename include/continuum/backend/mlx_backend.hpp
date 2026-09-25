#pragma once

#include <continuum/backend/backend.hpp>

namespace continuum::backend {

/// Tensor backend behind the `mlx` tensor type (`CONTINUUM_TENSOR_BACKEND=mlx`).
///
/// A portable, dependency-free C++ reference implementation of the tensor ops
/// on `MlxTensorValue` (row-major float32); it does not link Apple's MLX
/// framework, so it builds and runs on every platform. Ops: `identity` /
/// `input` / `id`, `relu`, `softmax` (attr 0 = dim, default -1), `add`
/// (same shape), `matmul` (1D x 1D dot, or 2D x 2D). Tensor-only: no token
/// path, no cache, no portable state. See the capability matrix in
/// docs/design/abi.md.
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
