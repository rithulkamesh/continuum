#pragma once

#include <continuum/backend/backend.hpp>

#include <string>

namespace continuum::backend {

/// Tensor backend behind the `mlx` tensor type (`CONTINUUM_TENSOR_BACKEND=mlx`).
///
/// Built with Apple's MLX (`CONTINUUM_USE_MLX`, on by default when the `mlx`
/// package is installed on Apple silicon), every op runs as an `mlx::core`
/// kernel: on the Metal GPU on Apple silicon (`CONTINUUM_MLX_DEVICE=cpu` to
/// pin the CPU), on the CPU with Linux wheels. Built without MLX, the same ops
/// fall back to portable C++ reference kernels. `runtime()` reports which is
/// in use. Tensors are row-major float32 `MlxTensorValue`s. Ops: `identity` /
/// `input` / `id`, `relu`, `softmax` (attr 0 = dim, default -1), `add`
/// (same shape), `matmul` (1D x 1D dot, or 2D x 2D). Tensor-only: no token
/// path, no cache, no portable state. See the capability matrix in
/// docs/design/abi.md.
class MLXBackend : public Backend {
 public:
  BackendCapabilities capabilities() const override;
  std::string tensor_backend_type() const override { return "mlx"; }
  /// "mlx <version> (gpu|cpu)" when built against MLX, else "reference (built without MLX)".
  static std::string runtime();
  BackendRunResult run_with_cache(
      const ir::Node& node,
      const std::vector<continuum::Value>& inputs,
      const std::optional<BackendState>& prefix_state,
      std::int32_t remaining_tokens) override;
};

}  // namespace continuum::backend
