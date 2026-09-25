// MLX backend suite. Test names start with "MLX" so CI can run them alone on
// the arm64 macOS runner: ctest -R '^MLX'.
#include <continuum/backend/conformance.hpp>
#include <continuum/backend/libtorch.hpp>
#include <continuum/backend/mlx_backend.hpp>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace {

using continuum::MlxTensorValue;
using continuum::TensorValue;
using continuum::Value;
using continuum::ir::Node;
using continuum::ir::NodeKind;
using continuum::ir::TensorOpPayload;

Node Op(const std::string& name, std::vector<std::int64_t> attrs = {}) {
  Node n;
  n.kind = NodeKind::TensorOp;
  n.payload = TensorOpPayload{name, std::move(attrs)};
  return n;
}

Value Torch(const torch::Tensor& t) { return TensorValue{t, "libtorch"}; }

// Run `op` on both backends and require matching output.
void ExpectParity(const std::string& op, const std::vector<torch::Tensor>& in, std::vector<std::int64_t> attrs = {}) {
  continuum::backend::MLXBackend mlx;
  continuum::backend::LibTorchBackend lt;
  std::vector<Value> values;
  for (const auto& t : in) values.push_back(Torch(t));
  const auto node = Op(op, std::move(attrs));
  const auto want = std::get<TensorValue>(lt.run_with_cache(node, values, std::nullopt, 0).output).tensor;
  const auto got_value = mlx.run_with_cache(node, values, std::nullopt, 0).output;
  ASSERT_TRUE(std::holds_alternative<MlxTensorValue>(got_value)) << op;
  const auto& got = std::get<MlxTensorValue>(got_value);
  const auto flat = want.flatten().to(torch::kFloat32).contiguous();
  ASSERT_EQ(got.data.size(), static_cast<std::size_t>(flat.numel())) << op;
  for (std::int64_t i = 0; i < flat.numel(); ++i) {
    EXPECT_NEAR(got.data[static_cast<std::size_t>(i)], flat[i].item<float>(), 1e-5f) << op << " element " << i;
  }
  if (want.dim() > 0) {
    std::vector<std::int64_t> shape(want.sizes().begin(), want.sizes().end());
    EXPECT_EQ(got.shape, shape) << op;
  }
}

}  // namespace

TEST(MLXBackend, CapabilitiesAndConformance) {
  continuum::backend::MLXBackend mlx;
  const auto caps = mlx.capabilities();
  EXPECT_TRUE(caps.supports_tensor);
  EXPECT_FALSE(caps.supports_token);
  EXPECT_FALSE(caps.supports_cache);
  EXPECT_EQ(mlx.tensor_backend_type(), "mlx");
  const auto report = continuum::backend::RunBackendConformance(mlx, "mlx");
  EXPECT_TRUE(report.passed()) << report.summary();
}

TEST(MLXBackend, ParityWithLibTorch) {
  torch::manual_seed(7);
  const auto a = torch::randn({3, 4});
  const auto b = torch::randn({3, 4});
  ExpectParity("relu", {a});
  ExpectParity("add", {a, b});
  ExpectParity("softmax", {a}, {-1});
  ExpectParity("softmax", {a}, {0});
  ExpectParity("softmax", {torch::randn({2, 3, 4})}, {1});
  ExpectParity("matmul", {a, torch::randn({4, 5})});
  ExpectParity("matmul", {torch::randn({6}), torch::randn({6})});
}

TEST(MLXBackend, SoftmaxIsNumericallyStable) {
  continuum::backend::MLXBackend mlx;
  const std::vector<Value> in{MlxTensorValue{{3}, {1000.0f, 1001.0f, 1002.0f}}};
  const auto out = std::get<MlxTensorValue>(mlx.run_with_cache(Op("softmax"), in, std::nullopt, 0).output);
  float sum = 0.0f;
  for (float x : out.data) {
    EXPECT_TRUE(std::isfinite(x));
    sum += x;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(MLXBackend, RejectsMalformedInputs) {
  continuum::backend::MLXBackend mlx;
  auto run = [&](const Node& n, std::vector<Value> in) { return mlx.run_with_cache(n, in, std::nullopt, 0); };
  // Shape / data disagreement would index out of bounds without validation.
  EXPECT_THROW(run(Op("relu"), {MlxTensorValue{{2, 3}, {1.0f}}}), std::runtime_error);
  EXPECT_THROW(run(Op("relu"), {MlxTensorValue{{-1}, {}}}), std::runtime_error);
  EXPECT_THROW(run(Op("add"), {MlxTensorValue{{2}, {1, 2}}}), std::runtime_error);  // missing input
  EXPECT_THROW(run(Op("identity"), {}), std::runtime_error);
  EXPECT_THROW(run(Op("add"), {MlxTensorValue{{2}, {1, 2}}, MlxTensorValue{{3}, {1, 2, 3}}}), std::runtime_error);
  EXPECT_THROW(run(Op("matmul"), {MlxTensorValue{{2, 3}, std::vector<float>(6)}, MlxTensorValue{{2, 3}, std::vector<float>(6)}}),
               std::runtime_error);
  EXPECT_THROW(run(Op("matmul"), {MlxTensorValue{{2}, {1, 2}}, MlxTensorValue{{3}, {1, 2, 3}}}), std::runtime_error);
  EXPECT_THROW(run(Op("matmul"), {MlxTensorValue{{1, 1, 1}, {1}}, MlxTensorValue{{1, 1, 1}, {1}}}), std::runtime_error);
  EXPECT_THROW(run(Op("softmax", {5}), {MlxTensorValue{{2}, {1, 2}}}), std::runtime_error);
  EXPECT_THROW(run(Op("softmax"), {MlxTensorValue{{}, {1}}}), std::runtime_error);
  EXPECT_THROW(run(Op("conv2d"), {MlxTensorValue{{1}, {1}}}), std::runtime_error);
  EXPECT_THROW(run(Op("relu"), {std::string{"not a tensor"}}), std::runtime_error);
  Node no_payload;
  no_payload.kind = NodeKind::TensorOp;
  no_payload.payload = continuum::ir::TokenOpPayload{};
  EXPECT_THROW(run(no_payload, {MlxTensorValue{{1}, {1}}}), std::runtime_error);
}

TEST(MLXBackend, EmptyAndNonTensorPaths) {
  continuum::backend::MLXBackend mlx;
  const std::vector<Value> empty{MlxTensorValue{{0}, {}}};
  const auto relu = std::get<MlxTensorValue>(mlx.run_with_cache(Op("relu"), empty, std::nullopt, 0).output);
  EXPECT_TRUE(relu.data.empty());
  Node token;
  token.kind = NodeKind::TokenOp;
  EXPECT_EQ(std::get<std::string>(mlx.run_with_cache(token, {}, std::nullopt, 0).output), "mlx");
}

TEST(MLXBackend, ReportsRuntime) {
  const auto runtime = continuum::backend::MLXBackend::runtime();
#ifdef CONTINUUM_HAVE_MLX
  EXPECT_EQ(runtime.rfind("mlx ", 0), 0u) << runtime;  // real Apple MLX kernels
#else
  EXPECT_EQ(runtime, "reference (built without MLX)");
#endif
}
