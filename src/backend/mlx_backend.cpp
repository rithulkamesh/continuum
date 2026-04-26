#include <continuum/backend/mlx_backend.hpp>

#include <torch/torch.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace continuum::backend {
namespace {

continuum::MlxTensorValue ToMlx(const continuum::Value& v) {
  if (const auto* mx = std::get_if<continuum::MlxTensorValue>(&v)) {
    return *mx;
  }
  const auto* tv = std::get_if<continuum::TensorValue>(&v);
  if (tv == nullptr) {
    throw std::runtime_error("mlx backend: expected TensorValue or MlxTensorValue");
  }
  auto flat = tv->tensor.flatten().to(torch::kFloat32).contiguous();
  std::vector<float> data(static_cast<std::size_t>(flat.numel()));
  std::memcpy(data.data(), flat.data_ptr<float>(), data.size() * sizeof(float));
  std::vector<std::int64_t> shape;
  shape.reserve(static_cast<std::size_t>(tv->tensor.dim()));
  for (std::int64_t i = 0; i < tv->tensor.dim(); ++i) {
    shape.push_back(tv->tensor.size(i));
  }
  return continuum::MlxTensorValue{std::move(shape), std::move(data), "mlx"};
}

continuum::MlxTensorValue Add(const continuum::MlxTensorValue& a, const continuum::MlxTensorValue& b) {
  if (a.shape != b.shape || a.data.size() != b.data.size()) {
    throw std::runtime_error("mlx backend: add expects same-shaped tensors");
  }
  continuum::MlxTensorValue out{a.shape, std::vector<float>(a.data.size(), 0.0f)};
  for (std::size_t i = 0; i < a.data.size(); ++i) {
    out.data[i] = a.data[i] + b.data[i];
  }
  return out;
}

continuum::MlxTensorValue Relu(const continuum::MlxTensorValue& a) {
  continuum::MlxTensorValue out{a.shape, std::vector<float>(a.data.size(), 0.0f)};
  for (std::size_t i = 0; i < a.data.size(); ++i) {
    out.data[i] = std::max(0.0f, a.data[i]);
  }
  return out;
}

continuum::MlxTensorValue Matmul2D(const continuum::MlxTensorValue& a, const continuum::MlxTensorValue& b) {
  if (a.shape.size() == 1 && b.shape.size() == 1) {
    if (a.shape[0] != b.shape[0]) {
      throw std::runtime_error("mlx backend: 1D matmul shape mismatch");
    }
    float acc = 0.0f;
    for (std::int64_t i = 0; i < a.shape[0]; ++i) {
      acc += a.data[static_cast<std::size_t>(i)] * b.data[static_cast<std::size_t>(i)];
    }
    return continuum::MlxTensorValue{{1}, {acc}, "mlx"};
  }
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::runtime_error("mlx backend: matmul currently supports 2D tensors only");
  }
  const auto m = a.shape[0];
  const auto k = a.shape[1];
  const auto k2 = b.shape[0];
  const auto n = b.shape[1];
  if (k != k2) {
    throw std::runtime_error("mlx backend: matmul shape mismatch");
  }
  continuum::MlxTensorValue out{{m, n}, std::vector<float>(static_cast<std::size_t>(m * n), 0.0f)};
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::int64_t p = 0; p < k; ++p) {
        acc += a.data[static_cast<std::size_t>(i * k + p)] * b.data[static_cast<std::size_t>(p * n + j)];
      }
      out.data[static_cast<std::size_t>(i * n + j)] = acc;
    }
  }
  return out;
}

continuum::MlxTensorValue Softmax(const continuum::MlxTensorValue& a, std::int64_t dim) {
  if (a.shape.empty()) {
    throw std::runtime_error("mlx backend: softmax expects rank >= 1");
  }
  const auto rank = static_cast<std::int64_t>(a.shape.size());
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    throw std::runtime_error("mlx backend: softmax dim out of range");
  }
  const auto axis = a.shape[static_cast<std::size_t>(dim)];
  std::int64_t inner = 1;
  for (std::int64_t i = dim + 1; i < rank; ++i) inner *= a.shape[static_cast<std::size_t>(i)];
  std::int64_t outer = 1;
  for (std::int64_t i = 0; i < dim; ++i) outer *= a.shape[static_cast<std::size_t>(i)];
  continuum::MlxTensorValue out{a.shape, std::vector<float>(a.data.size(), 0.0f)};
  for (std::int64_t o = 0; o < outer; ++o) {
    for (std::int64_t in = 0; in < inner; ++in) {
      const auto base = o * axis * inner + in;
      float max_v = -std::numeric_limits<float>::infinity();
      for (std::int64_t i = 0; i < axis; ++i) {
        max_v = std::max(max_v, a.data[static_cast<std::size_t>(base + i * inner)]);
      }
      float sum = 0.0f;
      for (std::int64_t i = 0; i < axis; ++i) {
        const float e = std::exp(a.data[static_cast<std::size_t>(base + i * inner)] - max_v);
        out.data[static_cast<std::size_t>(base + i * inner)] = e;
        sum += e;
      }
      for (std::int64_t i = 0; i < axis; ++i) {
        out.data[static_cast<std::size_t>(base + i * inner)] /= sum;
      }
    }
  }
  return out;
}

}  // namespace

BackendCapabilities MLXBackend::capabilities() const {
  return BackendCapabilities{true, false, false};
}

BackendRunResult MLXBackend::run_with_cache(
    const ir::Node& node,
    const std::vector<continuum::Value>& inputs,
    const std::optional<BackendState>&,
    std::int32_t) {
  BackendRunResult r;
  r.resulting_state = BackendState{nullptr};
  r.used_cached_state = false;
  r.reused_prefix_len = 0;
  r.compute_steps = 0;

  if (node.kind != ir::NodeKind::TensorOp) {
    if (!inputs.empty()) {
      r.output = inputs.front();
    } else {
      r.output = std::string("mlx");
    }
    return r;
  }

  const auto* payload = std::get_if<ir::TensorOpPayload>(&node.payload);
  if (payload == nullptr) {
    throw std::runtime_error("mlx backend: tensor op missing payload");
  }
  if (payload->op_name == "input" || payload->op_name == "identity" || payload->op_name == "id") {
    r.output = inputs.at(0);
    r.compute_steps = 1;
    return r;
  }

  auto a = ToMlx(inputs.at(0));
  if (payload->op_name == "relu") {
    r.output = Relu(a);
    r.compute_steps = 1;
    return r;
  }
  if (payload->op_name == "softmax") {
    const auto dim = payload->attrs.empty() ? -1 : payload->attrs[0];
    r.output = Softmax(a, dim);
    r.compute_steps = 1;
    return r;
  }
  if (payload->op_name == "add") {
    auto b = ToMlx(inputs.at(1));
    r.output = Add(a, b);
    r.compute_steps = 1;
    return r;
  }
  if (payload->op_name == "matmul") {
    auto b = ToMlx(inputs.at(1));
    r.output = Matmul2D(a, b);
    r.compute_steps = 1;
    return r;
  }

  throw std::runtime_error("mlx backend: unsupported tensor op: " + payload->op_name);
}

}  // namespace continuum::backend
