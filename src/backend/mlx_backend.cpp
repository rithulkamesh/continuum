#include <continuum/backend/mlx_backend.hpp>

#include <torch/torch.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef CONTINUUM_HAVE_MLX
#include <mlx/mlx.h>
#endif

namespace continuum::backend {
namespace {

#ifdef CONTINUUM_HAVE_MLX
namespace mx = mlx::core;

// Apple silicon runs on the Metal GPU by default; CONTINUUM_MLX_DEVICE=cpu
// pins the CPU (Linux wheels are CPU-only either way).
void SelectDeviceOnce() {
  static const bool selected = [] {
    const char* env = std::getenv("CONTINUUM_MLX_DEVICE");
    const std::string want = env == nullptr ? "" : env;
    if (want == "cpu") {
      mx::set_default_device(mx::Device::cpu);
    } else if (want == "gpu" || want.empty()) {
      if (mx::is_available(mx::Device::gpu)) {
        mx::set_default_device(mx::Device::gpu);
      } else if (want == "gpu") {
        throw std::runtime_error("mlx backend: CONTINUUM_MLX_DEVICE=gpu but no MLX GPU is available");
      }
    }
    return true;
  }();
  (void)selected;
}

mx::array ToArray(const continuum::MlxTensorValue& t) {
  SelectDeviceOnce();
  mx::Shape shape(t.shape.begin(), t.shape.end());
  return mx::array(t.data.begin(), std::move(shape), mx::float32);
}

continuum::MlxTensorValue FromArray(mx::array a) {
  a = mx::contiguous(a);
  mx::eval(a);
  continuum::MlxTensorValue out;
  out.shape.assign(a.shape().begin(), a.shape().end());
  const float* p = a.data<float>();
  out.data.assign(p, p + a.size());
  out.backend_type = "mlx";
  return out;
}
#endif

// Reject tensors whose shape disagrees with their data: every kernel below
// indexes `data` by shape, so a mismatch would read out of bounds.
void Validate(const continuum::MlxTensorValue& t) {
  std::size_t elems = 1;
  for (const auto d : t.shape) {
    if (d < 0) {
      throw std::runtime_error("mlx backend: negative dimension in tensor shape");
    }
    const auto ud = static_cast<std::size_t>(d);
    if (ud != 0 && elems > std::numeric_limits<std::size_t>::max() / ud) {
      throw std::runtime_error("mlx backend: tensor shape overflows");
    }
    elems *= ud;
  }
  if (elems != t.data.size()) {
    throw std::runtime_error("mlx backend: tensor shape describes " + std::to_string(elems) +
                             " elements but data holds " + std::to_string(t.data.size()));
  }
}

const continuum::Value& Input(const std::vector<continuum::Value>& inputs, std::size_t i, const std::string& op) {
  if (i >= inputs.size()) {
    throw std::runtime_error("mlx backend: " + op + " expects " + std::to_string(i + 1) + " input(s), got " +
                             std::to_string(inputs.size()));
  }
  return inputs[i];
}

continuum::MlxTensorValue ToMlx(const continuum::Value& v) {
  if (const auto* mx = std::get_if<continuum::MlxTensorValue>(&v)) {
    Validate(*mx);
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
#ifdef CONTINUUM_HAVE_MLX
  return FromArray(mx::add(ToArray(a), ToArray(b)));
#endif
  continuum::MlxTensorValue out{a.shape, std::vector<float>(a.data.size(), 0.0f)};
  for (std::size_t i = 0; i < a.data.size(); ++i) {
    out.data[i] = a.data[i] + b.data[i];
  }
  return out;
}

continuum::MlxTensorValue Relu(const continuum::MlxTensorValue& a) {
#ifdef CONTINUUM_HAVE_MLX
  return FromArray(mx::maximum(ToArray(a), mx::array(0.0f)));
#endif
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
#ifdef CONTINUUM_HAVE_MLX
    return FromArray(mx::reshape(mx::sum(mx::multiply(ToArray(a), ToArray(b))), mx::Shape{1}));
#endif
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
#ifdef CONTINUUM_HAVE_MLX
  return FromArray(mx::matmul(ToArray(a), ToArray(b)));
#endif
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
#ifdef CONTINUUM_HAVE_MLX
  return FromArray(mx::softmax(ToArray(a), static_cast<int>(dim), /*precise=*/true));
#endif
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

std::string MLXBackend::runtime() {
#ifdef CONTINUUM_HAVE_MLX
  SelectDeviceOnce();
  return std::string("mlx ") + mx::version() + " (" +
         (mx::default_device() == mx::Device::gpu ? "gpu" : "cpu") + ")";
#else
  return "reference (built without MLX)";
#endif
}

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
  const auto& op = payload->op_name;
  if (op == "input" || op == "identity" || op == "id") {
    r.output = Input(inputs, 0, op);
    r.compute_steps = 1;
    return r;
  }

  auto a = ToMlx(Input(inputs, 0, op));
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
    auto b = ToMlx(Input(inputs, 1, op));
    r.output = Add(a, b);
    r.compute_steps = 1;
    return r;
  }
  if (payload->op_name == "matmul") {
    auto b = ToMlx(Input(inputs, 1, op));
    r.output = Matmul2D(a, b);
    r.compute_steps = 1;
    return r;
  }

  throw std::runtime_error("mlx backend: unsupported tensor op: " + payload->op_name);
}

}  // namespace continuum::backend
