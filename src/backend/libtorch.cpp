#include <continuum/backend/libtorch.hpp>

#include <stdexcept>

namespace continuum::backend {

BackendCapabilities LibTorchBackend::capabilities() const {
  return BackendCapabilities{true, false, false};
}

BackendRunResult LibTorchBackend::run_with_cache(
    const ir::Node& node,
    const std::vector<continuum::Value>& inputs,
    const std::optional<BackendState>&,
    std::int32_t) {
  BackendRunResult r;
  r.resulting_state = BackendState{nullptr};
  r.used_cached_state = false;
  r.reused_prefix_len = 0;
  r.compute_steps = 0;
  if (node.kind == ir::NodeKind::TensorOp) {
    const auto* payload = std::get_if<ir::TensorOpPayload>(&node.payload);
    if (payload == nullptr) {
      throw std::runtime_error("libtorch backend: tensor op missing payload");
    }
    if (payload->op_name == "input" || payload->op_name == "identity" || payload->op_name == "id") {
      r.output = inputs.at(0);
      r.compute_steps = 1;
      return r;
    }
    const auto* a = std::get_if<continuum::TensorValue>(&inputs.at(0));
    if (a == nullptr) {
      throw std::runtime_error("libtorch backend: expected TensorValue input");
    }
    if (payload->op_name == "relu") {
      r.output = continuum::TensorValue{torch::relu(a->tensor), "libtorch"};
      r.compute_steps = 1;
      return r;
    }
    if (payload->op_name == "softmax") {
      const auto dim = payload->attrs.empty() ? -1 : static_cast<int64_t>(payload->attrs[0]);
      r.output = continuum::TensorValue{torch::softmax(a->tensor, dim), "libtorch"};
      r.compute_steps = 1;
      return r;
    }
    if (payload->op_name == "add") {
      const auto* b = std::get_if<continuum::TensorValue>(&inputs.at(1));
      if (b == nullptr) {
        throw std::runtime_error("libtorch backend: add expects two tensors");
      }
      r.output = continuum::TensorValue{a->tensor + b->tensor, "libtorch"};
      r.compute_steps = 1;
      return r;
    }
    if (payload->op_name == "matmul") {
      const auto* b = std::get_if<continuum::TensorValue>(&inputs.at(1));
      if (b == nullptr) {
        throw std::runtime_error("libtorch backend: matmul expects two tensors");
      }
      r.output = continuum::TensorValue{torch::matmul(a->tensor, b->tensor), "libtorch"};
      r.compute_steps = 1;
      return r;
    }
    throw std::runtime_error("libtorch backend: unsupported tensor op: " + payload->op_name);
  }
  if (!inputs.empty()) {
    r.output = inputs.front();
  } else {
    r.output = std::string("libtorch");
  }
  return r;
}

}  // namespace continuum::backend
