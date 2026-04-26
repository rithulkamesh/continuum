#pragma once

#include <torch/torch.h>

#include <string>
#include <variant>
#include <vector>

namespace continuum {

struct TensorValue {
  torch::Tensor tensor;
  std::string backend_type = "libtorch";
};

struct MlxTensorValue {
  std::vector<std::int64_t> shape;
  std::vector<float> data;
  std::string backend_type = "mlx";
};

struct TokensValue {
  std::vector<int> ids;
};

struct SchemaValue {
  std::string json;
};

using Value = std::variant<TensorValue, MlxTensorValue, TokensValue, SchemaValue, std::string, double, int64_t>;

}  // namespace continuum
