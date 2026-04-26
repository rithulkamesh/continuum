#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace continuum::ir {

enum class DType : std::uint8_t { F32, F16, BF16, I32, I64, U8, Bool };
enum class Device : std::uint8_t { CPU, CUDA, MPS, MLX, Remote };

struct TensorType {
  std::vector<std::int64_t> shape;
  DType dtype = DType::F32;
  Device device = Device::CPU;
};

struct TokensType {
  std::string vocab_id;
  std::int32_t max_len = 0;
  std::string model_family;
};

struct SchemaType {
  std::string canonical_json;
  std::uint64_t schema_hash = 0;
};

struct EffectType {
  std::uint8_t bits = 0;
};

using Type = std::variant<TensorType, TokensType, SchemaType, EffectType>;

bool is_subtype(const Type& a, const Type& b);
Type meet(const Type& a, const Type& b);

}  // namespace continuum::ir
