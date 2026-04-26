#include <continuum/ir/type.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace continuum::ir {
namespace {

bool TensorSubtype(const TensorType& a, const TensorType& b) {
  if (a.dtype != b.dtype || a.device != b.device || a.shape.size() != b.shape.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.shape.size(); ++i) {
    if (b.shape[i] != -1 && a.shape[i] != b.shape[i]) {
      return false;
    }
  }
  return true;
}

TensorType TensorMeet(const TensorType& a, const TensorType& b) {
  TensorType out;
  out.dtype = (a.dtype == b.dtype) ? a.dtype : DType::F32;
  out.device = (a.device == b.device) ? a.device : Device::CPU;
  if (a.shape.size() != b.shape.size()) {
    return out;
  }
  out.shape.resize(a.shape.size(), -1);
  for (std::size_t i = 0; i < a.shape.size(); ++i) {
    const auto da = a.shape[i];
    const auto db = b.shape[i];
    if (da == db) {
      out.shape[i] = da;
    } else if (da == -1) {
      out.shape[i] = db;
    } else if (db == -1) {
      out.shape[i] = da;
    }
  }
  return out;
}

}  // namespace

bool is_subtype(const Type& a, const Type& b) {
  if (a.index() != b.index()) {
    return false;
  }
  return std::visit(
      [&](const auto& lhs, const auto& rhs) -> bool {
        using L = std::decay_t<decltype(lhs)>;
        using R = std::decay_t<decltype(rhs)>;
        if constexpr (!std::is_same_v<L, R>) {
          return false;
        } else if constexpr (std::is_same_v<L, TensorType>) {
          return TensorSubtype(lhs, rhs);
        } else if constexpr (std::is_same_v<L, TokensType>) {
          return lhs.vocab_id == rhs.vocab_id && lhs.model_family == rhs.model_family && lhs.max_len <= rhs.max_len;
        } else if constexpr (std::is_same_v<L, SchemaType>) {
          return lhs.schema_hash == rhs.schema_hash && lhs.canonical_json == rhs.canonical_json;
        } else if constexpr (std::is_same_v<L, EffectType>) {
          return (lhs.bits & rhs.bits) == lhs.bits;
        }
      },
      a,
      b);
}

Type meet(const Type& a, const Type& b) {
  if (a.index() != b.index()) {
    return EffectType{0};
  }
  return std::visit(
      [&](const auto& lhs, const auto& rhs) -> Type {
        using L = std::decay_t<decltype(lhs)>;
        using R = std::decay_t<decltype(rhs)>;
        if constexpr (!std::is_same_v<L, R>) {
          return EffectType{0};
        } else if constexpr (std::is_same_v<L, TensorType>) {
          return TensorMeet(lhs, rhs);
        } else if constexpr (std::is_same_v<L, TokensType>) {
          if (lhs.vocab_id != rhs.vocab_id || lhs.model_family != rhs.model_family) {
            return TokensType{"", 0, ""};
          }
          return TokensType{lhs.vocab_id, std::min(lhs.max_len, rhs.max_len), lhs.model_family};
        } else if constexpr (std::is_same_v<L, SchemaType>) {
          if (lhs.schema_hash == rhs.schema_hash && lhs.canonical_json == rhs.canonical_json) {
            return lhs;
          }
          return SchemaType{"", 0};
        } else if constexpr (std::is_same_v<L, EffectType>) {
          return EffectType{static_cast<std::uint8_t>(lhs.bits & rhs.bits)};
        }
      },
      a,
      b);
}

}  // namespace continuum::ir
