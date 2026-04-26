#pragma once

#include <cstdint>

namespace continuum::ir {

enum class EffectBit : std::uint8_t {
  Pure = 1 << 0,
  Idem = 1 << 1,
  Net = 1 << 2,
  Mut = 1 << 3,
  Stoch = 1 << 4,
};

struct Effect {
  std::uint8_t bits = static_cast<std::uint8_t>(EffectBit::Pure);
};

}  // namespace continuum::ir
