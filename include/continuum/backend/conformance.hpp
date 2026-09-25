#pragma once

#include <continuum/backend/backend.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace continuum::backend {

/// Outcome of one conformance check.
enum class ConformanceStatus { Pass, Fail, Skip };

/// One named check and why it passed, failed, or was skipped.
struct ConformanceCheck {
  std::string name;
  ConformanceStatus status = ConformanceStatus::Skip;
  std::string detail;
};

/// Inputs the kit drives the backend with.
struct ConformanceOptions {
  std::string model_id = "conformance/model";
  /// Token-path prompt. Its first `prefix_len` bytes act as the cached prefix
  /// on the warm run (default: half the prompt).
  std::string prompt = "You are a careful assistant. Answer briefly. Question: what is a prefix cache?";
  std::int32_t prefix_len = -1;
  std::int32_t max_tokens = 16;
  /// Require identical output for identical cold inputs at temperature 0.
  bool expect_deterministic = true;
  /// Tensor-path op; the backend must return its single input unchanged.
  std::string tensor_op = "identity";
};

/// All checks for one backend.
struct ConformanceReport {
  std::string backend;
  std::vector<ConformanceCheck> checks;

  /// True when no check failed (skips are allowed).
  bool passed() const;
  /// Human-readable, one line per check.
  std::string summary() const;
};

/// Drive \p backend through capability declaration, a cold and a warm token
/// run (the `reused_prefix_len` / `tokens_saved` / `used_cached_state`
/// metric contract), state export/import round-trip, and a tensor identity
/// op, as applicable to its declared capabilities. This is the bar a new
/// backend must clear; see "Backend conformance" in docs/design/abi.md.
ConformanceReport RunBackendConformance(Backend& backend, const std::string& name,
                                        const ConformanceOptions& options = {});

}  // namespace continuum::backend
