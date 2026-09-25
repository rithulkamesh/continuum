#pragma once

#include <cstdint>
#include <string>

namespace continuum::runtime {

/// What a ReuseEvent describes.
enum class ReuseEventKind : std::uint8_t {
  TierLookup,     ///< One reuse-tier lookup for a TokenOp.
  NodeExecution,  ///< One executed node, whatever served it.
};

/// Structured record the interpreter emits for observability (tracing,
/// metrics). Times are Unix-epoch nanoseconds from the system clock.
struct ReuseEvent {
  ReuseEventKind kind = ReuseEventKind::TierLookup;
  /// TierLookup: "memo", "semantic", "prefix_kv", "layer_kv", "memory_graph".
  std::string tier;
  std::string node_name;
  /// "TokenOp", "TensorOp", "PromptOp", "ToolOp", "ControlOp".
  std::string node_kind;
  std::string backend;
  std::string model_id;
  std::string cache_namespace;
  /// TierLookup: the tier matched (for memory_graph: recalled >= 1 node).
  bool hit = false;
  /// NodeExecution: "memo", "semantic", "backend", or "passthrough".
  std::string served_by;
  /// Semantic: best similarity. Memory graph: top recalled similarity.
  float similarity = 0.0f;
  /// Prefix / layer KV: matched prefix length. Memory graph: nodes recalled.
  std::int32_t match_len = 0;
  std::int32_t total_tokens = 0;
  std::int32_t tokens_saved = 0;
  std::int32_t tokens_sent = 0;
  std::int32_t reused_prefix_len = 0;
  std::int32_t compute_steps = 0;
  bool used_cached_state = false;
  std::int64_t start_unix_ns = 0;
  std::int64_t end_unix_ns = 0;
};

/// Receives ReuseEvents from an Interpreter (via Session / DurableAgent).
/// For one node, its TierLookup events arrive before its NodeExecution event.
/// Implementations must not throw.
class ReuseObserver {
 public:
  virtual ~ReuseObserver() = default;
  virtual void on_event(const ReuseEvent& event) = 0;
};

}  // namespace continuum::runtime
