#pragma once

#include <continuum/ir/effect.hpp>
#include <continuum/ir/type.hpp>

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace continuum::ir {

using NodeId = std::uint64_t;

enum class NodeKind : std::uint8_t { TensorOp = 0, TokenOp = 1, PromptOp = 2, ToolOp = 3, ControlOp = 4 };

struct Schema {
  std::string canonical_json;
};

struct TensorOpPayload {
  std::string op_name;
  std::vector<std::int64_t> attrs;
};

struct TokenOpPayload {
  std::string op_name;
  std::string model_id;
  float temperature = 1.0f;
  std::int32_t max_tokens = 512;
};

struct PromptOpPayload {
  std::string template_id;
  std::vector<NodeId> slot_inputs;
};

struct ToolOpPayload {
  std::string tool_name;
  Schema input_schema;
  Schema output_schema;
};

struct ControlOpPayload {
  enum class Kind { If, While, ForEach, Parallel, TryCatch } kind;
  std::vector<NodeId> branch_entries;
};

using OpPayload = std::variant<TensorOpPayload, TokenOpPayload, PromptOpPayload, ToolOpPayload, ControlOpPayload>;

struct Node {
  NodeId id = 0;
  NodeKind kind = NodeKind::TensorOp;
  OpPayload payload = TensorOpPayload{};
  std::vector<NodeId> inputs;
  Type out_type = TensorType{};
  Effect effect{};
  std::string debug_name;
};

}  // namespace continuum::ir
