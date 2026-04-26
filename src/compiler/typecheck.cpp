#include <continuum/compiler/typecheck.hpp>

#include <string>
#include <unordered_map>

namespace continuum::compiler {
namespace {

bool IsTensor(const ir::Type& t) { return std::holds_alternative<ir::TensorType>(t); }

TypeError MakeError(ir::NodeId id, const std::string& msg) { return TypeError{id, msg}; }

}  // namespace

std::vector<TypeError> typecheck(ir::Graph& graph) {
  std::vector<TypeError> errors;
  std::unordered_map<ir::NodeId, ir::Type> inferred;

  for (const auto id : graph.topo_order()) {
    auto& node = graph.get_mut(id);
    std::vector<ir::Type> in_types;
    in_types.reserve(node.inputs.size());
    for (auto input_id : node.inputs) {
      auto it = inferred.find(input_id);
      if (it == inferred.end()) {
        errors.push_back(MakeError(id, "missing inferred input type"));
        continue;
      }
      in_types.push_back(it->second);
    }

    ir::Type out = node.out_type;
    switch (node.kind) {
      case ir::NodeKind::TensorOp: {
        const auto* payload = std::get_if<ir::TensorOpPayload>(&node.payload);
        const std::string op = payload == nullptr ? "id" : payload->op_name;
        if (op == "input" || op == "const") {
          if (!IsTensor(node.out_type)) {
            errors.push_back(MakeError(id, "tensor input/const op must declare TensorType output"));
          }
          break;
        }
        if (op == "id" || op == "identity" || op == "relu") {
          if (in_types.size() != 1 || !IsTensor(in_types[0])) {
            errors.push_back(MakeError(id, "unary tensor op requires exactly one tensor input"));
            break;
          }
          out = in_types[0];
          break;
        }
        if (op == "add" || op == "mul" || op == "matmul") {
          if (in_types.size() != 2 || !IsTensor(in_types[0]) || !IsTensor(in_types[1])) {
            errors.push_back(MakeError(id, "binary tensor op requires two tensor inputs"));
            break;
          }
          const auto& a = std::get<ir::TensorType>(in_types[0]);
          const auto& b = std::get<ir::TensorType>(in_types[1]);
          if (a.dtype != b.dtype) {
            errors.push_back(MakeError(id, "binary tensor op dtype mismatch"));
            break;
          }
          if (a.device != b.device) {
            errors.push_back(MakeError(id, "binary tensor op device mismatch"));
            break;
          }
          out = ir::meet(in_types[0], in_types[1]);
          break;
        }
        if (!in_types.empty()) {
          out = in_types[0];
        }
        break;
      }
      case ir::NodeKind::TokenOp: {
        if (!std::holds_alternative<ir::TokensType>(node.out_type) &&
            !std::holds_alternative<ir::SchemaType>(node.out_type)) {
          errors.push_back(MakeError(id, "token op output must be TokensType or SchemaType"));
        }
        break;
      }
      case ir::NodeKind::PromptOp: {
        if (!std::holds_alternative<ir::TokensType>(node.out_type)) {
          errors.push_back(MakeError(id, "prompt op output must be TokensType"));
        }
        break;
      }
      case ir::NodeKind::ToolOp: {
        if (!std::holds_alternative<ir::SchemaType>(node.out_type)) {
          errors.push_back(MakeError(id, "tool op output must be SchemaType"));
        }
        break;
      }
      case ir::NodeKind::ControlOp: {
        if (!in_types.empty()) {
          out = in_types.back();
        }
        break;
      }
    }

    node.out_type = out;
    inferred[id] = out;
  }
  return errors;
}

}  // namespace continuum::compiler
