#pragma once

#include <continuum/ir/graph.hpp>

#include <string>
#include <vector>

namespace continuum::compiler {

struct TypeError {
  ir::NodeId node_id = 0;
  std::string message;
};

std::vector<TypeError> typecheck(ir::Graph& graph);

}  // namespace continuum::compiler
