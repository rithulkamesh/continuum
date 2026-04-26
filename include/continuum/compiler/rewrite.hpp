#pragma once

#include <continuum/ir/graph.hpp>

namespace continuum::compiler {

void hoist_common_token_prefixes(ir::Graph& g);
void memoize_pure_tool_ops(ir::Graph& g);
void specialize_structured_outputs(ir::Graph& g);
void run_tier0(ir::Graph& g);

}  // namespace continuum::compiler
