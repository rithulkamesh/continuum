#pragma once

#include <continuum/ir/graph.hpp>

namespace continuum::runtime {

class Scheduler {
 public:
  using ExecutionPlan = std::vector<std::vector<ir::NodeId>>;
  ExecutionPlan schedule(const ir::Graph& graph) const;
};

}  // namespace continuum::runtime
