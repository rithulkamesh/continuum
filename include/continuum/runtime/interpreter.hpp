#pragma once

#include <continuum/backend/backend.hpp>
#include <continuum/ir/graph.hpp>
#include <continuum/runtime/checkpoint.hpp>
#include <continuum/runtime/cache.hpp>
#include <continuum/runtime/scheduler.hpp>

#include <optional>
#include <unordered_map>

namespace continuum::runtime {

continuum::Value convert_value_for_backend(const continuum::Value& value, const std::string& target_backend);

class Interpreter {
 public:
  explicit Interpreter(backend::BackendRegistry& backends, KVCacheIndex& cache);
  std::vector<continuum::Value> run(
      const ir::Graph& g, const std::unordered_map<ir::NodeId, continuum::Value>& inputs);
  Checkpoint run_until(ir::NodeId node_id);
  std::vector<continuum::Value> resume(const Checkpoint& checkpoint);
  void begin(const ir::Graph& g, const std::unordered_map<ir::NodeId, continuum::Value>& inputs);
  continuum::Value step(const ir::Node& n, const std::vector<continuum::Value>& input_values);

 private:
  struct ActiveExecution {
    ir::Graph graph;
    Scheduler::ExecutionPlan plan;
    std::unordered_map<ir::NodeId, continuum::Value> values;
    std::vector<continuum::Value> out;
    std::size_t next_group_index = 0;
    std::size_t next_node_in_group = 0;
    std::size_t executed_nodes = 0;
  };

  std::vector<continuum::Value> run_to_end();
  ir::NodeId next_planned_node(const ActiveExecution& state) const;
  void advance_plan_cursor(ActiveExecution& state);

  backend::BackendRegistry& backends_;
  KVCacheIndex& cache_;
  Scheduler scheduler_;
  std::optional<ActiveExecution> active_;
};

}  // namespace continuum::runtime
