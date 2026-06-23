#pragma once

#include <continuum/backend/backend.hpp>
#include <continuum/ir/graph.hpp>
#include <continuum/runtime/checkpoint.hpp>
#include <continuum/runtime/cache.hpp>
#include <continuum/runtime/scheduler.hpp>
#include <continuum/runtime/semantic_cache.hpp>
#include <continuum/runtime/layer_cache.hpp>
#include <continuum/runtime/memory_graph.hpp>

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace continuum::runtime {

struct ReusePolicy;
class MemoTable;

continuum::Value convert_value_for_backend(const continuum::Value& value, const std::string& target_backend);

class Interpreter {
 public:
  explicit Interpreter(backend::BackendRegistry& backends, KVCacheIndex& cache,
                       const ReusePolicy* policy = nullptr);
  std::vector<continuum::Value> run(
      const ir::Graph& g, const std::unordered_map<ir::NodeId, continuum::Value>& inputs);
  Checkpoint run_until(ir::NodeId node_id);
  std::vector<continuum::Value> resume(const Checkpoint& checkpoint);
  void begin(const ir::Graph& g, const std::unordered_map<ir::NodeId, continuum::Value>& inputs);
  continuum::Value step(const ir::Node& n, const std::vector<continuum::Value>& input_values);

  void set_memo_table(MemoTable* memo) { memo_table_ = memo; }
  void set_semantic_cache(SemanticCacheIndex* sc) { semantic_cache_ = sc; }
  void set_embedding_provider(EmbeddingProvider* ep) { embedder_ = ep; }
  void set_layer_cache(LayerKVCacheIndex* lc) { layer_cache_ = lc; }
  void set_memory_graph(MemoryGraphStore* mg) { memory_graph_ = mg; }

  MemoTable* memo_table() const { return memo_table_; }
  SemanticCacheIndex* semantic_cache() const { return semantic_cache_; }
  EmbeddingProvider* embedding_provider() const { return embedder_; }
  LayerKVCacheIndex* layer_cache() const { return layer_cache_; }
  MemoryGraphStore* memory_graph() const { return memory_graph_; }

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
  const ReusePolicy* policy_;
  Scheduler scheduler_;
  std::optional<ActiveExecution> active_;
  MemoTable* memo_table_ = nullptr;
  SemanticCacheIndex* semantic_cache_ = nullptr;
  EmbeddingProvider* embedder_ = nullptr;
  LayerKVCacheIndex* layer_cache_ = nullptr;
  MemoryGraphStore* memory_graph_ = nullptr;
};

}  // namespace continuum::runtime
