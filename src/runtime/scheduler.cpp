#include <continuum/runtime/scheduler.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <continuum/ir/effect.hpp>

namespace continuum::runtime {
namespace {

bool IsEffectConstrained(const ir::Node& node) {
  constexpr std::uint8_t kConstrainedBits = static_cast<std::uint8_t>(ir::EffectBit::Net) |
                                             static_cast<std::uint8_t>(ir::EffectBit::Mut) |
                                             static_cast<std::uint8_t>(ir::EffectBit::Stoch);
  return (node.effect.bits & kConstrainedBits) != 0;
}

}  // namespace

Scheduler::ExecutionPlan Scheduler::schedule(const ir::Graph& graph) const {
  const auto& topo = graph.topo_order();
  std::unordered_map<ir::NodeId, std::size_t> topo_pos;
  topo_pos.reserve(topo.size());
  for (std::size_t i = 0; i < topo.size(); ++i) {
    topo_pos[topo[i]] = i;
  }

  std::unordered_map<ir::NodeId, std::size_t> indegree;
  std::unordered_map<ir::NodeId, std::vector<ir::NodeId>> children;
  indegree.reserve(topo.size());
  children.reserve(topo.size());
  for (const auto id : topo) {
    indegree[id] = 0;
  }
  for (const auto id : topo) {
    const auto& node = graph.get(id);
    for (const auto parent : node.inputs) {
      auto parent_it = indegree.find(parent);
      if (parent_it == indegree.end()) {
        throw std::runtime_error("scheduler: missing parent in graph");
      }
      ++indegree[id];
      children[parent].push_back(id);
    }
  }

  std::vector<ir::NodeId> ready;
  ready.reserve(topo.size());
  for (const auto id : topo) {
    if (indegree[id] == 0) {
      ready.push_back(id);
    }
  }

  ExecutionPlan plan;
  std::size_t emitted = 0;
  while (!ready.empty()) {
    std::sort(ready.begin(), ready.end(), [&](ir::NodeId a, ir::NodeId b) { return topo_pos[a] < topo_pos[b]; });

    std::vector<ir::NodeId> group;
    std::size_t effectful_idx = ready.size();
    for (std::size_t i = 0; i < ready.size(); ++i) {
      if (IsEffectConstrained(graph.get(ready[i]))) {
        effectful_idx = i;
        break;
      }
    }

    if (effectful_idx != ready.size()) {
      group.push_back(ready[effectful_idx]);
      ready.erase(ready.begin() + static_cast<std::ptrdiff_t>(effectful_idx));
    } else {
      group = std::move(ready);
      ready.clear();
    }

    plan.push_back(group);
    emitted += group.size();

    for (const auto id : group) {
      auto it = children.find(id);
      if (it == children.end()) {
        continue;
      }
      for (const auto child : it->second) {
        auto in_it = indegree.find(child);
        if (in_it == indegree.end()) {
          throw std::runtime_error("scheduler: child missing indegree");
        }
        if (--in_it->second == 0) {
          ready.push_back(child);
        }
      }
    }
  }

  if (emitted != topo.size()) {
    throw std::runtime_error("scheduler: failed to produce full plan");
  }
  return plan;
}

}  // namespace continuum::runtime
