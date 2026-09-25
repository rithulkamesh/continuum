#include <continuum/backend/fake_llm.hpp>
#include <continuum/backend/libtorch.hpp>
#include <continuum/backend/vllm_shim.hpp>
#include <continuum/ir/graph.hpp>
#include <continuum/ir/node.hpp>
#include <continuum/runtime/checkpoint.hpp>
#include <continuum/runtime/interpreter.hpp>
#include <continuum/runtime/layer_cache.hpp>
#include <continuum/runtime/memo_table.hpp>
#include <continuum/runtime/memory_graph.hpp>
#include <continuum/runtime/scheduler.hpp>
#include <continuum/runtime/semantic_cache.hpp>
#include <continuum/runtime/session.hpp>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace {

using continuum::ir::Graph;
using continuum::ir::Node;
using continuum::ir::NodeId;
using continuum::ir::NodeKind;

Node MakeTensor(const std::string& op, std::vector<NodeId> inputs = {}) {
  Node n;
  n.kind = NodeKind::TensorOp;
  n.debug_name = op;
  n.payload = continuum::ir::TensorOpPayload{op, {}};
  n.inputs = std::move(inputs);
  return n;
}

}  // namespace

TEST(InterpreterDispatchTest, PromptOpPassesThroughInput) {
  Graph g;
  Node prompt;
  prompt.kind = NodeKind::PromptOp;
  prompt.debug_name = "p";
  prompt.payload = continuum::ir::PromptOpPayload{"tmpl", {}};
  const NodeId pid = g.add_node(prompt);

  continuum::backend::BackendRegistry registry;
  registry.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>());
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::Interpreter interp(registry, cache);

  std::unordered_map<NodeId, continuum::Value> inputs;
  inputs[pid] = std::string{"hello prompt"};
  const auto out = interp.run(g, inputs);
  ASSERT_EQ(out.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<std::string>(out.back()));
  EXPECT_EQ(std::get<std::string>(out.back()), "hello prompt");
}

TEST(InterpreterDispatchTest, ToolOpInvalidatesMemoAndDispatches) {
  Graph g;
  Node tool;
  tool.kind = NodeKind::ToolOp;
  tool.debug_name = "tool";
  tool.payload = continuum::ir::ToolOpPayload{"echo", continuum::ir::Schema{"{}"}, continuum::ir::Schema{"{}"}};
  g.add_node(tool);

  continuum::backend::BackendRegistry registry;
  registry.register_backend("default", std::make_shared<continuum::backend::FakeLLMBackend>(), 10);
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::MemoTable memo;
  continuum::runtime::Interpreter interp(registry, cache);
  interp.set_memo_table(&memo);

  // Seed a stale ToolOp memo entry that must be dropped on ToolOp execution.
  Node probe = tool;
  auto stale_key = memo.make_key(probe, {std::string{"x"}});
  memo.insert(stale_key, continuum::runtime::MemoEntry{
                             continuum::runtime::MemoTable::serialize_value(std::string{"stale"}), 0, 1, 0});
  ASSERT_TRUE(memo.lookup(stale_key).has_value());

  // Do not pre-seed the ToolOp value — otherwise step() is skipped.
  const auto out = interp.run(g, {});
  ASSERT_FALSE(out.empty());
  EXPECT_FALSE(memo.lookup(stale_key).has_value());
}

TEST(InterpreterDispatchTest, ControlOpFallsBackToLastInput) {
  Graph g;
  const NodeId a = g.add_node(MakeTensor("input"));
  Node ctrl;
  ctrl.kind = NodeKind::ControlOp;
  ctrl.debug_name = "ctrl";
  ctrl.payload = continuum::ir::ControlOpPayload{continuum::ir::ControlOpPayload::Kind::If, {}};
  ctrl.inputs = {a};
  g.add_node(ctrl);

  continuum::backend::BackendRegistry registry;
  registry.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>());
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::Interpreter interp(registry, cache);

  std::unordered_map<NodeId, continuum::Value> inputs;
  inputs[a] = continuum::TensorValue{torch::tensor({3.0f}), "libtorch"};
  const auto out = interp.run(g, inputs);
  ASSERT_EQ(out.size(), 2u);
  ASSERT_TRUE(std::holds_alternative<continuum::TensorValue>(out.back()));
}

TEST(SchedulerTest, DiamondDependencyPreservesTopoOrder) {
  //     a
  //    / \
  //   b   c
  //    \ /
  //     d
  Graph g;
  const NodeId a = g.add_node(MakeTensor("a"));
  const NodeId b = g.add_node(MakeTensor("b", {a}));
  const NodeId c = g.add_node(MakeTensor("c", {a}));
  const NodeId d = g.add_node(MakeTensor("d", {b, c}));

  continuum::runtime::Scheduler sched;
  const auto plan = sched.schedule(g);
  std::unordered_map<NodeId, std::size_t> pos;
  std::size_t i = 0;
  for (const auto& group : plan) {
    for (const auto id : group) {
      pos[id] = i++;
    }
  }
  ASSERT_EQ(pos.size(), 4u);
  EXPECT_LT(pos[a], pos[b]);
  EXPECT_LT(pos[a], pos[c]);
  EXPECT_LT(pos[b], pos[d]);
  EXPECT_LT(pos[c], pos[d]);
}

TEST(CheckpointTest, RoundTripPreservesKvStateBytes) {
  Graph g;
  Node prompt;
  prompt.kind = NodeKind::PromptOp;
  prompt.debug_name = "p";
  prompt.payload = continuum::ir::PromptOpPayload{"t", {}};
  const NodeId pid = g.add_node(prompt);
  Node tok;
  tok.kind = NodeKind::TokenOp;
  tok.debug_name = "gen";
  tok.payload = continuum::ir::TokenOpPayload{"generate", "fake/model", 0.0f, 8};
  tok.inputs = {pid};
  const NodeId tid = g.add_node(tok);

  continuum::backend::BackendRegistry registry;
  registry.register_backend("default", std::make_shared<continuum::backend::FakeLLMBackend>(), 10);
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::Interpreter interp(registry, cache);
  std::unordered_map<NodeId, continuum::Value> inputs;
  inputs[pid] = std::string{"prefix text for cache"};
  interp.begin(g, inputs);
  auto cp = interp.run_until(tid);
  ASSERT_FALSE(cp.cache_snapshot.empty());
  for (const auto& e : cp.cache_snapshot) {
    EXPECT_FALSE(e.state_bytes.empty());
  }

  const auto bytes = continuum::runtime::serialize_checkpoint(cp);
  const auto restored = continuum::runtime::deserialize_checkpoint(bytes);
  ASSERT_EQ(restored.cache_snapshot.size(), cp.cache_snapshot.size());
  for (std::size_t i = 0; i < cp.cache_snapshot.size(); ++i) {
    EXPECT_EQ(restored.cache_snapshot[i].state_bytes, cp.cache_snapshot[i].state_bytes);
    EXPECT_EQ(restored.cache_snapshot[i].tokens, cp.cache_snapshot[i].tokens);
    EXPECT_EQ(restored.cache_snapshot[i].model_id, cp.cache_snapshot[i].model_id);
  }
}

TEST(MemoTableTest, HitMissAndInvalidate) {
  continuum::runtime::MemoTable memo(16, 1);
  Graph g;
  Node n = MakeTensor("id");
  const NodeId id = g.add_node(n);
  const auto& node = g.get(id);
  auto key = memo.make_key(node, {continuum::TensorValue{torch::tensor({1.0f}), "libtorch"}});
  EXPECT_FALSE(memo.lookup(key).has_value());
  memo.insert(key, continuum::runtime::MemoEntry{
                       continuum::runtime::MemoTable::serialize_value(std::string{"hit"}), 1, 1, 0});
  ASSERT_TRUE(memo.lookup(key).has_value());
  memo.invalidate_node("TensorOp");
  EXPECT_FALSE(memo.lookup(key).has_value());
}

TEST(SemanticCacheTest, ThresholdHitAndMiss) {
  continuum::runtime::SemanticCacheIndex sc(16, 0.9f);
  sc.insert({1.0f, 0.0f}, "m", continuum::runtime::MemoTable::serialize_value(std::string{"a"}));
  auto hit = sc.lookup({1.0f, 0.0f}, "m");
  EXPECT_TRUE(hit.above_threshold);
  auto miss = sc.lookup({0.0f, 1.0f}, "m");
  EXPECT_FALSE(miss.above_threshold);
  auto other_model = sc.lookup({1.0f, 0.0f}, "other");
  EXPECT_FALSE(other_model.above_threshold);
}

TEST(LayerCacheTest, HitMissAndModelInvalidation) {
  continuum::runtime::LayerKVCacheIndex lc(16, 1024 * 1024);
  continuum::runtime::LayerCheckpoint cp;
  cp.tokens = {1, 2, 3};
  cp.model_id = "m";
  cp.decode_hash = "d";
  cp.layer_id = 1;
  cp.prefix_len = 3;
  cp.estimated_bytes = 32;
  cp.state.handle = reinterpret_cast<void*>(0x42);
  lc.insert(cp);

  EXPECT_TRUE(lc.find_deepest("m", "d", {1, 2, 3, 4}, 10, 0).found);
  EXPECT_FALSE(lc.find_deepest("m", "d", {9, 8, 7}, 10, 0).found);
  lc.invalidate_model("m");
  EXPECT_FALSE(lc.find_deepest("m", "d", {1, 2, 3, 4}, 10, 0).found);
}

TEST(MemoryGraphTest, InsertAndRetrieveSimilar) {
  continuum::runtime::MemoryGraphStore store(16);
  continuum::runtime::MemoryNode n;
  n.type = continuum::runtime::MemoryNodeType::Prompt;
  n.content = "hello";
  n.embedding = {1.0f, 0.0f};
  store.add_node(n);

  auto hits = store.retrieve_similar({1.0f, 0.0f}, 3, 0.5f);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits.front().node.content, "hello");
  EXPECT_TRUE(store.retrieve_similar({0.0f, 1.0f}, 3, 0.99f).empty());
}

TEST(EvictionTest, SemanticLookupRefreshesLru) {
  continuum::runtime::SemanticCacheIndex sc(2, 0.99f);
  sc.insert({1.0f, 0.0f, 0.0f}, "m", {1});
  sc.insert({0.0f, 1.0f, 0.0f}, "m", {2});
  ASSERT_TRUE(sc.lookup({1.0f, 0.0f, 0.0f}, "m").above_threshold);
  sc.insert({0.0f, 0.0f, 1.0f}, "m", {3});
  EXPECT_EQ(sc.size(), 2u);
  EXPECT_TRUE(sc.lookup({1.0f, 0.0f, 0.0f}, "m").above_threshold);
  EXPECT_FALSE(sc.lookup({0.0f, 1.0f, 0.0f}, "m").above_threshold);
  EXPECT_GT(sc.estimated_bytes(), 0u);
}

TEST(EvictionTest, MemoryGraphIsFifo) {
  continuum::runtime::MemoryGraphStore store(2);
  continuum::runtime::MemoryNode n;
  n.embedding = {1.0f};
  const auto first = store.add_node(n);
  const auto second = store.add_node(n);
  store.add_node(n);
  EXPECT_EQ(store.size(), 2u);
  EXPECT_FALSE(store.get_node(first).has_value());
  EXPECT_TRUE(store.get_node(second).has_value());
  EXPECT_GT(store.estimated_bytes(), 0u);

  continuum::runtime::MemoryGraphStore empty(0);
  empty.add_node(n);
  EXPECT_EQ(empty.size(), 0u);
}

TEST(EvictionTest, SessionReportsTierStats) {
  continuum::backend::BackendRegistry registry;
  continuum::runtime::Session session("s", registry, 4);
  continuum::runtime::MemoTable memo(3, 0);
  session.set_memo_table(&memo);
  const auto stats = session.cache_stats();
  ASSERT_EQ(stats.size(), 2u);
  EXPECT_EQ(stats[0].tier, "prefix_kv");
  EXPECT_EQ(stats[0].capacity, 4u);
  EXPECT_EQ(stats[1].tier, "memo");
  EXPECT_EQ(stats[1].capacity, 3u);
}

TEST(VllmShimTest, ExtractJsonStringDecodesEscapes) {
  using continuum::backend::VllmShimBackend;
  const std::string body =
      R"({"choices":[{"index":0,"text":"a\nb \"q\" \\ ✓ 😀 \/"}],)"
      R"("usage":{"prompt_tokens_details":{"cached_tokens": 42}}})";
  EXPECT_EQ(VllmShimBackend::ExtractJsonString(body, "text"),
            "a\nb \"q\" \\ \xE2\x9C\x93 \xF0\x9F\x98\x80 /");
  EXPECT_EQ(VllmShimBackend::ExtractJsonInt(body, "cached_tokens"), 42);
  EXPECT_EQ(VllmShimBackend::ExtractJsonString(body, "missing"), "");
  EXPECT_EQ(VllmShimBackend::ExtractJsonInt(body, "missing"), 0);
}
