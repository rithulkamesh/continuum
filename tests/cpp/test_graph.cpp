#include <continuum/ir/graph.hpp>
#include <continuum/ir/type.hpp>
#include <continuum/compiler/typecheck.hpp>
#include <continuum/compiler/rewrite.hpp>
#include <continuum/backend/openai.hpp>
#include <continuum/backend/anthropic.hpp>
#include <continuum/backend/vllm_shim.hpp>
#include <continuum/backend/libtorch.hpp>
#include <continuum/backend/mlx_backend.hpp>
#include <continuum/runtime/interpreter.hpp>
#include <continuum/runtime/scheduler.hpp>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstring>

namespace continuum::ir {
namespace {
continuum::runtime::DecodeParams DecodeFor(const std::string& op = "generate", float temp = 0.2f, int max_tokens = 32) {
  return continuum::runtime::DecodeParams{op, temp, max_tokens};
}


Node MakeTensorNode(std::string name, std::vector<NodeId> inputs = {}) {
  Node n;
  n.kind = NodeKind::TensorOp;
  n.payload = TensorOpPayload{std::move(name), {1, 2, 3}};
  n.inputs = std::move(inputs);
  n.out_type = TensorType{{2, 3}, DType::F32, Device::CPU};
  n.effect = Effect{static_cast<std::uint8_t>(EffectBit::Pure)};
  n.debug_name = "dbg";
  return n;
}

template <typename T>
T ReadPrimitive(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
  T out{};
  std::memcpy(&out, bytes.data() + offset, sizeof(T));
  offset += sizeof(T);
  return out;
}

std::string ReadString(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
  const auto n = ReadPrimitive<std::uint64_t>(bytes, offset);
  std::string out(reinterpret_cast<const char*>(bytes.data() + offset), static_cast<std::size_t>(n));
  offset += static_cast<std::size_t>(n);
  return out;
}

void SkipType(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
  const auto idx = ReadPrimitive<std::uint8_t>(bytes, offset);
  if (idx == 0) {
    const auto ndim = ReadPrimitive<std::uint64_t>(bytes, offset);
    for (std::uint64_t i = 0; i < ndim; ++i) (void)ReadPrimitive<std::int64_t>(bytes, offset);
    (void)ReadPrimitive<std::uint8_t>(bytes, offset);
    (void)ReadPrimitive<std::uint8_t>(bytes, offset);
    return;
  }
  if (idx == 1) {
    (void)ReadString(bytes, offset);
    (void)ReadPrimitive<std::int32_t>(bytes, offset);
    (void)ReadString(bytes, offset);
    return;
  }
  if (idx == 2) {
    (void)ReadString(bytes, offset);
    (void)ReadPrimitive<std::uint64_t>(bytes, offset);
    return;
  }
  if (idx == 3) {
    (void)ReadPrimitive<std::uint8_t>(bytes, offset);
    return;
  }
  throw std::runtime_error("test parser: unknown type idx");
}

void SkipPayload(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
  const auto idx = ReadPrimitive<std::uint8_t>(bytes, offset);
  if (idx == 0) {
    (void)ReadString(bytes, offset);
    const auto n = ReadPrimitive<std::uint64_t>(bytes, offset);
    for (std::uint64_t i = 0; i < n; ++i) (void)ReadPrimitive<std::int64_t>(bytes, offset);
    return;
  }
  if (idx == 1) {
    (void)ReadString(bytes, offset);
    (void)ReadString(bytes, offset);
    (void)ReadPrimitive<float>(bytes, offset);
    (void)ReadPrimitive<std::int32_t>(bytes, offset);
    return;
  }
  if (idx == 2) {
    (void)ReadString(bytes, offset);
    const auto n = ReadPrimitive<std::uint64_t>(bytes, offset);
    for (std::uint64_t i = 0; i < n; ++i) (void)ReadPrimitive<NodeId>(bytes, offset);
    return;
  }
  if (idx == 3) {
    (void)ReadString(bytes, offset);
    (void)ReadString(bytes, offset);
    (void)ReadString(bytes, offset);
    return;
  }
  if (idx == 4) {
    (void)ReadPrimitive<std::uint8_t>(bytes, offset);
    const auto n = ReadPrimitive<std::uint64_t>(bytes, offset);
    for (std::uint64_t i = 0; i < n; ++i) (void)ReadPrimitive<NodeId>(bytes, offset);
    return;
  }
  throw std::runtime_error("test parser: unknown payload idx");
}

TEST(GraphTest, TopoOrderRespectsDependencies) {
  Graph g;
  const NodeId a = g.add_node(MakeTensorNode("a"));
  const NodeId b = g.add_node(MakeTensorNode("b", {a}));
  const NodeId c = g.add_node(MakeTensorNode("c", {b}));

  const auto& topo = g.topo_order();
  ASSERT_EQ(topo.size(), 3U);
  EXPECT_EQ(topo[0], a);
  EXPECT_EQ(topo[1], b);
  EXPECT_EQ(topo[2], c);
}

TEST(GraphTest, StructuralHashChangesWhenGraphChanges) {
  Graph g1;
  const NodeId n1 = g1.add_node(MakeTensorNode("matmul"));
  g1.add_node(MakeTensorNode("relu", {n1}));

  Graph g2;
  const NodeId m1 = g2.add_node(MakeTensorNode("matmul"));
  g2.add_node(MakeTensorNode("sigmoid", {m1}));

  EXPECT_NE(g1.structural_hash(), g2.structural_hash());
}

TEST(GraphTest, SerializeDeserializeRoundtripPreservesNodes) {
  Graph g;
  const NodeId in = g.add_node(MakeTensorNode("input"));
  Node tok;
  tok.kind = NodeKind::TokenOp;
  tok.inputs = {in};
  tok.payload = TokenOpPayload{"generate", "openai/gpt-4o-mini", 0.7f, 128};
  tok.out_type = TokensType{"cl100k_base", 4096, "gpt"};
  tok.effect = Effect{static_cast<std::uint8_t>(EffectBit::Net) |
                      static_cast<std::uint8_t>(EffectBit::Stoch)};
  tok.debug_name = "token";
  const NodeId out = g.add_node(tok);

  const auto bytes = g.serialize();
  Graph restored = Graph::deserialize(bytes.data(), bytes.size());
  const auto& topo = restored.topo_order();
  ASSERT_EQ(topo.size(), 2U);

  const Node& restored_out = restored.get(topo[1]);
  ASSERT_TRUE(std::holds_alternative<TokenOpPayload>(restored_out.payload));
  const auto& payload = std::get<TokenOpPayload>(restored_out.payload);
  EXPECT_EQ(payload.op_name, "generate");
  EXPECT_EQ(payload.model_id, "openai/gpt-4o-mini");
  EXPECT_FLOAT_EQ(payload.temperature, 0.7f);
  EXPECT_EQ(payload.max_tokens, 128);
  EXPECT_EQ(restored_out.inputs.size(), 1U);
  EXPECT_EQ(restored_out.debug_name, "token");

  ASSERT_TRUE(std::holds_alternative<TokensType>(restored_out.out_type));
  const auto& out_type = std::get<TokensType>(restored_out.out_type);
  EXPECT_EQ(out_type.vocab_id, "cl100k_base");
  EXPECT_EQ(out_type.max_len, 4096);
  EXPECT_EQ(out_type.model_family, "gpt");
  EXPECT_EQ(restored_out.effect.bits,
            static_cast<std::uint8_t>(EffectBit::Net) | static_cast<std::uint8_t>(EffectBit::Stoch));
  (void)out;
}

TEST(GraphTest, SerializedBinaryMatchesCirSchemaLayout) {
  Graph g;
  const NodeId in = g.add_node(MakeTensorNode("input"));
  Node relu = MakeTensorNode("relu", {in});
  relu.debug_name = "relu-node";
  g.add_node(relu);

  const auto bytes = g.serialize();
  std::size_t offset = 0;
  const auto magic = ReadPrimitive<std::uint32_t>(bytes, offset);
  const auto version = ReadPrimitive<std::uint16_t>(bytes, offset);
  const auto node_count = ReadPrimitive<std::uint64_t>(bytes, offset);
  EXPECT_EQ(magic, 0x31495243U);  // CIR1
  EXPECT_EQ(version, 1);
  EXPECT_EQ(node_count, 2U);

  for (std::uint64_t i = 0; i < node_count; ++i) {
    (void)ReadPrimitive<NodeId>(bytes, offset);
    const auto kind = ReadPrimitive<std::uint8_t>(bytes, offset);
    EXPECT_LE(kind, 4);
    (void)ReadString(bytes, offset);
    const auto n_inputs = ReadPrimitive<std::uint64_t>(bytes, offset);
    for (std::uint64_t j = 0; j < n_inputs; ++j) (void)ReadPrimitive<NodeId>(bytes, offset);
    SkipType(bytes, offset);
    (void)ReadPrimitive<std::uint8_t>(bytes, offset);
    SkipPayload(bytes, offset);
  }
  EXPECT_EQ(offset, bytes.size());
}

TEST(SchedulerTest, EffectAwarePlanGroupsPureAndPreservesEffectOrder) {
  Graph g;
  const NodeId a = g.add_node(MakeTensorNode("input"));
  const NodeId b = g.add_node(MakeTensorNode("input"));

  Node p1 = MakeTensorNode("id", {a});
  p1.effect = Effect{static_cast<std::uint8_t>(EffectBit::Pure)};
  const NodeId p1_id = g.add_node(p1);

  Node n1;
  n1.kind = NodeKind::TokenOp;
  n1.inputs = {a};
  n1.payload = TokenOpPayload{"generate", "openai/gpt-4o-mini", 0.2f, 8};
  n1.out_type = TokensType{"cl100k_base", 128, "gpt"};
  n1.effect = Effect{static_cast<std::uint8_t>(EffectBit::Net)};
  const NodeId n1_id = g.add_node(n1);

  Node p2 = MakeTensorNode("id", {b});
  p2.effect = Effect{static_cast<std::uint8_t>(EffectBit::Idem)};
  const NodeId p2_id = g.add_node(p2);

  Node n2 = n1;
  n2.inputs = {b};
  n2.effect = Effect{static_cast<std::uint8_t>(EffectBit::Mut)};
  const NodeId n2_id = g.add_node(n2);

  continuum::runtime::Scheduler scheduler;
  const auto plan = scheduler.schedule(g);
  ASSERT_GE(plan.size(), 4U);

  EXPECT_EQ(plan[0].size(), 2U);
  EXPECT_EQ(plan[0][0], a);
  EXPECT_EQ(plan[0][1], b);
  EXPECT_EQ(plan[1].size(), 1U);
  EXPECT_EQ(plan[1][0], n1_id);
  EXPECT_EQ(plan[2].size(), 1U);
  EXPECT_EQ(plan[2][0], n2_id);
  ASSERT_EQ(plan[3].size(), 2U);
  EXPECT_EQ(plan[3][0], p1_id);
  EXPECT_EQ(plan[3][1], p2_id);

  std::vector<NodeId> flattened;
  for (const auto& group : plan) {
    for (auto id : group) flattened.push_back(id);
  }
  const auto& topo = g.topo_order();
  EXPECT_NE(flattened, topo);
}

TEST(TypeTest, TensorSubtypeSupportsWildcardsAndExactMatches) {
  const Type dynamic = TensorType{{-1, 64}, DType::F32, Device::CUDA};
  const Type concrete = TensorType{{16, 64}, DType::F32, Device::CUDA};
  const Type wrong_dtype = TensorType{{16, 64}, DType::F16, Device::CUDA};

  EXPECT_TRUE(is_subtype(concrete, dynamic));
  EXPECT_FALSE(is_subtype(dynamic, concrete));
  EXPECT_FALSE(is_subtype(wrong_dtype, dynamic));
}

TEST(TypeTest, MeetIntersectsTensorShapesAndEffectBits) {
  const Type a = TensorType{{16, -1}, DType::F32, Device::CPU};
  const Type b = TensorType{{16, 32}, DType::F32, Device::CPU};
  const Type m = meet(a, b);
  ASSERT_TRUE(std::holds_alternative<TensorType>(m));
  const auto& mt = std::get<TensorType>(m);
  ASSERT_EQ(mt.shape.size(), 2U);
  EXPECT_EQ(mt.shape[0], 16);
  EXPECT_EQ(mt.shape[1], 32);

  const Type e1 = EffectType{static_cast<std::uint8_t>(EffectBit::Pure) |
                             static_cast<std::uint8_t>(EffectBit::Idem)};
  const Type e2 = EffectType{static_cast<std::uint8_t>(EffectBit::Pure)};
  const Type em = meet(e1, e2);
  ASSERT_TRUE(std::holds_alternative<EffectType>(em));
  EXPECT_EQ(std::get<EffectType>(em).bits, static_cast<std::uint8_t>(EffectBit::Pure));
}

class RecordingBackend final : public continuum::backend::Backend {
 public:
  explicit RecordingBackend(continuum::backend::BackendCapabilities caps = {false, true, true})
      : caps_(caps) {}

  continuum::backend::BackendCapabilities capabilities() const override { return caps_; }

  continuum::Value response = std::string("token-result");
  int calls = 0;
  continuum::backend::BackendRunResult run_with_cache(
      const ir::Node&,
      const std::vector<continuum::Value>&,
      const std::optional<continuum::backend::BackendState>&,
      std::int32_t) override {
    ++calls;
    continuum::backend::BackendRunResult r;
    r.output = response;
    r.resulting_state = continuum::backend::BackendState{reinterpret_cast<void*>(0x1)};
    return r;
  }

 private:
  continuum::backend::BackendCapabilities caps_;
};

class CacheAwareBackend final : public continuum::backend::Backend {
 public:
  continuum::backend::BackendCapabilities capabilities() const override {
    return continuum::backend::BackendCapabilities{false, true, true};
  }

  int calls = 0;
  int total_work_tokens = 0;
  std::vector<int> compute_steps_history;
  std::vector<int> used_cached_state_history;

  continuum::backend::BackendRunResult run_with_cache(
      const ir::Node& node,
      const std::vector<continuum::Value>& inputs,
      const std::optional<continuum::backend::BackendState>& prefix_state,
      std::int32_t remaining_tokens) override {
    ++calls;
    const auto* t = !inputs.empty() ? std::get_if<continuum::TokensValue>(&inputs[0]) : nullptr;
    const auto prompt_len = t == nullptr ? 0 : static_cast<int>(t->ids.size());
    const int decode_tokens = std::get<ir::TokenOpPayload>(node.payload).max_tokens;
    const int compute_steps = remaining_tokens + decode_tokens;
    total_work_tokens += compute_steps;
    compute_steps_history.push_back(compute_steps);
    used_cached_state_history.push_back(prefix_state.has_value() ? 1 : 0);

    const auto* payload = std::get_if<TokenOpPayload>(&node.payload);
    const std::int32_t target = payload == nullptr ? 0 : payload->max_tokens;
    continuum::TokensValue out;
    out.ids.reserve(static_cast<std::size_t>(target));
    for (std::int32_t i = 0; i < target; ++i) {
      out.ids.push_back(1000 + ((i + prompt_len) % 97));
    }
    continuum::backend::BackendRunResult r;
    r.output = out;
    r.resulting_state = continuum::backend::BackendState{reinterpret_cast<void*>(0xBEEF)};
    r.used_cached_state = prefix_state.has_value();
    r.compute_steps = compute_steps;
    r.reused_prefix_len = prefix_state.has_value() ? std::max<std::int32_t>(0, prompt_len - remaining_tokens) : 0;
    return r;
  }
};

class TensorPassthroughBackend final : public continuum::backend::Backend {
 public:
  continuum::backend::BackendCapabilities capabilities() const override {
    return continuum::backend::BackendCapabilities{true, false, false};
  }

  continuum::backend::BackendRunResult run_with_cache(
      const ir::Node&,
      const std::vector<continuum::Value>& inputs,
      const std::optional<continuum::backend::BackendState>&,
      std::int32_t) override {
    continuum::backend::BackendRunResult r;
    r.output = inputs.empty() ? continuum::Value{std::string("empty")} : inputs.front();
    return r;
  }
};

TEST(InterpreterTest, RunUsesInputValuesAndTensorPassThrough) {
  Graph g;
  Node input = MakeTensorNode("input");
  const NodeId in_id = g.add_node(input);
  const NodeId passthrough = g.add_node(MakeTensorNode("id", {in_id}));

  continuum::backend::BackendRegistry registry;
  registry.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>());
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::Interpreter interp(registry, cache);

  continuum::TensorValue tv{torch::tensor({1.0f, 2.0f, 3.0f})};
  std::unordered_map<NodeId, continuum::Value> inputs;
  inputs[in_id] = tv;
  const auto out = interp.run(g, inputs);
  ASSERT_EQ(out.size(), 2U);
  ASSERT_TRUE(std::holds_alternative<continuum::TensorValue>(out[1]));
  EXPECT_EQ(std::get<continuum::TensorValue>(out[1]).tensor.numel(), 3);
  (void)passthrough;
}

TEST(InterpreterTest, TokenNodeDispatchesByModelFamilyBackend) {
  Graph g;
  const NodeId in_id = g.add_node(MakeTensorNode("input"));
  Node tok;
  tok.kind = NodeKind::TokenOp;
  tok.inputs = {in_id};
  tok.payload = TokenOpPayload{"generate", "openai/gpt-4o-mini", 0.1f, 8};
  tok.out_type = TokensType{"cl100k_base", 128, "gpt"};
  tok.effect = Effect{static_cast<std::uint8_t>(EffectBit::Net)};
  const NodeId tok_id = g.add_node(tok);

  continuum::backend::BackendRegistry registry;
  auto openai = std::make_shared<RecordingBackend>();
  openai->response = continuum::TokensValue{{11, 22, 33}};
  registry.register_backend("openai", openai, 100);
  registry.register_backend("default", std::make_shared<RecordingBackend>(), 10);
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::Interpreter interp(registry, cache);

  std::unordered_map<NodeId, continuum::Value> inputs;
  inputs[in_id] = continuum::TensorValue{torch::tensor({9.0f}), "libtorch"};
  const auto out = interp.run(g, inputs);
  ASSERT_EQ(out.size(), 2U);
  ASSERT_TRUE(std::holds_alternative<continuum::TokensValue>(out[1]));
  EXPECT_EQ(std::get<continuum::TokensValue>(out[1]).ids.size(), 3U);
  EXPECT_EQ(openai->calls, 1);
  (void)tok_id;
}

TEST(InterpreterTest, TokenNodeCacheHitReducesBackendWork) {
  Node tok;
  tok.kind = NodeKind::TokenOp;
  tok.payload = TokenOpPayload{"generate", "openai/gpt-4o-mini", 0.1f, 6};
  tok.out_type = TokensType{"cl100k_base", 128, "gpt"};
  tok.effect = Effect{static_cast<std::uint8_t>(EffectBit::Net)};

  continuum::backend::BackendRegistry registry;
  auto openai = std::make_shared<CacheAwareBackend>();
  registry.register_backend("openai", openai, 100);
  registry.register_backend("default", std::make_shared<CacheAwareBackend>(), 10);
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::Interpreter interp(registry, cache);

  std::vector<continuum::Value> inputs{continuum::TokensValue{{1, 2, 3, 4}}};
  auto first = interp.step(tok, inputs);
  auto second = interp.step(tok, inputs);

  ASSERT_TRUE(std::holds_alternative<continuum::TokensValue>(first));
  ASSERT_TRUE(std::holds_alternative<continuum::TokensValue>(second));
  EXPECT_EQ(std::get<continuum::TokensValue>(first).ids, std::get<continuum::TokensValue>(second).ids);
  ASSERT_EQ(openai->compute_steps_history.size(), 2U);
  EXPECT_LT(openai->compute_steps_history[1], openai->compute_steps_history[0]);
  ASSERT_EQ(openai->used_cached_state_history.size(), 2U);
  EXPECT_EQ(openai->used_cached_state_history[0], 0);
  EXPECT_EQ(openai->used_cached_state_history[1], 1);
  EXPECT_GT(cache.size(), 0U);
}

TEST(CheckpointTest, ValueRoundtripSupportsTensorTokensAndScalars) {
  const continuum::Value tensor_value = continuum::TensorValue{torch::tensor({1.5f, -2.0f}, torch::kFloat32), "libtorch"};
  const continuum::Value token_value = continuum::TokensValue{{1, 2, 3, 4}};
  const continuum::Value scalar_value = 42.25;

  const auto tensor_bytes = continuum::runtime::serialize_value(tensor_value);
  const auto token_bytes = continuum::runtime::serialize_value(token_value);
  const auto scalar_bytes = continuum::runtime::serialize_value(scalar_value);

  const auto tensor_back = continuum::runtime::deserialize_value(tensor_bytes.data(), tensor_bytes.size());
  const auto token_back = continuum::runtime::deserialize_value(token_bytes.data(), token_bytes.size());
  const auto scalar_back = continuum::runtime::deserialize_value(scalar_bytes.data(), scalar_bytes.size());

  ASSERT_TRUE(std::holds_alternative<continuum::TensorValue>(tensor_back));
  ASSERT_TRUE(std::holds_alternative<continuum::TokensValue>(token_back));
  ASSERT_TRUE(std::holds_alternative<double>(scalar_back));
  EXPECT_TRUE(torch::allclose(
      std::get<continuum::TensorValue>(tensor_value).tensor,
      std::get<continuum::TensorValue>(tensor_back).tensor));
  EXPECT_EQ(
      std::get<continuum::TokensValue>(token_value).ids,
      std::get<continuum::TokensValue>(token_back).ids);
  EXPECT_DOUBLE_EQ(std::get<double>(scalar_value), std::get<double>(scalar_back));
}

TEST(InterpreterTest, RunUntilAndResumeMatchFullRunAcrossCheckpointBytes) {
  Graph g;
  const NodeId in_id = g.add_node(MakeTensorNode("input"));
  const NodeId mid_id = g.add_node(MakeTensorNode("id", {in_id}));
  const NodeId out_id = g.add_node(MakeTensorNode("id", {mid_id}));
  (void)out_id;

  std::unordered_map<NodeId, continuum::Value> inputs;
  inputs[in_id] = continuum::TensorValue{torch::tensor({2.0f, 4.0f, 8.0f}, torch::kFloat32), "libtorch"};

  continuum::backend::BackendRegistry registry_full;
  registry_full.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>());
  continuum::runtime::KVCacheIndex cache_full;
  continuum::runtime::Interpreter full_interp(registry_full, cache_full);
  const auto full_out = full_interp.run(g, inputs);

  continuum::backend::BackendRegistry registry_a;
  registry_a.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>());
  continuum::runtime::KVCacheIndex cache_a;
  continuum::runtime::Interpreter first_interp(registry_a, cache_a);
  first_interp.begin(g, inputs);
  const auto checkpoint = first_interp.run_until(mid_id);
  const auto checkpoint_bytes = continuum::runtime::serialize_checkpoint(checkpoint);

  continuum::backend::BackendRegistry registry_b;
  registry_b.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>());
  continuum::runtime::KVCacheIndex cache_b;
  continuum::runtime::Interpreter resumed_interp(registry_b, cache_b);
  const auto restored_checkpoint = continuum::runtime::deserialize_checkpoint(checkpoint_bytes);
  const auto resumed_out = resumed_interp.resume(restored_checkpoint);

  ASSERT_EQ(full_out.size(), resumed_out.size());
  ASSERT_TRUE(std::holds_alternative<continuum::TensorValue>(full_out.back()));
  ASSERT_TRUE(std::holds_alternative<continuum::TensorValue>(resumed_out.back()));
  EXPECT_TRUE(torch::allclose(
      std::get<continuum::TensorValue>(full_out.back()).tensor,
      std::get<continuum::TensorValue>(resumed_out.back()).tensor));
}

TEST(CacheTest, LongestPrefixFindsDeepestMatch) {
  continuum::runtime::KVCacheIndex cache(16);
  cache.insert(continuum::runtime::CacheEntry{
                   0, "openai/gpt-4o-mini", DecodeFor(), 2, continuum::backend::BackendState{reinterpret_cast<void*>(0x1)}, 0},
               {10, 20});
  cache.insert(continuum::runtime::CacheEntry{
                   0, "openai/gpt-4o-mini", DecodeFor(), 4, continuum::backend::BackendState{reinterpret_cast<void*>(0x2)}, 0},
               {10, 20, 30, 40});

  auto hit = cache.longest_prefix("openai/gpt-4o-mini", DecodeFor(), {10, 20, 30, 40, 50});
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(hit->second, 4);
  EXPECT_EQ(hit->first.backend_state.handle, reinterpret_cast<void*>(0x2));
}

TEST(CacheTest, LruEvictionDropsLeastRecentlyUsedEntry) {
  continuum::runtime::KVCacheIndex cache(2);
  cache.insert(
      continuum::runtime::CacheEntry{0, "m", DecodeFor(), 1, continuum::backend::BackendState{reinterpret_cast<void*>(0x1)}, 0},
      {1});
  cache.insert(
      continuum::runtime::CacheEntry{0, "m", DecodeFor(), 1, continuum::backend::BackendState{reinterpret_cast<void*>(0x2)}, 0},
      {2});
  ASSERT_TRUE(cache.longest_prefix("m", DecodeFor(), {1}).has_value());  // mark handle 0x1 as recent

  cache.insert(
      continuum::runtime::CacheEntry{0, "m", DecodeFor(), 1, continuum::backend::BackendState{reinterpret_cast<void*>(0x3)}, 0},
      {3});
  EXPECT_FALSE(cache.longest_prefix("m", DecodeFor(), {2}).has_value());
  EXPECT_TRUE(cache.longest_prefix("m", DecodeFor(), {1}).has_value());
  EXPECT_TRUE(cache.longest_prefix("m", DecodeFor(), {3}).has_value());
}

TEST(CacheTest, InvalidateRemovesAllEntriesForBackendHandle) {
  continuum::runtime::KVCacheIndex cache(8);
  const auto* handle = reinterpret_cast<void*>(0xABCD);
  cache.insert(
      continuum::runtime::CacheEntry{0, "a", DecodeFor(), 2, continuum::backend::BackendState{const_cast<void*>(handle)}, 0},
      {7, 8});
  cache.insert(
      continuum::runtime::CacheEntry{0, "a", DecodeFor(), 3, continuum::backend::BackendState{const_cast<void*>(handle)}, 0},
      {7, 8, 9});
  cache.insert(
      continuum::runtime::CacheEntry{0, "a", DecodeFor(), 1, continuum::backend::BackendState{reinterpret_cast<void*>(0x99)}, 0},
      {5});
  cache.invalidate(const_cast<void*>(handle));

  EXPECT_FALSE(cache.longest_prefix("a", DecodeFor(), {7, 8, 9}).has_value());
  EXPECT_TRUE(cache.longest_prefix("a", DecodeFor(), {5, 6}).has_value());
}

TEST(CacheTest, DecodeParamsAffectCacheKey) {
  continuum::runtime::KVCacheIndex cache(8);
  cache.insert(
      continuum::runtime::CacheEntry{
          0, "openai/gpt-4o-mini", DecodeFor("generate", 0.1f, 64), 4,
          continuum::backend::BackendState{reinterpret_cast<void*>(0x1)}, 0},
      {1, 2, 3, 4});
  EXPECT_TRUE(cache.longest_prefix("openai/gpt-4o-mini", DecodeFor("generate", 0.1f, 64), {1, 2, 3, 4, 9}).has_value());
  EXPECT_FALSE(cache.longest_prefix("openai/gpt-4o-mini", DecodeFor("generate", 0.9f, 64), {1, 2, 3, 4, 9}).has_value());
}

TEST(TypecheckTest, ReportsBinaryTensorOpMismatch) {
  Graph g;
  Node a = MakeTensorNode("input");
  a.out_type = TensorType{{2, 4}, DType::F32, Device::CPU};
  const NodeId a_id = g.add_node(a);
  Node b = MakeTensorNode("input");
  b.out_type = TensorType{{2, 4}, DType::F16, Device::CPU};
  const NodeId b_id = g.add_node(b);
  Node add = MakeTensorNode("add", {a_id, b_id});
  add.out_type = TensorType{{2, 4}, DType::F32, Device::CPU};
  const NodeId add_id = g.add_node(add);

  auto errors = continuum::compiler::typecheck(g);
  ASSERT_FALSE(errors.empty());
  EXPECT_EQ(errors[0].node_id, add_id);
  EXPECT_NE(errors[0].message.find("dtype"), std::string::npos);
}

TEST(TypecheckTest, RefinesTensorOutputTypeFromInputs) {
  Graph g;
  Node a = MakeTensorNode("input");
  a.out_type = TensorType{{-1, 64}, DType::F32, Device::CPU};
  const NodeId a_id = g.add_node(a);
  Node b = MakeTensorNode("input");
  b.out_type = TensorType{{16, 64}, DType::F32, Device::CPU};
  const NodeId b_id = g.add_node(b);
  Node add = MakeTensorNode("add", {a_id, b_id});
  add.out_type = TensorType{{-1, 64}, DType::F32, Device::CPU};
  const NodeId add_id = g.add_node(add);

  auto errors = continuum::compiler::typecheck(g);
  EXPECT_TRUE(errors.empty());
  const auto& out = std::get<TensorType>(g.get(add_id).out_type);
  EXPECT_EQ(out.shape[0], 16);
  EXPECT_EQ(out.shape[1], 64);
}

TEST(RewriteTest, HoistsCommonTokenPrefixesByRewiringUses) {
  Graph g;
  const NodeId in = g.add_node(MakeTensorNode("input"));

  Node t1;
  t1.kind = NodeKind::TokenOp;
  t1.inputs = {in};
  t1.payload = TokenOpPayload{"generate", "openai/gpt-4o-mini", 0.2f, 32};
  t1.out_type = TokensType{"cl100k_base", 128, "gpt"};
  t1.debug_name = "Answer: X";
  const NodeId t1_id = g.add_node(t1);

  Node t2 = t1;
  t2.debug_name = "Answer: Y";
  const NodeId t2_id = g.add_node(t2);

  Node consumer = MakeTensorNode("id", {t2_id});
  const NodeId c_id = g.add_node(consumer);

  continuum::compiler::hoist_common_token_prefixes(g);
  EXPECT_EQ(g.get(t2_id).inputs[0], t1_id);
  EXPECT_EQ(g.get(c_id).inputs[0], t2_id);
}

TEST(CacheTest, SharedSystemPromptYieldsHighHitRate) {
  continuum::runtime::KVCacheIndex cache(128);
  const auto decode = DecodeFor("generate", 0.2f, 64);
  const std::vector<std::int32_t> shared_prefix = {'S', 'y', 's', ':', ' '};
  int hits = 0;
  const int calls = 10;
  for (int i = 0; i < calls; ++i) {
    std::vector<std::int32_t> prompt = shared_prefix;
    prompt.push_back(static_cast<std::int32_t>('0' + (i % 10)));
    auto hit = cache.longest_prefix("openai/gpt-4o-mini", decode, prompt);
    if (hit.has_value()) {
      ++hits;
    }
    cache.insert(
        continuum::runtime::CacheEntry{
            0, "openai/gpt-4o-mini", decode, static_cast<std::int32_t>(shared_prefix.size()),
            continuum::backend::BackendState{reinterpret_cast<void*>(0x1234)}, 0},
        shared_prefix);
  }
  const double hit_rate = static_cast<double>(hits) / static_cast<double>(calls - 1);
  EXPECT_GE(hit_rate, 0.8);
}

TEST(RewriteTest, MemoizesPureToolOpsByRewiringUses) {
  Graph g;
  const NodeId in = g.add_node(MakeTensorNode("input"));

  Node tool;
  tool.kind = NodeKind::ToolOp;
  tool.inputs = {in};
  tool.payload = ToolOpPayload{"search", Schema{"{\"q\":\"string\"}"}, Schema{"{\"ok\":\"bool\"}"}};
  tool.out_type = SchemaType{"{\"ok\":\"bool\"}", 42};
  tool.effect.bits = static_cast<std::uint8_t>(EffectBit::Pure) | static_cast<std::uint8_t>(EffectBit::Idem);
  const NodeId t1_id = g.add_node(tool);
  const NodeId t2_id = g.add_node(tool);
  (void)t1_id;

  Node consumer = MakeTensorNode("id", {t2_id});
  const NodeId c_id = g.add_node(consumer);

  continuum::compiler::memoize_pure_tool_ops(g);
  EXPECT_EQ(g.get(c_id).inputs[0], t1_id);
}

TEST(RewriteTest, SpecializesStructuredTokenOutputs) {
  Graph g;
  const NodeId in = g.add_node(MakeTensorNode("input"));
  Node tok;
  tok.kind = NodeKind::TokenOp;
  tok.inputs = {in};
  tok.payload = TokenOpPayload{"generate", "openai/gpt-4o-mini", 0.2f, 32};
  tok.out_type = SchemaType{"{\"name\":\"string\"}", 77};
  const NodeId tok_id = g.add_node(tok);
  (void)tok_id;

  continuum::compiler::specialize_structured_outputs(g);
  const auto& payload = std::get<TokenOpPayload>(g.get(tok_id).payload);
  EXPECT_EQ(payload.op_name, "generate:structured");
}

TEST(BackendTest, TokenBackendsReturnTokenValues) {
  Node tok;
  tok.kind = NodeKind::TokenOp;
  tok.payload = TokenOpPayload{"generate", "openai/gpt-4o-mini", 0.2f, 8};
  tok.out_type = TokensType{"cl100k_base", 32, "gpt"};
  std::vector<continuum::Value> inputs{std::string("hello world")};

  continuum::backend::OpenAIBackend openai;
  auto out1 = openai.run_with_cache(tok, inputs, std::nullopt, 0);
  ASSERT_TRUE(std::holds_alternative<continuum::TokensValue>(out1.output));
  EXPECT_FALSE(std::get<continuum::TokensValue>(out1.output).ids.empty());

  continuum::backend::AnthropicBackend anthropic;
  auto out2 = anthropic.run_with_cache(tok, inputs, std::nullopt, 0);
  ASSERT_TRUE(std::holds_alternative<continuum::TokensValue>(out2.output));
  EXPECT_FALSE(std::get<continuum::TokensValue>(out2.output).ids.empty());

  continuum::backend::VllmShimBackend vllm;
  auto out3 = vllm.run_with_cache(tok, inputs, std::nullopt, 0);
  ASSERT_TRUE(std::holds_alternative<continuum::TokensValue>(out3.output));
  EXPECT_FALSE(std::get<continuum::TokensValue>(out3.output).ids.empty());
}

TEST(BackendTest, CachedStateProducesSameOutputAsFullRun) {
  Node tok;
  tok.kind = NodeKind::TokenOp;
  tok.payload = TokenOpPayload{"generate", "openai/gpt-4o-mini", 0.2f, 8};
  tok.out_type = TokensType{"cl100k_base", 32, "gpt"};
  std::vector<continuum::Value> inputs{continuum::TokensValue{{1, 2, 3, 4, 5}}};

  continuum::backend::OpenAIBackend openai;
  auto full = openai.run_with_cache(tok, inputs, std::nullopt, 5);
  auto resumed = openai.run_with_cache(tok, inputs, full.resulting_state, 0);
  ASSERT_TRUE(std::holds_alternative<continuum::TokensValue>(full.output));
  ASSERT_TRUE(std::holds_alternative<continuum::TokensValue>(resumed.output));
  EXPECT_EQ(std::get<continuum::TokensValue>(full.output).ids, std::get<continuum::TokensValue>(resumed.output).ids);
  EXPECT_TRUE(resumed.used_cached_state);
}

TEST(BackendRegistryTest, SelectsBackendByCapabilityAndPriority) {
  continuum::backend::BackendRegistry registry;
  auto token_low = std::make_shared<RecordingBackend>(continuum::backend::BackendCapabilities{false, true, true});
  auto token_high = std::make_shared<RecordingBackend>(continuum::backend::BackendCapabilities{false, true, true});
  auto tensor_backend = std::make_shared<continuum::backend::MLXBackend>();
  registry.register_backend("token_low", token_low, 10);
  registry.register_backend("token_high", token_high, 50);
  registry.register_backend("tensor", tensor_backend, 40);

  continuum::ir::Node token_node;
  token_node.kind = NodeKind::TokenOp;
  auto token_selected = registry.select_backend(token_node);
  EXPECT_EQ(token_selected.name, "token_high");
  EXPECT_TRUE(token_selected.capabilities.supports_token);

  continuum::ir::Node tensor_node;
  tensor_node.kind = NodeKind::TensorOp;
  auto tensor_selected = registry.select_backend(tensor_node);
  EXPECT_EQ(tensor_selected.name, "tensor");
  EXPECT_TRUE(tensor_selected.capabilities.supports_tensor);
}

TEST(InterpreterTest, TensorBackendAutoSelectionHonorsPriority) {
  continuum::backend::BackendRegistry registry;
  registry.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>(), 10);
  registry.register_backend("mlx", std::make_shared<continuum::backend::MLXBackend>(), 100);
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::Interpreter interp(registry, cache);

  continuum::ir::Node n;
  n.kind = NodeKind::TensorOp;
  n.payload = TensorOpPayload{"relu", {}};

  std::vector<continuum::Value> inputs;
  inputs.emplace_back(continuum::TensorValue{torch::tensor({-1.0f, 2.0f}), "libtorch"});
  auto out = interp.step(n, inputs);
  ASSERT_TRUE(std::holds_alternative<continuum::MlxTensorValue>(out));
  const auto& mx = std::get<continuum::MlxTensorValue>(out);
  ASSERT_EQ(mx.data.size(), 2U);
  EXPECT_FLOAT_EQ(mx.data[0], 0.0f);
  EXPECT_FLOAT_EQ(mx.data[1], 2.0f);
}

TEST(InterpreterTest, MixedTensorInputsAreExplicitlyConvertedForSelectedBackend) {
  continuum::backend::BackendRegistry registry;
  registry.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>(), 10);
  registry.register_backend("mlx", std::make_shared<continuum::backend::MLXBackend>(), 100);
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::Interpreter interp(registry, cache);

  continuum::ir::Node n;
  n.kind = NodeKind::TensorOp;
  n.payload = TensorOpPayload{"add", {}};

  std::vector<continuum::Value> inputs;
  inputs.emplace_back(continuum::TensorValue{torch::tensor({1.0f, 2.0f}), "libtorch"});
  inputs.emplace_back(continuum::MlxTensorValue{{2}, {3.0f, 4.0f}, "mlx"});
  auto out = interp.step(n, inputs);
  ASSERT_TRUE(std::holds_alternative<continuum::MlxTensorValue>(out));
  const auto& mx = std::get<continuum::MlxTensorValue>(out);
  ASSERT_EQ(mx.backend_type, "mlx");
  ASSERT_EQ(mx.data.size(), 2U);
  EXPECT_FLOAT_EQ(mx.data[0], 4.0f);
  EXPECT_FLOAT_EQ(mx.data[1], 6.0f);
}

TEST(InterpreterTest, UnsupportedTensorConversionThrowsExplicitly) {
  continuum::backend::BackendRegistry registry;
  registry.register_backend("custom", std::make_shared<TensorPassthroughBackend>(), 100);
  continuum::runtime::KVCacheIndex cache;
  continuum::runtime::Interpreter interp(registry, cache);

  continuum::ir::Node n;
  n.kind = NodeKind::TensorOp;
  n.payload = TensorOpPayload{"id", {}};

  std::vector<continuum::Value> inputs;
  inputs.emplace_back(continuum::TensorValue{torch::tensor({1.0f}), "libtorch"});
  EXPECT_THROW(
      {
        try {
          (void)interp.step(n, inputs);
        } catch (const std::runtime_error& e) {
          EXPECT_NE(std::string(e.what()).find("tensor conversion not implemented"), std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

}  // namespace
}  // namespace continuum::ir
