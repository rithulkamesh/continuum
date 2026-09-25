#include <continuum/backend/backend.hpp>
#include <continuum/backend/conformance.hpp>
#include <continuum/backend/fake_llm.hpp>
#include <continuum/backend/libtorch.hpp>
#include <continuum/backend/vllm_shim.hpp>
#include <continuum/runtime/session.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <variant>

namespace be = continuum::backend;

namespace {

void ExpectConformant(be::Backend& backend, const std::string& name) {
  const auto report = be::RunBackendConformance(backend, name);
  EXPECT_TRUE(report.passed()) << report.summary();
}

// v1 vtable with a deliberately broken metric contract: claims reuse on a
// cold run and never returns state.
continuum_backend_caps_t LiarCaps(void*) { return {0, 1, 1}; }
continuum_backend_run_result_t LiarRun(void*, continuum_backend_node_meta_t, const continuum_backend_value_t*,
                                       size_t, const continuum_backend_state_t*, int32_t) {
  continuum_backend_run_result_t r{};
  r.output.kind = CONTINUUM_BACKEND_VALUE_STRING;
  r.output.string_data = "x";
  r.used_cached_state = 1;
  r.reused_prefix_len = 5;
  return r;
}

}  // namespace

TEST(BackendConformance, BuiltinsPass) {
  be::FakeLLMBackend fake;
  ExpectConformant(fake, "fake_llm");
  be::LibTorchBackend torch_backend;
  ExpectConformant(torch_backend, "libtorch");
}

TEST(BackendConformance, FlagsContractViolations) {
  auto liar = be::MakeBackendFromAbi({1u, nullptr, LiarCaps, nullptr, LiarRun, nullptr, nullptr, nullptr});
  const auto report = be::RunBackendConformance(*liar, "liar");
  EXPECT_FALSE(report.passed());
  bool cold_flagged = false, state_flagged = false;
  for (const auto& c : report.checks) {
    if (c.name == "token.cold_metrics") cold_flagged = c.status == be::ConformanceStatus::Fail;
    if (c.name == "cache.state_handle") state_flagged = c.status == be::ConformanceStatus::Fail;
  }
  EXPECT_TRUE(cold_flagged);
  EXPECT_TRUE(state_flagged);
  EXPECT_NE(report.summary().find("NOT CONFORMANT"), std::string::npos);
}

TEST(BackendPlugins, LoadsEchoPluginAndPassesConformance) {
  auto echo = be::LoadBackendPlugin(CONTINUUM_ECHO_PLUGIN);
  const auto caps = echo->capabilities();
  EXPECT_TRUE(caps.supports_token);
  EXPECT_FALSE(caps.supports_tensor);
  const auto report = be::RunBackendConformance(*echo, "echo");
  EXPECT_TRUE(report.passed()) << report.summary();
  bool roundtrip = false;
  for (const auto& c : report.checks) {
    if (c.name == "state.roundtrip") roundtrip = c.status == be::ConformanceStatus::Pass;
  }
  EXPECT_TRUE(roundtrip) << report.summary();
}

TEST(BackendPlugins, RegistryRunsPluginThroughSession) {
  be::BackendRegistry registry;
  registry.register_backend("fake", std::make_shared<be::FakeLLMBackend>(), 10);
  registry.load_plugin("echo", CONTINUUM_ECHO_PLUGIN, 100);
  ASSERT_TRUE(registry.has("echo"));

  continuum::ir::Graph g;
  continuum::ir::Node p;
  p.kind = continuum::ir::NodeKind::PromptOp;
  const auto pid = g.add_node(p);
  continuum::ir::Node t;
  t.kind = continuum::ir::NodeKind::TokenOp;
  t.payload = continuum::ir::TokenOpPayload{"generate", "echo/model", 0.0f, 8};
  t.inputs.push_back(pid);
  g.add_node(t);

  continuum::runtime::Session session("plugin", registry);
  std::unordered_map<continuum::ir::NodeId, continuum::Value> inputs{{pid, std::string{"hello plugin"}}};
  const auto out = session.run(g, inputs);
  ASSERT_TRUE(std::holds_alternative<std::string>(out.back()));
  EXPECT_EQ(std::get<std::string>(out.back()), "echo: hello plugin");
}

TEST(BackendPlugins, LoadErrorsAreDescriptive) {
  try {
    be::LoadBackendPlugin("/nonexistent/libnope.so");
    FAIL() << "expected throw";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("cannot load backend plugin"), std::string::npos);
  }
  EXPECT_THROW(be::MakeBackendFromAbi({99u, nullptr, LiarCaps, nullptr, LiarRun, nullptr, nullptr, nullptr}),
               std::runtime_error);
}

TEST(BackendPlugins, ParsesSpecsAndEnv) {
  const auto specs = be::ParsePluginSpecs(" a=/x/liba.so ; b@-3=/y/libb.so;;");
  ASSERT_EQ(specs.size(), 2u);
  EXPECT_EQ(specs[0].name, "a");
  EXPECT_EQ(specs[0].path, "/x/liba.so");
  EXPECT_EQ(specs[0].priority, 0);
  EXPECT_EQ(specs[1].name, "b");
  EXPECT_EQ(specs[1].priority, -3);
  EXPECT_THROW(be::ParsePluginSpecs("noequals"), std::runtime_error);
  EXPECT_THROW(be::ParsePluginSpecs("a@x=/p.so"), std::runtime_error);

  be::BackendRegistry registry;
  const std::string spec = std::string("echo@7=") + CONTINUUM_ECHO_PLUGIN;
  ::setenv("CONTINUUM_TEST_PLUGINS", spec.c_str(), 1);
  EXPECT_EQ(registry.load_plugins_from_env("CONTINUUM_TEST_PLUGINS"), 1u);
  EXPECT_TRUE(registry.has("echo"));
  ::unsetenv("CONTINUUM_TEST_PLUGINS");
  EXPECT_EQ(registry.load_plugins_from_env("CONTINUUM_TEST_PLUGINS"), 0u);
}
