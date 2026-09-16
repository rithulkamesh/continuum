#include <continuum/ir/graph.hpp>
#include <continuum/ir/node.hpp>
#include <continuum/ir/type.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using continuum::ir::Graph;
using continuum::ir::Node;
using continuum::ir::NodeKind;

namespace {

Graph MakeTensorGraph() {
  Graph g;
  Node n;
  n.kind = NodeKind::TensorOp;
  n.debug_name = "matmul";
  n.payload = continuum::ir::TensorOpPayload{"matmul", {2, 4}};
  n.out_type = continuum::ir::TensorType{{-1, 4}, continuum::ir::DType::F32, continuum::ir::Device::CPU};
  g.add_node(n);
  return g;
}

Graph MakeTokenGraph() {
  Graph g;
  Node n;
  n.kind = NodeKind::TokenOp;
  n.debug_name = "generate";
  n.payload = continuum::ir::TokenOpPayload{"generate", "fake/model", 0.2f, 32};
  n.out_type = continuum::ir::TokensType{"vocab", 128, "fake"};
  g.add_node(n);
  return g;
}

Graph MakePromptGraph() {
  Graph g;
  Node n;
  n.kind = NodeKind::PromptOp;
  n.debug_name = "prompt";
  n.payload = continuum::ir::PromptOpPayload{"tmpl-1", {}};
  n.out_type = continuum::ir::SchemaType{"{}", 42};
  g.add_node(n);
  return g;
}

Graph MakeToolGraph() {
  Graph g;
  Node n;
  n.kind = NodeKind::ToolOp;
  n.debug_name = "tool";
  n.payload = continuum::ir::ToolOpPayload{"search", continuum::ir::Schema{"{\"q\":1}"}, continuum::ir::Schema{"{\"r\":1}"}};
  n.out_type = continuum::ir::EffectType{1};
  g.add_node(n);
  return g;
}

Graph MakeControlGraph() {
  Graph g;
  Node n;
  n.kind = NodeKind::ControlOp;
  n.debug_name = "ctrl";
  n.payload = continuum::ir::ControlOpPayload{continuum::ir::ControlOpPayload::Kind::While, {}};
  g.add_node(n);
  return g;
}

struct CorpusCase {
  const char* name;
  Graph (*make)();
};

const CorpusCase kCases[] = {
    {"tensor_op", &MakeTensorGraph},
    {"token_op", &MakeTokenGraph},
    {"prompt_op", &MakePromptGraph},
    {"tool_op", &MakeToolGraph},
    {"control_op", &MakeControlGraph},
};

fs::path CorpusDir() {
  // Prefer checked-in path relative to source; fall back to CWD.
  const fs::path candidates[] = {
      fs::path("tests/data/cir"),
      fs::path("../tests/data/cir"),
      fs::path("../../tests/data/cir"),
      fs::path(CONTINUUM_SOURCE_DIR) / "tests" / "data" / "cir",
  };
  for (const auto& p : candidates) {
    if (fs::is_directory(p)) {
      return p;
    }
  }
  return fs::path("tests/data/cir");
}

std::vector<std::uint8_t> ReadFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  EXPECT_TRUE(in.good()) << "missing corpus file: " << path;
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}

}  // namespace

TEST(CirCorpusTest, GoldenFilesRoundtripByteIdentical) {
  const auto dir = CorpusDir();
  ASSERT_TRUE(fs::is_directory(dir)) << "CIR corpus directory missing: " << dir
                                      << " (run tests/cpp/gen_cir_corpus)";

  for (const auto& c : kCases) {
    const auto path = dir / (std::string(c.name) + ".cir");
    ASSERT_TRUE(fs::exists(path)) << path;
    const auto golden = ReadFile(path);
    ASSERT_FALSE(golden.empty()) << path;

    auto graph = Graph::deserialize(golden.data(), golden.size());
    const auto again = graph.serialize();
    EXPECT_EQ(again, golden) << "byte drift for " << c.name
                             << "; if intentional, regenerate corpus and document migration in docs/design/ir.md";

    // Also ensure the in-memory builder still matches the golden (schema lock).
    const auto rebuilt = c.make().serialize();
    EXPECT_EQ(rebuilt, golden) << "builder diverged from golden for " << c.name;
  }
}
