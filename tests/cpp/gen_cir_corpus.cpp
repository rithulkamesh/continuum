// One-shot generator for tests/data/cir/*.cir golden binaries.
// Build: cmake --build build-tests --target gen_cir_corpus
// Run from repo root: ./build-tests/tests/cpp/gen_cir_corpus

#include <continuum/ir/graph.hpp>
#include <continuum/ir/node.hpp>
#include <continuum/ir/type.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using continuum::ir::Graph;
using continuum::ir::Node;
using continuum::ir::NodeKind;

static void Write(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  std::cout << "wrote " << path << " (" << bytes.size() << " bytes)\n";
}

int main(int argc, char** argv) {
  fs::path out_dir = "tests/data/cir";
  if (argc > 1) {
    out_dir = argv[1];
  }

  {
    Graph g;
    Node n;
    n.kind = NodeKind::TensorOp;
    n.debug_name = "matmul";
    n.payload = continuum::ir::TensorOpPayload{"matmul", {2, 4}};
    n.out_type = continuum::ir::TensorType{{-1, 4}, continuum::ir::DType::F32, continuum::ir::Device::CPU};
    g.add_node(n);
    Write(out_dir / "tensor_op.cir", g.serialize());
  }
  {
    Graph g;
    Node n;
    n.kind = NodeKind::TokenOp;
    n.debug_name = "generate";
    n.payload = continuum::ir::TokenOpPayload{"generate", "fake/model", 0.2f, 32};
    n.out_type = continuum::ir::TokensType{"vocab", 128, "fake"};
    g.add_node(n);
    Write(out_dir / "token_op.cir", g.serialize());
  }
  {
    Graph g;
    Node n;
    n.kind = NodeKind::PromptOp;
    n.debug_name = "prompt";
    n.payload = continuum::ir::PromptOpPayload{"tmpl-1", {}};
    n.out_type = continuum::ir::SchemaType{"{}", 42};
    g.add_node(n);
    Write(out_dir / "prompt_op.cir", g.serialize());
  }
  {
    Graph g;
    Node n;
    n.kind = NodeKind::ToolOp;
    n.debug_name = "tool";
    n.payload = continuum::ir::ToolOpPayload{"search", continuum::ir::Schema{"{\"q\":1}"}, continuum::ir::Schema{"{\"r\":1}"}};
    n.out_type = continuum::ir::EffectType{1};
    g.add_node(n);
    Write(out_dir / "tool_op.cir", g.serialize());
  }
  {
    Graph g;
    Node n;
    n.kind = NodeKind::ControlOp;
    n.debug_name = "ctrl";
    n.payload = continuum::ir::ControlOpPayload{continuum::ir::ControlOpPayload::Kind::While, {}};
    g.add_node(n);
    Write(out_dir / "control_op.cir", g.serialize());
  }
  return 0;
}
