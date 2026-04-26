#include <continuum/ir/graph.hpp>
#include <continuum/ir/node.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_ir(py::module_& m) {
  py::enum_<continuum::ir::NodeKind>(m, "NodeKind")
      .value("TensorOp", continuum::ir::NodeKind::TensorOp)
      .value("TokenOp", continuum::ir::NodeKind::TokenOp)
      .value("PromptOp", continuum::ir::NodeKind::PromptOp)
      .value("ToolOp", continuum::ir::NodeKind::ToolOp)
      .value("ControlOp", continuum::ir::NodeKind::ControlOp);

  py::class_<continuum::ir::Node>(m, "Node")
      .def(py::init<>())
      .def_readwrite("id", &continuum::ir::Node::id)
      .def_readwrite("kind", &continuum::ir::Node::kind)
      .def_readwrite("inputs", &continuum::ir::Node::inputs)
      .def_readwrite("debug_name", &continuum::ir::Node::debug_name);

  py::class_<continuum::ir::Graph>(m, "Graph")
      .def(py::init<>())
      .def("add_node", &continuum::ir::Graph::add_node)
      .def("topo_order", &continuum::ir::Graph::topo_order, py::return_value_policy::copy)
      .def("structural_hash", &continuum::ir::Graph::structural_hash)
      .def("serialize", [](const continuum::ir::Graph& g) {
        const auto bytes = g.serialize();
        return py::bytes(reinterpret_cast<const char*>(bytes.data()), static_cast<py::ssize_t>(bytes.size()));
      })
      .def_static("deserialize", [](py::bytes data) {
        std::string buf = data;
        return continuum::ir::Graph::deserialize(
            reinterpret_cast<const std::uint8_t*>(buf.data()),
            static_cast<std::size_t>(buf.size()));
      });

  m.def("version", []() { return "0.1"; });
}
