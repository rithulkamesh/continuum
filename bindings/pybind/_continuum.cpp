#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_ir(py::module_& m);
void bind_runtime(py::module_& m);
void bind_backend(py::module_& m);

PYBIND11_MODULE(_continuum, m) {
  m.doc() = "Continuum native module";
  auto ir = m.def_submodule("ir");
  auto runtime = m.def_submodule("runtime");
  auto backend = m.def_submodule("backend");
  bind_ir(ir);
  bind_runtime(runtime);
  bind_backend(backend);
}
