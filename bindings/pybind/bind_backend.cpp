#include <continuum/backend/anthropic.hpp>
#include <continuum/backend/azure_openai.hpp>
#include <continuum/backend/backend.hpp>
#include <continuum/backend/fake_llm.hpp>
#include <continuum/backend/libtorch.hpp>
#include <continuum/backend/mlx_backend.hpp>
#include <continuum/backend/openai.hpp>
#include <continuum/backend/vllm_shim.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using continuum::backend::BackendRegistry;

void bind_backend(py::module_& m) {
  py::class_<BackendRegistry>(m, "BackendRegistry")
      .def(py::init<>())
      .def("register_default_libtorch", [](BackendRegistry& r) {
        r.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>());
      })
      .def("register_mlx", [](BackendRegistry& r) {
        r.register_backend("mlx", std::make_shared<continuum::backend::MLXBackend>());
      })
      .def("register_openai", [](BackendRegistry& r) {
        r.register_backend("openai", std::make_shared<continuum::backend::OpenAIBackend>());
      })
      .def("register_azure", [](BackendRegistry& r) {
        r.register_backend("azure", std::make_shared<continuum::backend::AzureOpenAIBackend>());
      })
      .def("register_fake_llm", [](BackendRegistry& r) {
        r.register_backend("fake", std::make_shared<continuum::backend::FakeLLMBackend>());
      })
      .def("register_anthropic", [](BackendRegistry& r) {
        r.register_backend("anthropic", std::make_shared<continuum::backend::AnthropicBackend>());
      })
      .def("register_vllm", [](BackendRegistry& r) {
        r.register_backend("vllm", std::make_shared<continuum::backend::VllmShimBackend>());
      });
}
