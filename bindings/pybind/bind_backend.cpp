#include <continuum/backend/anthropic.hpp>
#include <continuum/backend/azure_openai.hpp>
#include <continuum/backend/backend.hpp>
#include <continuum/backend/conformance.hpp>
#include <continuum/backend/fake_llm.hpp>
#include <continuum/backend/libtorch.hpp>
#include <continuum/backend/mlx_backend.hpp>
#include <continuum/backend/openai.hpp>
#include <continuum/backend/vllm_shim.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using continuum::backend::BackendRegistry;

namespace {

bool LooksLikePath(const std::string& target) {
  auto ends_with = [&](const char* suffix) {
    const std::string s(suffix);
    return target.size() >= s.size() && target.compare(target.size() - s.size(), s.size(), s) == 0;
  };
  return target.find('/') != std::string::npos || target.find('\\') != std::string::npos ||
         ends_with(".so") || ends_with(".dylib") || ends_with(".dll");
}

// A built-in backend by name, a plugin path, or a plugin name listed in
// CONTINUUM_BACKEND_PLUGINS.
std::shared_ptr<continuum::backend::Backend> ResolveBackend(const std::string& target) {
  namespace be = continuum::backend;
  if (LooksLikePath(target)) return be::LoadBackendPlugin(target);
  if (target == "fake" || target == "fake_llm") return std::make_shared<be::FakeLLMBackend>();
  if (target == "libtorch") return std::make_shared<be::LibTorchBackend>();
  if (target == "vllm") return std::make_shared<be::VllmShimBackend>();
  if (target == "mlx") return std::make_shared<be::MLXBackend>();
  if (target == "openai") return std::make_shared<be::OpenAIBackend>();
  if (target == "azure") return std::make_shared<be::AzureOpenAIBackend>();
  if (target == "anthropic") return std::make_shared<be::AnthropicBackend>();
  if (const char* env = std::getenv("CONTINUUM_BACKEND_PLUGINS")) {
    for (const auto& spec : be::ParsePluginSpecs(env)) {
      if (spec.name == target) return be::LoadBackendPlugin(spec.path);
    }
  }
  throw std::runtime_error("unknown backend '" + target +
                           "': not a built-in, a plugin path, or a CONTINUUM_BACKEND_PLUGINS name");
}

}  // namespace

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
      })
      .def("load_plugin", &BackendRegistry::load_plugin, py::arg("name"), py::arg("path"),
           py::arg("priority") = 0,
           "Load a shared-library backend built against backend_abi.h and register it.")
      .def("load_plugins_from_env", [](BackendRegistry& r, const std::string& env_var) {
             return r.load_plugins_from_env(env_var.c_str());
           }, py::arg("env_var") = "CONTINUUM_BACKEND_PLUGINS",
           "Load every `name[@priority]=path` entry (';'-separated) in the variable.")
      .def("names", &BackendRegistry::names)
      .def("has", &BackendRegistry::has, py::arg("name"));

  m.def("check_backend", [](const std::string& target, bool expect_deterministic,
                             const std::string& prompt, const std::string& tensor_op) {
          auto backend = ResolveBackend(target);
          continuum::backend::ConformanceOptions opts;
          opts.expect_deterministic = expect_deterministic;
          if (!prompt.empty()) opts.prompt = prompt;
          opts.tensor_op = tensor_op;
          const auto report = continuum::backend::RunBackendConformance(*backend, target, opts);
          py::list checks;
          for (const auto& c : report.checks) {
            py::dict d;
            d["name"] = c.name;
            d["status"] = c.status == continuum::backend::ConformanceStatus::Pass   ? "pass"
                          : c.status == continuum::backend::ConformanceStatus::Fail ? "fail"
                                                                                    : "skip";
            d["detail"] = c.detail;
            checks.append(d);
          }
          py::dict out;
          out["backend"] = report.backend;
          out["passed"] = report.passed();
          out["checks"] = checks;
          out["summary"] = report.summary();
          return out;
        },
        py::arg("target"), py::arg("expect_deterministic") = true, py::arg("prompt") = "",
        py::arg("tensor_op") = "identity",
        "Run the backend conformance kit against a built-in backend name, a plugin path, "
        "or a CONTINUUM_BACKEND_PLUGINS name.");
}
