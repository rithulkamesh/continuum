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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace py = pybind11;
using continuum::backend::BackendRegistry;

namespace {

// A token backend implemented by a Python callable, so Python code (an HTTP
// upstream, a LangChain model, ...) can sit behind the full reuse stack.
//
// The callable receives one dict: prompt_parts (list[str]), model_id,
// op_name, max_tokens, temperature, has_prefix_state, remaining_tokens,
// prompt_len. It returns the output string, or a dict with "output" and
// optional "tokens_sent" / "tokens_saved" overrides.
class PyCallableBackend final : public continuum::backend::Backend {
 public:
  PyCallableBackend(py::object fn, bool supports_cache) : fn_(std::move(fn)), supports_cache_(supports_cache) {}

  ~PyCallableBackend() override {
    py::gil_scoped_acquire gil;
    fn_ = py::object();
  }

  continuum::backend::BackendCapabilities capabilities() const override {
    return continuum::backend::BackendCapabilities{false, true, supports_cache_};
  }

  std::vector<std::uint8_t> export_state(const continuum::backend::BackendState& state) const override {
    const auto id = reinterpret_cast<std::uintptr_t>(state.handle);
    if (id == 0) return {};
    std::vector<std::uint8_t> out(sizeof(id));
    std::memcpy(out.data(), &id, sizeof(id));
    return out;
  }

  std::optional<continuum::backend::BackendState> import_state(const std::vector<std::uint8_t>& bytes) override {
    if (bytes.size() != sizeof(std::uintptr_t)) return std::nullopt;
    std::uintptr_t id = 0;
    std::memcpy(&id, bytes.data(), sizeof(id));
    if (id == 0) return std::nullopt;
    return continuum::backend::BackendState{reinterpret_cast<void*>(id)};
  }

  continuum::backend::BackendRunResult run_with_cache(
      const continuum::ir::Node& node, const std::vector<continuum::Value>& inputs,
      const std::optional<continuum::backend::BackendState>& prefix_state, std::int32_t remaining_tokens) override {
    const auto* payload = std::get_if<continuum::ir::TokenOpPayload>(&node.payload);
    std::vector<std::string> parts;
    std::int32_t prompt_len = 0;
    for (const auto& v : inputs) {
      std::string part;
      if (const auto* s = std::get_if<std::string>(&v)) part = *s;
      else if (const auto* t = std::get_if<continuum::TokensValue>(&v)) {
        for (int id : t->ids) part.push_back(static_cast<char>(id));
      } else if (const auto* sch = std::get_if<continuum::SchemaValue>(&v)) part = sch->json;
      else if (const auto* d = std::get_if<double>(&v)) part = std::to_string(*d);
      else if (const auto* n = std::get_if<std::int64_t>(&v)) part = std::to_string(*n);
      prompt_len += static_cast<std::int32_t>(part.size());
      parts.push_back(std::move(part));
    }
    const bool warm = prefix_state.has_value() && prefix_state->handle != nullptr;
    const std::int32_t remaining = std::clamp<std::int32_t>(remaining_tokens, 0, prompt_len);

    continuum::backend::BackendRunResult r;
    r.reused_prefix_len = warm ? prompt_len - remaining : 0;
    r.tokens_saved = r.reused_prefix_len;
    r.tokens_sent = warm ? remaining : prompt_len;
    r.compute_steps = r.tokens_sent;
    r.used_cached_state = warm;
    r.resulting_state.handle = reinterpret_cast<void*>(static_cast<std::uintptr_t>(next_state_++));

    py::gil_scoped_acquire gil;
    py::dict req;
    req["prompt_parts"] = parts;
    req["model_id"] = payload == nullptr ? std::string{} : payload->model_id;
    req["op_name"] = payload == nullptr ? std::string{} : payload->op_name;
    req["max_tokens"] = payload == nullptr ? 0 : payload->max_tokens;
    req["temperature"] = payload == nullptr ? 0.0f : payload->temperature;
    req["has_prefix_state"] = warm;
    req["remaining_tokens"] = remaining;
    req["prompt_len"] = prompt_len;
    py::object result = fn_(req);
    if (py::isinstance<py::dict>(result)) {
      py::dict d = result;
      r.output = d["output"].cast<std::string>();
      if (d.contains("tokens_sent")) r.tokens_sent = d["tokens_sent"].cast<std::int32_t>();
      if (d.contains("tokens_saved")) r.tokens_saved = d["tokens_saved"].cast<std::int32_t>();
    } else {
      r.output = result.cast<std::string>();
    }
    return r;
  }

 private:
  py::object fn_;
  bool supports_cache_;
  std::atomic<std::uint64_t> next_state_{1};
};

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
      .def("register_python", [](BackendRegistry& r, const std::string& name, py::function fn, int priority,
                                  bool supports_cache) {
             r.register_backend(name, std::make_shared<PyCallableBackend>(std::move(fn), supports_cache), priority);
           }, py::arg("name"), py::arg("fn"), py::arg("priority") = 0, py::arg("supports_cache") = true,
           "Register a Python callable as a token backend; see continuum.backends.")
      .def("names", &BackendRegistry::names)
      .def("has", &BackendRegistry::has, py::arg("name"));

  m.def("mlx_runtime", &continuum::backend::MLXBackend::runtime,
        "Which kernels the mlx backend runs: 'mlx <version> (gpu|cpu)' or 'reference (built without MLX)'.");

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
