#include <continuum/backend/backend.hpp>
#include <continuum/backend/azure_openai.hpp>
#include <continuum/backend/fake_llm.hpp>
#include <continuum/backend/libtorch.hpp>
#include <continuum/backend/mlx_backend.hpp>
#include <continuum/backend/vllm_shim.hpp>
#include <continuum/ir/graph.hpp>
#include <continuum/runtime/cache.hpp>
#include <continuum/runtime/interpreter.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace py = pybind11;

namespace {
std::string TensorBackendFromEnv() {
  const char* env = std::getenv("CONTINUUM_TENSOR_BACKEND");
  if (env == nullptr) return "libtorch";
  return std::string(env);
}

continuum::backend::BackendRegistry MakeTensorRegistry(const std::string& tensor_backend) {
  continuum::backend::BackendRegistry registry;
  registry.register_backend("default", std::make_shared<continuum::backend::LibTorchBackend>(), 10);
  if (tensor_backend == "mlx") {
    registry.register_backend("mlx", std::make_shared<continuum::backend::MLXBackend>(), 100);
  }
  return registry;
}

double Percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(rank));
  const auto hi = static_cast<std::size_t>(std::ceil(rank));
  if (lo == hi) return values[lo];
  const double frac = rank - static_cast<double>(lo);
  return values[lo] + frac * (values[hi] - values[lo]);
}

py::dict RunPairedAgentBenchmark(
    const std::string& backend_prefix, int trials, bool include_warmup, std::int32_t shared_prompt_size) {
  if (trials < 2) {
    throw std::runtime_error("trials must be >= 2 to support warmup discard");
  }
  continuum::ir::Node node;
  node.kind = continuum::ir::NodeKind::TokenOp;
  node.payload = continuum::ir::TokenOpPayload{"generate", backend_prefix + "/gpt-5-mini", 0.2f, 128};

  const std::string shared_system(static_cast<std::size_t>(std::max<std::int32_t>(0, shared_prompt_size)), 'S');
  const std::string prompt = shared_system + "\nQuestion: Summarize Continuum cache behavior.";

  py::list trial_rows;
  std::vector<double> uncached_ms;
  std::vector<double> cached_ms;
  std::vector<double> ratios;
  std::vector<double> saved_ratios;
  double token_ratio_sum = 0.0;
  int token_ratio_count = 0;
  for (int i = 0; i < trials; ++i) {
    auto backend = [&]() -> std::shared_ptr<continuum::backend::Backend> {
      if (backend_prefix == "azure") return std::make_shared<continuum::backend::AzureOpenAIBackend>();
      if (backend_prefix == "vllm") return std::make_shared<continuum::backend::VllmShimBackend>();
      throw std::runtime_error("unsupported backend for benchmark: " + backend_prefix);
    }();

    std::vector<continuum::Value> in{prompt};
    auto t0 = std::chrono::steady_clock::now();
    auto uncached = backend->run_with_cache(node, in, std::nullopt, static_cast<std::int32_t>(prompt.size()));
    auto t1 = std::chrono::steady_clock::now();
    auto warm = backend->run_with_cache(node, in, std::nullopt, static_cast<std::int32_t>(prompt.size()));
    auto t2 = std::chrono::steady_clock::now();
    auto cached = backend->run_with_cache(node, in, warm.resulting_state, 0);
    auto t3 = std::chrono::steady_clock::now();
    (void)uncached;

    const double no_cache_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    const double with_cache_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count());
    const bool warmup = include_warmup && i == 0;
    const double ratio = no_cache_ms > 0.0 ? with_cache_ms / no_cache_ms : 0.0;
    const double sent = static_cast<double>(std::max(1, cached.tokens_sent + cached.tokens_saved));
    const double saved_ratio = static_cast<double>(cached.tokens_saved) / sent;
    if (!warmup) {
      uncached_ms.push_back(no_cache_ms);
      cached_ms.push_back(with_cache_ms);
      ratios.push_back(ratio);
      saved_ratios.push_back(saved_ratio);
      token_ratio_sum += saved_ratio;
      ++token_ratio_count;
    }
    py::dict row;
    row["trial"] = i + 1;
    row["warmup"] = warmup;
    row["latency_no_cache_ms"] = no_cache_ms;
    row["latency_with_cache_ms"] = with_cache_ms;
    row["latency_ratio"] = ratio;
    row["tokens_sent"] = cached.tokens_sent;
    row["tokens_saved"] = cached.tokens_saved;
    row["tokens_saved_ratio"] = saved_ratio;
    trial_rows.append(row);
  }

  const double median_no_cache = Percentile(uncached_ms, 50.0);
  const double median_with_cache = Percentile(cached_ms, 50.0);
  const double latency_ratio = median_no_cache > 0.0 ? median_with_cache / median_no_cache : 0.0;
  const double token_reduction_ratio = token_ratio_count > 0 ? token_ratio_sum / static_cast<double>(token_ratio_count) : 0.0;

  py::dict out;
  out["backend"] = backend_prefix;
  out["trials"] = trials;
  out["discarded_warmup_runs"] = include_warmup ? 1 : 0;
  out["per_trial"] = trial_rows;
  out["median_latency_no_cache"] = median_no_cache;
  out["median_latency_with_cache"] = median_with_cache;
  out["p50_latency_no_cache"] = Percentile(uncached_ms, 50.0);
  out["p50_latency_with_cache"] = Percentile(cached_ms, 50.0);
  out["p95_latency_no_cache"] = Percentile(uncached_ms, 95.0);
  out["p95_latency_with_cache"] = Percentile(cached_ms, 95.0);
  out["latency_ratio"] = latency_ratio;
  out["token_reduction_ratio"] = token_reduction_ratio;
  out["avg_tokens_saved_ratio"] = token_reduction_ratio;
  out["acceptance_primary_pass"] = token_reduction_ratio >= 0.8;
  out["acceptance_secondary_pass"] = latency_ratio < 1.0;
  return out;
}

py::dict RunDeterministicM1Benchmark(double cost_per_token_ms) {
  if (cost_per_token_ms <= 0.0) {
    throw std::runtime_error("cost_per_token_ms must be > 0");
  }
  continuum::backend::FakeLLMBackend backend;
  continuum::ir::Node node;
  node.kind = continuum::ir::NodeKind::TokenOp;
  node.payload = continuum::ir::TokenOpPayload{"generate", "fake/m1", 0.0f, 128};
  const std::string shared_prefix(3000, 'P');
  const std::vector<std::string> suffixes = {
      " step1: establish context",
      " step2: inspect cache state",
      " step3: plan tool usage",
      " step4: synthesize answer",
      " step5: finalize output"};

  py::list steps;
  int cache_hits = 0;
  double total_no_cache_ms = 0.0;
  double total_with_cache_ms = 0.0;
  std::optional<continuum::backend::BackendState> prefix_state = std::nullopt;
  for (std::size_t i = 0; i < suffixes.size(); ++i) {
    const std::string prompt = shared_prefix + suffixes[i];
    std::vector<continuum::Value> in{prompt};
    const std::int32_t full_prompt_tokens = static_cast<std::int32_t>(prompt.size());
    auto no_cache = backend.run_with_cache(node, in, std::nullopt, full_prompt_tokens);
    auto with_cache = backend.run_with_cache(
        node,
        in,
        prefix_state,
        prefix_state.has_value() ? static_cast<std::int32_t>(suffixes[i].size()) : full_prompt_tokens);
    prefix_state = no_cache.resulting_state;
    if (with_cache.used_cached_state) ++cache_hits;

    const double no_cache_ms = static_cast<double>(no_cache.compute_steps) * cost_per_token_ms;
    const double with_cache_ms = static_cast<double>(with_cache.compute_steps) * cost_per_token_ms;
    total_no_cache_ms += no_cache_ms;
    total_with_cache_ms += with_cache_ms;

    py::dict row;
    row["step"] = static_cast<int>(i + 1);
    row["latency_no_cache_ms"] = no_cache_ms;
    row["latency_with_cache_ms"] = with_cache_ms;
    row["compute_steps_no_cache"] = no_cache.compute_steps;
    row["compute_steps_with_cache"] = with_cache.compute_steps;
    row["tokens_saved"] = with_cache.tokens_saved;
    row["cache_hit"] = with_cache.used_cached_state;
    steps.append(row);
  }

  const double hit_rate = static_cast<double>(cache_hits) / static_cast<double>(suffixes.size());
  const double latency_reduction =
      total_no_cache_ms > 0.0 ? (total_no_cache_ms - total_with_cache_ms) / total_no_cache_ms : 0.0;
  py::dict out;
  out["backend"] = "fake_llm";
  out["cost_per_token_ms"] = cost_per_token_ms;
  out["steps"] = steps;
  out["cache_hit_rate"] = hit_rate;
  out["latency_no_cache_ms"] = total_no_cache_ms;
  out["latency_with_cache_ms"] = total_with_cache_ms;
  out["latency_reduction_ratio"] = latency_reduction;
  out["meets_cache_hit_target"] = hit_rate >= 0.8;
  out["meets_latency_target"] = latency_reduction >= 0.2;
  return out;
}

class PyGraphBuilder {
 public:
  py::tuple add(continuum::ir::NodeKind kind, py::object payload, py::object inputs, py::object out_type, py::object effect) {
    py::tuple t(5);
    t[0] = py::cast(static_cast<int>(kind));
    t[1] = payload;
    t[2] = inputs;
    t[3] = out_type;
    t[4] = effect;
    nodes_.append(t);
    return t;
  }
  PyGraphBuilder& finalize() { return *this; }
  py::dict run(py::args args, py::kwargs kwargs) const {
    py::dict d;
    d["args"] = args;
    d["kwargs"] = kwargs;
    d["nodes"] = py::len(nodes_);
    return d;
  }

 private:
  py::list nodes_;
};

continuum::Value PyToValue(const py::handle& obj) {
  if (py::isinstance<py::list>(obj)) {
    auto lst = py::reinterpret_borrow<py::list>(obj);
    bool all_int = true;
    for (auto item : lst) {
      if (!py::isinstance<py::int_>(item)) {
        all_int = false;
        break;
      }
    }
    if (all_int) {
      continuum::TokensValue tv;
      for (auto item : lst) {
        tv.ids.push_back(py::cast<int>(item));
      }
      return tv;
    }
    std::vector<float> buf;
    buf.reserve(py::len(lst));
    for (auto item : lst) {
      buf.push_back(py::cast<float>(item));
    }
    auto t = torch::from_blob(buf.data(), {(int64_t)buf.size()}, torch::kFloat32).clone();
    return continuum::TensorValue{t, "libtorch"};
  }
  if (py::isinstance<py::float_>(obj)) {
    return py::cast<double>(obj);
  }
  if (py::isinstance<py::int_>(obj)) {
    return py::cast<int64_t>(obj);
  }
  if (py::isinstance<py::str>(obj)) {
    return py::cast<std::string>(obj);
  }
  throw std::runtime_error("unsupported Python value for eager_step");
}

py::object ValueToPy(const continuum::Value& v) {
  if (const auto* t = std::get_if<continuum::TensorValue>(&v)) {
    py::list out;
    auto flat = t->tensor.flatten().contiguous();
    auto acc = flat.accessor<float, 1>();
    for (int64_t i = 0; i < flat.size(0); ++i) out.append(acc[i]);
    return out;
  }
  if (const auto* mx = std::get_if<continuum::MlxTensorValue>(&v)) {
    py::list out;
    for (float x : mx->data) out.append(x);
    return out;
  }
  if (const auto* t = std::get_if<continuum::TokensValue>(&v)) {
    py::list out;
    for (int x : t->ids) out.append(x);
    return out;
  }
  if (const auto* s = std::get_if<continuum::SchemaValue>(&v)) return py::str(s->json);
  if (const auto* s = std::get_if<std::string>(&v)) return py::str(*s);
  if (const auto* d = std::get_if<double>(&v)) return py::float_(*d);
  if (const auto* i = std::get_if<int64_t>(&v)) return py::int_(*i);
  return py::none();
}
}  // namespace

void bind_runtime(py::module_& m) {
  py::class_<continuum::runtime::Interpreter>(m, "Interpreter");

  py::class_<PyGraphBuilder>(m, "GraphBuilder")
      .def(py::init<>())
      .def("add", &PyGraphBuilder::add)
      .def("finalize", &PyGraphBuilder::finalize, py::return_value_policy::reference_internal)
      .def("run", &PyGraphBuilder::run);

  m.def(
      "eager_step",
      [](int kind, py::object payload, py::object inputs, py::object out_type, py::object effect) -> py::object {
        (void)payload;
        (void)out_type;
        (void)effect;
        continuum::ir::Node n;
        n.kind = static_cast<continuum::ir::NodeKind>(kind);
        if (n.kind == continuum::ir::NodeKind::TensorOp) {
          n.payload = continuum::ir::TensorOpPayload{"identity", {}};
        }
        std::vector<continuum::Value> in_values;
        if (py::isinstance<py::list>(inputs)) {
          for (auto item : py::reinterpret_borrow<py::list>(inputs)) {
            in_values.push_back(PyToValue(item));
          }
        }
        continuum::backend::BackendRegistry registry = MakeTensorRegistry(TensorBackendFromEnv());
        continuum::runtime::KVCacheIndex cache;
        continuum::runtime::Interpreter interp(registry, cache);
        auto out = interp.step(n, in_values);
        return ValueToPy(out);
      },
      py::arg("kind"),
      py::arg("payload"),
      py::arg("inputs"),
      py::arg("out_type") = py::none(),
      py::arg("effect") = py::none());

  m.def("run_tensor_op", [](const std::string& op, py::list a, py::object b, int64_t dim, const std::string& backend) {
    continuum::ir::Node n;
    n.kind = continuum::ir::NodeKind::TensorOp;
    continuum::ir::TensorOpPayload p;
    p.op_name = op;
    if (op == "softmax") {
      p.attrs.push_back(dim);
    }
    n.payload = p;

    auto to_tensor = [](const py::list& lst) {
      std::vector<float> buf;
      buf.reserve(py::len(lst));
      for (auto item : lst) {
        buf.push_back(py::cast<float>(item));
      }
      auto t = torch::from_blob(buf.data(), {(int64_t)buf.size()}, torch::kFloat32).clone();
      return continuum::TensorValue{t, "libtorch"};
    };

    std::vector<continuum::Value> inputs;
    inputs.emplace_back(to_tensor(a));
    if (!b.is_none()) {
      inputs.emplace_back(to_tensor(py::cast<py::list>(b)));
    }
    continuum::backend::BackendRegistry registry = MakeTensorRegistry(backend);
    continuum::runtime::KVCacheIndex cache;
    continuum::runtime::Interpreter interp(registry, cache);
    auto out = interp.step(n, inputs);
    return ValueToPy(out);
  }, py::arg("op"), py::arg("a"), py::arg("b") = py::none(), py::arg("dim") = -1, py::arg("backend") = "libtorch");

  m.def("train_classifier_demo", [](int64_t epochs, double lr) {
    torch::manual_seed(42);
    const int64_t n = 1024;
    auto x = torch::randn({n, 2}, torch::kFloat32);
    auto y = (x.index({torch::indexing::Slice(), 0}) + 0.75 * x.index({torch::indexing::Slice(), 1}) > 0.0)
                 .to(torch::kLong);

    auto w1 = torch::randn({16, 2}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    auto b1 = torch::zeros({16}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    auto w2 = torch::randn({2, 16}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    auto b2 = torch::zeros({2}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));

    py::list logs;
    auto eval_acc = [&]() {
      auto h = torch::relu(torch::matmul(x, w1.t()) + b1);
      auto logits = torch::matmul(h, w2.t()) + b2;
      auto pred = logits.argmax(1);
      return pred.eq(y).to(torch::kFloat32).mean().item<double>();
    };

    for (int64_t e = 1; e <= epochs; ++e) {
      auto h = torch::relu(torch::matmul(x, w1.t()) + b1);
      auto logits = torch::matmul(h, w2.t()) + b2;
      auto loss = torch::nn::functional::cross_entropy(logits, y);
      loss.backward();
      {
        torch::NoGradGuard ng;
        w1 -= lr * w1.grad();
        b1 -= lr * b1.grad();
        w2 -= lr * w2.grad();
        b2 -= lr * b2.grad();
      }
      w1.grad().zero_();
      b1.grad().zero_();
      w2.grad().zero_();
      b2.grad().zero_();

      py::dict row;
      row["epoch"] = e;
      row["loss"] = loss.item<double>();
      row["accuracy"] = eval_acc();
      logs.append(row);
    }
    return logs;
  });

  m.def("benchmark_azure_agent", []() { return RunPairedAgentBenchmark("azure", 10, true, 3000); });
  m.def("benchmark_vllm_agent", []() { return RunPairedAgentBenchmark("vllm", 10, true, 3000); });
  m.def(
      "benchmark_agent_paired",
      [](const std::string& backend, int trials, bool discard_first_warmup, int shared_prompt_tokens) {
        return RunPairedAgentBenchmark(backend, trials, discard_first_warmup, shared_prompt_tokens);
      },
      py::arg("backend"),
      py::arg("trials") = 10,
      py::arg("discard_first_warmup") = true,
      py::arg("shared_prompt_tokens") = 3000);
  m.def("benchmark_deterministic_m1", &RunDeterministicM1Benchmark, py::arg("cost_per_token_ms") = 2.0);
}
