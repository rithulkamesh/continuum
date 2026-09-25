#include <continuum/backend/conformance.hpp>

#include <torch/torch.h>

#include <exception>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace continuum::backend {
namespace {

class Recorder {
 public:
  explicit Recorder(ConformanceReport& report) : report_(report) {}

  void pass(std::string name, std::string detail = {}) {
    report_.checks.push_back({std::move(name), ConformanceStatus::Pass, std::move(detail)});
  }
  void fail(std::string name, std::string detail) {
    report_.checks.push_back({std::move(name), ConformanceStatus::Fail, std::move(detail)});
  }
  void skip(std::string name, std::string detail) {
    report_.checks.push_back({std::move(name), ConformanceStatus::Skip, std::move(detail)});
  }
  void expect(std::string name, bool ok, const std::string& detail) {
    ok ? pass(std::move(name), detail) : fail(std::move(name), detail);
  }

 private:
  ConformanceReport& report_;
};

std::string Metrics(const BackendRunResult& r) {
  std::ostringstream ss;
  ss << "used_cached_state=" << (r.used_cached_state ? 1 : 0) << " reused_prefix_len=" << r.reused_prefix_len
     << " tokens_saved=" << r.tokens_saved << " tokens_sent=" << r.tokens_sent
     << " compute_steps=" << r.compute_steps;
  return ss.str();
}

bool IsTokenOutput(const continuum::Value& v) {
  return std::holds_alternative<std::string>(v) || std::holds_alternative<continuum::TokensValue>(v);
}

bool SameTokenOutput(const continuum::Value& a, const continuum::Value& b) {
  if (const auto* sa = std::get_if<std::string>(&a)) {
    const auto* sb = std::get_if<std::string>(&b);
    return sb != nullptr && *sa == *sb;
  }
  if (const auto* ta = std::get_if<continuum::TokensValue>(&a)) {
    const auto* tb = std::get_if<continuum::TokensValue>(&b);
    return tb != nullptr && ta->ids == tb->ids;
  }
  return false;
}

template <typename Fn>
std::optional<BackendRunResult> Attempt(Recorder& rec, const std::string& name, Fn&& fn) {
  try {
    return fn();
  } catch (const std::exception& e) {
    rec.fail(name, std::string("threw: ") + e.what());
  } catch (...) {
    rec.fail(name, "threw a non-standard exception");
  }
  return std::nullopt;
}

void CheckWarm(Recorder& rec, const std::string& prefix, const BackendRunResult& r, std::int32_t total) {
  rec.expect(prefix + ".used_cached_state", r.used_cached_state,
             "a run handed prefix state must report used_cached_state; " + Metrics(r));
  rec.expect(prefix + ".bounds",
             r.reused_prefix_len >= 0 && r.reused_prefix_len <= total && r.tokens_saved >= 0 &&
                 r.tokens_saved <= total && r.tokens_sent >= 0 && r.compute_steps >= 0,
             "need 0 <= reused_prefix_len, tokens_saved <= prompt length (" + std::to_string(total) +
                 ") and non-negative tokens_sent / compute_steps; " + Metrics(r));
  rec.expect(prefix + ".saves_tokens", r.tokens_saved > 0,
             "a cache-capable backend must save work on a matching prefix; " + Metrics(r));
}

}  // namespace

bool ConformanceReport::passed() const {
  for (const auto& c : checks) {
    if (c.status == ConformanceStatus::Fail) return false;
  }
  return true;
}

std::string ConformanceReport::summary() const {
  std::ostringstream ss;
  int pass = 0, fail = 0, skip = 0;
  for (const auto& c : checks) {
    const char* tag = "SKIP";
    if (c.status == ConformanceStatus::Pass) { tag = "PASS"; ++pass; }
    else if (c.status == ConformanceStatus::Fail) { tag = "FAIL"; ++fail; }
    else { ++skip; }
    ss << tag << "  " << c.name;
    if (!c.detail.empty()) ss << "  (" << c.detail << ")";
    ss << "\n";
  }
  ss << "backend " << backend << ": " << (passed() ? "CONFORMANT" : "NOT CONFORMANT") << " (" << pass
     << " passed, " << fail << " failed, " << skip << " skipped)\n";
  return ss.str();
}

ConformanceReport RunBackendConformance(Backend& backend, const std::string& name,
                                        const ConformanceOptions& options) {
  ConformanceReport report;
  report.backend = name;
  Recorder rec(report);

  // --- capability declaration ---------------------------------------------
  const auto caps = backend.capabilities();
  rec.expect("capabilities.declared", caps.supports_tensor || caps.supports_token,
             "a backend must support at least one of tensor / token execution");
  rec.expect("capabilities.cache_implies_token", !caps.supports_cache || caps.supports_token,
             "supports_cache applies to the token path and needs supports_token");

  // --- token path -----------------------------------------------------------
  const auto total = static_cast<std::int32_t>(options.prompt.size());
  const std::int32_t prefix =
      options.prefix_len >= 0 ? std::min(options.prefix_len, total) : total / 2;
  ir::Node token_node;
  token_node.kind = ir::NodeKind::TokenOp;
  token_node.debug_name = "conformance_generate";
  token_node.payload = ir::TokenOpPayload{"generate", options.model_id, 0.0f, options.max_tokens};
  const std::vector<continuum::Value> token_inputs{continuum::Value{options.prompt}};

  std::optional<BackendRunResult> cold;
  if (!caps.supports_token) {
    rec.skip("token.*", "backend does not declare supports_token");
  } else {
    cold = Attempt(rec, "token.cold_run", [&] {
      return backend.run_with_cache(token_node, token_inputs, std::nullopt, total);
    });
    if (cold.has_value()) {
      rec.expect("token.cold_run", IsTokenOutput(cold->output), "output must be a string or token ids");
      rec.expect("token.cold_metrics",
                 !cold->used_cached_state && cold->reused_prefix_len == 0 && cold->tokens_saved == 0 &&
                     cold->tokens_sent >= 0 && cold->compute_steps >= 0,
                 "a run without prefix state must report no reuse; " + Metrics(*cold));
      if (options.expect_deterministic) {
        auto again = Attempt(rec, "token.deterministic", [&] {
          return backend.run_with_cache(token_node, token_inputs, std::nullopt, total);
        });
        if (again.has_value()) {
          rec.expect("token.deterministic", SameTokenOutput(cold->output, again->output),
                     "identical cold inputs at temperature 0 must give identical output");
        }
      } else {
        rec.skip("token.deterministic", "disabled by options");
      }
    }
  }

  // --- cache path -----------------------------------------------------------
  if (!caps.supports_cache) {
    rec.skip("cache.*", "backend does not declare supports_cache");
    rec.skip("state.*", "backend does not declare supports_cache");
  } else if (cold.has_value()) {
    const BackendState state = cold->resulting_state;
    rec.expect("cache.state_handle", state.handle != nullptr,
               "a cache-capable backend must return a non-null resulting_state");
    if (state.handle != nullptr) {
      auto warm = Attempt(rec, "cache.warm_run", [&] {
        return backend.run_with_cache(token_node, token_inputs, state, total - prefix);
      });
      if (warm.has_value()) {
        rec.pass("cache.warm_run");
        CheckWarm(rec, "cache.warm", *warm, total);
        if (options.expect_deterministic) {
          rec.expect("cache.warm_output_matches_cold", SameTokenOutput(cold->output, warm->output),
                     "reusing a prefix must not change the output");
        }
      }

      // --- state portability (optional) ------------------------------------
      std::vector<std::uint8_t> bytes;
      try {
        bytes = backend.export_state(state);
      } catch (const std::exception& e) {
        rec.fail("state.export", std::string("threw: ") + e.what());
      }
      if (bytes.empty()) {
        rec.skip("state.*", "export_state returned no bytes: state is not portable (optional)");
      } else {
        rec.pass("state.export", std::to_string(bytes.size()) + " bytes");
        std::optional<BackendState> imported;
        try {
          imported = backend.import_state(bytes);
        } catch (const std::exception& e) {
          rec.fail("state.import", std::string("threw: ") + e.what());
        }
        rec.expect("state.import", imported.has_value() && imported->handle != nullptr,
                   "import_state must accept bytes produced by export_state");
        if (imported.has_value() && imported->handle != nullptr) {
          rec.expect("state.roundtrip", backend.export_state(*imported) == bytes,
                     "export(import(bytes)) must reproduce bytes");
          auto resumed = Attempt(rec, "state.imported_warm_run", [&] {
            return backend.run_with_cache(token_node, token_inputs, *imported, total - prefix);
          });
          if (resumed.has_value()) {
            rec.pass("state.imported_warm_run");
            CheckWarm(rec, "state.imported_warm", *resumed, total);
          }
        }
      }
    }
  }

  // --- tensor path ----------------------------------------------------------
  if (!caps.supports_tensor) {
    rec.skip("tensor.*", "backend does not declare supports_tensor");
  } else {
    ir::Node tensor_node;
    tensor_node.kind = ir::NodeKind::TensorOp;
    tensor_node.debug_name = "conformance_tensor";
    tensor_node.payload = ir::TensorOpPayload{options.tensor_op, {}};
    const auto input = torch::tensor({1.0f, 2.0f, 3.0f});
    const std::vector<continuum::Value> tensor_inputs{continuum::TensorValue{input, "libtorch"}};
    auto out = Attempt(rec, "tensor." + options.tensor_op, [&] {
      return backend.run_with_cache(tensor_node, tensor_inputs, std::nullopt, 0);
    });
    if (out.has_value()) {
      const auto* t = std::get_if<continuum::TensorValue>(&out->output);
      rec.expect("tensor." + options.tensor_op, t != nullptr && torch::equal(t->tensor, input),
                 "must return its input tensor unchanged");
      rec.expect("tensor.no_reuse_claimed", !out->used_cached_state && out->reused_prefix_len == 0,
                 "a tensor op run without state must not claim reuse; " + Metrics(*out));
    }
  }
  return report;
}

}  // namespace continuum::backend
