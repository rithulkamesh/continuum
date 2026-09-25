#include <continuum/backend/backend.hpp>
#include <continuum/backend/backend_abi.h>
#include <continuum/ir/node.hpp>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace continuum::backend {
namespace {

BackendCapabilities FromAbiCaps(const continuum_backend_caps_t& caps) {
  return BackendCapabilities{
      static_cast<bool>(caps.supports_tensor),
      static_cast<bool>(caps.supports_token),
      static_cast<bool>(caps.supports_cache)};
}

continuum_backend_value_t ToAbiValue(const continuum::Value& value) {
  if (const auto* s = std::get_if<std::string>(&value)) {
    return continuum_backend_value_t{CONTINUUM_BACKEND_VALUE_STRING, s->c_str(), nullptr, 0};
  }
  if (const auto* t = std::get_if<continuum::TokensValue>(&value)) {
    return continuum_backend_value_t{
        CONTINUUM_BACKEND_VALUE_TOKENS, nullptr, reinterpret_cast<const int32_t*>(t->ids.data()), t->ids.size()};
  }
  return continuum_backend_value_t{CONTINUUM_BACKEND_VALUE_NONE, nullptr, nullptr, 0};
}

continuum::Value FromAbiValue(const continuum_backend_value_t& value) {
  if (value.kind == CONTINUUM_BACKEND_VALUE_STRING) {
    return value.string_data == nullptr ? std::string{} : std::string(value.string_data);
  }
  if (value.kind == CONTINUUM_BACKEND_VALUE_TOKENS) {
    continuum::TokensValue out;
    out.ids.reserve(value.token_count);
    for (size_t i = 0; i < value.token_count; ++i) {
      out.ids.push_back(static_cast<int>(value.token_ids[i]));
    }
    return out;
  }
  return std::string{};
}

continuum_backend_node_meta_t ToAbiNodeMeta(const ir::Node& node) {
  continuum_backend_node_meta_t out{};
  out.node_kind = static_cast<uint8_t>(node.kind);
  if (const auto* payload = std::get_if<ir::TensorOpPayload>(&node.payload)) {
    out.op_name = payload->op_name.c_str();
    out.attrs = payload->attrs.data();
    out.attr_count = payload->attrs.size();
  } else if (const auto* payload = std::get_if<ir::TokenOpPayload>(&node.payload)) {
    out.op_name = payload->op_name.c_str();
    out.model_id = payload->model_id.c_str();
  }
  return out;
}

}  // namespace

class BackendAbiAdapter final : public Backend {
 public:
  /// \p library keeps a dynamically loaded plugin mapped for as long as the
  /// adapter lives; it is released only after `destroy` has run.
  explicit BackendAbiAdapter(continuum_backend_vtable_t vtable, std::shared_ptr<void> library = nullptr)
      : library_(std::move(library)), vtable_(vtable) {
    if (vtable_.abi_version < CONTINUUM_BACKEND_ABI_MIN_VERSION ||
        vtable_.abi_version > CONTINUUM_BACKEND_ABI_VERSION) {
      throw std::runtime_error("backend ABI version mismatch: got " + std::to_string(vtable_.abi_version) +
                               ", host supports " + std::to_string(CONTINUUM_BACKEND_ABI_MIN_VERSION) + "-" +
                               std::to_string(CONTINUUM_BACKEND_ABI_VERSION));
    }
    if (vtable_.abi_version < 2) {
      // v1 vtables end at run_with_cache; never read fields they do not have.
      vtable_.destroy = nullptr;
      vtable_.export_state = nullptr;
      vtable_.import_state = nullptr;
    }
  }

  ~BackendAbiAdapter() override {
    if (vtable_.destroy != nullptr) {
      vtable_.destroy(vtable_.instance);
    }
  }

  BackendAbiAdapter(const BackendAbiAdapter&) = delete;
  BackendAbiAdapter& operator=(const BackendAbiAdapter&) = delete;

  std::vector<std::uint8_t> export_state(const BackendState& state) const override {
    if (vtable_.export_state == nullptr) {
      return {};
    }
    const continuum_backend_state_t abi_state{state.handle};
    const std::size_t needed = vtable_.export_state(vtable_.instance, abi_state, nullptr, 0);
    if (needed == 0) {
      return {};
    }
    std::vector<std::uint8_t> out(needed);
    const std::size_t written = vtable_.export_state(vtable_.instance, abi_state, out.data(), out.size());
    if (written != needed) {
      return {};
    }
    return out;
  }

  std::optional<BackendState> import_state(const std::vector<std::uint8_t>& bytes) override {
    if (vtable_.import_state == nullptr || bytes.empty()) {
      return std::nullopt;
    }
    continuum_backend_state_t out{};
    if (vtable_.import_state(vtable_.instance, bytes.data(), bytes.size(), &out) == 0) {
      return std::nullopt;
    }
    return BackendState{out.handle};
  }

  BackendCapabilities capabilities() const override {
    if (vtable_.capabilities == nullptr) {
      return BackendCapabilities{};
    }
    return FromAbiCaps(vtable_.capabilities(vtable_.instance));
  }

  std::string tensor_backend_type() const override {
    if (vtable_.tensor_backend_type == nullptr) {
      return "";
    }
    const char* name = vtable_.tensor_backend_type(vtable_.instance);
    return name == nullptr ? std::string{} : std::string(name);
  }

  BackendRunResult run_with_cache(
      const ir::Node& node,
      const std::vector<continuum::Value>& inputs,
      const std::optional<BackendState>& prefix_state,
      std::int32_t remaining_tokens) override {
    if (vtable_.run_with_cache == nullptr) {
      throw std::runtime_error("backend ABI vtable missing run_with_cache");
    }
    std::vector<continuum_backend_value_t> abi_inputs;
    abi_inputs.reserve(inputs.size());
    for (const auto& v : inputs) {
      abi_inputs.push_back(ToAbiValue(v));
    }
    const auto abi_node = ToAbiNodeMeta(node);
    const continuum_backend_state_t* abi_prefix = nullptr;
    continuum_backend_state_t abi_prefix_value{};
    if (prefix_state.has_value()) {
      abi_prefix_value.handle = prefix_state->handle;
      abi_prefix = &abi_prefix_value;
    }
    const auto abi_out = vtable_.run_with_cache(
        vtable_.instance,
        abi_node,
        abi_inputs.empty() ? nullptr : abi_inputs.data(),
        abi_inputs.size(),
        abi_prefix,
        remaining_tokens);
    BackendRunResult out;
    out.output = FromAbiValue(abi_out.output);
    out.resulting_state = BackendState{abi_out.resulting_state.handle};
    out.reused_prefix_len = abi_out.reused_prefix_len;
    out.compute_steps = abi_out.compute_steps;
    out.tokens_sent = abi_out.tokens_sent;
    out.tokens_saved = abi_out.tokens_saved;
    out.used_cached_state = static_cast<bool>(abi_out.used_cached_state);
    return out;
  }

 private:
  std::shared_ptr<void> library_;  // declared first: unmapped after destroy() runs
  continuum_backend_vtable_t vtable_{};
};

namespace {

std::string LastLoaderError() {
#if defined(_WIN32)
  return "LoadLibrary error " + std::to_string(GetLastError());
#else
  const char* err = dlerror();
  return err == nullptr ? std::string("unknown error") : std::string(err);
#endif
}

std::shared_ptr<void> OpenLibrary(const std::string& path) {
#if defined(_WIN32)
  HMODULE handle = LoadLibraryA(path.c_str());
  if (handle == nullptr) {
    throw std::runtime_error("cannot load backend plugin " + path + ": " + LastLoaderError());
  }
  return std::shared_ptr<void>(handle, [](void* h) { FreeLibrary(static_cast<HMODULE>(h)); });
#else
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    throw std::runtime_error("cannot load backend plugin " + path + ": " + LastLoaderError());
  }
  return std::shared_ptr<void>(handle, [](void* h) { dlclose(h); });
#endif
}

void* FindSymbol(const std::shared_ptr<void>& library, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library.get()), name));
#else
  dlerror();
  return dlsym(library.get(), name);
#endif
}

std::string Trim(const std::string& s) {
  const auto begin = s.find_first_not_of(" \t\n\r");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = s.find_last_not_of(" \t\n\r");
  return s.substr(begin, end - begin + 1);
}

}  // namespace

std::shared_ptr<Backend> MakeBackendFromAbi(continuum_backend_vtable_t vtable) {
  return std::make_shared<BackendAbiAdapter>(vtable);
}

std::shared_ptr<Backend> LoadBackendPlugin(const std::string& path) {
  auto library = OpenLibrary(path);
  void* sym = FindSymbol(library, CONTINUUM_BACKEND_PLUGIN_INIT_SYMBOL);
  if (sym == nullptr) {
    throw std::runtime_error("backend plugin " + path + " does not export " +
                             std::string(CONTINUUM_BACKEND_PLUGIN_INIT_SYMBOL));
  }
  auto init = reinterpret_cast<continuum_backend_plugin_init_fn>(sym);
  continuum_backend_vtable_t vtable{};
  const int rc = init(CONTINUUM_BACKEND_ABI_VERSION, &vtable);
  if (rc != 0) {
    throw std::runtime_error("backend plugin " + path + " init failed with code " + std::to_string(rc));
  }
  if (vtable.run_with_cache == nullptr) {
    if (vtable.destroy != nullptr) {
      vtable.destroy(vtable.instance);
    }
    throw std::runtime_error("backend plugin " + path + " vtable missing run_with_cache");
  }
  try {
    return std::make_shared<BackendAbiAdapter>(vtable, std::move(library));
  } catch (...) {
    // Version rejected: the adapter never took ownership of the instance.
    if (vtable.abi_version >= 2 && vtable.destroy != nullptr) {
      vtable.destroy(vtable.instance);
    }
    throw;
  }
}

std::vector<PluginSpec> ParsePluginSpecs(const std::string& spec) {
  std::vector<PluginSpec> out;
  std::size_t start = 0;
  while (start <= spec.size()) {
    auto end = spec.find(';', start);
    if (end == std::string::npos) end = spec.size();
    const std::string entry = Trim(spec.substr(start, end - start));
    start = end + 1;
    if (entry.empty()) {
      continue;
    }
    const auto eq = entry.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == entry.size()) {
      throw std::runtime_error("bad plugin spec '" + entry + "': expected name[@priority]=path");
    }
    PluginSpec ps;
    std::string head = Trim(entry.substr(0, eq));
    ps.path = Trim(entry.substr(eq + 1));
    const auto at = head.find('@');
    if (at != std::string::npos) {
      const std::string prio = head.substr(at + 1);
      char* parse_end = nullptr;
      const long value = std::strtol(prio.c_str(), &parse_end, 10);
      if (prio.empty() || parse_end == nullptr || *parse_end != '\0') {
        throw std::runtime_error("bad plugin priority '" + prio + "' in '" + entry + "'");
      }
      ps.priority = static_cast<int>(value);
      head = head.substr(0, at);
    }
    ps.name = head;
    out.push_back(std::move(ps));
  }
  return out;
}

}  // namespace continuum::backend
