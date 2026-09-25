#pragma once

#include <continuum/ir/node.hpp>
#include <continuum/ir/value.hpp>
#include <continuum/backend/backend_abi.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace continuum::backend {

struct BackendState {
  void* handle = nullptr;  // Opaque backend-owned state (e.g. KV cache handle).
};

struct BackendRunResult {
  continuum::Value output;
  BackendState resulting_state{};
  std::int32_t reused_prefix_len = 0;
  std::int32_t compute_steps = 0;
  std::int32_t tokens_sent = 0;
  std::int32_t tokens_saved = 0;
  bool used_cached_state = false;
};

struct BackendCapabilities {
  bool supports_tensor = false;
  bool supports_token = false;
  bool supports_cache = false;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual BackendCapabilities capabilities() const = 0;
  virtual std::string tensor_backend_type() const { return ""; }
  // Optional cross-process state portability. Backends that can externalize a
  // state handle return its bytes from export_state and rebuild a live handle
  // in import_state. Default: not portable (empty / nullopt); checkpoints then
  // drop the state and resume falls back to a cold cache.
  virtual std::vector<std::uint8_t> export_state(const BackendState& /*state*/) const { return {}; }
  virtual std::optional<BackendState> import_state(const std::vector<std::uint8_t>& /*bytes*/) { return std::nullopt; }
  virtual BackendRunResult run_with_cache(
      const ir::Node& node,
      const std::vector<continuum::Value>& inputs,
      const std::optional<BackendState>& prefix_state,
      std::int32_t remaining_tokens) = 0;
};

class BackendRegistry {
 public:
  struct BackendSelection {
    std::string name;
    std::shared_ptr<Backend> backend;
    BackendCapabilities capabilities;
    int priority = 0;
  };

  void register_backend(const std::string& name, std::shared_ptr<Backend> backend, int priority = 0);
  /// Load a shared-library backend (see LoadBackendPlugin) and register it.
  void load_plugin(const std::string& name, const std::string& path, int priority = 0);
  /// Load every plugin listed in environment variable \p env_var
  /// (format: see ParsePluginSpecs). Returns the number loaded; an unset or
  /// empty variable loads nothing.
  std::size_t load_plugins_from_env(const char* env_var = "CONTINUUM_BACKEND_PLUGINS");
  /// Registered backend names, in no particular order.
  std::vector<std::string> names() const;
  std::shared_ptr<Backend> get(const std::string& name) const;
  bool has(const std::string& name) const;
  /// Returns a backend that supports \p kind, or nullptr if none is registered.
  std::shared_ptr<Backend> get_backend_for(ir::NodeKind kind) const;
  BackendSelection select_backend(const ir::Node& node) const;

 private:
  std::unordered_map<std::string, BackendSelection> backends_;
};

/// Wrap a C ABI vtable (v1 or v2) as a Backend. Throws on an unsupported
/// `abi_version`.
std::shared_ptr<Backend> MakeBackendFromAbi(continuum_backend_vtable_t vtable);

/// `dlopen` / `LoadLibrary` \p path, resolve CONTINUUM_BACKEND_PLUGIN_INIT_SYMBOL,
/// initialize it, and check the vtable's ABI version. The library stays
/// loaded until the returned backend is destroyed. Throws std::runtime_error
/// with the reason on any failure.
std::shared_ptr<Backend> LoadBackendPlugin(const std::string& path);

/// One entry of a plugin list.
struct PluginSpec {
  std::string name;
  std::string path;
  int priority = 0;
};

/// Parse `name[@priority]=path` entries separated by `;`, e.g.
/// `echo@50=/opt/plugins/libecho.so;other=/x/libother.so`. Throws on a
/// malformed entry.
std::vector<PluginSpec> ParsePluginSpecs(const std::string& spec);

}  // namespace continuum::backend
