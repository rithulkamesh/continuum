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
  std::shared_ptr<Backend> get(const std::string& name) const;
  bool has(const std::string& name) const;
  std::shared_ptr<Backend> get_backend_for(ir::NodeKind kind) const;
  BackendSelection select_backend(const ir::Node& node) const;

 private:
  std::unordered_map<std::string, BackendSelection> backends_;
};

std::shared_ptr<Backend> MakeBackendFromAbi(continuum_backend_vtable_t vtable);

}  // namespace continuum::backend
