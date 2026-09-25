#include <continuum/backend/backend.hpp>

#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace continuum::backend {

void BackendRegistry::register_backend(const std::string& name, std::shared_ptr<Backend> backend, int priority) {
  BackendSelection entry;
  entry.name = name;
  entry.priority = priority;
  entry.capabilities = backend->capabilities();
  entry.backend = std::move(backend);
  backends_[name] = std::move(entry);
}

void BackendRegistry::load_plugin(const std::string& name, const std::string& path, int priority) {
  register_backend(name, LoadBackendPlugin(path), priority);
}

std::size_t BackendRegistry::load_plugins_from_env(const char* env_var) {
  const char* raw = std::getenv(env_var);
  if (raw == nullptr) {
    return 0;
  }
  const auto specs = ParsePluginSpecs(raw);
  for (const auto& spec : specs) {
    load_plugin(spec.name, spec.path, spec.priority);
  }
  return specs.size();
}

std::vector<std::string> BackendRegistry::names() const {
  std::vector<std::string> out;
  out.reserve(backends_.size());
  for (const auto& [name, entry] : backends_) {
    out.push_back(name);
  }
  return out;
}

std::shared_ptr<Backend> BackendRegistry::get(const std::string& name) const {
  auto it = backends_.find(name);
  if (it == backends_.end()) {
    throw std::runtime_error("backend not found: " + name);
  }
  return it->second.backend;
}

bool BackendRegistry::has(const std::string& name) const {
  return backends_.find(name) != backends_.end();
}

std::shared_ptr<Backend> BackendRegistry::get_backend_for(ir::NodeKind kind) const {
  ir::Node probe;
  probe.kind = kind;
  BackendSelection best{};
  int best_priority = std::numeric_limits<int>::min();
  for (const auto& [name, entry] : backends_) {
    bool supports = false;
    switch (kind) {
      case ir::NodeKind::TensorOp:
        supports = entry.capabilities.supports_tensor;
        break;
      case ir::NodeKind::TokenOp:
        supports = entry.capabilities.supports_token;
        break;
      default:
        supports = false;
        break;
    }
    if (!supports) {
      continue;
    }
    if (entry.priority > best_priority) {
      best_priority = entry.priority;
      best = entry;
    }
  }
  return best.backend;
}

BackendRegistry::BackendSelection BackendRegistry::select_backend(const ir::Node& node) const {
  BackendSelection best{};
  int best_priority = std::numeric_limits<int>::min();
  for (const auto& [name, entry] : backends_) {
    bool supports = false;
    switch (node.kind) {
      case ir::NodeKind::TensorOp:
        supports = entry.capabilities.supports_tensor;
        break;
      case ir::NodeKind::TokenOp:
        supports = entry.capabilities.supports_token;
        break;
      default:
        supports = false;
        break;
    }
    if (!supports) {
      continue;
    }
    if (entry.priority > best_priority) {
      best_priority = entry.priority;
      best = entry;
    }
  }
  if (best.backend == nullptr) {
    throw std::runtime_error("no backend supports requested node kind");
  }
  return best;
}

}  // namespace continuum::backend
