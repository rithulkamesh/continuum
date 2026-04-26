#include <continuum/utils/logging.hpp>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace continuum::log {
namespace {

std::once_flag g_init_once;
std::shared_ptr<spdlog::logger> g_core;
std::shared_ptr<spdlog::logger> g_runtime;
std::shared_ptr<spdlog::logger> g_backend;
std::shared_ptr<spdlog::logger> g_compiler;

spdlog::level::level_enum ToSpdLevel(LogLevel level) {
  switch (level) {
    case LogLevel::kTrace:
      return spdlog::level::trace;
    case LogLevel::kDebug:
      return spdlog::level::debug;
    case LogLevel::kInfo:
      return spdlog::level::info;
    case LogLevel::kWarn:
      return spdlog::level::warn;
    case LogLevel::kError:
      return spdlog::level::err;
    case LogLevel::kCritical:
      return spdlog::level::critical;
    case LogLevel::kOff:
      return spdlog::level::off;
  }
  return spdlog::level::info;
}

void InitOnce(LogLevel level) {
  auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] [%n] %v");

  g_core = std::make_shared<spdlog::logger>("core", sink);
  g_runtime = std::make_shared<spdlog::logger>("runtime", sink);
  g_backend = std::make_shared<spdlog::logger>("backend", sink);
  g_compiler = std::make_shared<spdlog::logger>("compiler", sink);
  const auto spd_level = ToSpdLevel(level);
  for (const auto& logger : std::vector<std::shared_ptr<spdlog::logger>>{g_core, g_runtime, g_backend, g_compiler}) {
    logger->set_level(spd_level);
    logger->flush_on(spdlog::level::warn);
    spdlog::register_logger(logger);
  }
}

void EnsureInitialized() { std::call_once(g_init_once, []() { InitOnce(LogLevel::kInfo); }); }

spdlog::logger& Require(const std::shared_ptr<spdlog::logger>& logger, const char* name) {
  if (!logger) {
    throw std::runtime_error(std::string("logger not initialized: ") + name);
  }
  return *logger;
}

}  // namespace

void init(LogLevel level) { std::call_once(g_init_once, [level]() { InitOnce(level); }); }

spdlog::logger& core() {
  EnsureInitialized();
  return Require(g_core, "core");
}

spdlog::logger& runtime() {
  EnsureInitialized();
  return Require(g_runtime, "runtime");
}

spdlog::logger& backend() {
  EnsureInitialized();
  return Require(g_backend, "backend");
}

spdlog::logger& compiler() {
  EnsureInitialized();
  return Require(g_compiler, "compiler");
}

}  // namespace continuum::log
