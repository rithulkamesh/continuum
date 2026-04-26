#pragma once

#include <spdlog/logger.h>

namespace continuum::log {

enum class LogLevel { kTrace, kDebug, kInfo, kWarn, kError, kCritical, kOff };

void init(LogLevel level = LogLevel::kInfo);
spdlog::logger& core();
spdlog::logger& runtime();
spdlog::logger& backend();
spdlog::logger& compiler();

}  // namespace continuum::log

#define LOG_TRACE(subsystem, ...) continuum::log::subsystem().trace(__VA_ARGS__)
#define LOG_DEBUG(subsystem, ...) continuum::log::subsystem().debug(__VA_ARGS__)
#define LOG_INFO(subsystem, ...) continuum::log::subsystem().info(__VA_ARGS__)
#define LOG_WARN(subsystem, ...) continuum::log::subsystem().warn(__VA_ARGS__)
#define LOG_ERROR(subsystem, ...) continuum::log::subsystem().error(__VA_ARGS__)
#define LOG_CRITICAL(subsystem, ...) continuum::log::subsystem().critical(__VA_ARGS__)
