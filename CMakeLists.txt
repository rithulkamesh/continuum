cmake_minimum_required(VERSION 3.24)
project(continuum CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CONTINUUM_BUILD_TESTS "Build C++ tests" OFF)
option(CONTINUUM_BUILD_PYTHON "Build Python bindings" ON)
option(CONTINUUM_USE_LIBTORCH "Enable libtorch backend" ON)
option(CONTINUUM_USE_ABSL "Use absl containers/maps" OFF)
option(CONTINUUM_USE_SPDLOG "Use spdlog logging" ON)
option(CONTINUUM_USE_CURL "Use libcurl for remote backends" ON)

if(CONTINUUM_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
endif()
if(CONTINUUM_USE_LIBTORCH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(NOT Torch_DIR)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE TORCH_PREFIX_RESULT
        )
        if(TORCH_PREFIX_RESULT EQUAL 0 AND TORCH_CMAKE_PREFIX)
            list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
        endif()
    endif()
    find_package(Torch REQUIRED)
endif()
if(CONTINUUM_USE_ABSL)
    find_package(absl CONFIG REQUIRED)
endif()
if(CONTINUUM_USE_SPDLOG)
    find_package(spdlog CONFIG QUIET)
    if(NOT spdlog_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.14.1
        )
        FetchContent_MakeAvailable(spdlog)
    endif()
endif()
if(CONTINUUM_USE_CURL)
    find_package(CURL REQUIRED)
endif()

add_library(continuum_core
    src/backend/backend.cpp
    src/backend/backend_abi_adapter.cpp
    src/ir/node.cpp
    src/ir/graph.cpp
    src/ir/type.cpp
    src/ir/effect.cpp
    src/runtime/interpreter.cpp
    src/runtime/scheduler.cpp
    src/runtime/cache.cpp
    src/runtime/checkpoint.cpp
    src/utils/logging.cpp
    src/compiler/typecheck.cpp
    src/compiler/rewrite.cpp
    src/backend/libtorch.cpp
    src/backend/mlx_backend.cpp
    src/backend/openai.cpp
    src/backend/fake_llm.cpp
    src/backend/azure_openai.cpp
    src/backend/anthropic.cpp
    src/backend/vllm_shim.cpp
)
target_include_directories(continuum_core PUBLIC include)
if(CONTINUUM_USE_LIBTORCH)
    target_link_libraries(continuum_core PUBLIC ${TORCH_LIBRARIES})
endif()
if(CONTINUUM_USE_ABSL)
    target_link_libraries(continuum_core PUBLIC absl::flat_hash_map)
endif()
if(CONTINUUM_USE_SPDLOG)
    target_link_libraries(continuum_core PUBLIC spdlog::spdlog)
endif()
if(CONTINUUM_USE_CURL)
    target_link_libraries(continuum_core PUBLIC CURL::libcurl)
endif()
target_compile_options(continuum_core PRIVATE -Wall -Wextra -Wpedantic)

if(CONTINUUM_BUILD_PYTHON)
    pybind11_add_module(_continuum
        bindings/pybind/_continuum.cpp
        bindings/pybind/bind_ir.cpp
        bindings/pybind/bind_runtime.cpp
        bindings/pybind/bind_backend.cpp
    )
    target_link_libraries(_continuum PRIVATE continuum_core)
    install(TARGETS _continuum DESTINATION continuum)
endif()

if(CONTINUUM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
