cmake_minimum_required(VERSION 3.24)
project(continuum CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CONTINUUM_BUILD_TESTS "Build C++ tests" OFF)
option(CONTINUUM_BUILD_PYTHON "Build Python bindings" ON)
option(CONTINUUM_USE_LIBTORCH "Enable libtorch backend" ON)
option(CONTINUUM_USE_ABSL "Use absl containers/maps" OFF)
option(CONTINUUM_USE_SPDLOG "Use spdlog logging" ON)
option(CONTINUUM_USE_CURL "Use libcurl for remote backends" ON)
option(CONTINUUM_WERROR "Treat warnings as errors" OFF)
option(CONTINUUM_COVERAGE "Build with coverage instrumentation" OFF)
# Apple MLX for the `mlx` tensor backend. AUTO: on when the `mlx` Python
# package (which ships libmlx + its CMake config) is importable on Apple
# silicon; ON: require it anywhere (Linux has CPU wheels: pip install "mlx[cpu]").
set(CONTINUUM_USE_MLX "AUTO" CACHE STRING "Use Apple MLX for the mlx backend (AUTO/ON/OFF)")

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
set(CONTINUUM_HAVE_MLX OFF)
if(CONTINUUM_USE_MLX STREQUAL "ON" OR (CONTINUUM_USE_MLX STREQUAL "AUTO" AND APPLE
        AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64"))
    if(NOT MLX_DIR)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import mlx, os; print(os.path.join(list(mlx.__path__)[0], 'share', 'cmake', 'MLX'))"
                OUTPUT_VARIABLE MLX_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE MLX_DIR_RESULT
                ERROR_QUIET
            )
            if(MLX_DIR_RESULT EQUAL 0 AND EXISTS "${MLX_CMAKE_DIR}")
                set(MLX_DIR "${MLX_CMAKE_DIR}")
            endif()
        endif()
    endif()
    if(CONTINUUM_USE_MLX STREQUAL "ON")
        find_package(MLX CONFIG REQUIRED)
    else()
        find_package(MLX CONFIG QUIET)
    endif()
    if(MLX_FOUND)
        set(CONTINUUM_HAVE_MLX ON)
        message(STATUS "Continuum: mlx backend uses Apple MLX from ${MLX_DIR}")
    else()
        message(STATUS "Continuum: MLX not found; mlx backend uses reference kernels")
    endif()
endif()
if(CONTINUUM_USE_SPDLOG)
    find_package(spdlog CONFIG QUIET)
    if(NOT spdlog_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            # 1.15.x bundles fmt 11, which compiles cleanly under C++20.
            # 1.14.1's bundled fmt trips "consteval ... not a constant expression".
            GIT_TAG v1.15.3
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
    src/backend/conformance.cpp
    src/ir/node.cpp
    src/ir/graph.cpp
    src/ir/type.cpp
    src/ir/effect.cpp
    src/runtime/interpreter.cpp
    src/runtime/scheduler.cpp
    src/runtime/kv_prefix_cache.cpp
    src/runtime/checkpoint.cpp
    src/runtime/session.cpp
    src/runtime/memo_table.cpp
    src/runtime/semantic_cache.cpp
    src/runtime/memory_graph.cpp
    src/runtime/layer_cache.cpp
    src/runtime/prefetch.cpp
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
# dlopen for runtime backend plugins (LoadBackendPlugin).
target_link_libraries(continuum_core PUBLIC ${CMAKE_DL_LIBS})
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
if(CONTINUUM_HAVE_MLX)
    target_link_libraries(continuum_core PUBLIC mlx)
    target_compile_definitions(continuum_core PUBLIC CONTINUUM_HAVE_MLX=1)
    if(MLX_CXX_FLAGS)
        target_compile_options(continuum_core PUBLIC ${MLX_CXX_FLAGS})
    endif()
    # libmlx lives inside the mlx wheel; find it at runtime without LD_LIBRARY_PATH.
    set_target_properties(continuum_core PROPERTIES INSTALL_RPATH "${MLX_LIBRARY_DIRS}")
endif()
target_compile_options(continuum_core PRIVATE -Wall -Wextra -Wpedantic)
if(CONTINUUM_WERROR)
    target_compile_options(continuum_core PRIVATE -Werror)
endif()
if(CONTINUUM_COVERAGE)
    target_compile_options(continuum_core PRIVATE --coverage -O0 -g)
    target_link_options(continuum_core PUBLIC --coverage)
endif()

if(CONTINUUM_BUILD_PYTHON)
    pybind11_add_module(_continuum
        bindings/pybind/_continuum.cpp
        bindings/pybind/bind_ir.cpp
        bindings/pybind/bind_runtime.cpp
        bindings/pybind/bind_backend.cpp
    )
    target_link_libraries(_continuum PRIVATE continuum_core)
    if(CONTINUUM_HAVE_MLX)
        set_target_properties(_continuum PROPERTIES BUILD_RPATH "${MLX_LIBRARY_DIRS}"
                                                    INSTALL_RPATH "${MLX_LIBRARY_DIRS}")
    endif()
    install(TARGETS _continuum DESTINATION continuum)
endif()

if(CONTINUUM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
