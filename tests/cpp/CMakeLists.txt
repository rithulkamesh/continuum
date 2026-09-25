include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
  continuum_cpp_tests
  test_graph.cpp
  test_runtime_extras.cpp
  test_cir_corpus.cpp
  test_backend_plugins.cpp
  test_mlx_backend.cpp
)
target_link_libraries(continuum_cpp_tests PRIVATE continuum_core GTest::gtest_main)
target_compile_definitions(continuum_cpp_tests PRIVATE CONTINUUM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Example out-of-tree plugin, built from the public C header only.
enable_language(C)
add_library(continuum_echo_backend MODULE ${CMAKE_SOURCE_DIR}/examples/plugins/echo_backend/echo_backend.c)
target_include_directories(continuum_echo_backend PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(continuum_echo_backend PROPERTIES C_VISIBILITY_PRESET hidden)
add_dependencies(continuum_cpp_tests continuum_echo_backend)
target_compile_definitions(continuum_cpp_tests PRIVATE
  CONTINUUM_ECHO_PLUGIN="$<TARGET_FILE:continuum_echo_backend>")

add_executable(gen_cir_corpus gen_cir_corpus.cpp)
target_link_libraries(gen_cir_corpus PRIVATE continuum_core)

include(GoogleTest)
gtest_discover_tests(continuum_cpp_tests)
