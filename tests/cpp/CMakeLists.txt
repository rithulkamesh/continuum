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
)
target_link_libraries(continuum_cpp_tests PRIVATE continuum_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(continuum_cpp_tests)
