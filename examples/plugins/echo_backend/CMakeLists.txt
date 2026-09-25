# Standalone build of the example plugin. Needs only the public ABI header:
#
#   cmake -S examples/plugins/echo_backend -B build-echo \
#         -DCONTINUUM_INCLUDE_DIR=$PWD/include
#   cmake --build build-echo
#
# The Continuum C++ test suite also builds this target (tests/cpp).
cmake_minimum_required(VERSION 3.16)
project(continuum_echo_backend C)

if(NOT CONTINUUM_INCLUDE_DIR)
  get_filename_component(CONTINUUM_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../include" ABSOLUTE)
endif()

add_library(continuum_echo_backend MODULE echo_backend.c)
target_include_directories(continuum_echo_backend PRIVATE ${CONTINUUM_INCLUDE_DIR})
set_target_properties(continuum_echo_backend PROPERTIES C_VISIBILITY_PRESET hidden)
