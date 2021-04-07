# ########################################################################
# Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# The ROCm platform requires at least Ubuntu 16.04 or Fedora 24, which have cmake 3.5
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# Adding CMAKE_PREFIX_PATH, needed for static builds
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake /opt/rocm/hip/cmake)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if(POLICY CMP0066)
  cmake_policy(SET CMP0066 NEW)
endif()

# rocSPARSE project
project(rocsparse LANGUAGES CXX Fortran)

# Determine if CXX Compiler is hip-clang
execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--version" OUTPUT_VARIABLE CXX_OUTPUT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "[A-Za-z]* ?clang version" TMP_CXX_VERSION ${CXX_OUTPUT})
string(REGEX MATCH "[A-Za-z]+" CXX_VERSION_STRING ${TMP_CXX_VERSION})

# Print compiler information
if(CXX_VERSION_STRING MATCHES "clang" OR 
   CXX_VERSION_STRING MATCHES "AMD")
  message(STATUS "Using hip-clang to build for amdgpu backend")
else()
  message(FATAL_ERROR "'hipcc' compiler required to compile for ROCm platform.")
endif()

# Force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Build flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build rocSPARSE as a shared library" ON)
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" ON)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(BUILD_CODE_COVERAGE "Build rocSPARSE with code coverage enabled" OFF)
option(BUILD_ADDRESS_SANITIZER "Build rocSPARSE with address sanitizer enabled" OFF)

if(BUILD_CODE_COVERAGE)
  add_compile_options(-g -fprofile-arcs -ftest-coverage)
  add_link_options(-g --coverage)
endif()

if(BUILD_ADDRESS_SANITIZER)
  add_compile_options(-g -fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-g -fsanitize=address -fuse-ld=lld)
endif()

# Detect compiler support for target ID
if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
    OUTPUT_VARIABLE CXX_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

# Use target ID syntax if supported for AMDGPU_TARGETS
if(TARGET_ID_SUPPORT)
  # AMD targets
  set(AMDGPU_TARGETS gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx908:xnack+;gfx1030 CACHE STRING "List of specific machine types for library to target")
else()
  set(AMDGPU_TARGETS gfx803;gfx900;gfx906;gfx908;gfx1030 CACHE STRING "List of specific machine types for library to target")
endif()

# Dependencies
include(cmake/Dependencies.cmake)

# Setup version
set(VERSION_STRING "1.20.1")
rocm_setup_version(VERSION ${VERSION_STRING})
set(rocsparse_SOVERSION 0.1)

# rocSPARSE library
add_subdirectory(library)

# Trigger client builds if selected
if(BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  add_subdirectory(clients)
endif()

#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
# > make coverage GTEST_FILTER=<>
# will run:
#  > make coverage_analysis (analyze tests)
#  > make coverage_output (generate html documentation)
#  > make coverage_cleanup (clean coverage related files.)
#

#
# Run coverage analysis
#
add_custom_target(coverage_analysis
   COMMAND mkdir -p gcoverage
   COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
   COMMAND ./clients/staging/rocsparse-test --gtest_filter=\"\${GTEST_FILTER}\"
   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
 )

add_dependencies(coverage_analysis rocsparse)

#
# Prepare coverage output
#
add_custom_target(coverage_output
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
    )

#
# Generate coverage output.
#
add_custom_command(TARGET coverage_output
  COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
  COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
  COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage
    )

add_custom_target(coverage DEPENDS coverage_analysis coverage_output)

#
# Coverage cleanup
#
add_custom_target(coverage_cleanup
    COMMAND rm -f `find ${CMAKE_BINARY_DIR} -name *.gcda`
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
