cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )
endif( )

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  /opt/rocm/hip/cmake
)

# Find HIP package
find_package(HIP REQUIRED)

# Set compiler
if(HIP_PLATFORM STREQUAL "nvcc")
  message("-- PLATFORM = nvcc currently unsupported")
elseif(HIP_PLATFORM STREQUAL "hcc")
  message("-- PLATFORM = hcc")
  find_program(HIP_HCC_EXECUTABLE NAMES hcc PATHS
               "${HIP_ROOT_DIR}"
               ENV ROCM_PATH
               ENV HIP_PATH
               /opt/rocm
               /opt/rocm/hip
               PATH_SUFFIXES bin
               NO_DEFAULT_PATH
  )
  if(NOT HIP_HCC_EXECUTABLE)
    find_program(HIP_HCC_EXECUTABLE hcc)
  endif()
  mark_as_advanced(HIP_HCC_EXECUTABLE)
  set(CMAKE_CXX_COMPILER ${HIP_HCC_EXECUTABLE})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip)
else()
  message(FATAL_ERROR "HIP_PLATFORM must be 'hcc/nvcc' (AMD ROCm platform).")
endif()

# rocSPARSE project
project(rocsparse VERSION 0.1.0.0 LANGUAGES CXX)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# CXX Build flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# HIP configuration
if(HIP_PLATFORM STREQUAL "hcc")
  # Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
  # (Thanks to rocBLAS devs for finding workaround for this problem!)
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip)
  # Ignore hcc warning: argument unused during compilation: '-isystem /opt/rocm/hip/include'
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
  find_package(hcc REQUIRED CONFIG PATHS /opt/rocm)
  find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
endif()

# Build options
option(BUILD_SHARED_LIBS "Build rocSPARSE as a shared library" ON)
option(BUILD_VERBOSE "Output additional build information" ON)
option(BUILD_TEST "Build tests (requires googletest)" ON)
option(BUILD_BENCHMARK "Build benchmarks (requires googlebenchmark)" ON)
option(BUILD_EXAMPLE "Build examples" ON)

# Test dependencies
if(BUILD_TEST)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(GTest QUIET)
  endif()
  if(NOT GTEST_FOUND)
    message(STATUS "GTest not found. Downloading and building GTest.")
    include(cmake/DownloadProject.cmake)
    find_package(Git REQUIRED)
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
    download_project(PROJ        googletest
             GIT_REPOSITORY      https://github.com/google/googletest.git
             GIT_TAG             master
             INSTALL_DIR         ${GTEST_ROOT}
             CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             LOG_DOWNLOAD        TRUE
             LOG_CONFIGURE       TRUE
             LOG_BUILD           TRUE
             LOG_INSTALL         TRUE
             BUILD_PROJECT       TRUE
             UPDATE_DISCONNECTED TRUE
    )
  endif()
  find_package(GTest REQUIRED)
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(benchmark QUIET)
  endif()
  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found. Downloading and building Google Benchmark.")
    include(cmake/DownloadProject.cmake)
    find_package(Git REQUIRED)
    set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark CACHE PATH "")
    download_project(PROJ        googlebenchmark
             GIT_REPOSITORY      https://github.com/google/benchmark.git
             GIT_TAG             master
             INSTALL_DIR         ${GOOGLEBENCHMARK_ROOT}
             CMAKE_ARGS          -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_ENABLE_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             LOG_DOWNLOAD        TRUE
             LOG_CONFIGURE       TRUE
             LOG_BUILD           TRUE
             LOG_INSTALL         TRUE
             BUILD_PROJECT       TRUE
             UPDATE_DISCONNECTED TRUE
    )
  endif()
  find_package(benchmark REQUIRED CONFIG PATHS ${GOOGLEBENCHMARK_ROOT})
endif()

# ROCm cmake project
find_package(ROCM QUIET CONFIG PATHS /opt/rocm)
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(
    DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  find_package(ROCM REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900 CACHE STRING "List of specific machine types for library to target")

# rocSPARSE library
add_subdirectory(library)

# Tests
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks
if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

# Examples
if(BUILD_EXAMPLE)
  add_subdirectory(example)
endif()
