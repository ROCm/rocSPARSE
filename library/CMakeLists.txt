# ########################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc.
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

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# Get full git commit SHA-1
set(GIT_COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD)
execute_process(COMMAND ${GIT_COMMAND}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE rocsparse_GIT_REVISION
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)

# Configure a header file to pass the rocSPARSE version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocsparse-version.h.in"
               "${PROJECT_BINARY_DIR}/include/rocsparse-version.h"
               @ONLY
)

# Public rocSPARSE headers
set(rocsparse_headers_public
  include/rocsparse-auxiliary.h
  include/rocsparse-functions.h
  include/rocsparse-types.h
  include/rocsparse.h
  ${PROJECT_BINARY_DIR}/include/rocsparse-version.h
  ${PROJECT_BINARY_DIR}/include/rocsparse-export.h
)

# Include sources
include(src/CMakeLists.txt)

# Create rocSPARSE library
add_library(rocsparse ${rocsparse_source} ${rocsparse_headers_public})
add_library(roc::rocsparse ALIAS rocsparse)

# Target compile options
if(NOT HIP_PLATFORM STREQUAL "nvcc")
  target_compile_options(rocsparse PRIVATE -fno-gpu-rdc)
  foreach(target ${AMDGPU_TARGETS})
    target_compile_options(rocsparse PRIVATE --amdgpu-target=${target})
  endforeach()
endif()

# Target link libraries
if(NOT HIP_PLATFORM STREQUAL "nvcc")
  target_link_libraries(rocsparse PRIVATE hip::device)
  foreach(target ${AMDGPU_TARGETS})
    target_link_libraries(rocsparse PRIVATE --amdgpu-target=${target})
  endforeach()
endif()

# Target include directories
target_include_directories(rocsparse
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                                   $<BUILD_INTERFACE:${ROCPRIM_INCLUDE_DIRS}>
                                   $<BUILD_INTERFACE:${HIPCUB_INCLUDE_DIRS}>
                           PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<INSTALL_INTERFACE:include>
)

# Target properties
if(NOT HIP_PLATFORM STREQUAL "nvcc")
  set_target_properties(rocsparse PROPERTIES VERSION ${rocsparse_VERSION} SOVERSION ${rocsparse_SOVERSION})
  set_target_properties(rocsparse PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
endif()
set_target_properties(rocsparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_properties(rocsparse PROPERTIES DEBUG_POSTFIX "-d")

# Generate export header
include(GenerateExportHeader)
generate_export_header(rocsparse EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocsparse-export.h)

# Install targets
rocm_install_targets(TARGETS rocsparse
                     INCLUDE
                       ${CMAKE_SOURCE_DIR}/library/include
                       ${CMAKE_BINARY_DIR}/include
                     PREFIX rocsparse
)

# Export targets
rocm_export_targets(TARGETS roc::rocsparse
                    PREFIX rocsparse
                    DEPENDS PACKAGE hip
                    NAMESPACE roc::
)

# Symbolic links
rocm_install_symlink_subdir(rocsparse)

# Package specific CPACK vars
set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.5.19055)") # 1.5.19055 is HIP version in ROCm 2.2
set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.5.19055")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

# Package name differs for CUDA backend
if(HIP_PLATFORM STREQUAL "nvcc")
  set(package_name rocsparse-alt)
else()
  set(package_name rocsparse)
endif()

set(ROCSPARSE_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute SPARSE library"
  MAINTAINER "Nico Trost"
  LDCONFIG
  LDCONFIG_DIR ${ROCSPARSE_CONFIG_DIR}
)
