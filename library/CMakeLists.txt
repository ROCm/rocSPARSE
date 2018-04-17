# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Package that helps me set visibility for function names exported from shared library
include(GenerateExportHeader)

# Verbose build info
if(BUILD_VERBOSE)
  message(STATUS "rocsparse_VERSION: ${rocsparse_VERSION}")
  message(STATUS "\t==>CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
  message(STATUS "\t==>BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
  message(STATUS "\t==>CMAKE_INSTALL_PREFIX link: " ${CMAKE_INSTALL_PREFIX})
  message(STATUS "\t==>CMAKE_MODULE_PATH link: " ${CMAKE_MODULE_PATH})
  message(STATUS "\t==>CMAKE_PREFIX_PATH link: " ${CMAKE_PREFIX_PATH})
  message(STATUS "==============")
  message(STATUS "\t==>CMAKE_CXX_COMPILER: " ${CMAKE_CXX_FLAGS})
  message(STATUS "\t==>CMAKE_CXX_COMPILER debug: " ${CMAKE_CXX_FLAGS_DEBUG})
  message(STATUS "\t==>CMAKE_CXX_COMPILER release: " ${CMAKE_CXX_FLAGS_RELEASE})
  message(STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  message(STATUS "\t==>CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
  message(STATUS "\t==>CMAKE_EXE_LINKER_FLAGS_RELEASE: " ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
  message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: " ${CMAKE_SHARED_LINKER_FLAGS})
  message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: " ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
  message(STATUS "==============")
  message(STATUS "\t==>CMAKE_SHARED_LIBRARY_C_FLAGS: ${CMAKE_SHARED_LIBRARY_C_FLAGS}")
  message(STATUS "\t==>CMAKE_SHARED_LIBRARY_CXX_FLAGS: ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}")
  message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
  message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_DEBUG: ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
  message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
endif()

# .so version
set(rocsparse_SOVERSION 0)

# Configure a header file to pass the rocSPARSE version
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/rocsparse_version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/rocsparse_version.h"
  @ONLY
)

# Include sources
include(src/CMakeLists.txt)

# Create library from sources
if(BUILD_SHARED_LIBS)
  add_library(rocsparse SHARED ${rocsparse_source})
else()
  add_library(rocsparse STATIC ${rocsparse_source})
endif()
add_library(roc::rocsparse ALIAS rocsparse)

if(HIP_PLATFORM STREQUAL "hcc")
  # Linker targets
  target_link_libraries(rocsparse PRIVATE hip::hip_hcc hip::hip_device hcc::hccshared)

  # GPU targets
  foreach(target ${AMDGPU_TARGETS})
    target_link_libraries(rocsparse PRIVATE --amdgpu-target=${target})
  endforeach()
endif()

# Include directories
target_include_directories(rocsparse
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
	  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:include>
)

# Target properties
set_target_properties(rocsparse PROPERTIES VERSION ${rocsparse_VERSION} SOVERSION ${rocsparse_SOVERSION} CXX_EXTENSIONS NO)
set_target_properties(rocsparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_properties(rocsparse PROPERTIES DEBUG_POSTFIX "-d")
set_target_properties(rocsparse PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
generate_export_header(rocsparse EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocsparse_export.h)

# Installation
rocm_install_targets(
  TARGETS rocsparse
  INCLUDE
    ${CMAKE_SOURCE_DIR}/library/include
    ${CMAKE_BINARY_DIR}/include
  PREFIX rocsparse
)

rocm_export_targets(
  TARGETS rocsparse-targets
  PREFIX rocsparse
  DEPENDS PACKAGE hip
  NAMESPACE roc::
)

# Package specific CPACK vars
set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.3)")
set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.3")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

# Package name
set(package_name rocsparse)

set(ROCSPARSE_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute SPARSE library"
  MAINTAINER "Nico Trost <nico.trost@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCSPARSE_CONFIG_DIR}
)
