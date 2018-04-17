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
