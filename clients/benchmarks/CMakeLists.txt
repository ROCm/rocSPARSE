# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

set(Boost_USE_MULTITHREADED ON)
set(Boost_DETAILED_FAILURE_MSG ON)
set(Boost_ADDITIONAL_VERSIONS 1.65.1 1.65)
set(Boost_USE_STATIC_LIBS OFF)

find_package(Boost COMPONENTS program_options)

if(NOT Boost_FOUND)
  message(STATUS "Dynamic boost libraries not found. Attempting to find static libraries")
  set(Boost_USE_STATIC_LIBS ON)
  find_package(Boost COMPONENTS program_options)

  if(NOT Boost_FOUND)
    message(FATAL_ERROR "boost is a required dependency and is not found;  try adding boost path to CMAKE_PREFIX_PATH")
  endif()
endif()

set(ROCSPARSE_CLIENTS_COMMON
  ../common/arg_check.cpp
  ../common/unit.cpp
  ../common/utility.cpp
  ../common/rocsparse_template_specialization.cpp
)

add_executable(rocsparse-bench client.cpp ${ROCSPARSE_CLIENTS_COMMON})

target_include_directories(rocsparse-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_include_directories(rocsparse-bench
  SYSTEM
    PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
)

if(HIP_PLATFORM STREQUAL "hcc")
  target_link_libraries(rocsparse-bench
    PRIVATE
      ${Boost_LIBRARIES}
      roc::rocsparse
      hip::hip_hcc
      hip::hip_device
  )

  foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(rocsparse-bench
      PRIVATE
        --amdgpu-target=${amdgpu_target}
    )
  endforeach()
endif()

if(HIP_PLATFORM STREQUAL "nvcc")
  target_link_libraries(rocsparse-bench
    PRIVATE
      ${Boost_LIBRARIES}
      roc::rocsparse
  )
endif()
