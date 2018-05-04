# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

find_package(GTest REQUIRED)

set(ROCSPARSE_TEST_SOURCES
  rocsparse_gtest_main.cpp
  test_axpyi.cpp
  test_csrmv.cpp
)

set(ROCSPARSE_CLIENTS_COMMON
  ../common/arg_check.cpp
  ../common/unit.cpp
  ../common/utility.cpp
  ../common/rocsparse_template_specialization.cpp
)

add_executable(rocsparse-test ${ROCSPARSE_TEST_SOURCES} ${ROCSPARSE_CLIENTS_COMMON})

target_compile_definitions(rocsparse-test PRIVATE GOOGLE_TEST)

target_include_directories(rocsparse-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_include_directories(rocsparse-test
  SYSTEM
    PRIVATE
      $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

target_link_libraries(rocsparse-test
  PRIVATE
    roc::rocsparse
    ${GTEST_BOTH_LIBRARIES}
    hip::hip_hcc
    hip::hip_device
)

foreach(amdgpu_target ${AMDGPU_TARGETS})
  target_link_libraries(rocsparse-test
    PRIVATE
      --amdgpu-target=${amdgpu_target}
  )
endforeach()

add_test(rocsparse-test rocsparse-test)
