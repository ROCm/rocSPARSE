# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Function to add rocsparse tests
function(add_rocsparse_test TEST_NAME TEST_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  target_include_directories(${TEST_TARGET} SYSTEM
    PUBLIC
      ${GTEST_INCLUDE_DIRS}
      ${CMAKE_SOURCE_DIR}/utils
  )
  if(HIP_PLATFORM STREQUAL "hcc")
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        rocsparse
        ${GTEST_BOTH_LIBRARIES}
        hip::hip_hcc
        hip::hip_device
    )
    foreach(amdgpu_target ${AMDGPU_TARGETS})
      target_link_libraries(${TEST_TARGET}
        PRIVATE
          --amdgpu-target=${amdgpu_target}
      )
    endforeach()
  endif()
  set_target_properties(${TEST_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

# Tests
add_rocsparse_test("device_apis" test_device_apis.cpp)
add_rocsparse_test("rocsparse.handle" test_rocsparse_handle.cpp)
add_rocsparse_test("rocsparse.csrmv" test_rocsparse_csrmv.cpp)
