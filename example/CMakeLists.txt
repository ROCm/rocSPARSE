# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Function to add rocsparse examples
function(add_rocsparse_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})
  if(HIP_PLATFORM STREQUAL "hcc")
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
        rocsparse
        hip::hip_hcc
    )
    foreach(amdgpu_target ${AMDGPU_TARGETS})
      target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
          --amdgpu-target=${amdgpu_target}
      )
    endforeach()
  endif()
  set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
  )
endfunction()

# Examples
add_rocsparse_example(rocsparse_handle.cpp)
