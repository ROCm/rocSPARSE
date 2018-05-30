# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

set(ROCSPARSE_CLIENTS_COMMON
  ../common/utility.cpp
)

# Function to add rocsparse examples
function(add_rocsparse_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE} ${ROCSPARSE_CLIENTS_COMMON})

  target_include_directories(${EXAMPLE_TARGET}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )

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

  if(HIP_PLATFORM STREQUAL "nvcc")
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
        rocsparse
    )
  endif()
  set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
  )
endfunction()

# Examples
add_rocsparse_example(example_handle.cpp)
add_rocsparse_example(example_coomv.cpp)
add_rocsparse_example(example_csrmv.cpp)
add_rocsparse_example(example_ellmv.cpp)
add_rocsparse_example(example_hybmv.cpp)
