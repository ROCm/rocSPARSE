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

  if(NOT TARGET rocsparse)
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${ROCSPARSE_LIBRARIES})
  else()
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE roc::rocsparse)
  endif()

  if(NOT HIP_PLATFORM STREQUAL "nvcc")
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${hip_LIBRARIES})
    foreach(amdgpu_target ${AMDGPU_TARGETS})
      target_link_libraries(${EXAMPLE_TARGET} PRIVATE --amdgpu-target=${amdgpu_target})
    endforeach()
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
