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

get_target_property(HIPHCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE)

set(ROCSPARSE_CLIENTS_COMMON ../common/utility.cpp)

# Function to add rocsparse examples
function(add_rocsparse_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE} ${ROCSPARSE_CLIENTS_COMMON})

  # rocSPARSE headers
  target_include_directories(${EXAMPLE_TARGET}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )

  # rocSPARSE library
  if(NOT TARGET rocsparse)
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${ROCSPARSE_LIBRARIES})
  else()
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE roc::rocsparse)
  endif()

  if(CMAKE_CXX_COMPILER MATCHES ".*/hcc$")
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${HIP_LIBRARIES})
    foreach(target ${AMDGPU_TARGETS})
      target_link_libraries(${EXAMPLE_TARGET} PRIVATE --amdgpu-target=${target})
    endforeach()
  elseif(NOT CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
    target_compile_definitions(${EXAMPLE_TARGET} PRIVATE __HIP_PLATFORM_HCC__)
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${HIPHCC_LOCATION})
  endif()

  set_target_properties(${EXAMPLE_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
endfunction()

# Examples
add_rocsparse_example(example_handle.cpp)
add_rocsparse_example(example_coomv.cpp)
add_rocsparse_example(example_csrmv.cpp)
add_rocsparse_example(example_ellmv.cpp)
add_rocsparse_example(example_hybmv.cpp)
