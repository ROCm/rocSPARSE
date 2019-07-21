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

# Internal header includes
target_include_directories(rocsparse-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# External header includes included as system files
target_include_directories(rocsparse-bench SYSTEM PRIVATE $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)

target_link_libraries(rocsparse-bench PRIVATE ${Boost_LIBRARIES})

if(NOT TARGET rocsparse)
  target_link_libraries(rocsparse-bench PRIVATE ${ROCSPARSE_LIBRARIES})
else()
  target_link_libraries(rocsparse-bench PRIVATE roc::rocsparse)
endif()

if(CMAKE_CXX_COMPILER MATCHES ".*/hcc$")
  target_link_libraries(rocsparse-bench PRIVATE ${HIP_LIBRARIES})
  foreach(target ${AMDGPU_TARGETS})
    target_link_libraries(rocsparse-bench PRIVATE --amdgpu-target=${target})
  endforeach()
elseif(NOT CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
  target_compile_definitions(rocsparse-bench PRIVATE __HIP_PLATFORM_HCC__)
  get_target_property(HIPHCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE)
  target_link_libraries(rocsparse-bench PRIVATE ${HIPHCC_LOCATION})
endif( )

set_target_properties(rocsparse-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
