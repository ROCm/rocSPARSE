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

# Download some test matrices
set(TEST_MATRICES
  Bova/rma10
  JGD_BIBD/bibd_22_8
  Williams/mac_econ_fwd500
  Williams/mc2depi
  Hamm/scircuit
  Sandia/ASIC_320k
  GHS_psdef/bmwcra_1
  HB/nos1
  HB/nos2
  HB/nos3
  HB/nos4
  HB/nos5
  HB/nos6
  HB/nos7
)

if(NOT TARGET rocsparse)
  set(CONVERT ${CMAKE_SOURCE_DIR}/../deps/convert)
else()
  set(CONVERT ${CMAKE_SOURCE_DIR}/deps/convert)
endif()

foreach(m ${TEST_MATRICES})
  string(REPLACE "/" ";" sep_m ${m})
  list(GET sep_m 0 dir)
  list(GET sep_m 1 mat)

  # Download test matrices if not already downloaded
  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/matrices/${mat}.bin")
    message("  Downloading and extracting test matrix ${m}.tar.gz")
    file(DOWNLOAD http://www.cise.ufl.edu/research/sparse/MM/${m}.tar.gz
         ${CMAKE_CURRENT_BINARY_DIR}/matrices/${mat}.tar.gz)
    execute_process(COMMAND tar xf ${mat}.tar.gz
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matrices)
    execute_process(COMMAND mv ${mat}/${mat}.mtx .
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matrices)
    execute_process(COMMAND ${CONVERT} ${mat}.mtx ${mat}.bin
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matrices)
    execute_process(COMMAND rm ${mat}.tar.gz ${mat} ${mat}.mtx -rf
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matrices)
  endif()
endforeach()


find_package(GTest REQUIRED)

set(ROCSPARSE_TEST_SOURCES
  rocsparse_gtest_main.cpp
  test_axpyi.cpp
  test_doti.cpp
  test_gthr.cpp
  test_gthrz.cpp
  test_roti.cpp
  test_sctr.cpp
  test_coomv.cpp
  test_csrmv.cpp
  test_csrsv.cpp
  test_ellmv.cpp
  test_hybmv.cpp
  test_csrmm.cpp
  test_csrilu0.cpp
  test_csr2coo.cpp
  test_csr2csc.cpp
  test_csr2ell.cpp
  test_csr2hyb.cpp
  test_coo2csr.cpp
  test_ell2csr.cpp
  test_identity.cpp
  test_csrsort.cpp
  test_coosort.cpp
  test_csrilusv.cpp
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
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
)

if(NOT TARGET rocsparse)
  target_link_libraries(rocsparse-test PRIVATE ${ROCSPARSE_LIBRARIES})
else()
  target_link_libraries(rocsparse-test PRIVATE roc::rocsparse)
endif()

if(HIP_PLATFORM STREQUAL "nvcc")
  target_link_libraries(rocsparse-test PRIVATE ${GTEST_BOTH_LIBRARIES})
else()
  target_link_libraries(rocsparse-test PRIVATE ${GTEST_BOTH_LIBRARIES} ${hip_LIBRARIES})

  foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(rocsparse-test PRIVATE --amdgpu-target=${amdgpu_target})
  endforeach()
endif()

add_test(rocsparse-test rocsparse-test)
