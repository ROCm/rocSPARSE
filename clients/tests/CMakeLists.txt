# ########################################################################
# Copyright (c) 2019 Advanced Micro Devices, Inc.
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
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost)

if(NOT Boost_FOUND)
  message(STATUS "Dynamic boost libraries not found. Attempting to find static libraries...")
  set(Boost_USE_STATIC_LIBS ON)
  find_package(Boost)

  if(NOT Boost_FOUND)
    message(FATAL_ERROR "boost is a required dependency and is not found; try adding boost path to CMAKE_PREFIX_PATH")
  endif()
endif()

# Download some test matrices
set(TEST_MATRICES
  SNAP/amazon0312
  Muite/Chebyshev4
  FEMLAB/sme3Dc
  Williams/webbase-1M
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
  DNVS/shipsec1
  Cote/mplate
  Bai/qc2534
  Chevron/Chevron2
  Chevron/Chevron3
  Chevron/Chevron4
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
  set(CMAKE_MATRICES_DIR ${PROJECT_BINARY_DIR}/matrices)
  if(NOT EXISTS "${CMAKE_MATRICES_DIR}/${mat}.csr")
    message("  Downloading and extracting test matrix ${m}.tar.gz")
    file(DOWNLOAD http://www.cise.ufl.edu/research/sparse/MM/${m}.tar.gz
         ${CMAKE_MATRICES_DIR}/${mat}.tar.gz)
    execute_process(COMMAND tar xf ${mat}.tar.gz
                    WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})
    execute_process(COMMAND mv ${mat}/${mat}.mtx .
                    WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})
    execute_process(COMMAND ${CONVERT} ${mat}.mtx ${mat}.csr
                    WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})
    execute_process(COMMAND rm ${mat}.tar.gz ${mat} ${mat}.mtx -rf
                    WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})
  endif()
endforeach()

set(ROCSPARSE_TEST_SOURCES
  rocsparse_test_main.cpp
  test_axpyi.cpp
  test_doti.cpp
  test_dotci.cpp
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
  test_csrgemm.cpp
  test_csrilu0.cpp
  test_csr2coo.cpp
  test_csr2csc.cpp
  test_csr2ell.cpp
  test_csr2hyb.cpp
  test_coo2csr.cpp
  test_ell2csr.cpp
  test_identity.cpp
  test_csrsort.cpp
  test_cscsort.cpp
  test_coosort.cpp
  test_csrilusv.cpp
)

set(ROCSPARSE_CLIENTS_COMMON
  ../common/utility.cpp
  ../common/rocsparse_parse_data.cpp
  ../common/rocsparse_template_specialization.cpp
)

add_executable(rocsparse-test ${ROCSPARSE_TEST_SOURCES} ${ROCSPARSE_CLIENTS_COMMON})

# Compile options set to enable fma on host path
target_compile_options(rocsparse-test PRIVATE -mfma)

# Set GOOGLE_TEST definition
target_compile_definitions(rocsparse-test PRIVATE GOOGLE_TEST)

# Internal header includes
target_include_directories(rocsparse-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# External header includes included as system files
target_include_directories(rocsparse-test
  SYSTEM
    PRIVATE
      $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
)

# OpenMP cmake fix for cmake <= 3.9
if(OPENMP_FOUND)
  if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
  endif()
endif()

if(NOT TARGET rocsparse)
  target_link_libraries(rocsparse-test PRIVATE ${ROCSPARSE_LIBRARIES})
else()
  target_link_libraries(rocsparse-test PRIVATE roc::rocsparse)
endif()

# Add amdgpu targets
if(CMAKE_CXX_COMPILER MATCHES ".*/hcc$")
  foreach(target ${AMDGPU_TARGETS})
    target_link_libraries(rocsparse-test PRIVATE --amdgpu-target=${target})
  endforeach()
endif()

target_link_libraries(rocsparse-test PRIVATE ${GTEST_BOTH_LIBRARIES} Threads::Threads hip::host)

if(OPENMP_FOUND)
  target_link_libraries(rocsparse-test PRIVATE OpenMP::OpenMP_CXX)
endif()

# Prepare testing data
set(ROCSPARSE_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocsparse_test.data")
add_custom_command(OUTPUT "${ROCSPARSE_TEST_DATA}"
                   COMMAND ../common/rocsparse_gentest.py -I ../include rocsparse_test.yaml -o "${ROCSPARSE_TEST_DATA}"
                   DEPENDS ../common/rocsparse_gentest.py rocsparse_test.yaml ../include/rocsparse_common.yaml known_bugs.yaml test_axpyi.yaml test_doti.yaml test_dotci.yaml test_gthr.yaml test_gthrz.yaml test_roti.yaml test_sctr.yaml test_coomv.yaml test_csrmv.yaml test_csrsv.yaml test_ellmv.yaml test_hybmv.yaml test_csrmm.yaml test_csrgemm.yaml test_csrilu0.yaml test_csr2coo.yaml test_csr2csc.yaml test_csr2ell.yaml test_csr2hyb.yaml test_coo2csr.yaml test_ell2csr.yaml test_identity.yaml test_csrsort.yaml test_cscsort.yaml test_coosort.yaml test_csrilusv.yaml
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(rocsparse-test-data
                  DEPENDS "${ROCSPARSE_TEST_DATA}" )

add_dependencies(rocsparse-test rocsparse-test-data rocsparse-common)

set_target_properties(rocsparse-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
