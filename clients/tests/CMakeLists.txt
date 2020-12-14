# ########################################################################
# Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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

find_package(GTest REQUIRED)
find_package(Boost COMPONENTS program_options)

if(NOT Boost_FOUND)
  message(STATUS "Dynamic boost libraries not found. Attempting to find static libraries...")
  set(Boost_USE_STATIC_LIBS ON)
  find_package(Boost REQUIRED COMPONENTS program_options)
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

set(TEST_MD5HASH
  f567e5f5029d052e3004bc69bb3f13f5
  e39879103dafab21f4cf942e0fe42a85
  a95eee14d980a9cfbbaf5df4a3c64713
  2d4c239daad6f12d66a1e6a2af44cbdb
  a899a0c48b9a58d081c52ffd88a84955
  455d5b699ea10232bbab5bc002219ae6
  f1b0e56fbb75d1d6862874e3d7d33060
  8c8633eada6455c1784269b213c85ea6
  3e62f7ea83914f7e20019aefb2a5176f
  fcfaf8a25c8f49b8d29f138f3c65c08f
  8a3cf5448a4fe73dcbdb5a16b326715f
  b203f7605cb1f20f83280061068f7ec7
  b0f812ffcc9469f0bf9be701205522c4
  f185514062a0eeabe86d2909275fe1dc
  04b781415202db404733ca0c159acbef
  c98e35f1cfd1ee8177f37bdae155a6e7
  c39375226aa5c495293003a5f637598f
  9a6481268847e6cf0d70671f2ff1ddcd
  73372e7d6a0848f8b19d64a924fab73e
  ad5963d0a39a943fcb0dc2b119d5b22a
  fda33f178963fbb39dfc8c051fd0279e
  c093666487879a4e44409eb7be1c0348
  5e784e1f8c6341287a2842bd188b347a
  01e49e63fa0ac2204baef0f5f33974ad
)


if(NOT TARGET rocsparse)
  set(CONVERT ${CMAKE_SOURCE_DIR}/../deps/convert)
else()
  set(CONVERT ${CMAKE_SOURCE_DIR}/deps/convert)
endif()

list(LENGTH TEST_MATRICES len)
math(EXPR len1 "${len} - 1")

foreach(i RANGE 0 ${len1})
  list(GET TEST_MATRICES ${i} m)
  list(GET TEST_MD5HASH ${i} md5)

  string(REPLACE "/" ";" sep_m ${m})
  list(GET sep_m 0 dir)
  list(GET sep_m 1 mat)

  # Download test matrices if not already downloaded
  set(CMAKE_MATRICES_DIR ${PROJECT_BINARY_DIR}/matrices)
  if(NOT EXISTS "${CMAKE_MATRICES_DIR}/${mat}.csr")
    message("-- Downloading and extracting test matrix ${m}.tar.gz")
    file(DOWNLOAD https://sparse.tamu.edu/MM/${m}.tar.gz ${CMAKE_MATRICES_DIR}/${mat}.tar.gz
         INACTIVITY_TIMEOUT 3
         STATUS DL)

    list(GET DL 0 stat)
    list(GET DL 1 msg)

    if(NOT stat EQUAL 0)
      message("-- Timeout has been reached, trying mirror ...")
      # Try again using ufl links
      file(DOWNLOAD https://www.cise.ufl.edu/research/sparse/MM/${m}.tar.gz ${CMAKE_MATRICES_DIR}/${mat}.tar.gz
           INACTIVITY_TIMEOUT 3
           STATUS DL)

      list(GET DL 0 stat)
      list(GET DL 1 msg)

      if(NOT stat EQUAL 0)
        message(FATAL_ERROR "${msg}")
      endif()
    endif()

    # Check MD5 hash before continuing
    file(MD5 ${CMAKE_MATRICES_DIR}/${mat}.tar.gz hash)

    # Compare hash
    if(NOT hash STREQUAL md5)
      message(FATAL_ERROR "${mat}.tar.gz is corrupted")
    endif()

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
  test_axpby.cpp
  test_axpyi.cpp
  test_doti.cpp
  test_dotci.cpp
  test_gather.cpp
  test_scatter.cpp
  test_gthr.cpp
  test_gthrz.cpp
  test_rot.cpp
  test_roti.cpp
  test_sctr.cpp
  test_bsrmv.cpp
  test_bsrsv.cpp
  test_coomv.cpp
  test_csrmv.cpp
  test_csrsv.cpp
  test_ellmv.cpp
  test_hybmv.cpp
  test_gebsrmv.cpp
  test_bsrmm.cpp
  test_csrmm.cpp
  test_csrsm.cpp
  test_gemmi.cpp
  test_csrgeam.cpp
  test_csrgemm.cpp
  test_bsric0.cpp
  test_bsrilu0.cpp
  test_csric0.cpp
  test_csrilu0.cpp
  test_csr2coo.cpp
  test_csr2csc.cpp
  test_gebsr2gebsc.cpp
  test_csr2ell.cpp
  test_csr2hyb.cpp
  test_csr2bsr.cpp
  test_csr2gebsr.cpp
  test_coo2csr.cpp
  test_ell2csr.cpp
  test_hyb2csr.cpp
  test_bsr2csr.cpp
  test_gebsr2csr.cpp
  test_gebsr2gebsr.cpp
  test_csr2csr_compress.cpp
  test_prune_csr2csr.cpp
  test_prune_csr2csr_by_percentage.cpp
  test_identity.cpp
  test_csrsort.cpp
  test_cscsort.cpp
  test_coosort.cpp
  test_csricsv.cpp
  test_csrilusv.cpp
  test_nnz.cpp
  test_dense2csr.cpp
  test_dense2coo.cpp
  test_prune_dense2csr.cpp
  test_prune_dense2csr_by_percentage.cpp
  test_dense2csc.cpp
  test_csr2dense.cpp
  test_csc2dense.cpp
  test_coo2dense.cpp
  test_spvec_descr.cpp
  test_spmat_descr.cpp
  test_dnvec_descr.cpp
  test_dnmat_descr.cpp
  test_spmv_coo.cpp
  test_spmv_coo_aos.cpp
  test_spmv_csr.cpp
  test_spmv_ell.cpp
  test_spvv.cpp
)

set(ROCSPARSE_TEST_SOURCES_TEMPLATE_INSTANCES
  testing_axpyi.instances.cpp
)

set(ROCSPARSE_CLIENTS_TESTINGS
../testings/testing_axpby.cpp
../testings/testing_axpyi.cpp
../testings/testing_doti.cpp
../testings/testing_dotci.cpp
../testings/testing_gather.cpp
../testings/testing_scatter.cpp
../testings/testing_gthr.cpp
../testings/testing_gthrz.cpp
../testings/testing_rot.cpp
../testings/testing_roti.cpp
../testings/testing_sctr.cpp
../testings/testing_bsrmv.cpp
../testings/testing_bsrsv.cpp
../testings/testing_coomv.cpp
../testings/testing_csrmv.cpp
../testings/testing_csrsv.cpp
../testings/testing_ellmv.cpp
../testings/testing_hybmv.cpp
../testings/testing_gebsrmv.cpp
../testings/testing_bsrmm.cpp
../testings/testing_csrmm.cpp
../testings/testing_csrsm.cpp
../testings/testing_gemmi.cpp
../testings/testing_csrgeam.cpp
../testings/testing_csrgemm.cpp
../testings/testing_bsric0.cpp
../testings/testing_bsrilu0.cpp
../testings/testing_csric0.cpp
../testings/testing_csrilu0.cpp
../testings/testing_csr2coo.cpp
../testings/testing_csr2csc.cpp
../testings/testing_gebsr2gebsc.cpp
../testings/testing_gebsr2gebsr.cpp
../testings/testing_csr2ell.cpp
../testings/testing_csr2hyb.cpp
../testings/testing_csr2bsr.cpp
../testings/testing_csr2gebsr.cpp
../testings/testing_coo2csr.cpp
../testings/testing_ell2csr.cpp
../testings/testing_hyb2csr.cpp
../testings/testing_bsr2csr.cpp
../testings/testing_gebsr2csr.cpp
../testings/testing_csr2csr_compress.cpp
../testings/testing_prune_csr2csr.cpp
../testings/testing_prune_csr2csr_by_percentage.cpp
../testings/testing_identity.cpp
../testings/testing_csrsort.cpp
../testings/testing_cscsort.cpp
../testings/testing_coosort.cpp
../testings/testing_csricsv.cpp
../testings/testing_csrilusv.cpp
../testings/testing_nnz.cpp
../testings/testing_dense2csr.cpp
../testings/testing_dense2coo.cpp
../testings/testing_prune_dense2csr.cpp
../testings/testing_prune_dense2csr_by_percentage.cpp
../testings/testing_dense2csc.cpp
../testings/testing_csr2dense.cpp
../testings/testing_csc2dense.cpp
../testings/testing_coo2dense.cpp
../testings/testing_spvec_descr.cpp
../testings/testing_spmat_descr.cpp
../testings/testing_dnvec_descr.cpp
../testings/testing_dnmat_descr.cpp
../testings/testing_spmv_coo.cpp
../testings/testing_spmv_coo_aos.cpp
../testings/testing_spmv_csr.cpp
../testings/testing_spmv_ell.cpp
../testings/testing_spvv.cpp
  )


#
# remove pragma once
# remove ifndef
# remove define
# remove last line
# add specialization
#


set(ROCSPARSE_CLIENTS_COMMON
  ../common/utility.cpp
  ../common/rocsparse_parse_data.cpp
  ../common/rocsparse_template_specialization.cpp
  ../common/rocsparse_init.cpp
  ../common/rocsparse_host.cpp
)

add_executable(rocsparse-test rocsparse_test_main.cpp ${ROCSPARSE_TEST_SOURCES} ${ROCSPARSE_CLIENTS_COMMON} ${ROCSPARSE_CLIENTS_TESTINGS})

# Set GOOGLE_TEST definition
target_compile_definitions(rocsparse-test PRIVATE GOOGLE_TEST)

# Target compile options
target_compile_options(rocsparse-test PRIVATE -ffp-contract=on -mfma -Wno-unused-command-line-argument -Wall)

# Internal common header
target_include_directories(rocsparse-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# Target link libraries
target_link_libraries(rocsparse-test PRIVATE GTest::GTest Boost::program_options roc::rocsparse hip::host)

# Add OpenMP if available
if(OPENMP_FOUND)
  target_link_libraries(rocsparse-test PRIVATE OpenMP::OpenMP_CXX -Wl,-rpath=${HIP_CLANG_ROOT}/lib)
endif()

# Set test output directory
set_target_properties(rocsparse-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

# Prepare testing data
set(ROCSPARSE_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocsparse_test.data")
add_custom_command(OUTPUT "${ROCSPARSE_TEST_DATA}"
                   COMMAND ../common/rocsparse_gentest.py -I ../include rocsparse_test.yaml -o "${ROCSPARSE_TEST_DATA}"
                   DEPENDS ../common/rocsparse_gentest.py rocsparse_test.yaml ../include/rocsparse_common.yaml known_bugs.yaml test_axpby.yaml test_axpyi.yaml test_doti.yaml test_dotci.yaml test_gather.yaml test_scatter.yaml test_gthr.yaml test_gthrz.yaml test_rot.yaml test_roti.yaml test_sctr.yaml test_bsrmv.yaml test_bsrsv.yaml test_coomv.yaml test_csrmv.yaml test_csrsv.yaml test_ellmv.yaml test_hybmv.yaml test_gebsrmv.yaml test_bsrmm.yaml test_csrmm.yaml test_csrsm.yaml test_gemmi.yaml test_csrgeam.yaml test_csrgemm.yaml test_bsric0.yaml test_bsrilu0.yaml test_csric0.yaml test_csrilu0.yaml test_csr2coo.yaml test_csr2csc.yaml test_gebsr2gebsc.yaml test_csr2ell.yaml test_csr2hyb.yaml test_bsr2csr.yaml test_csr2bsr.yaml test_csr2gebsr.yaml test_coo2csr.yaml test_ell2csr.yaml test_hyb2csr.yaml test_identity.yaml test_csrsort.yaml test_cscsort.yaml test_coosort.yaml test_csricsv.yaml test_csrilusv.yaml test_nnz.yaml test_dense2csr.yaml test_dense2coo.yaml test_prune_dense2csr.yaml test_prune_dense2csr_by_percentage.yaml test_dense2csc.yaml test_csr2dense.yaml test_csc2dense.yaml test_coo2dense.yaml test_csr2csr_compress.yaml test_prune_csr2csr.yaml test_prune_csr2csr_by_percentage.yaml test_gebsr2gebsr.yaml test_spvec_descr.yaml test_spmat_descr.yaml test_dnvec_descr.yaml test_dnmat_descr.yaml test_spmv_coo.yaml test_spmv_coo_aos.yaml test_spmv_csr.yaml test_spmv_ell.yaml test_spvv.yaml
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(rocsparse-test-data
                  DEPENDS "${ROCSPARSE_TEST_DATA}" )

add_dependencies(rocsparse-test rocsparse-test-data rocsparse-common)
