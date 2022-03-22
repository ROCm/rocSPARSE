# ########################################################################
# Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
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

find_package(GTest 1.11.0 REQUIRED)
find_package(rocsparseio QUIET)

#
# Client matrices.
#

if(NOT EXISTS "${CMAKE_MATRICES_DIR}")
  #
  # Download.
  #
  set(CMAKE_MATRICES_DIR ${PROJECT_BINARY_DIR}/matrices CACHE STRING "Matrices directory.")

  if(NOT TARGET rocsparse)
    set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/../deps/convert.cpp CACHE STRING "Convert tool mtx2csr.")
    include(${CMAKE_SOURCE_DIR}/../cmake/ClientMatrices.cmake)
  else()
   set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/deps/convert.cpp CACHE STRING "Convert tool mtx2csr.")
    include(${CMAKE_SOURCE_DIR}/cmake/ClientMatrices.cmake)
  endif()

else()

  #
  # Copy.
  #
  if(NOT CMAKE_MATRICES_DIR STREQUAL "${PROJECT_BINARY_DIR}/matrices")
    execute_process(COMMAND cp -r ${CMAKE_MATRICES_DIR} ${PROJECT_BINARY_DIR}/matrices)
  endif()

endif()

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
  test_bsrxmv.cpp
  test_bsrsv.cpp
  test_coomv.cpp
  test_csrmv.cpp
  test_csrmv_managed.cpp
  test_csrsv.cpp
  test_ellmv.cpp
  test_hybmv.cpp
  test_gebsrmv.cpp
  test_bsrmm.cpp
  test_gebsrmm.cpp
  test_csrmm.cpp
  test_csrsm.cpp
  test_bsrsm.cpp
  test_gemmi.cpp
  test_csrgeam.cpp
  test_csrgemm.cpp
  test_csrgemm_reuse.cpp
  test_bsric0.cpp
  test_bsrilu0.cpp
  test_csric0.cpp
  test_csrilu0.cpp
  test_gtsv_no_pivot.cpp
  test_gtsv_no_pivot_strided_batch.cpp
  test_gtsv_interleaved_batch.cpp
  test_gpsv_interleaved_batch.cpp
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
  test_spsv_csr.cpp
  test_spsv_coo.cpp
  test_spsm_csr.cpp
  test_spsm_coo.cpp
  test_spmm_csr.cpp
  test_spmm_coo.cpp
  test_spmm_bell.cpp
  test_spmm_batched_csr.cpp
  test_spmm_batched_coo.cpp
  test_spmm_batched_bell.cpp
  test_spvv.cpp
  test_sparse_to_dense_coo.cpp
  test_sparse_to_dense_csr.cpp
  test_sparse_to_dense_csc.cpp
  test_dense_to_sparse_coo.cpp
  test_dense_to_sparse_csr.cpp
  test_dense_to_sparse_csc.cpp
  test_spgemm_csr.cpp
  test_gtsv.cpp
  test_gemvi.cpp
  test_sddmm.cpp
  test_csrcolor.cpp
  test_copy_info.cpp
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
../testings/testing_bsrxmv.cpp
../testings/testing_bsrsv.cpp
../testings/testing_coomv.cpp
../testings/testing_csrmv.cpp
../testings/testing_csrmv_managed.cpp
../testings/testing_csrsv.cpp
../testings/testing_ellmv.cpp
../testings/testing_hybmv.cpp
../testings/testing_gebsrmv.cpp
../testings/testing_bsrmm.cpp
../testings/testing_gebsrmm.cpp
../testings/testing_csrmm.cpp
../testings/testing_csrsm.cpp
../testings/testing_bsrsm.cpp
../testings/testing_gemmi.cpp
../testings/testing_csrgeam.cpp
../testings/testing_csrgemm.cpp
../testings/testing_csrgemm_reuse.cpp
../testings/testing_bsric0.cpp
../testings/testing_bsrilu0.cpp
../testings/testing_csric0.cpp
../testings/testing_csrilu0.cpp
../testings/testing_gtsv_no_pivot.cpp
../testings/testing_gtsv_no_pivot_strided_batch.cpp
../testings/testing_gtsv_interleaved_batch.cpp
../testings/testing_gpsv_interleaved_batch.cpp
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
../testings/testing_spsv_csr.cpp
../testings/testing_spsv_coo.cpp
../testings/testing_spsm_csr.cpp
../testings/testing_spsm_coo.cpp
../testings/testing_spmm_csr.cpp
../testings/testing_spmm_coo.cpp
../testings/testing_spmm_bell.cpp
../testings/testing_spmm_batched_csr.cpp
../testings/testing_spmm_batched_coo.cpp
../testings/testing_spmm_batched_bell.cpp
../testings/testing_spvv.cpp
../testings/testing_sparse_to_dense_coo.cpp
../testings/testing_sparse_to_dense_csr.cpp
../testings/testing_sparse_to_dense_csc.cpp
../testings/testing_dense_to_sparse_coo.cpp
../testings/testing_dense_to_sparse_csr.cpp
../testings/testing_dense_to_sparse_csc.cpp
../testings/testing_spgemm_csr.cpp
../testings/testing_gtsv.cpp
../testings/testing_gemvi.cpp
../testings/testing_sddmm.cpp
../testings/testing_csrcolor.cpp
../testings/testing_copy_info.cpp
  )


set(ROCSPARSE_CLIENTS_COMMON
  ../common/utility.cpp
  ../common/rocsparse_check.cpp
  ../common/rocsparse_parse_data.cpp
  ../common/rocsparse_enum.cpp
  ../common/rocsparse_init.cpp
  ../common/rocsparse_host.cpp
  ../common/rocsparse_matrix_factory.cpp
  ../common/rocsparse_matrix_factory_laplace2d.cpp
  ../common/rocsparse_matrix_factory_laplace3d.cpp
  ../common/rocsparse_matrix_factory_zero.cpp
  ../common/rocsparse_matrix_factory_random.cpp
  ../common/rocsparse_matrix_factory_file.cpp
  ../common/rocsparse_exporter_rocsparseio.cpp
  ../common/rocsparse_exporter_rocalution.cpp
  ../common/rocsparse_exporter_matrixmarket.cpp
  ../common/rocsparse_exporter_ascii.cpp
  ../common/rocsparse_importer.cpp
  ../common/rocsparse_importer_rocalution.cpp
  ../common/rocsparse_importer_rocsparseio.cpp
  ../common/rocsparse_importer_matrixmarket.cpp
  )

add_executable(rocsparse-test rocsparse_test_main.cpp ${ROCSPARSE_TEST_SOURCES} ${ROCSPARSE_CLIENTS_COMMON} ${ROCSPARSE_CLIENTS_TESTINGS})

# Set GOOGLE_TEST definition
target_compile_definitions(rocsparse-test PRIVATE GOOGLE_TEST)

# Target compile options
target_compile_options(rocsparse-test PRIVATE -ffp-contract=on -mfma -Wno-unused-command-line-argument -Wall)
if (rocsparseio_FOUND)
  message("Compiles rocSPARSE client with rocSPARSEIO to import/export matrices.")
  target_compile_options(rocsparse-test PRIVATE -DROCSPARSEIO)
endif()

# Internal common header
target_include_directories(rocsparse-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# Target link libraries
target_link_libraries(rocsparse-test PRIVATE GTest::GTest roc::rocsparse hip::host hip::device)
if (rocsparseio_FOUND)
  target_link_libraries(rocsparse-test PRIVATE roc::rocsparseio)
endif()

# Add OpenMP if available
if(OPENMP_FOUND)
if (NOT WIN32)
   target_link_libraries(rocsparse-test PRIVATE OpenMP::OpenMP_CXX -Wl,-rpath=${HIP_CLANG_ROOT}/lib)
  else()
   target_link_libraries(rocsparse-test PRIVATE libomp)
  endif()
endif()


# Set test output directory
set_target_properties(rocsparse-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS rocsparse-test COMPONENT tests)

string(REGEX REPLACE ".cpp" ".yaml" ROCSPARSE_TEST_YAMLS "${ROCSPARSE_TEST_SOURCES}")

# Prepare testing data
set(ROCSPARSE_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocsparse_test.data")
add_custom_command(OUTPUT "${ROCSPARSE_TEST_DATA}"
                   COMMAND ${python} ../common/rocsparse_gentest.py -I ../include rocsparse_test.yaml -o "${ROCSPARSE_TEST_DATA}"
                   DEPENDS ../common/rocsparse_gentest.py rocsparse_test.yaml ../include/rocsparse_common.yaml known_bugs.yaml ${ROCSPARSE_TEST_YAMLS}
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(rocsparse-test-data
  DEPENDS "${ROCSPARSE_TEST_DATA}" )

add_dependencies(rocsparse-test  rocsparse-test-data rocsparse-common)

rocm_install(FILES ${ROCSPARSE_TEST_DATA} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT tests)

if (WIN32)
  # for now adding in all .dll as dependency chain is not cmake based on win32
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    $ENV{HIP_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/hipinfo.exe
    ${CMAKE_SOURCE_DIR}/rtest.*
    C:/Windows/System32/libomp140*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET rocsparse-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
  endforeach( file_i )
endif()
