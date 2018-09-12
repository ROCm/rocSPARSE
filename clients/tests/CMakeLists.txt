# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
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
    execute_process(COMMAND ${CMAKE_SOURCE_DIR}/deps/convert ${mat}.mtx ${mat}.bin
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
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

if(HIP_PLATFORM STREQUAL "hcc")
  target_link_libraries(rocsparse-test
    PRIVATE
      roc::rocsparse
      ${GTEST_BOTH_LIBRARIES}
      hip::hip_hcc
      hip::hip_device
  )

  foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(rocsparse-test
      PRIVATE
        --amdgpu-target=${amdgpu_target}
    )
  endforeach()
endif()

if(HIP_PLATFORM STREQUAL "nvcc")
  target_link_libraries(rocsparse-test
    PRIVATE
      roc::rocsparse
      ${GTEST_BOTH_LIBRARIES}
  )
endif()

add_test(rocsparse-test rocsparse-test)
