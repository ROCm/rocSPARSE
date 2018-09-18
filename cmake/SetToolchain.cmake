# Find HIP package
find_package(HIP 1.5.18353 REQUIRED) # ROCm 1.9

# Select toolchain
if(HIP_PLATFORM STREQUAL "nvcc" OR HIP_COMPILER STREQUAL "clang")
  # Find HIPCC executable
  find_program(
      HIP_HIPCC_EXECUTABLE
      NAMES hipcc
      PATHS
      "${HIP_ROOT_DIR}"
      ENV ROCM_PATH
      ENV HIP_PATH
      /opt/rocm
      /opt/rocm/hip
      PATH_SUFFIXES bin
      NO_DEFAULT_PATH
      )
  if(NOT HIP_HIPCC_EXECUTABLE)
      # Now search in default paths
      find_program(HIP_HIPCC_EXECUTABLE hipcc)
  endif()
  mark_as_advanced(HIP_HIPCC_EXECUTABLE)
  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
elseif(HIP_PLATFORM STREQUAL "hcc")
  # Find HCC executable
  find_program(
      HIP_HCC_EXECUTABLE
      NAMES hcc
      PATHS
      "${HIP_ROOT_DIR}"
      ENV ROCM_PATH
      ENV HIP_PATH
      /opt/rocm
      /opt/rocm/hip
      PATH_SUFFIXES bin
      NO_DEFAULT_PATH
      )
  if(NOT HIP_HCC_EXECUTABLE)
      # Now search in default paths
      find_program(HIP_HCC_EXECUTABLE hcc)
  endif()
  mark_as_advanced(HIP_HCC_EXECUTABLE)
  set(CMAKE_CXX_COMPILER ${HIP_HCC_EXECUTABLE})
else()
  message(FATAL_ERROR "HIP_PLATFORM must be 'hcc/nvcc' (AMD ROCm platform).")
endif()
