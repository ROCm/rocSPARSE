# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Function to add rocsparse benchmarks
function(add_rocsparse_benchmark BENCHMARK_SOURCE)
  get_filename_component(BENCHMARK_TARGET ${BENCHMARK_SOURCE} NAME_WE)
  add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})
  if(HIP_PLATFORM STREQUAL "hcc")
    target_link_libraries(${BENCHMARK_TARGET}
      PRIVATE
        rocsparse
        hip::hip_hcc
        benchmark::benchmark
    )
    foreach(amdgpu_target ${AMDGPU_TARGETS})
      target_link_libraries(${BENCHMARK_TARGET}
        PRIVATE
          --amdgpu-target=${amdgpu_target}
      )
    endforeach()
  endif()
  set_target_properties(${BENCHMARK_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
  )
endfunction()

# Benchmarks
add_rocsparse_benchmark(benchmark_spmv.cpp)
