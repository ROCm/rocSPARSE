# rocSPARSE
rocSPARSE exposes a common interface that provides Basic Linear Algebra Subroutines for sparse computation implemented on top of AMD's Radeon Open Compute [ROCm][] runtime and toolchains. rocSPARSE is created using the [HIP][] programming language and optimized for AMD's latest discrete GPUs.

## Requirements
* Git
* CMake (3.5 or later)
* AMD [ROCm] platform

Optional:
* [GTest][]
  * Required for tests.
  * Use GTEST_ROOT to specify GTest location.
  * If [GTest][] is not found, it will be downloaded and built automatically.
* [Google Benchmark][]
  * Required for benchmarks.
  * If [Google Benchmark][] is not found, it will be downloaded and built automatically.

## Quickstart rocSPARSE build and install

#### CMake
All compiler specifications are determined automatically. The compilation process can be performed by
```
# Clone rocSPARSE using git
git clone https://github.com/ROCmSoftwarePlatform/rocSparse.git

# Go to rocSPARSE directory, create and go to the build directory
cd rocSPARSE; mkdir build; cd build

# Configure rocSPARSE
# Build options:
#   BUILD_TEST        - build tests using [GTest][] (OFF)
#   BUILD_BENCHMARK   - build benchmarks using [Google Benchmark][] (OFF)
#   BUILD_EXAMPLE     - build examples (ON)
#   BUILD_VERBOSE     - verbose output (OFF)
#   BUILD_SHARED_LIBS - build rocSPARSE as a shared library (ON)
cmake -DBUILD_TEST=ON ..

# Build
make

# Install
[sudo] make install
```

## Unit tests
To run unit tests, rocSPARSE has to be built with option -DBUILD_TEST=ON.
```
# Go to rocSPARSE build directory
cd rocSPARSE; cd build

# Run all tests
ctest
```

## Benchmarks
To run benchmarks, rocSPARSE has to be built with option -DBUILD_BENCHMARK=ON.
```
# Go to rocSPARSE build directory
cd rocSPARSE/build

# Run benchmark
./benchmark/benchmark_csrmv <size> <trials> <batch_size>
```

## Support
Please use [the issue tracker][] for bugs and feature requests.

## License
The [license file][] can be found in the main repository.



[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/
[GTest]: https://github.com/google/googletest
[Google Benchmark]: https://github.com/google/benchmark
[the issue tracker]: https://github.com/ROCmSoftwarePlatform/rocSparse/issues
[license file]: ./LICENSE.md
