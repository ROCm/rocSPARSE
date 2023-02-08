# Change Log for rocSPARSE

Full documentation for rocSPARSE is available at [rocsparse.readthedocs.io](https://rocsparse.readthedocs.io/en/latest/).

## rocSPARSE 2.6.0
### Added
- Added rocsparse_inverse_permutation
- Added mixed precisions for SpVV
- Added uniform int8 precision for Gather and Scatter
### Changed
- Removed old deprecated rocsparse_spmv, deprecated current rocsparse_spmv_ex, and added new rocsparse_spmv routine
- Removed old deprecated rocsparse_xbsrmv routines, deprecated current rocsparse_xbsrmv_ex routines, and added new rocsparse_xbsrmv routines
- Removed old deprecated rocsparse_spmm_ex routine
- doti, dotci, spvv, and csr2ell now require calling hipStreamSynchronize after when using host pointer mode
### Improved
- Optimization to doti routine
- Fixed a bug in csrsm and bsrsm
### Known Issues
- none

## rocSPARSE 2.5.0 for ROCm 5.5.0
### Added
- Added bsrgemm and spgemm for BSR format
- Added bsrgeam
- Added build support for Navi32
- Added experimental hipGraph support for some rocSPARSE routines
- Added csritsv, spitsv csr iterative triangular solve
- Added mixed precisions for SpMV
- Added batched SpMM for transpose A in COO format with atomic atomic algorithm
### Improved
- Optimization to csr2bsr
- Optimization to csr2csr_compress
- Optimization to csr2coo
- Optimization to gebsr2csr
- Optimization to csr2gebsr
- Fixes to documentation
- Fixes a bug in COO SpMV gridsize
- Fixes a bug in SpMM gridsize when using very large matrices
### Known Issues
- none

## rocSPARSE 2.4.0 for ROCm 5.4.0
### Added
- Added rocsparse_spmv_ex routine
- Added rocsparse_bsrmv_ex_analysis and rocsparse_bsrmv_ex routines
- Added csritilu0 routine
- Added build support for Navi31 and Navi 33
### Improved
- Optimization to segmented algorithm for COO SpMV by performing analysis
- Improve performance when generating random matrices.
- Fixed bug in ellmv
- Optimized bsr2csr routine
- Fixed integer overflow bugs

## rocSPARSE 2.3.2 for ROCm 5.3.0
### Added
- Transpose A for SpMM COO format
- Added matrix checker routines for verifying matrix data
- Added atomic algorithm for COO SpMV
- Added bsrpad routine
### Improved
- Fixed a bug in csrilu0 which could cause a deadlock
- Fixed a bug where asynchronous memcpy would use wrong stream
- Fixed potential size overflows

## rocSPARSE 2.2.0 for ROCm 5.2.0
### Added
- Batched SpMM for CSR, CSC, and COO formats.
- Packages for test and benchmark executables on all supported OSes using CPack.
- Clients file importers and exporters.
### Improved
- Clients code size reduction.
- Clients error handling.
- Clients benchmarking for performance tracking.
### Changed
- Test adjustments due to roundoff errors.
- Fixing API calls compatibility with rocPRIM.
### Known Issues
- none

## rocSPARSE 2.1.0 for ROCm 5.1.0
### Added
- gtsv_interleaved_batch
- gpsv_interleaved_batch
- SpGEMM_reuse
- Allow copying of mat info struct
### Improved
- Optimization for SDDMM
- Allow unsorted matrices in csrgemm multipass algorithm
### Known Issues
- none

## rocSPARSE 2.0.0 for ROCm 5.0.0
### Added
- csrmv, coomv, ellmv, hybmv for (conjugate) transposed matrices
- csrmv for symmetric matrices
- Packages for test and benchmark executables on all supported OSes using CPack.
### Changed
- spmm\_ex is now deprecated and will be removed in the next major release
### Improved
- Optimization for gtsv

## rocSPARSE 1.22.2 for ROCm 4.5.0
### Added
- Triangular solve for multiple right-hand sides using BSR format
- SpMV for BSRX format
- SpMM in CSR format enhanced to work with transposed A
- Matrix coloring for CSR matrices
- Added batched tridiagonal solve (gtsv\_strided\_batch)
- SpMM for BLOCKED ELL format
- Generic routines for SpSV and SpSM
- Enabling beta support for Windows 10
- Additional atomic based algorithms for SpMM in COO format
- Extended version of SpMM
- Additional algorithm for SpMM in CSR format
- Added (conjugate) transpose support for csrmv and SpMV (CSR) routines
### Changed
- Packaging split into a runtime package called rocsparse and a development package called rocsparse-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.
### Improved
- Fixed a bug with gemvi on Navi21
- Fixed a bug with adaptive csrmv
- Optimization for pivot based gtsv
### Known Issues
- none

## rocSPARSE 1.20.2 for ROCm 4.3.0
### Added
- (batched) tridiagonal solver with and without pivoting
- dense matrix sparse vector multiplication (gemvi)
- support for gfx90a
- sampled dense-dense matrix multiplication (sddmm)
### Improved
- client matrix download mechanism
- boost dependency in clients removed
### Known Issues
- none

## rocSPARSE 1.19.5 for ROCm 4.2.0
### Added
- SpMM (CSR, COO)
- Code coverage analysis
### Improved
- Install script
- Level 2/3 unit tests
- rocsparse-bench does not depend on boost anymore
### Known Issues
- none

## rocSPARSE 1.19.4 for ROCm 4.1.0
### Added
- gebsrmm
- gebsrmv
- gebsrsv
- coo2dense and dense2coo
- generic API including axpby, gather, scatter, rot, spvv, spmv, spgemm, sparsetodense, densetosparse
- support for mixed indexing types in matrix formats

## rocSPARSE 1.18.4 for ROCm 4.0.0
### Added
- Add changelog
- csr2gebsr
- gebsr2gebsc
- gebsr2gebsr
- treating filename as regular expression for yaml-based testing generation.
### Optimized
- bsric0
### Improved
- gfx1030 adjustment to the latest compiler.
- Replace old xnack off compiler flag with new version.
- Updates to debian package name.
### Documentation
- gebsr2csr

## rocSPARSE 1.17.6 for ROCm 3.9
### Added
- prune_csr2csr, prune_dense2csr_percentage and prune_csr2csr_percentage added
- bsrilu0 added
- csrilu0_numeric_boost functionality added
### Known Issues
- none

## rocSPARSE 1.16.1 for ROCm 3.8
### Added
- bsric0 added.
### Known Issues
- none

## rocSPARSE 1.14.3 for ROCm 3.7
### Added
- Fortran bindings
- CentOS 6 support.
- Triangular solve for BSR format (bsrsv)
- Default compiler switched to hipclang
### Optimized
- bsrmv
### Known Issues
- none

## rocSPARSE 1.14.3 for ROCm 3.6
### Added
- Fortran bindings
- CentOS 6 support.
- Triangular solve for BSR format (bsrsv)
- Default compiler switched to hipclang
### Optimized
- bsrmv routine
### Known Issues
- none

## rocSPARSE 1.12.10 for ROCm 3.5
### Added
- Switched to hipclang as default compiler
- csr2dense, csc2dense, csr2csr_compress, nnz_compress, bsr2csr, csr2bsr, bsrmv, csrgeam
- Triangular solve for BSR format (bsrsv)
- Options for static build
- Examples
### Optimized
- dense2csr, dense2csc
- installation process.
### Known Issues
- none
