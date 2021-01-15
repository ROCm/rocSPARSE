# Change Log for rocSPARSE

Full documentation for rocSPARSE is available at [rocsparse.readthedocs.io](https://rocsparse.readthedocs.io/en/latest/).

## [(Unreleased) rocSPARSE 1.19.4 for ROCm 4.1.0]
### Added
- gebsrmm
- gebsrmv
- gebsrsv
- coo2dense and dense2coo
- generic API including axpby, gather, scatter, rot, spvv, spmv, spgemm, sparsetodense, densetosparse
- support for mixed indexing types in matrix formats


## [rocSPARSE 1.18.4 for ROCm 4.0.0]
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

## [rocSPARSE 1.17.6 for ROCm 3.9]
### Added
- prune_csr2csr, prune_dense2csr_percentage and prune_csr2csr_percentage added
- bsrilu0 added
- csrilu0_numeric_boost functionality added
### Known Issues
- none

## [rocSPARSE 1.16.1 for ROCm 3.8]
### Added
- bsric0 added.
### Known Issues
- none

## [rocSPARSE 1.14.3 for ROCm 3.7]
### Added
- Fortran bindings
- CentOS 6 support.
- Triangular solve for BSR format (bsrsv)
- Default compiler switched to hipclang
### Optimized
- bsrmv
### Known Issues
- none

## [rocSPARSE 1.14.3 for ROCm 3.6]
### Added
- Fortran bindings
- CentOS 6 support.
- Triangular solve for BSR format (bsrsv)
- Default compiler switched to hipclang
### Optimized
- bsrmv routine
### Known Issues
- none

## [rocSPARSE 1.12.10 for ROCm 3.5]
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
