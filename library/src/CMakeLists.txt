# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

# rocSPARSE source
set(rocsparse_source
  src/handle.cpp
  src/status.cpp
  src/rocsparse_auxiliary.cpp
  src/level1/rocsparse_axpyi.cpp
  src/level2/rocsparse_csrmv.cpp
  src/level2/rocsparse_hybmv.cpp
  src/conversion/rocsparse_csr2hyb.cpp
)
