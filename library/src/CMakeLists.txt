# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

# rocSPARSE source
set(rocsparse_source
  src/handle.cpp
  src/status.cpp
  src/rocsparse_auxiliary.cpp

# Level1
  src/level1/rocsparse_axpyi.cpp
  src/level1/rocsparse_doti.cpp
  src/level1/rocsparse_dotci.cpp
  src/level1/rocsparse_gthr.cpp
  src/level1/rocsparse_gthrz.cpp
  src/level1/rocsparse_roti.cpp
  src/level1/rocsparse_sctr.cpp

# Level2
  src/level2/rocsparse_coomv.cpp
  src/level2/rocsparse_csrmv.cpp
  src/level2/rocsparse_csrmv_adaptive.cpp
  src/level2/rocsparse_ellmv.cpp
  src/level2/rocsparse_hybmv.cpp

# Conversion
  src/conversion/rocsparse_csr2coo.cpp
  src/conversion/rocsparse_csr2csc.cpp
  src/conversion/rocsparse_csr2ell.cpp
  src/conversion/rocsparse_csr2hyb.cpp
  src/conversion/rocsparse_coo2csr.cpp
  src/conversion/rocsparse_identity.cpp
  src/conversion/rocsparse_csrsort.cpp
  src/conversion/rocsparse_coosort.cpp
)
