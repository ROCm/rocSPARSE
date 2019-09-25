# ########################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc.
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
  src/level2/rocsparse_csrsv.cpp
  src/level2/rocsparse_ellmv.cpp
  src/level2/rocsparse_hybmv.cpp

# Level3
  src/level3/rocsparse_csrmm.cpp

# Extra
  src/extra/rocsparse_csrgemm.cpp

# Preconditioner
  src/precond/rocsparse_csrilu0.cpp

# Conversion
  src/conversion/rocsparse_csr2coo.cpp
  src/conversion/rocsparse_csr2csc.cpp
  src/conversion/rocsparse_csr2ell.cpp
  src/conversion/rocsparse_csr2hyb.cpp
  src/conversion/rocsparse_coo2csr.cpp
  src/conversion/rocsparse_ell2csr.cpp
  src/conversion/rocsparse_identity.cpp
  src/conversion/rocsparse_csrsort.cpp
  src/conversion/rocsparse_cscsort.cpp
  src/conversion/rocsparse_coosort.cpp
)
