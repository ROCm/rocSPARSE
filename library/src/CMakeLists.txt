# ########################################################################
# Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

# rocSPARSE Fortran source
set(rocsparse_fortran_source
  src/rocsparse_enums.f
  src/rocsparse.f
)

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
  src/level1/rocsparse_axpby.cpp
  src/level1/rocsparse_gather.cpp
  src/level1/rocsparse_scatter.cpp
  src/level1/rocsparse_rot.cpp
  src/level1/rocsparse_spvv.cpp

# Level2
  src/level2/rocsparse_bsrmv.cpp
  src/level2/rocsparse_bsrmv_spzl_2x2.cpp
  src/level2/rocsparse_bsrmv_spzl_3x3.cpp
  src/level2/rocsparse_bsrmv_spzl_4x4.cpp
  src/level2/rocsparse_bsrmv_spzl_5x5.cpp
  src/level2/rocsparse_bsrmv_spzl_8x8.cpp
  src/level2/rocsparse_bsrmv_spzl_16x16.cpp
  src/level2/rocsparse_bsrmv_spzl_17_32.cpp
  src/level2/rocsparse_bsrmv_spzl_general.cpp
  src/level2/rocsparse_bsrsv.cpp
  src/level2/rocsparse_bsrsv_analysis.cpp
  src/level2/rocsparse_bsrsv_buffer_size.cpp
  src/level2/rocsparse_bsrsv_solve.cpp
  src/level2/rocsparse_coomv.cpp
  src/level2/rocsparse_coomv_aos.cpp
  src/level2/rocsparse_csrmv.cpp
  src/level2/rocsparse_csrsv.cpp
  src/level2/rocsparse_csrsv_analysis.cpp
  src/level2/rocsparse_csrsv_buffer_size.cpp
  src/level2/rocsparse_csrsv_solve.cpp
  src/level2/rocsparse_ellmv.cpp
  src/level2/rocsparse_hybmv.cpp
  src/level2/rocsparse_spmv.cpp
  src/level2/rocsparse_gebsrmv.cpp
  src/level2/rocsparse_gebsrmv_template_row_block_dim_1.cpp
  src/level2/rocsparse_gebsrmv_template_row_block_dim_2.cpp
  src/level2/rocsparse_gebsrmv_template_row_block_dim_3.cpp
  src/level2/rocsparse_gebsrmv_template_row_block_dim_4.cpp
  src/level2/rocsparse_gebsrmv_template_row_block_dim_5_8.cpp
  src/level2/rocsparse_gebsrmv_template_row_block_dim_9_12.cpp
  src/level2/rocsparse_gebsrmv_template_row_block_dim_13_16.cpp
  src/level2/rocsparse_gebsrmv_template_row_block_dim_17_inf.cpp
  src/level2/rocsparse_gemvi.cpp

# Level3
  src/level3/rocsparse_gebsrmm_template_small.cpp
  src/level3/rocsparse_gebsrmm_template_large_ext.cpp
  src/level3/rocsparse_gebsrmm_template_general.cpp
  src/level3/rocsparse_gebsrmm.cpp
  src/level3/rocsparse_bsrmm_template_small.cpp
  src/level3/rocsparse_bsrmm_template_large_ext.cpp
  src/level3/rocsparse_bsrmm_template_general.cpp
  src/level3/rocsparse_bsrmm.cpp
  src/level3/rocsparse_csrmm.cpp
  src/level3/rocsparse_coomm.cpp
  src/level3/rocsparse_spmm.cpp
  src/level3/rocsparse_csrsm.cpp
  src/level3/rocsparse_gemmi.cpp
  src/level3/rocsparse_sddmm.cpp
  src/level3/rocsparse_sddmm_coo.cpp
  src/level3/rocsparse_sddmm_coo_aos.cpp
  src/level3/rocsparse_sddmm_csr.cpp
  src/level3/rocsparse_sddmm_csc.cpp
  src/level3/rocsparse_sddmm_ell.cpp

# Extra
  src/extra/rocsparse_csrgeam.cpp
  src/extra/rocsparse_csrgemm.cpp
  src/extra/rocsparse_csrgemm_nnz.cpp
  src/extra/rocsparse_spgemm.cpp

# Preconditioner
  src/precond/rocsparse_bsric0.cpp
  src/precond/rocsparse_bsrilu0.cpp
  src/precond/rocsparse_csric0.cpp
  src/precond/rocsparse_csrilu0.cpp
  src/precond/rocsparse_gtsv_no_pivot.cpp

# Conversion
  src/conversion/rocsparse_nnz.cpp
  src/conversion/rocsparse_dense2csr.cpp
  src/conversion/rocsparse_dense2coo.cpp
  src/conversion/rocsparse_prune_dense2csr.cpp
  src/conversion/rocsparse_prune_dense2csr_by_percentage.cpp
  src/conversion/rocsparse_dense2csc.cpp
  src/conversion/rocsparse_csr2dense.cpp
  src/conversion/rocsparse_csc2dense.cpp
  src/conversion/rocsparse_coo2dense.cpp
  src/conversion/rocsparse_nnz_compress.cpp
  src/conversion/rocsparse_csr2coo.cpp
  src/conversion/rocsparse_csr2csc.cpp
  src/conversion/rocsparse_gebsr2gebsc.cpp
  src/conversion/rocsparse_csr2bsr.cpp
  src/conversion/rocsparse_csr2gebsr.cpp
  src/conversion/rocsparse_csr2ell.cpp
  src/conversion/rocsparse_csr2hyb.cpp
  src/conversion/rocsparse_csr2csr_compress.cpp
  src/conversion/rocsparse_prune_csr2csr.cpp
  src/conversion/rocsparse_prune_csr2csr_by_percentage.cpp
  src/conversion/rocsparse_coo2csr.cpp
  src/conversion/rocsparse_ell2csr.cpp
  src/conversion/rocsparse_hyb2csr.cpp
  src/conversion/rocsparse_bsr2csr.cpp
  src/conversion/rocsparse_gebsr2csr.cpp
  src/conversion/rocsparse_gebsr2gebsr.cpp
  src/conversion/rocsparse_identity.cpp
  src/conversion/rocsparse_csrsort.cpp
  src/conversion/rocsparse_cscsort.cpp
  src/conversion/rocsparse_coosort.cpp
  src/conversion/rocsparse_sparse_to_dense.cpp
  src/conversion/rocsparse_dense_to_sparse.cpp
)
