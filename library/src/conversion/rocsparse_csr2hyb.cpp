/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_csr2hyb.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsr2hyb(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               const rocsparse_mat_descr descr,
                                               const float* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_hyb_mat hyb,
                                               rocsparse_int user_ell_width,
                                               rocsparse_hyb_partition partition_type)
{
    return rocsparse_csr2hyb_template(handle,
                                      m,
                                      n,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      hyb,
                                      user_ell_width,
                                      partition_type);
}

extern "C" rocsparse_status rocsparse_dcsr2hyb(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               const rocsparse_mat_descr descr,
                                               const double* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_hyb_mat hyb,
                                               rocsparse_int user_ell_width,
                                               rocsparse_hyb_partition partition_type)
{
    return rocsparse_csr2hyb_template(handle,
                                      m,
                                      n,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      hyb,
                                      user_ell_width,
                                      partition_type);
}
