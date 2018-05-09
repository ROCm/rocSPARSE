/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.hpp"

#include <rocsparse.h>

namespace rocsparse {

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const float *alpha,
                                 const float *x_val,
                                 const rocsparse_int *x_ind,
                                 float *y,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_saxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const double *alpha,
                                 const double *x_val,
                                 const rocsparse_int *x_ind,
                                 double *y,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_daxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_csrmv(rocsparse_handle handle,
                                 rocsparse_operation trans, 
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int nnz,
                                 const float *alpha,
                                 const rocsparse_mat_descr descr,
                                 const float *csr_val,
                                 const rocsparse_int *csr_row_ptr,
                                 const rocsparse_int *csr_col_ind,
                                 const float *x,
                                 const float *beta,
                                 float *y)
{
    return rocsparse_scsrmv(handle, trans, m, n, nnz, alpha,
                            descr, csr_val, csr_row_ptr, csr_col_ind,
                            x, beta, y);
}

template <>
rocsparse_status rocsparse_csrmv(rocsparse_handle handle,
                                 rocsparse_operation trans, 
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int nnz,
                                 const double *alpha,
                                 const rocsparse_mat_descr descr,
                                 const double *csr_val,
                                 const rocsparse_int *csr_row_ptr,
                                 const rocsparse_int *csr_col_ind,
                                 const double *x,
                                 const double *beta,
                                 double *y)
{
    return rocsparse_dcsrmv(handle, trans, m, n, nnz, alpha,
                            descr, csr_val, csr_row_ptr, csr_col_ind,
                            x, beta, y);
}

template <>
rocsparse_status rocsparse_hybmv(rocsparse_handle handle,
                                 rocsparse_operation trans, 
                                 const float *alpha,
                                 const rocsparse_mat_descr descr,
                                 const rocsparse_hyb_mat hyb,
                                 const float *x,
                                 const float *beta,
                                 float *y)
{
    return rocsparse_shybmv(handle, trans, alpha, descr,
                            hyb, x, beta, y);
}

template <>
rocsparse_status rocsparse_hybmv(rocsparse_handle handle,
                                 rocsparse_operation trans, 
                                 const double *alpha,
                                 const rocsparse_mat_descr descr,
                                 const rocsparse_hyb_mat hyb,
                                 const double *x,
                                 const double *beta,
                                 double *y)
{
    return rocsparse_dhybmv(handle, trans, alpha, descr,
                            hyb, x, beta, y);
}

} // namespace rocsparse
