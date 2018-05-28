/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_ellmv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sellmv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int nnz,
                                             const float* alpha,
                                             const rocsparse_mat_descr descr,
                                             const float* ell_val,
                                             const rocsparse_int* ell_col_ind,
                                             const float* x,
                                             const float* beta,
                                             float* y)
{
    return rocsparse_ellmv_template<float>(
        handle, trans, m, n, nnz, alpha, descr, ell_val, ell_col_ind, x, beta, y);
}

extern "C" rocsparse_status rocsparse_dellmv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int nnz,
                                             const double* alpha,
                                             const rocsparse_mat_descr descr,
                                             const double* ell_val,
                                             const rocsparse_int* ell_col_ind,
                                             const double* x,
                                             const double* beta,
                                             double* y)
{
    return rocsparse_ellmv_template<double>(
        handle, trans, m, n, nnz, alpha, descr, ell_val, ell_col_ind, x, beta, y);
}
