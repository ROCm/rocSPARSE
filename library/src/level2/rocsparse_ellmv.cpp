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
                                             const float* alpha,
                                             const rocsparse_mat_descr descr,
                                             const float* ell_val,
                                             const rocsparse_int* ell_col_ind,
                                             rocsparse_int ell_width,
                                             const float* x,
                                             const float* beta,
                                             float* y)
{
    return rocsparse_ellmv_template<float>(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

extern "C" rocsparse_status rocsparse_dellmv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             const double* alpha,
                                             const rocsparse_mat_descr descr,
                                             const double* ell_val,
                                             const rocsparse_int* ell_col_ind,
                                             rocsparse_int ell_width,
                                             const double* x,
                                             const double* beta,
                                             double* y)
{
    return rocsparse_ellmv_template<double>(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}
