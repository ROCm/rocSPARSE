/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_coomv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scoomv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int nnz,
                                             const float* alpha,
                                             const rocsparse_mat_descr descr,
                                             const float* coo_val,
                                             const rocsparse_int* coo_row_ind,
                                             const rocsparse_int* coo_col_ind,
                                             const float* x,
                                             const float* beta,
                                             float* y)
{
    return rocsparse_coomv_template<float>(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

extern "C" rocsparse_status rocsparse_dcoomv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int nnz,
                                             const double* alpha,
                                             const rocsparse_mat_descr descr,
                                             const double* coo_val,
                                             const rocsparse_int* coo_row_ind,
                                             const rocsparse_int* coo_col_ind,
                                             const double* x,
                                             const double* beta,
                                             double* y)
{
    return rocsparse_coomv_template<double>(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}
