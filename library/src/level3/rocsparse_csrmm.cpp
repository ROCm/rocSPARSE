/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_csrmm.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrmm(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int k,
                                             rocsparse_int nnz,
                                             const float* alpha,
                                             const rocsparse_mat_descr descr,
                                             const float* csr_val,
                                             const rocsparse_int* csr_row_ptr,
                                             const rocsparse_int* csr_col_ind,
                                             const float* B,
                                             rocsparse_int ldb,
                                             const float* beta,
                                             float* C,
                                             rocsparse_int ldc)
{
    return rocsparse_csrmm_template<float>(
        handle, trans, m, n, k, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, beta, C, ldc);
}

extern "C" rocsparse_status rocsparse_dcsrmm(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int k,
                                             rocsparse_int nnz,
                                             const double* alpha,
                                             const rocsparse_mat_descr descr,
                                             const double* csr_val,
                                             const rocsparse_int* csr_row_ptr,
                                             const rocsparse_int* csr_col_ind,
                                             const double* B,
                                             rocsparse_int ldb,
                                             const double* beta,
                                             double* C,
                                             rocsparse_int ldc)
{
    return rocsparse_csrmm_template<double>(
        handle, trans, m, n, k, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, beta, C, ldc);
}
