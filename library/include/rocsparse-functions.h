/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/*!\file
 * \brief rocsparse-functions.h provides Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3, using HIP optimized for AMD HCC-based GPU hardware.
 *  This library can also run on CUDA-based NVIDIA GPUs.
*/

#pragma once
#ifndef _ROCSPARSE_FUNCTIONS_H_
#define _ROCSPARSE_FUNCTIONS_H_

#include "rocsparse-types.h"
#include "rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */

/*! \brief SPARSE Level 1 API

    \details
    axpyi multiplies the sparse vector x with scalar alpha and adds the
    result to the dense vector y

        y := y + alpha * x

    @param[in]
    handle      rocsparse_handle.
                handle to the rocsparse library context queue.
    @param[in]
    nnz         number of non-zero entries of x.
    @param[in]
    alpha       scalar alpha.
    @param[in]
    x_val       array of nnz values.
    @param[in]
    x_ind       array of nnz elements containing the indices of the non-zero
                values of x.
    @param[inout]
    y           array of values in dense format.
    @param[in]
    idx_base    rocsparse_index_base_zero or rocsparse_index_base_one.

    ********************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_saxpyi(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  const float* alpha,
                                  const float* x_val,
                                  const rocsparse_int* x_ind,
                                  float* y,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_daxpyi(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  const double* alpha,
                                  const double* x_val,
                                  const rocsparse_int* x_ind,
                                  double* y,
                                  rocsparse_index_base idx_base);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_caxpyi(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_float_complex* x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_float_complex* y,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zaxpyi(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_double_complex* x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_double_complex* y,
                                  rocsparse_index_base idx_base);
*/

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */

/*! \brief SPARSE Level 2 API

    \details
    coomv multiplies the dense vector x with scalar alpha and sparse m x n
    matrix A that is defined in COO storage format and adds the result to the
    dense vector y that is multiplied by beta

        y := alpha * op(A) * x + beta * y

    @param[in]
    handle      rocsparse_handle.
                handle to the rocsparse library context queue.
    @param[in]
    trans       operation type of A.
    @param[in]
    m           number of rows of A.
    @param[in]
    n           number of columns of A.
    @param[in]
    nnz         number of non-zero entries of A.
    @param[in]
    alpha       scalar alpha.
    @param[in]
    descr       descriptor of A.
    @param[in]
    coo_val     array of nnz elements of A.
    @param[in]
    coo_row_ind array of nnz elements containing the row indices of A.
    @param[in]
    coo_col_ind array of nnz elements containing the column indices of A.
    @param[in]
    x           array of n elements (op(A) = A) or m elements (op(A) = A^T or
                op(A) = A^H).
    @param[in]
    beta        scalar beta.
    @param[inout]
    y           array of m elements (op(A) = A) or n elements (op(A) = A^T or
                op(A) = A^H).

    ********************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scoomv(rocsparse_handle handle,
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
                                  float* y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcoomv(rocsparse_handle handle,
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
                                  double* y);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccoomv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_float_complex* coo_val,
                                  const rocsparse_int* coo_row_ind,
                                  const rocsparse_int* coo_col_ind,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex* y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcoomv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_double_complex* coo_val,
                                  const rocsparse_int* coo_row_ind,
                                  const rocsparse_int* coo_col_ind,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex* y);
*/

/*! \brief SPARSE Level 2 API

    \details
    csrmv multiplies the dense vector x with scalar alpha and sparse m x n
    matrix A that is defined in CSR storage format and adds the result to the
    dense vector y that is multiplied by beta

        y := alpha * op(A) * x + beta * y

    @param[in]
    handle      rocsparse_handle.
                handle to the rocsparse library context queue.
    @param[in]
    trans       operation type of A.
    @param[in]
    m           number of rows of A.
    @param[in]
    n           number of columns of A.
    @param[in]
    nnz         number of non-zero entries of A.
    @param[in]
    alpha       scalar alpha.
    @param[in]
    descr       descriptor of A.
    @param[in]
    csr_val     array of nnz elements of A.
    @param[in]
    csr_row_ptr array of m+1 elements that point to the start
                of every row of A.
    @param[in]
    csr_col_ind array of nnz elements containing the column indices of A.
    @param[in]
    x           array of n elements (op(A) = A) or m elements (op(A) = A^T or
                op(A) = A^H).
    @param[in]
    beta        scalar beta.
    @param[inout]
    y           array of m elements (op(A) = A) or n elements (op(A) = A^T or
                op(A) = A^H).

    ********************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int nnz,
                                  const float* alpha,
                                  const rocsparse_mat_descr descr,
                                  const float* csr_val,
                                  const rocsparse_int* csr_row_ptr,
                                  const rocsparse_int* csr_col_ind,
                                  const float* x,
                                  const float* beta,
                                  float* y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int nnz,
                                  const double* alpha,
                                  const rocsparse_mat_descr descr,
                                  const double* csr_val,
                                  const rocsparse_int* csr_row_ptr,
                                  const rocsparse_int* csr_col_ind,
                                  const double* x,
                                  const double* beta,
                                  double* y);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_float_complex* csr_val,
                                  const rocsparse_int* csr_row_ptr,
                                  const rocsparse_int* csr_col_ind,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex* y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_double_complex* csr_val,
                                  const rocsparse_int* csr_row_ptr,
                                  const rocsparse_int* csr_col_ind,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex* y);
*/

/*! \brief SPARSE Level 2 API

    \details
    hybmv  multiplies the dense vector x[i] with scalar alpha and sparse m x n
    matrix A that is defined in HYB storage format and add the result to y[i]
    that is multiplied by beta, for  i = 1 , … , n

        y := alpha * op(A) * x + beta * y,

    @param[in]
    handle      rocsparse_handle.
                handle to the rocsparse library context queue.
    @param[in]
    trans       operation type of A.
    @param[in]
    alpha       scalar alpha.
    @param[in]
    descr       descriptor of A.
    @param[in]
    hyb         matrix in HYB storage format.
    @param[in]
    x           array of n elements (op(A) = A) or m elements (op(A) = A^T or
                op(A) = A^H).
    @param[in]
    beta        scalar beta.
    @param[inout]
    y           array of m elements (op(A) = A) or n elements (op(A) = A^T or
                op(A) = A^H).

    ********************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_shybmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  const float* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat hyb,
                                  const float* x,
                                  const float* beta,
                                  float* y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dhybmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  const double* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat hyb,
                                  const double* x,
                                  const double* beta,
                                  double* y);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_shybmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat hyb,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex* y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dhybmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat hyb,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex* y);
*/
/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */

/*
 * ===========================================================================
 *    Sparse Format Conversions
 * ===========================================================================
 */

/*! \brief SPARSE Format Conversions API

    \details
    csr2coo converts the CSR array containing the row offset pointers into a
    COO array of row indices.

    @param[in]
    handle      rocsparse_handle.
                handle to the rocsparse library context queue.
    @param[in]
    csr_row_ptr array of m+1 elements that point to the start of every row
                of A.
    @param[in]
    nnz         number of non-zero entries of the sparse matrix A.
    @param[in]
    m           number of rows of the sparse matrix A.
    @param[out]
    coo_row_ind array of nnz elements containing the row indices of A.
    @param[in]
    idx_base    rocsparse_index_base_zero or rocsparse_index_base_one.

    ********************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2coo(rocsparse_handle handle,
                                   const rocsparse_int* csr_row_ptr,
                                   rocsparse_int nnz,
                                   rocsparse_int m,
                                   rocsparse_int* coo_row_ind,
                                   rocsparse_index_base idx_base);

/*! \brief SPARSE Format Conversions API

    \details
    coo2csr converts the COO array containing the row indices into a
    CSR array of row offset pointers.

    @param[in]
    handle      rocsparse_handle.
                handle to the rocsparse library context queue.
    @param[in]
    coo_row_ind array of nnz elements containing the row indices of A.
    @param[in]
    nnz         number of non-zero entries of the sparse matrix A.
    @param[in]
    m           number of rows of the sparse matrix A.
    @param[out]
    csr_row_ptr array of m+1 elements that point to the start of every row
                of A.
    @param[in]
    idx_base    rocsparse_index_base_zero or rocsparse_index_base_one.

    ********************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo2csr(rocsparse_handle handle,
                                   const rocsparse_int* coo_row_ind,
                                   rocsparse_int nnz,
                                   rocsparse_int m,
                                   rocsparse_int* csr_row_ptr,
                                   rocsparse_index_base idx_base);

/*! \brief SPARSE Format Conversions API

    \details
    csr2hyb converts a CSR matrix into a HYB matrix.

    @param[in]
    handle          rocsparse_handle.
                    handle to the rocsparse library context queue.
    @param[in]
    m               number of rows of A.
    @param[in]
    n               number of columns of A.
    @param[in]
    descr           descriptor of A.
    @param[in]
    csr_val         array of nnz elements of A.
    @param[in]
    csr_row_ptr     array of m+1 elements that point to the start
                    of every row of A.
    @param[in]
    csr_col_ind     array of nnz elements containing the column indices
                    of A.
    @param[out]
    hyb             sparse matrix in HYB format
    @param[in]
    user_ell_width  width of the ELL part of the HYB matrix (only
                    required if
                    partition_type == rocsparse_hyb_partition_user)
    @param[in]
    partition_type  partitioning method can be
                    rocsparse_hyb_partition_auto (default)
                    rocsparse_hyb_partition_user
                    rocsparse_hyb_partition_max

    ********************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2hyb(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr descr,
                                    const float* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    rocsparse_hyb_mat hyb,
                                    rocsparse_int user_ell_width,
                                    rocsparse_hyb_partition partition_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2hyb(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr descr,
                                    const double* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    rocsparse_hyb_mat hyb,
                                    rocsparse_int user_ell_width,
                                    rocsparse_hyb_partition partition_type);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2hyb(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_float_complex* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    rocsparse_hyb_mat hyb,
                                    rocsparse_int user_ell_width,
                                    rocsparse_hyb_partition partition_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2hyb(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_double_complex* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    rocsparse_hyb_mat hyb,
                                    rocsparse_int user_ell_width,
                                    rocsparse_hyb_partition partition_type);
*/
#ifdef __cplusplus
}
#endif

#endif // _ROCSPARSE_FUNCTIONS_H_
