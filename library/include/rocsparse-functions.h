/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/*!\file
 * \brief rocsparse-functions.h provides Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3, using HIP optimized for AMD HCC-based GPU hardware.
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

/*! \brief Scale a sparse vector and add it to a dense vector.
 *
 *  \details
 *  \p rocsparse_axpyi multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
 *  adds the result to the dense vector \f$y\f$, such that
 *
 *  \f$
 *      y := y + \alpha \cdot x
 *  \f$
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          y[x_ind[i]] = y[x_ind[i]] + alpha * x_val[i];
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of vector \f$x\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  x_val       array of \p nnz elements containing the values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[inout]
 *  y           array of values in dense format.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was not
 *              initialized. <br>
 *              \ref rocsparse_status_invalid_value \p idx_base is invalid. <br>
 *              \ref rocsparse_status_invalid_size \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p alpha, \p x_val, \p x_ind or
 *              \p y pointer is invalid.
 */
///@{
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
///@}

/*! \brief Compute the dot product of a sparse vector with a dense vector.
 *
 *  \details
 *  \p rocsparse_doti computes the dot product of the sparse vector \f$x\f$ with the
 *  dense vector \f$y\f$, such that
 *
 *      \f$\text{result} := y^T x\f$
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          result += x_val[i] * y[x_ind[i]];
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of vector \f$x\f$.
 *  @param[in]
 *  x_val       array of \p nnz values.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[in]
 *  y           array of values in dense format.
 *  @param[out]
 *  result      pointer to the result, can be host or device memory
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was not
 *              initialized. <br>
 *              \ref rocsparse_status_invalid_value \p idx_base is invalid. <br>
 *              \ref rocsparse_status_invalid_size \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p x_val, \p x_ind, \p y or
 *              \p result pointer is invalid. <br>
 *              \ref rocsparse_status_memory_error the buffer for the dot product
 *              reduction could not be allocated. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sdoti(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const float* x_val,
                                 const rocsparse_int* x_ind,
                                 const float* y,
                                 float* result,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ddoti(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const double* x_val,
                                 const rocsparse_int* x_ind,
                                 const double* y,
                                 double* result,
                                 rocsparse_index_base idx_base);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cdoti(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const rocsparse_float_complex* x_val,
                                 const rocsparse_int* x_ind,
                                 const rocsparse_float_complex* y,
                                 rocsparse_float_complex* result,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zdoti(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const rocsparse_double_complex* x_val,
                                 const rocsparse_int* x_ind,
                                 const rocsparse_double_complex* y,
                                 rocsparse_double_complex* result,
                                 rocsparse_index_base idx_base);
*/
///@}

/*! \brief Gather elements from a dense vector and store them into a sparse vector.
 *
 *  \details
 *  \p rocsparse_gthr gathers the elements that are listed in \p x_ind from the dense
 *  vector \f$y\f$ and stores them in the sparse vector \f$x\f$.
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          x_val[i] = y[x_ind[i]];
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *  @param[in]
 *  y           array of values in dense format.
 *  @param[out]
 *  x_val       array of \p nnz elements containing the values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was not
 *              initialized. <br>
 *              \ref rocsparse_status_invalid_value \p idx_base is invalid. <br>
 *              \ref rocsparse_status_invalid_size \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p y, \p x_val or \p x_ind pointer
 *              is invalid.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgthr(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const float* y,
                                 float* x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgthr(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const double* y,
                                 double* x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgthr(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const rocsparse_float_complex* y,
                                 rocsparse_float_complex* x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgthr(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const rocsparse_double_complex* y,
                                 rocsparse_double_complex* x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);
*/
///@}

/*! \brief Gather and zero out elements from a dense vector and store them into a sparse
 *  vector.
 *
 *  \details
 *  \p rocsparse_gthrz gathers the elements that are listed in \p x_ind from the dense
 *  vector \f$y\f$ and stores them in the sparse vector \f$x\f$. The gathered elements
 *  in \f$y\f$ are replaced by zero.
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          x_val[i]    = y[x_ind[i]];
 *          y[x_ind[i]] = 0;
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *  @param[inout]
 *  y           array of values in dense format.
 *  @param[out]
 *  x_val       array of \p nnz elements containing the non-zero values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was not
 *              initialized. <br>
 *              \ref rocsparse_status_invalid_value \p idx_base is invalid. <br>
 *              \ref rocsparse_status_invalid_size \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p y, \p x_val or \p x_ind pointer
 *              is invalid.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgthrz(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  float* y,
                                  float* x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgthrz(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  double* y,
                                  double* x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_index_base idx_base);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgthrz(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  rocsparse_float_complex* y,
                                  rocsparse_float_complex* x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgthrz(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  rocsparse_double_complex* y,
                                  rocsparse_double_complex* x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_index_base idx_base);
*/
///@}

/*! \brief Apply Givens rotation to a dense and a sparse vector.
 *
 *  \details
 *  \p rocsparse_roti applies the Givens rotation matrix \f$G\f$ to the sparse vector
 *  \f$x\f$ and the dense vector \f$y\f$, where
 *
 *      \f$G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}\f$
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          x_tmp = x_val[i];
 *          y_tmp = y[x_ind[i]];
 *
 *          x_val[i]    = c * x_tmp + s * y_tmp;
 *          y[x_ind[i]] = c * y_tmp - s * x_tmp;
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *  @param[inout]
 *  x_val       array of \p nnz elements containing the non-zero values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[inout]
 *  y           array of values in dense format.
 *  @param[in]
 *  c           pointer to the cosine element of \f$G\f$, can be on host or device.
 *  @param[in]
 *  s           pointer to the sine element of \f$G\f$, can be on host or device.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was not
 *              initialized. <br>
 *              \ref rocsparse_status_invalid_value \p idx_base is invalid. <br>
 *              \ref rocsparse_status_invalid_size \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p c, \p s, \p x_val, \p x_ind or
 *              \p y pointer is invalid.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sroti(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 float* x_val,
                                 const rocsparse_int* x_ind,
                                 float* y,
                                 const float* c,
                                 const float* s,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_droti(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 double* x_val,
                                 const rocsparse_int* x_ind,
                                 double* y,
                                 const double* c,
                                 const double* s,
                                 rocsparse_index_base idx_base);
///@}

/*! \brief Scatter elements from a dense vector across a sparse vector.
 *
 *  \details
 *  \p rocsparse_sctr scatters the elements that are listed in \p x_ind from the sparse
 *  vector \f$x\f$ into the dense vector \f$y\f$. Indices of \f$y\f$ that are not listed
 *  in \p x_ind remain unchanged.
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          y[x_ind[i]] = x_val[i];
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *  @param[in]
 *  x_val       array of \p nnz elements containing the non-zero values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of x.
 *  @param[inout]
 *  y           array of values in dense format.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was not
 *              initialized. <br>
 *              \ref rocsparse_status_invalid_value \p idx_base is invalid. <br>
 *              \ref rocsparse_status_invalid_size \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p x_val, \p x_ind or \p y pointer
 *              is invalid.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ssctr(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const float* x_val,
                                 const rocsparse_int* x_ind,
                                 float* y,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dsctr(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const double* x_val,
                                 const rocsparse_int* x_ind,
                                 double* y,
                                 rocsparse_index_base idx_base);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csctr(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const rocsparse_float_complex* x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_float_complex* y,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zsctr(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const rocsparse_double_complex* x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_double_complex* y,
                                 rocsparse_index_base idx_base);
*/
///@}

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */

/*! \brief Sparse matrix vector multiplication using \p COO storage format
 *
 *  \details
 *  \p rocsparse_coomv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in \p COO storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  
 *      \f$y := \alpha \cdot op(A) \cdot x + \beta \cdot y\f$, with
 *
 *      \f$
 *          op(A) = \left\{
 *          \begin{array}{ll}
 *              A,   & \text{if trans == rocsparse_operation_none} \\
 *              A^T, & \text{if trans == rocsparse_operation_transpose} \\
 *              A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
 *          \end{array}
 *          \right.
 *      \f$
 *
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  The \p COO matrix has to be sorted by row indices. This can be achieved by using
 *  rocsparse_coosort_by_row().
 *
 *  \code{.c}
 *      for(i = 0; i < m; ++i)
 *      {
 *          y[i] = beta * y[i];
 *      }
 *
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          y[coo_row_ind[i]] += alpha * coo_val[i] * x[coo_col_ind[i]];
 *      }
 *  \endcode
 *  
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse \p COO matrix.
 *  @param[in]
 *  n           number of columns of the sparse \p COO matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse \p COO matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse \p COO matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  coo_val     array of \p nnz elements of the sparse \p COO matrix.
 *  @param[in]
 *  coo_row_ind array of \p nnz elements containing the row indices of the sparse \p COO
 *              matrix.
 *  @param[in]
 *  coo_col_ind array of \p nnz elements containing the column indices of the sparse
 *              \p COO matrix.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) = A\f$) or \p m elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) = A\f$) or \p n elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p descr, \p alpha, \p coo_val,
 *              \p coo_row_ind, \p coo_col_ind, \p x, \p beta or \p y pointer is
 *              invalid. <br>
 *              \ref rocsparse_status_arch_mismatch the device is not supported. <br>
 *              \ref rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
///@{
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
///@}

/*! \brief Sparse matrix vector multiplication using \p CSR storage format
 *
 *  \details
 *  \p rocsparse_csrmv_analysis performs the analysis step for rocsparse_scsrmv() and
 *  rocsparse_dcsrmv(). It is expected that this function will be executed only once for
 *  a given matrix and particular operation type. Note that if the matrix sparsity
 *  pattern changes, the gathered information will become invalid. The gathered analysis
 *  meta data can be cleared by rocsparse_csrmv_clear().
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse \p CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse \p CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse \p CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse \p CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              \p CSR matrix.
 *  @param[out]
 *  info        structure that holds the information collected during
 *              the analysis step.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p descr, \p csr_row_ptr,
 *              \p csr_col_ind or \p info pointer is invalid. <br>
 *              \ref rocsparse_status_memory_error the buffer for the gathered
 *              information could not be allocated. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred. <br>
 *              \ref rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrmv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int nnz,
                                          const rocsparse_mat_descr descr,
                                          const rocsparse_int* csr_row_ptr,
                                          const rocsparse_int* csr_col_ind,
                                          rocsparse_mat_info info);

/*! \brief Sparse matrix vector multiplication using \p CSR storage format
 *
 *  \details
 *  \p rocsparse_csrmv_clear deallocates all memory that was allocated by
 *  rocsparse_csrmv_analysis(). This is especially useful, if memory is an issue and the
 *  analysis data is not required anymore for further computation, e.g. when switching
 *  to another sparse matrix format. Calling \p rocsparse_csrmv_clear is optional. All
 *  allocated resources will be cleared, when the opaque \ref rocsparse_mat_info struct is
 *  destroyed using rocsparse_destroy_mat_info().
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[inout]
 *  info        structure that holds the information collected during analysis step.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_pointer \p info pointer is invalid. <br>
 *              \ref rocsparse_status_memory_error the buffer for the gathered
 *              information could not be deallocated. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred.
 * */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrmv_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \brief Sparse matrix vector multiplication using \p CSR storage format
 *
 *  \details
 *  \p rocsparse_csrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in \p CSR storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  
 *      \f$y := \alpha \cdot op(A) \cdot x + \beta \cdot y\f$, with
 *  
 *      \f$
 *          op(A) = \left\{
 *          \begin{array}{ll}
 *              A,   & \text{if trans == rocsparse_operation_none} \\
 *              A^T, & \text{if trans == rocsparse_operation_transpose} \\
 *              A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
 *          \end{array}
 *          \right.
 *      \f$
 *
 *  The \p info parameter is optional and contains information collected by
 *  rocsparse_csrmv_analysis(). If present, the information will be used to speed up the
 *  \p csrmv computation. If \p info == \p NULL, general \p csrmv routine will be used
 *  instead.
 *
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  \code{.c}
 *      for(i = 0; i < m; ++i)
 *      {
 *          y[i] = beta * y[i];
 *
 *          for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
 *          {
 *              y[i] = y[i] + alpha * csr_val[j] * x[csr_col_ind[j]];
 *          }
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse \p CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse \p CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse \p CSR matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse \p CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              \p CSR matrix.
 *  @param[in]
 *  info        information collected by rocsparse_csrmv_analysis(), can be \p NULL if
 *              no information is available.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p x, \p beta or \p y pointer is
 *              invalid. <br>
 *              \ref rocsparse_status_arch_mismatch the device is not supported. <br>
 *              \ref rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
///@{
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
                                  rocsparse_mat_info info,
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
                                  rocsparse_mat_info info,
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
                                  rocsparse_mat_info info,
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
                                  rocsparse_mat_info info,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex* y);
*/
///@}









ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int nnz,
                                             const rocsparse_mat_descr descr,
                                             const rocsparse_int* csr_row_ptr,
                                             const rocsparse_int* csr_col_ind,
                                             rocsparse_mat_info info,
                                             size_t* buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          rocsparse_int m,
                                          rocsparse_int nnz,
                                          const rocsparse_mat_descr descr,
                                          const rocsparse_int* csr_row_ptr,
                                          const rocsparse_int* csr_col_ind,
                                          rocsparse_mat_info info,
                                          rocsparse_solve_policy solve,
                                          rocsparse_analysis_policy analysis,
                                          void* temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrsv_clear(const rocsparse_mat_descr descr,
                                       rocsparse_mat_info info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrsv_solve(rocsparse_handle handle,
                                        rocsparse_operation trans,
                                        rocsparse_int m,
                                        rocsparse_int nnz,
                                        const float* alpha,
                                        const rocsparse_mat_descr descr,
                                        const float* csr_val,
                                        const rocsparse_int* csr_row_ptr,
                                        const rocsparse_int* csr_col_ind,
                                        rocsparse_mat_info info,
                                        const float* x,
                                        float* y,
                                        rocsparse_solve_policy policy,
                                        void* temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrsv_solve(rocsparse_handle handle,
                                        rocsparse_operation trans,
                                        rocsparse_int m,
                                        rocsparse_int nnz,
                                        const double* alpha,
                                        const rocsparse_mat_descr descr,
                                        const double* csr_val,
                                        const rocsparse_int* csr_row_ptr,
                                        const rocsparse_int* csr_col_ind,
                                        rocsparse_mat_info info,
                                        const double* x,
                                        double* y,
                                        rocsparse_solve_policy policy,
                                        void* temp_buffer);













/*! \brief Sparse matrix vector multiplication using \p ELL storage format
 *
 *  \details
 *  \p rocsparse_ellmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in \p ELL storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  
 *      \f$y := \alpha \cdot op(A) \cdot x + \beta \cdot y\f$, with
 *  
 *      \f$
 *          op(A) = \left\{
 *          \begin{array}{ll}
 *              A,   & \text{if trans == rocsparse_operation_none} \\
 *              A^T, & \text{if trans == rocsparse_operation_transpose} \\
 *              A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
 *          \end{array}
 *          \right.
 *      \f$
 *
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  \code{.c}
 *      for(i = 0; i < m; ++i)
 *      {
 *          y[i] = beta * y[i];
 *
 *          for(p = 0; p < ell_width; ++p)
 *          {
 *              idx = p * m + i;
 *
 *              if((ell_col_ind[idx] >= 0) && (ell_col_ind[idx] < n))
 *              {
 *                  y[i] = y[i] + alpha * ell_val[idx] * x[ell_col_ind[idx]];
 *              }
 *          }
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse \p ELL matrix.
 *  @param[in]
 *  n           number of columns of the sparse \p ELL matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse \p ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  ell_val     array that contains the elements of the sparse \p ELL matrix. Padded
 *              elements should be zero.
 *  @param[in]
 *  ell_col_ind array that contains the column indices of the sparse \p ELL matrix.
 *              Padded column indices should be -1.
 *  @param[in]
 *  ell_width   number of non-zero elements per row of the sparse \p ELL matrix.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p ell_width is
 *              invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p descr, \p alpha, \p ell_val,
 *              \p ell_col_ind, \p x, \p beta or \p y pointer is invalid. <br>
 *              \ref rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sellmv(rocsparse_handle handle,
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
                                  float* y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dellmv(rocsparse_handle handle,
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
                                  double* y);

/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sellmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_float_complex* ell_val,
                                  const rocsparse_int* ell_col_ind,
                                  rocsparse_int ell_width,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex* y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_sellmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_double_complex* ell_val,
                                  const rocsparse_int* ell_col_ind,
                                  rocsparse_int ell_width,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex* y);
*/
///@}

/*! \brief Sparse matrix vector multiplication using \p HYB storage format
 *
 *  \details
 *  \p rocsparse_hybmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in \p HYB storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  
 *      \f$y := \alpha \cdot op(A) \cdot x + \beta \cdot y\f$, with
 *  
 *      \f$
 *          op(A) = \left\{
 *          \begin{array}{ll}
 *              A,   & \text{if trans == rocsparse_operation_none} \\
 *              A^T, & \text{if trans == rocsparse_operation_transpose} \\
 *              A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
 *          \end{array}
 *          \right.
 *      \f$
 *
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse \p HYB matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  hyb         matrix in \p HYB storage format.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p hyb structure was not initialized
 *              with valid matrix sizes. <br>
 *              \ref rocsparse_status_invalid_pointer \p descr, \p alpha, \p hyb, \p x,
 *              \p beta or \p y pointer is invalid. <br>
 *              \ref rocsparse_status_invalid_value \p hyb structure was not initialized
 *              with a valid partitioning type. <br>
 *              \ref rocsparse_status_arch_mismatch the device is not supported. <br>
 *              \ref rocsparse_status_memory_error the buffer could not be
 *              allocated. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred. <br>
 *              \ref rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
///@{
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
///@}

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */

/*! \brief Sparse matrix dense matrix multiplication using \p CSR storage format
 *
 *  \details
 *  \p rocsparse_csrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
 *  matrix \f$A\f$, defined in \p CSR storage format, and the dense \f$k \times n\f$
 *  matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
 *  is multiplied by the scalar \f$\beta\f$, such that
 *  
 *      \f$C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C\f$, with
 *  
 *      \f$
 *          op(A) = \left\{
 *          \begin{array}{ll}
 *              A,   & \text{if trans_A == rocsparse_operation_none} \\
 *              A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
 *              A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
 *          \end{array}
 *          \right.
 *      \f$
 *
 *  and
 *
 *      \f$
 *          op(B) = \left\{
 *          \begin{array}{ll}
 *              B,   & \text{if trans_B == rocsparse_operation_none} \\
 *              B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
 *              B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
 *          \end{array}
 *          \right.
 *      \f$
 *
 *  Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
 *
 *  \code{.c}
 *      for(i = 0; i < ldc; ++i)
 *      {
 *          for(j = 0; j < n; ++j)
 *          {
 *              C[i][j] = beta * C[i][j];
 *              
 *              for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
 *              {
 *                  C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
 *              }
 *          }
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans_A     matrix \f$A\f$ operation type.
 *  @param[in]
 *  trans_B     matrix \f$B\f$ operation type.
 *  @param[in]
 *  m           number of rows of the sparse \p CSR matrix \f$A\f$.
 *  @param[in]
 *  n           number of columns of the dense matrix \f$op(B)\f$ and \f$C\f$.
 *  @param[in]
 *  k           number of columns of the sparse \p CSR matrix \f$A\f$.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse \p CSR matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse \p CSR matrix \f$A\f$. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse \p CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse \p CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              \p CSR matrix \f$A\f$.
 *  @param[in]
 *  B           array of dimension \f$ldb \times n\f$ (\f$op(B) == B\f$) or
 *              \f$ldb \times k\f$ (\f$op(B) == B^T\f$ or \f$op(B) == B^H\f$).
 *  @param[in]
 *  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$
 *              (\f$op(A) == A\f$) or \f$\max{(1, m)}\f$ (\f$op(A) == A^T\f$ or
 *              \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  C           array of dimension \f$ldc \times n\f$.
 *  @param[in]
 *  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$
 *              (\f$op(A) == A\f$) or \f$\max{(1, k)}\f$ (\f$op(A) == A^T\f$ or
 *              \f$op(A) == A^H\f$).
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz, \p ldb or
 *              \p ldc is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p B, \p beta or \p C pointer is
 *              invalid. <br>
 *              \ref rocsparse_status_arch_mismatch the device is not supported. <br>
 *              \ref rocsparse_status_not_implemented
 *              \p trans_A != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrmm(rocsparse_handle handle,
                                  rocsparse_operation trans_A,
                                  rocsparse_operation trans_B,
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
                                  rocsparse_int ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrmm(rocsparse_handle handle,
                                  rocsparse_operation trans_A,
                                  rocsparse_operation trans_B,
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
                                  rocsparse_int ldc);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrmm(rocsparse_handle handle,
                                  rocsparse_operation trans_A,
                                  rocsparse_operation trans_B,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int k,
                                  rocsparse_int nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_float_complex* csr_val,
                                  const rocsparse_int* csr_row_ptr,
                                  const rocsparse_int* csr_col_ind,
                                  const rocsparse_float_complex* B,
                                  rocsparse_int ldb,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex* C,
                                  rocsparse_int ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrmm(rocsparse_handle handle,
                                  rocsparse_operation trans_A,
                                  rocsparse_operation trans_B,
                                  rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int k,
                                  rocsparse_int nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_double_complex* csr_val,
                                  const rocsparse_int* csr_row_ptr,
                                  const rocsparse_int* csr_col_ind,
                                  const rocsparse_double_complex* B,
                                  rocsparse_int ldb,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex* C,
                                  rocsparse_int ldc);
*/
///@}







//TODO
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int nnz,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_mat_info info,
                                               size_t* buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle handle,
                                            rocsparse_int m,
                                            rocsparse_int nnz,
                                            const rocsparse_mat_descr descr,
                                            const rocsparse_int* csr_row_ptr,
                                            const rocsparse_int* csr_col_ind,
                                            rocsparse_mat_info info,
                                            rocsparse_solve_policy solve,
                                            rocsparse_analysis_policy analysis,
                                            void* temp_buffer);


ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrilu0_clear(rocsparse_mat_info info);



ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrilu0(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int nnz,
                                    const rocsparse_mat_descr descr,
                                    float* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    rocsparse_mat_info info,
                                    rocsparse_solve_policy policy,
                                    void* temp_buffer);


ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int nnz,
                                    const rocsparse_mat_descr descr,
                                    double* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    rocsparse_mat_info info,
                                    rocsparse_solve_policy policy,
                                    void* temp_buffer);













/*
 * ===========================================================================
 *    Sparse Format Conversions
 * ===========================================================================
 */

/*! \brief Convert a sparse \p CSR matrix into sparse \p COO matrix
 *
 *  \details
 *  \p rocsparse_csr2coo converts the \p CSR array containing the row offsets, that point
 *  to the start of every row, into a \p COO array of row indices. It can also be used
 *  to convert a \p CSC array containing the column offsets into a \p COO array of column
 *  indices.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row
 *              of the sparse \p CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse \p CSR matrix.
 *  @param[in]
 *  m           number of rows of the sparse \p CSR matrix.
 *  @param[out]
 *  coo_row_ind array of \p nnz elements containing the row indices of the sparse \p COO
 *              matrix.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p csr_row_ptr or \p coo_row_ind
 *              pointer is invalid. <br>
 *              \ref rocsparse_status_arch_mismatch the device is not supported.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2coo(rocsparse_handle handle,
                                   const rocsparse_int* csr_row_ptr,
                                   rocsparse_int nnz,
                                   rocsparse_int m,
                                   rocsparse_int* coo_row_ind,
                                   rocsparse_index_base idx_base);

/*! \brief Convert a sparse \p CSR matrix into sparse \p CSC matrix
 *
 *  \details
 *  \p rocsparse_csr2csc_buffer_size returns the size of the temporary storage buffer
 *  required by rocsparse_csr2csc(). The temporary storage buffer must be allocated by
 *  the user.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse \p CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse \p CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse \p CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              \p CSR matrix.
 *  @param[in]
 *  copy_values \ref rocsparse_action_symbolic or \ref rocsparse_action_numeric.
 *  @param[out]
 *  buffer_size number of bytes of the temporary storage buffer required by
 *              sparse_csr2csc().
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p csr_row_ptr, \p csr_col_ind or
 *              \p buffer_size pointer is invalid. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2csc_buffer_size(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int nnz,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_action copy_values,
                                               size_t* buffer_size);

/*! \brief Convert a sparse \p CSR matrix into sparse \p CSC matrix
 *
 *  \details
 *  \p rocsparse_csr2csc converts a \p CSR matrix info a \p CSC matrix. The resulting
 *  matrix can also be seen as the transpose of the input matrix. \p rocsparse_csr2csc
 *  can also be used to convert a \p CSC matrix into a \p CSR matrix. \p copy_values
 *  decides whether \p csc_val is being filled during conversion
 *  (\ref rocsparse_action_numeric) or not (\ref rocsparse_action_symbolic).
 *
 *  \p rocsparse_csr2csc requires extra temporary storage buffer that has to be
 *  allocated by the user. Storage buffer size can be determined by
 *  rocsparse_csr2csc_buffer_size().
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse \p CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse \p CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse \p CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              \p CSR matrix.
 *  @param[out]
 *  csc_val     array of \p nnz elements of the sparse \p CSC matrix.
 *  @param[out]
 *  csc_row_ind array of \p nnz elements containing the row indices of the sparse \p CSC
 *              matrix.
 *  @param[out]
 *  csc_col_ptr array of \p n+1 elements that point to the start of every column of the
 *              sparse \p CSC matrix.
 *  @param[in]
 *  copy_values \ref rocsparse_action_symbolic or \ref rocsparse_action_numeric.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user, size is returned by
 *              rocsparse_csr2csc_buffer_size().
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p csr_val, \p csr_row_ptr,
 *              \p csr_col_ind, \p csc_val, \p csc_row_ind, \p csc_col_ptr or
 *              \p temp_buffer pointer is invalid. <br>
 *              \ref rocsparse_status_arch_mismatch the device is not supported. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2csc(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    rocsparse_int nnz,
                                    const float* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    float* csc_val,
                                    rocsparse_int* csc_row_ind,
                                    rocsparse_int* csc_col_ptr,
                                    rocsparse_action copy_values,
                                    rocsparse_index_base idx_base,
                                    void* temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2csc(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    rocsparse_int nnz,
                                    const double* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    double* csc_val,
                                    rocsparse_int* csc_row_ind,
                                    rocsparse_int* csc_col_ptr,
                                    rocsparse_action copy_values,
                                    rocsparse_index_base idx_base,
                                    void* temp_buffer);
///@}

/*! \brief Convert a sparse \p CSR matrix into sparse \p ELL matrix
 *
 *  \details
 *  \p rocsparse_csr2ell_width computes the maximum of the per row non-zero elements
 *  over all rows, the \p ELL \p width, for a given \p CSR matrix.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_descr   descriptor of the sparse \p CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse \p CSR matrix.
 *  @param[in]
 *  ell_descr   descriptor of the sparse \p ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  ell_width   pointer to the number of non-zero elements per row in \p ELL storage
 *              format.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p csr_descr, \p csr_row_ptr, or
 *              \p ell_width pointer is invalid. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred. <br>
 *              \ref rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2ell_width(rocsparse_handle handle,
                                         rocsparse_int m,
                                         const rocsparse_mat_descr csr_descr,
                                         const rocsparse_int* csr_row_ptr,
                                         const rocsparse_mat_descr ell_descr,
                                         rocsparse_int* ell_width);

/*! \brief Convert a sparse \p CSR matrix into sparse \p ELL matrix
 *
 *  \details
 *  \p rocsparse_csr2ell converts a \p CSR matrix into an \p ELL matrix. It is assumed,
 *  that \p ell_val and \p ell_col_ind are allocated. Allocation size is computed by the
 *  number of rows times the number of \p ELL non-zero elements per row, such that
 *  \f$\text{nnz}_{\text{ELL}} = m \cdot \text{ell_width}\f$. The number of \p ELL
 *  non-zero elements per row is obtained by rocsparse_csr2ell_width().
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_descr   descriptor of the sparse \p CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val     array containing the values of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse \p CSR matrix.
 *  @param[in]
 *  csr_col_ind array containing the column indices of the sparse \p CSR matrix.
 *  @param[in]
 *  ell_descr   descriptor of the sparse \p ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  ell_width   number of non-zero elements per row in \p ELL storage format.
 *  @param[out]
 *  ell_val     array of \p m times \p ell_width elements of the sparse \p ELL matrix.
 *  @param[out]
 *  ell_col_ind array of \p m times \p ell_width elements containing the column indices
 *              of the sparse \p ELL matrix.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m or \p ell_width is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p csr_descr, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p ell_descr, \p ell_val or
 *              \p ell_col_ind pointer is invalid. <br>
 *              \ref rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2ell(rocsparse_handle handle,
                                    rocsparse_int m,
                                    const rocsparse_mat_descr csr_descr,
                                    const float* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int ell_width,
                                    float* ell_val,
                                    rocsparse_int* ell_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2ell(rocsparse_handle handle,
                                    rocsparse_int m,
                                    const rocsparse_mat_descr csr_descr,
                                    const double* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int ell_width,
                                    double* ell_val,
                                    rocsparse_int* ell_col_ind);

/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2ell(rocsparse_handle handle,
                                    rocsparse_int m,
                                    const rocsparse_mat_descr csr_descr,
                                    const rocsparse_float_complex* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int ell_width,
                                    rocsparse_float_complex* ell_val,
                                    rocsparse_int* ell_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2ell(rocsparse_handle handle,
                                    rocsparse_int m,
                                    const rocsparse_mat_descr csr_descr,
                                    const rocsparse_double_complex* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int ell_width,
                                    rocsparse_double_complex* ell_val,
                                    rocsparse_int* ell_col_ind);
*/
///@}

/*! \brief Convert a sparse \p CSR matrix into sparse \p HYB matrix
 *
 *  \details
 *  \p rocsparse_csr2hyb converts a \p CSR matrix into a \p HYB matrix. It is assumed
 *  that \p hyb has been initialized with rocsparse_create_hyb_mat().
 *
 *  This function requires a significant amount of storage for the \p HYB matrix,
 *  depending on the matrix structure.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse \p CSR matrix.
 *  @param[in]
 *  n               number of columns of the sparse \p CSR matrix.
 *  @param[in]
 *  descr           descriptor of the sparse \p CSR matrix. Currently, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val         array containing the values of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
 *                  sparse \p CSR matrix.
 *  @param[in]
 *  csr_col_ind     array containing the column indices of the sparse \p CSR matrix.
 *  @param[out]
 *  hyb             sparse matrix in \p HYB format.
 *  @param[in]
 *  user_ell_width  width of the \p ELL part of the \p HYB matrix (only required if
 *                  \p partition_type == \ref rocsparse_hyb_partition_user).
 *  @param[in]
 *  partition_type  \ref rocsparse_hyb_partition_auto (recommended),
 *                  \ref rocsparse_hyb_partition_user or
 *                  \ref rocsparse_hyb_partition_max.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p user_ell_width is
 *              invalid. <br>
 *              \ref rocsparse_status_invalid_value \p partition_type is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p descr, \p hyb, \p csr_val,
 *              \p csr_row_ptr or \p csr_col_ind pointer is invalid. <br>
 *              \ref rocsparse_status_memory_error the buffer for the \p HYB matrix
 *              could not be allocated. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred. <br>
 *              \ref rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
///@{
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
///@}

/*! \brief Convert a sparse \p COO matrix into sparse \p CSR matrix
 *
 *  \details
 *  \p rocsparse_coo2csr converts the \p COO array containing the row indices into a
 *  \p CSR array of row offsets, that point to the start of every row. It can also be
 *  used, to convert a \p COO array containing the column indices into a \p CSC array
 *  of column offsets, that point to the start of every column.
 *
 *  It is assumed that the \p COO row index array is sorted.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  coo_row_ind array of \p nnz elements containing the row indices of the sparse \p COO
 *              matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse \p CSR matrix.
 *  @param[in]
 *  m           number of rows of the sparse \p CSR matrix.
 *  @param[out]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse \p CSR matrix.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p coo_row_ind or \p csr_row_ptr
 *              pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo2csr(rocsparse_handle handle,
                                   const rocsparse_int* coo_row_ind,
                                   rocsparse_int nnz,
                                   rocsparse_int m,
                                   rocsparse_int* csr_row_ptr,
                                   rocsparse_index_base idx_base);

/*! \brief Convert a sparse \p ELL matrix into sparse \p CSR matrix
 *
 *  \details
 *  \p rocsparse_ell2csr_nnz computes the total \p CSR non-zero elements and the \p CSR
 *  row offsets, that point to the start of every row of the sparse \p CSR matrix, for
 *  a given \p ELL matrix. It is assumed that \p csr_row_ptr has been allocated with
 *  size \p m + 1.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse \p ELL matrix.
 *  @param[in]
 *  n           number of columns of the sparse \p ELL matrix.
 *  @param[in]
 *  ell_descr   descriptor of the sparse \p ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  ell_width   number of non-zero elements per row in \p ELL storage format.
 *  @param[in]
 *  ell_col_ind array of \p m times \p ell_width elements containing the column indices
 *              of the sparse \p ELL matrix.
 *  @param[in]
 *  csr_descr   descriptor of the sparse \p CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse \p CSR matrix.
 *  @param[out]
 *  csr_nnz     pointer to the total number of non-zero elements in \p CSR storage
 *              format.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p ell_width is
 *              invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p ell_descr, \p ell_col_ind,
 *              \p csr_descr, \p csr_row_ptr or \p csr_nnz pointer is invalid. <br>
 *              \ref rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ell2csr_nnz(rocsparse_handle handle,
                                       rocsparse_int m,
                                       rocsparse_int n,
                                       const rocsparse_mat_descr ell_descr,
                                       rocsparse_int ell_width,
                                       const rocsparse_int* ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       rocsparse_int* csr_row_ptr,
                                       rocsparse_int* csr_nnz);

/*! \brief Convert a sparse \p ELL matrix into sparse \p CSR matrix
 *
 *  \details
 *  \p rocsparse_ell2csr converts an \p ELL matrix into a \p CSR matrix. It is assumed
 *  that \p csr_row_ptr has already been filled and that \p csr_val and \p csr_col_ind
 *  are allocated by the user. \p csr_row_ptr and allocation size of \p csr_col_ind and
 *  \p csr_val is defined by the number of CSR non-zero elements. Both can be obtained
 *  by rocsparse_ell2csr_nnz().
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse \p ELL matrix.
 *  @param[in]
 *  n           number of columns of the sparse \p ELL matrix.
 *  @param[in]
 *  ell_descr   descriptor of the sparse \p ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  ell_width   number of non-zero elements per row in \p ELL storage format.
 *  @param[in]
 *  ell_val     array of \p m times \p ell_width elements of the sparse \p ELL matrix.
 *  @param[in]
 *  ell_col_ind array of \p m times \p ell_width elements containing the column indices
 *              of the sparse \p ELL matrix.
 *  @param[in]
 *  csr_descr   descriptor of the sparse \p CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_val     array containing the values of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse \p CSR matrix.
 *  @param[out]
 *  csr_col_ind array containing the column indices of the sparse \p CSR matrix.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p ell_width is
 *              invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p csr_descr, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p ell_descr, \p ell_val or
 *              \p ell_col_ind pointer is invalid. <br>
 *              \ref rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
///@{
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sell2csr(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int ell_width,
                                    const float* ell_val,
                                    const rocsparse_int* ell_col_ind,
                                    const rocsparse_mat_descr csr_descr,
                                    float* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    rocsparse_int* csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dell2csr(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int ell_width,
                                    const double* ell_val,
                                    const rocsparse_int* ell_col_ind,
                                    const rocsparse_mat_descr csr_descr,
                                    double* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    rocsparse_int* csr_col_ind);
/*
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cell2csr(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int ell_width,
                                    const rocsparse_float_complex* ell_val,
                                    const rocsparse_int* ell_col_ind,
                                    const rocsparse_mat_descr csr_descr,
                                    rocsparse_float_complex* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    rocsparse_int* csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zell2csr(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int ell_width,
                                    const rocsparse_double_complex* ell_val,
                                    const rocsparse_int* ell_col_ind,
                                    const rocsparse_mat_descr csr_descr,
                                    rocsparse_double_complex* csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    rocsparse_int* csr_col_ind);
*/
///@}

/*! \brief Create the identity map
 *
 *  \details
 *  \p rocsparse_create_identity_permutation stores the identity map in \p p, such that
 *  \f$p = 0:1:(n-1)\f$.
 *
 *  \code{.c}
 *      for(i = 0; i < n; ++i)
 *      {
 *          p[i] = i;
 *      }
 *  \endcode
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  n           size of the map \p p.
 *  @param[out]
 *  p           array of \p n integers containing the map.
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p n is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p p pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status
rocsparse_create_identity_permutation(rocsparse_handle handle, rocsparse_int n, rocsparse_int* p);

/*! \brief Sort a sparse \p CSR matrix
 *
 *  \details
 *  \p rocsparse_csrsort_buffer_size returns the size of the temporary storage buffer
 *  required by rocsparse_csrsort(). The temporary storage buffer must be allocated by
 *  the user.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse \p CSR matrix.
 *  @param[in]
 *  n               number of columns of the sparse \p CSR matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse \p CSR matrix.
 *  @param[in]
 *  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
 *                  sparse \p CSR matrix.
 *  @param[in]
 *  csr_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  \p CSR matrix.
 *  @param[out]
 *  buffer_size     number of bytes of the temporary storage buffer required by
 *                  rocsparse_csrsort().
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p csr_row_ptr, \p csr_col_ind or
 *              \p buffer_size pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrsort_buffer_size(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int nnz,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               size_t* buffer_size);

/*! \brief Sort a sparse \p CSR matrix
 *
 *  \details
 *  \p rocsparse_csrsort sorts a matrix in \p CSR format. The sorted permutation vector
 *  \p perm can be used to obtain sorted \p csr_val array. In this case, \p perm must be
 *  initialized as the identity permutation, see rocsparse_create_identity_permutation().
 *
 *  rocsparse_csrsort requires extra temporary storage buffer that has to be allocated by
 *  the user. Storage buffer size can be determined by rocsparse_csrsort_buffer_size().
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse \p CSR matrix.
 *  @param[in]
 *  n               number of columns of the sparse \p CSR matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse \p CSR matrix.
 *  @param[in]
 *  descr           descriptor of the sparse \p CSR matrix. Currently, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
 *                  sparse \p CSR matrix.
 *  @param[inout]
 *  csr_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  \p CSR matrix.
 *  @param[inout]
 *  perm            array of \p nnz integers containing the unsorted map indices, can be
 *                  \p NULL.
 *  @param[in]
 *  temp_buffer     temporary storage buffer allocated by the user, size is returned by
 *                  rocsparse_csrsort_buffer_size().
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p descr, \p csr_row_ptr,
 *              \p csr_col_ind or \p temp_buffer pointer is invalid. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred. <br>
 *              \ref rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  \code{.c}
 *      //     1 2 3
 *      // A = 4 5 6
 *      //     7 8 9
 *      rocsparse_int m   = 3;
 *      rocsparse_int n   = 3;
 *      rocsparse_int nnz = 9;
 *      
 *      csr_row_ptr[m + 1] = {0, 3, 6, 9}                 // device memory
 *      csr_col_ind[nnz]   = {2, 0, 1, 0, 1, 2, 0, 2, 1}; // device memory
 *      csr_val[nnz]       = {3, 1, 2, 4, 5, 6, 7, 9, 8}; // device memory
 *      
 *      // Allocate temporary buffer
 *      size_t buffer_size = 0;
 *      void* temp_buffer  = NULL;
 *      rocsparse_csrsort_buffer_size(handle, m, n, nnz, csr_row_ptr, csr_col_ind, &buffer_size);
 *      hipMalloc(&temp_buffer, sizeof(char) * buffer_size);
 *      
 *      // Create permutation vector perm as the identity map
 *      rocsparse_int* perm = NULL;
 *      hipMalloc((void**)&perm, sizeof(rocsparse_int) * nnz);
 *      rocsparse_create_identity_permutation(handle, nnz, perm);
 *      
 *      // Sort the CSR matrix
 *      rocsparse_csrsort(handle, m, n, nnz, descr, csr_row_ptr, csr_col_ind, perm, temp_buffer);
 *      
 *      // Gather sorted csr_val array
 *      float* csr_val_sorted = NULL;
 *      hipMalloc((void**)&csr_val_sorted, sizeof(float) * nnz);
 *      rocsparse_sgthr(handle, nnz, csr_val, csr_val_sorted, perm, rocsparse_index_base_zero);
 *      
 *      // Clean up
 *      hipFree(temp_buffer);
 *      hipFree(perm);
 *      hipFree(csr_val);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrsort(rocsparse_handle handle,
                                   rocsparse_int m,
                                   rocsparse_int n,
                                   rocsparse_int nnz,
                                   const rocsparse_mat_descr descr,
                                   const rocsparse_int* csr_row_ptr,
                                   rocsparse_int* csr_col_ind,
                                   rocsparse_int* perm,
                                   void* temp_buffer);

/*! \brief Sort a sparse \p COO matrix
 *
 *  \details
 *  coosort_buffer_size returns the size of the temporary storage buffer
 *  that is required by coosort. The temporary storage buffer has to be
 *  allocated by the user.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse \p COO matrix.
 *  @param[in]
 *  n               number of columns of the sparse \p COO matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse \p COO matrix.
 *  @param[in]
 *  coo_row_ind     array of \p nnz elements containing the row indices of the sparse
 *                  \p COO matrix.
 *  @param[in]
 *  coo_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  \p COO matrix.
 *  @param[out]
 *  buffer_size     number of bytes of the temporary storage buffer required by
 *                  rocsparse_coosort_by_row() and rocsparse_coosort_by_column().
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p coo_row_ind, \p coo_col_ind or
 *              \p buffer_size pointer is invalid. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coosort_buffer_size(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int nnz,
                                               const rocsparse_int* coo_row_ind,
                                               const rocsparse_int* coo_col_ind,
                                               size_t* buffer_size);

/*! \brief Sort a sparse \p COO matrix by row
 *
 *  \details
 *  \p rocsparse_coosort_by_row sorts a matrix in \p COO format by row. The sorted
 *  permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
 *  case, \p perm must be initialized as the identity permutation, see
 *  rocsparse_create_identity_permutation().
 *
 *  rocsparse_coosort_by_row requires extra temporary storage buffer that has to be
 *  allocated by the user. Storage buffer size can be determined by
 *  rocsparse_coosort_buffer_size().
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse \p COO matrix.
 *  @param[in]
 *  n               number of columns of the sparse \p COO matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse \p COO matrix.
 *  @param[inout]
 *  coo_row_ind     array of \p nnz elements containing the row indices of the sparse
 *                  \p COO matrix.
 *  @param[inout]
 *  coo_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  \p COO matrix.
 *  @param[inout]
 *  perm            array of \p nnz integers containing the unsorted map indices, can be
 *                  \p NULL.
 *  @param[in]
 *  temp_buffer     temporary storage buffer allocated by the user, size is returned by
 *                  rocsparse_coosort_buffer_size().
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p coo_row_ind, \p coo_col_ind or
 *              \p temp_buffer pointer is invalid. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coosort_by_row(rocsparse_handle handle,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int nnz,
                                          rocsparse_int* coo_row_ind,
                                          rocsparse_int* coo_col_ind,
                                          rocsparse_int* perm,
                                          void* temp_buffer);

/*! \brief Sort a sparse \p COO matrix by column
 *
 *  \details
 *  \p rocsparse_coosort_by_column sorts a matrix in \p COO format by column. The sorted
 *  permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
 *  case, \p perm must be initialized as the identity permutation, see
 *  rocsparse_create_identity_permutation().
 *
 *  rocsparse_coosort_by_column requires extra temporary storage buffer that has to be
 *  allocated by the user. Storage buffer size can be determined by
 *  rocsparse_coosort_buffer_size().
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse \p COO matrix.
 *  @param[in]
 *  n               number of columns of the sparse \p COO matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse \p COO matrix.
 *  @param[inout]
 *  coo_row_ind     array of \p nnz elements containing the row indices of the sparse
 *                  \p COO matrix.
 *  @param[inout]
 *  coo_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  \p COO matrix.
 *  @param[inout]
 *  perm            array of \p nnz integers containing the unsorted map indices, can be
 *                  \p NULL.
 *  @param[in]
 *  temp_buffer     temporary storage buffer allocated by the user, size is returned by
 *                  rocsparse_coosort_buffer_size().
 *
 *  \returns    \ref rocsparse_status_success the operation completed successfully. <br>
 *              \ref rocsparse_status_invalid_handle the library context was
 *              not initialized. <br>
 *              \ref rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid. <br>
 *              \ref rocsparse_status_invalid_pointer \p coo_row_ind, \p coo_col_ind or
 *              \p temp_buffer pointer is invalid. <br>
 *              \ref rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coosort_by_column(rocsparse_handle handle,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int nnz,
                                             rocsparse_int* coo_row_ind,
                                             rocsparse_int* coo_col_ind,
                                             rocsparse_int* perm,
                                             void* temp_buffer);

#ifdef __cplusplus
}
#endif

#endif // _ROCSPARSE_FUNCTIONS_H_
