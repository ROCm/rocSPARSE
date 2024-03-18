/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_BSRMV_H
#define ROCSPARSE_BSRMV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p rocsparse_bsrmv_ex_analysis performs the analysis step for rocsparse_sbsrmv(),
*  rocsparse_dbsrmv(), rocsparse_cbsrmv() and rocsparse_zbsrmv(). It is expected that
*  this function will be executed only once for a given matrix and particular operation
*  type. The gathered analysis meta data can be cleared by rocsparse_bsrmv_ex_clear().
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  block_dim     block dimension of the sparse BSR matrix.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb or \p nnzb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
*              \p bsr_col_ind or \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer for the gathered information
*              could not be allocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_sbsrmv_analysis instead.")))
ROCSPARSE_EXPORT rocsparse_status
    rocsparse_sbsrmv_ex_analysis(rocsparse_handle          handle,
                                 rocsparse_direction       dir,
                                 rocsparse_operation       trans,
                                 rocsparse_int             mb,
                                 rocsparse_int             nb,
                                 rocsparse_int             nnzb,
                                 const rocsparse_mat_descr descr,
                                 const float*              bsr_val,
                                 const rocsparse_int*      bsr_row_ptr,
                                 const rocsparse_int*      bsr_col_ind,
                                 rocsparse_int             block_dim,
                                 rocsparse_mat_info        info);

__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_dbsrmv_analysis instead.")))
ROCSPARSE_EXPORT rocsparse_status
    rocsparse_dbsrmv_ex_analysis(rocsparse_handle          handle,
                                 rocsparse_direction       dir,
                                 rocsparse_operation       trans,
                                 rocsparse_int             mb,
                                 rocsparse_int             nb,
                                 rocsparse_int             nnzb,
                                 const rocsparse_mat_descr descr,
                                 const double*             bsr_val,
                                 const rocsparse_int*      bsr_row_ptr,
                                 const rocsparse_int*      bsr_col_ind,
                                 rocsparse_int             block_dim,
                                 rocsparse_mat_info        info);

__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_cbsrmv_analysis instead.")))
ROCSPARSE_EXPORT rocsparse_status
    rocsparse_cbsrmv_ex_analysis(rocsparse_handle               handle,
                                 rocsparse_direction            dir,
                                 rocsparse_operation            trans,
                                 rocsparse_int                  mb,
                                 rocsparse_int                  nb,
                                 rocsparse_int                  nnzb,
                                 const rocsparse_mat_descr      descr,
                                 const rocsparse_float_complex* bsr_val,
                                 const rocsparse_int*           bsr_row_ptr,
                                 const rocsparse_int*           bsr_col_ind,
                                 rocsparse_int                  block_dim,
                                 rocsparse_mat_info             info);

__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_zbsrmv_analysis instead.")))
ROCSPARSE_EXPORT rocsparse_status
    rocsparse_zbsrmv_ex_analysis(rocsparse_handle                handle,
                                 rocsparse_direction             dir,
                                 rocsparse_operation             trans,
                                 rocsparse_int                   mb,
                                 rocsparse_int                   nb,
                                 rocsparse_int                   nnzb,
                                 const rocsparse_mat_descr       descr,
                                 const rocsparse_double_complex* bsr_val,
                                 const rocsparse_int*            bsr_row_ptr,
                                 const rocsparse_int*            bsr_col_ind,
                                 rocsparse_int                   block_dim,
                                 rocsparse_mat_info              info);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p rocsparse_bsrmv_ex multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})\f$
*  matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref rocsparse_operation_none is supported.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  block_dim     block dimension of the sparse BSR matrix.
*  @param[in]
*  x           array of \p nb*block_dim elements (\f$op(A) = A\f$) or \p mb*block_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p mb*block_dim elements (\f$op(A) = A\f$) or \p nb*block_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb, \p nnzb or \p block_dim is
*              invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
*              \p bsr_row_ind, \p bsr_col_ind, \p x, \p beta or \p y pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_sbsrmv instead."))) ROCSPARSE_EXPORT rocsparse_status
    rocsparse_sbsrmv_ex(rocsparse_handle          handle,
                        rocsparse_direction       dir,
                        rocsparse_operation       trans,
                        rocsparse_int             mb,
                        rocsparse_int             nb,
                        rocsparse_int             nnzb,
                        const float*              alpha,
                        const rocsparse_mat_descr descr,
                        const float*              bsr_val,
                        const rocsparse_int*      bsr_row_ptr,
                        const rocsparse_int*      bsr_col_ind,
                        rocsparse_int             block_dim,
                        rocsparse_mat_info        info,
                        const float*              x,
                        const float*              beta,
                        float*                    y);

__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_dbsrmv instead."))) ROCSPARSE_EXPORT rocsparse_status
    rocsparse_dbsrmv_ex(rocsparse_handle          handle,
                        rocsparse_direction       dir,
                        rocsparse_operation       trans,
                        rocsparse_int             mb,
                        rocsparse_int             nb,
                        rocsparse_int             nnzb,
                        const double*             alpha,
                        const rocsparse_mat_descr descr,
                        const double*             bsr_val,
                        const rocsparse_int*      bsr_row_ptr,
                        const rocsparse_int*      bsr_col_ind,
                        rocsparse_int             block_dim,
                        rocsparse_mat_info        info,
                        const double*             x,
                        const double*             beta,
                        double*                   y);

__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_cbsrmv instead."))) ROCSPARSE_EXPORT rocsparse_status
    rocsparse_cbsrmv_ex(rocsparse_handle               handle,
                        rocsparse_direction            dir,
                        rocsparse_operation            trans,
                        rocsparse_int                  mb,
                        rocsparse_int                  nb,
                        rocsparse_int                  nnzb,
                        const rocsparse_float_complex* alpha,
                        const rocsparse_mat_descr      descr,
                        const rocsparse_float_complex* bsr_val,
                        const rocsparse_int*           bsr_row_ptr,
                        const rocsparse_int*           bsr_col_ind,
                        rocsparse_int                  block_dim,
                        rocsparse_mat_info             info,
                        const rocsparse_float_complex* x,
                        const rocsparse_float_complex* beta,
                        rocsparse_float_complex*       y);

__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_zbsrmv instead."))) ROCSPARSE_EXPORT rocsparse_status
    rocsparse_zbsrmv_ex(rocsparse_handle                handle,
                        rocsparse_direction             dir,
                        rocsparse_operation             trans,
                        rocsparse_int                   mb,
                        rocsparse_int                   nb,
                        rocsparse_int                   nnzb,
                        const rocsparse_double_complex* alpha,
                        const rocsparse_mat_descr       descr,
                        const rocsparse_double_complex* bsr_val,
                        const rocsparse_int*            bsr_row_ptr,
                        const rocsparse_int*            bsr_col_ind,
                        rocsparse_int                   block_dim,
                        rocsparse_mat_info              info,
                        const rocsparse_double_complex* x,
                        const rocsparse_double_complex* beta,
                        rocsparse_double_complex*       y);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p rocsparse_bsrmv_ex_clear deallocates all memory that was allocated by
*  rocsparse_sbsrmv_ex_analysis(), rocsparse_dbsrmv_ex_analysis(), rocsparse_cbsrmv_ex_analysis()
*  or rocsparse_zbsrmv_ex_analysis(). This is especially useful, if memory is an issue and
*  the analysis data is not required anymore for further computation, e.g. when
*  switching to another sparse matrix format.
*
*  \note
*  Calling \p rocsparse_bsrmv_ex_clear is optional. All allocated resources will be
*  cleared, when the opaque \ref rocsparse_mat_info struct is destroyed using
*  rocsparse_destroy_mat_info().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[inout]
*  info        structure that holds the information collected during analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer for the gathered information
*              could not be deallocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
* */
__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_bsrmv_clear instead."))) ROCSPARSE_EXPORT rocsparse_status
    rocsparse_bsrmv_ex_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p rocsparse_bsrmv_analysis performs the analysis step for rocsparse_sbsrmv(),
*  rocsparse_dbsrmv(), rocsparse_cbsrmv() and rocsparse_zbsrmv(). It is expected that
*  this function will be executed only once for a given matrix and particular operation
*  type. The gathered analysis meta data can be cleared by rocsparse_bsrmv_clear().
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  block_dim     block dimension of the sparse BSR matrix.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb or \p nnzb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
*              \p bsr_col_ind or \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer for the gathered information
*              could not be allocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrmv_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans,
                                           rocsparse_int             mb,
                                           rocsparse_int             nb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           const float*              bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             block_dim,
                                           rocsparse_mat_info        info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrmv_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans,
                                           rocsparse_int             mb,
                                           rocsparse_int             nb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           const double*             bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             block_dim,
                                           rocsparse_mat_info        info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrmv_analysis(rocsparse_handle               handle,
                                           rocsparse_direction            dir,
                                           rocsparse_operation            trans,
                                           rocsparse_int                  mb,
                                           rocsparse_int                  nb,
                                           rocsparse_int                  nnzb,
                                           const rocsparse_mat_descr      descr,
                                           const rocsparse_float_complex* bsr_val,
                                           const rocsparse_int*           bsr_row_ptr,
                                           const rocsparse_int*           bsr_col_ind,
                                           rocsparse_int                  block_dim,
                                           rocsparse_mat_info             info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrmv_analysis(rocsparse_handle                handle,
                                           rocsparse_direction             dir,
                                           rocsparse_operation             trans,
                                           rocsparse_int                   mb,
                                           rocsparse_int                   nb,
                                           rocsparse_int                   nnzb,
                                           const rocsparse_mat_descr       descr,
                                           const rocsparse_double_complex* bsr_val,
                                           const rocsparse_int*            bsr_row_ptr,
                                           const rocsparse_int*            bsr_col_ind,
                                           rocsparse_int                   block_dim,
                                           rocsparse_mat_info              info);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p rocsparse_bsrmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})\f$
*  matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref rocsparse_operation_none is supported.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  block_dim     block dimension of the sparse BSR matrix.
*  @param[in]
*  x           array of \p nb*block_dim elements (\f$op(A) = A\f$) or \p mb*block_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p mb*block_dim elements (\f$op(A) = A\f$) or \p nb*block_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb, \p nnzb or \p block_dim is
*              invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
*              \p bsr_row_ind, \p bsr_col_ind, \p x, \p beta or \p y pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example performs a sparse matrix vector multiplication in BSR format.
*  \code{.c}
*      // rocSPARSE handle
*      rocsparse_handle handle;
*      rocsparse_create_handle(&handle);
*
*      // alpha * ( 1.0  0.0  2.0 ) * ( 1.0 ) + beta * ( 4.0 ) = (  31.1 )
*      //         ( 3.0  0.0  4.0 ) * ( 2.0 )          ( 5.0 ) = (  62.0 )
*      //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) = (  70.7 )
*      //         ( 7.0  0.0  8.0 ) *                  ( 7.0 ) = ( 123.8 )
*
*      // BSR block dimension
*      rocsparse_int bsr_dim = 2;
*
*      // Number of block rows and columns
*      rocsparse_int mb = 2;
*      rocsparse_int nb = 2;
*
*      // Number of non-zero blocks
*      rocsparse_int nnzb = 4;
*
*      // BSR row pointers
*      rocsparse_int hbsr_row_ptr[3] = {0, 2, 4};
*
*      // BSR column indices
*      rocsparse_int hbsr_col_ind[4] = {0, 1, 0, 1};
*
*      // BSR values
*      double hbsr_val[16]
*        = {1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0, 0.0, 0.0};
*
*      // Block storage in column major
*      rocsparse_direction dir = rocsparse_direction_column;
*
*      // Transposition of the matrix
*      rocsparse_operation trans = rocsparse_operation_none;
*
*      // Scalar alpha and beta
*      double alpha = 3.7;
*      double beta  = 1.3;
*
*      // x and y
*      double hx[4] = {1.0, 2.0, 3.0, 0.0};
*      double hy[4] = {4.0, 5.0, 6.0, 7.0};
*
*      // Matrix descriptor
*      rocsparse_mat_descr descr;
*      rocsparse_create_mat_descr(&descr);
*
*      // Offload data to device
*      rocsparse_int* dbsr_row_ptr;
*      rocsparse_int* dbsr_col_ind;
*      double*        dbsr_val;
*      double*        dx;
*      double*        dy;
*
*      hipMalloc((void**)&dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1));
*      hipMalloc((void**)&dbsr_col_ind, sizeof(rocsparse_int) * nnzb);
*      hipMalloc((void**)&dbsr_val, sizeof(double) * nnzb * bsr_dim * bsr_dim);
*      hipMalloc((void**)&dx, sizeof(double) * nb * bsr_dim);
*      hipMalloc((void**)&dy, sizeof(double) * mb * bsr_dim);
*
*      hipMemcpy(dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice);
*      hipMemcpy(dbsr_val, hbsr_val, sizeof(double) * nnzb * bsr_dim * bsr_dim, hipMemcpyHostToDevice);
*      hipMemcpy(dx, hx, sizeof(double) * nb * bsr_dim, hipMemcpyHostToDevice);
*      hipMemcpy(dy, hy, sizeof(double) * mb * bsr_dim, hipMemcpyHostToDevice);
*
*      rocsparse_mat_info info;
*      rocsparse_create_mat_info(&info);
*
*      // Call dbsrmv_analysis (Optional)
*      rocsparse_dbsrmv_analysis(handle,
*                                dir,
*                                trans,
*                                mb,
*                                nb,
*                                nnzb,
*                                descr,
*                                dbsr_val,
*                                dbsr_row_ptr,
*                                dbsr_col_ind,
*                                bsr_dim,
*                                info);
*
*      // Call dbsrmv to perform y = alpha * A x + beta * y
*      rocsparse_dbsrmv(handle,
*                       dir,
*                       trans,
*                       mb,
*                       nb,
*                       nnzb,
*                       &alpha,
*                       descr,
*                       dbsr_val,
*                       dbsr_row_ptr,
*                       dbsr_col_ind,
*                       bsr_dim,
*                       info,
*                       dx,
*                       &beta,
*                       dy);
*
*      // Copy result back to host
*      hipMemcpy(hy, dy, sizeof(double) * mb * bsr_dim, hipMemcpyDeviceToHost);
*
*      // Clear rocSPARSE
*      rocsparse_destroy_mat_descr(descr);
*      rocsparse_destroy_handle(handle);
*      rocsparse_destroy_mat_info(info);
*
*      // Clear device memory
*      hipFree(dbsr_row_ptr);
*      hipFree(dbsr_col_ind);
*      hipFree(dbsr_val);
*      hipFree(dx);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrmv(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans,
                                  rocsparse_int             mb,
                                  rocsparse_int             nb,
                                  rocsparse_int             nnzb,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  rocsparse_mat_info        info,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrmv(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans,
                                  rocsparse_int             mb,
                                  rocsparse_int             nb,
                                  rocsparse_int             nnzb,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  rocsparse_mat_info        info,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrmv(rocsparse_handle               handle,
                                  rocsparse_direction            dir,
                                  rocsparse_operation            trans,
                                  rocsparse_int                  mb,
                                  rocsparse_int                  nb,
                                  rocsparse_int                  nnzb,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* bsr_val,
                                  const rocsparse_int*           bsr_row_ptr,
                                  const rocsparse_int*           bsr_col_ind,
                                  rocsparse_int                  block_dim,
                                  rocsparse_mat_info             info,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrmv(rocsparse_handle                handle,
                                  rocsparse_direction             dir,
                                  rocsparse_operation             trans,
                                  rocsparse_int                   mb,
                                  rocsparse_int                   nb,
                                  rocsparse_int                   nnzb,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* bsr_val,
                                  const rocsparse_int*            bsr_row_ptr,
                                  const rocsparse_int*            bsr_col_ind,
                                  rocsparse_int                   block_dim,
                                  rocsparse_mat_info              info,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p rocsparse_bsrmv_clear deallocates all memory that was allocated by
*  rocsparse_sbsrmv_analysis(), rocsparse_dbsrmv_analysis(), rocsparse_cbsrmv_analysis()
*  or rocsparse_zbsrmv_analysis(). This is especially useful, if memory is an issue and
*  the analysis data is not required anymore for further computation, e.g. when
*  switching to another sparse matrix format.
*
*  \note
*  Calling \p rocsparse_bsrmv_clear is optional. All allocated resources will be
*  cleared, when the opaque \ref rocsparse_mat_info struct is destroyed using
*  rocsparse_destroy_mat_info().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[inout]
*  info        structure that holds the information collected during analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer for the gathered information
*              could not be deallocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
* */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsrmv_clear(rocsparse_handle handle, rocsparse_mat_info info);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSRMV_H */
