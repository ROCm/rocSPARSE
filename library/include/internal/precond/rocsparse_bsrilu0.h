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

#ifndef ROCSPARSE_BSRILU0_H
#define ROCSPARSE_BSRILU0_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
 *  structural or numerical zero has been found during rocsparse_sbsrilu0(),
 *  rocsparse_dbsrilu0(), rocsparse_cbsrilu0() or rocsparse_zbsrilu0() computation.
 *  The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same
 *  index base as the BSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
 *
 *  \note
 *  If a zero pivot is found, \p position \f$=j\f$ means that either the diagonal block
 *  \f$A_{j,j}\f$ is missing (structural zero) or the diagonal block \f$A_{j,j}\f$ is not
 *  invertible (numerical zero).
 *
 *  \note \p rocsparse_bsrilu0_zero_pivot is a blocking function. It might influence
 *  performance negatively.
 *
 *  \note
 *  This routine does not support execution in a hipGraph context.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[inout]
 *  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
 *              invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_zero_pivot zero pivot has been found.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsrilu0_zero_pivot(rocsparse_handle   handle,
                                              rocsparse_mat_info info,
                                              rocsparse_int*     position);

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_numeric_boost enables the user to replace a numerical value in
 *  an incomplete LU factorization. \p tol is used to determine whether a numerical value
 *  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
 *  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
 *
 *  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
 *  setting \p enable_boost to 0.
 *
 *  \note \p tol and \p boost_val can be in host or device memory.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  This routine supports execution in a hipGraph context.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  info            structure that holds the information collected during the analysis step.
 *  @param[in]
 *  enable_boost    enable/disable numeric boost.
 *  @param[in]
 *  boost_tol       tolerance to determine whether a numerical value is replaced or not.
 *  @param[in]
 *  boost_val       boost value to replace a numerical value.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info, \p tol or \p boost_val pointer
 *              is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const float*       boost_tol,
                                                  const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const double*      boost_tol,
                                                  const double*      boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrilu0_numeric_boost(rocsparse_handle               handle,
                                                  rocsparse_mat_info             info,
                                                  int                            enable_boost,
                                                  const float*                   boost_tol,
                                                  const rocsparse_float_complex* boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrilu0_numeric_boost(rocsparse_handle                handle,
                                                  rocsparse_mat_info              info,
                                                  int                             enable_boost,
                                                  const double*                   boost_tol,
                                                  const rocsparse_double_complex* boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dsbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                   rocsparse_mat_info info,
                                                   int                enable_boost,
                                                   const double*      boost_tol,
                                                   const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcbsrilu0_numeric_boost(rocsparse_handle               handle,
                                                   rocsparse_mat_info             info,
                                                   int                            enable_boost,
                                                   const double*                  boost_tol,
                                                   const rocsparse_float_complex* boost_val);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_buffer_size returns the size of the temporary storage buffer
 *  that is required by rocsparse_sbsrilu0_analysis(), rocsparse_dbsrilu0_analysis(),
 *  rocsparse_cbsrilu0_analysis(), rocsparse_zbsrilu0_analysis(), rocsparse_sbsrilu0(),
 *  rocsparse_dbsrilu0(), rocsparse_sbsrilu0() and rocsparse_dbsrilu0(). The temporary
 *  storage buffer must be allocated by the user. The size of the temporary storage
 *  buffer is identical to the size returned by rocsparse_sbsrsv_buffer_size(),
 *  rocsparse_dbsrsv_buffer_size(), rocsparse_cbsrsv_buffer_size(), rocsparse_zbsrsv_buffer_size(),
 *  rocsparse_sbsric0_buffer_size(), rocsparse_dbsric0_buffer_size(), rocsparse_cbsric0_buffer_size()
 *  and rocsparse_zbsric0_buffer_size() if the matrix sparsity pattern is identical. The user
 *  allocated buffer can thus be shared between subsequent calls to those functions.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  This routine supports execution in a hipGraph context.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  dir         direction that specifies whether to count nonzero elements by \ref rocsparse_direction_row or by
 *              \ref rocsparse_direction_column.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsr_val     array of length \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  block_dim   the block dimension of the BSR matrix. Between 1 and m where \p m=mb*block_dim.
 *  @param[out]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[out]
 *  buffer_size number of bytes of the temporary storage buffer required by
 *              rocsparse_sbsrilu0_analysis(), rocsparse_dbsrilu0_analysis(),
 *              rocsparse_cbsrilu0_analysis(), rocsparse_zbsrilu0_analysis(),
 *              rocsparse_sbsrilu0(), rocsparse_dbsrilu0(), rocsparse_cbsrilu0()
 *              and rocsparse_zbsrilu0().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p nnzb, or \p block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
 *              \p bsr_col_ind, \p info or \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_int             mb,
                                                rocsparse_int             nnzb,
                                                const rocsparse_mat_descr descr,
                                                const float*              bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_int             mb,
                                                rocsparse_int             nnzb,
                                                const rocsparse_mat_descr descr,
                                                const double*             bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrilu0_buffer_size(rocsparse_handle               handle,
                                                rocsparse_direction            dir,
                                                rocsparse_int                  mb,
                                                rocsparse_int                  nnzb,
                                                const rocsparse_mat_descr      descr,
                                                const rocsparse_float_complex* bsr_val,
                                                const rocsparse_int*           bsr_row_ptr,
                                                const rocsparse_int*           bsr_col_ind,
                                                rocsparse_int                  block_dim,
                                                rocsparse_mat_info             info,
                                                size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrilu0_buffer_size(rocsparse_handle                handle,
                                                rocsparse_direction             dir,
                                                rocsparse_int                   mb,
                                                rocsparse_int                   nnzb,
                                                const rocsparse_mat_descr       descr,
                                                const rocsparse_double_complex* bsr_val,
                                                const rocsparse_int*            bsr_row_ptr,
                                                const rocsparse_int*            bsr_col_ind,
                                                rocsparse_int                   block_dim,
                                                rocsparse_mat_info              info,
                                                size_t*                         buffer_size);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_analysis performs the analysis step for rocsparse_sbsrilu0()
 *  rocsparse_dbsrilu0(), rocsparse_cbsrilu0(), and rocsparse_zbsrilu0(). It is expected
 *  that this function will be executed only once for a given matrix. The analysis meta
 *  data can be cleared by rocsparse_bsrilu0_clear().
 *
 *  \p rocsparse_bsrilu0_analysis can share its meta data with
 *  rocsparse_sbsric0_analysis(), rocsparse_dbsric0_analysis(),
 *  rocsparse_cbsric0_analysis(), rocsparse_zbsric0_analysis(),
 *  rocsparse_sbsrsv_analysis(), rocsparse_dbsrsv_analysis(),
 *  rocsparse_cbsrsv_analysis(), rocsparse_zbsrsv_analysis(),
 *  rocsparse_sbsrsm_analysis(), rocsparse_dbsrsm_analysis(),
 *  rocsparse_cbsrsm_analysis() and rocsparse_zbsrsm_analysis(). Selecting
 *  \ref rocsparse_analysis_policy_reuse policy can greatly improve computation
 *  performance of meta data. However, the user need to make sure that the sparsity
 *  pattern remains unchanged. If this cannot be assured,
 *  \ref rocsparse_analysis_policy_force has to be used.
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
 *  dir         direction that specified whether to count nonzero elements by
 *              \ref rocsparse_direction_row or by \ref rocsparse_direction_column.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsr_val     array of length \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  block_dim   the block dimension of the BSR matrix. Between 1 and m where \p m=mb*block_dim.
 *  @param[out]
 *  info        structure that holds the information collected during
 *              the analysis step.
 *  @param[in]
 *  analysis    \ref rocsparse_analysis_policy_reuse or
 *              \ref rocsparse_analysis_policy_force.
 *  @param[in]
 *  solve       \ref rocsparse_solve_policy_auto.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p nnzb, or \p block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
 *              \p bsr_col_ind, \p info or \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrilu0_analysis(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_int             mb,
                                             rocsparse_int             nnzb,
                                             const rocsparse_mat_descr descr,
                                             const float*              bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             block_dim,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrilu0_analysis(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_int             mb,
                                             rocsparse_int             nnzb,
                                             const rocsparse_mat_descr descr,
                                             const double*             bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             block_dim,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrilu0_analysis(rocsparse_handle               handle,
                                             rocsparse_direction            dir,
                                             rocsparse_int                  mb,
                                             rocsparse_int                  nnzb,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* bsr_val,
                                             const rocsparse_int*           bsr_row_ptr,
                                             const rocsparse_int*           bsr_col_ind,
                                             rocsparse_int                  block_dim,
                                             rocsparse_mat_info             info,
                                             rocsparse_analysis_policy      analysis,
                                             rocsparse_solve_policy         solve,
                                             void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrilu0_analysis(rocsparse_handle                handle,
                                             rocsparse_direction             dir,
                                             rocsparse_int                   mb,
                                             rocsparse_int                   nnzb,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* bsr_val,
                                             const rocsparse_int*            bsr_row_ptr,
                                             const rocsparse_int*            bsr_col_ind,
                                             rocsparse_int                   block_dim,
                                             rocsparse_mat_info              info,
                                             rocsparse_analysis_policy       analysis,
                                             rocsparse_solve_policy          solve,
                                             void*                           temp_buffer);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_clear deallocates all memory that was allocated by
 *  rocsparse_sbsrilu0_analysis(), rocsparse_dbsrilu0_analysis(), rocsparse_cbsrilu0_analysis()
 *  or rocsparse_zbsrilu0_analysis(). This is especially useful, if memory is an issue and
 *  the analysis data is not required for further computation.
 *
 *  \note
 *  Calling \p rocsparse_bsrilu0_clear is optional. All allocated resources will be
 *  cleared, when the opaque \ref rocsparse_mat_info struct is destroyed using
 *  rocsparse_destroy_mat_info().
 *
 *  \note
 *  This routine does not support execution in a hipGraph context.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[inout]
 *  info        structure that holds the information collected during the analysis step.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
 *  \retval     rocsparse_status_memory_error the buffer holding the meta data could not
 *              be deallocated.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsrilu0_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0 computes the incomplete LU factorization with 0 fill-ins and no
 *  pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LU
 *  \f]
 *
 *  \p rocsparse_bsrilu0 requires a user allocated temporary buffer. Its size is returned
 *  by rocsparse_sbsrilu0_buffer_size(), rocsparse_dbsrilu0_buffer_size(),
 *  rocsparse_cbsrilu0_buffer_size() or rocsparse_zbsrilu0_buffer_size(). Furthermore,
 *  analysis meta data is required. It can be obtained by rocsparse_sbsrilu0_analysis(),
 *  rocsparse_dbsrilu0_analysis(), rocsparse_cbsrilu0_analysis() or
 *  rocsparse_zbsrilu0_analysis(). \p rocsparse_bsrilu0 reports the first zero pivot
 *  (either numerical or structural zero). The zero pivot status can be obtained by
 *  calling rocsparse_bsrilu0_zero_pivot().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  This routine supports execution in a hipGraph context.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  dir         direction that specified whether to count nonzero elements by
 *              \ref rocsparse_direction_row or by \ref rocsparse_direction_column.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix.
 *  @param[inout]
 *  bsr_val     array of length \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  block_dim   the block dimension of the BSR matrix. Between 1 and m where \p m=mb*block_dim.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy      \ref rocsparse_solve_policy_auto.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p nnzb, or \p block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr
 *              or \p bsr_col_ind pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  Consider the sparse \f$m \times m\f$ matrix \f$A\f$, stored in BSR
 *  storage format. The following example computes the incomplete LU factorization
 *  \f$M \approx LU\f$ and solves the preconditioned system \f$My = x\f$.
 *  \code{.c}
 *      // Create rocSPARSE handle
 *      rocsparse_handle handle;
 *      rocsparse_create_handle(&handle);
 *
 *      // Create matrix descriptor for M
 *      rocsparse_mat_descr descr_M;
 *      rocsparse_create_mat_descr(&descr_M);
 *
 *      // Create matrix descriptor for L
 *      rocsparse_mat_descr descr_L;
 *      rocsparse_create_mat_descr(&descr_L);
 *      rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower);
 *      rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_unit);
 *
 *      // Create matrix descriptor for U
 *      rocsparse_mat_descr descr_U;
 *      rocsparse_create_mat_descr(&descr_U);
 *      rocsparse_set_mat_fill_mode(descr_U, rocsparse_fill_mode_upper);
 *      rocsparse_set_mat_diag_type(descr_U, rocsparse_diag_type_non_unit);
 *
 *      // Create matrix info structure
 *      rocsparse_mat_info info;
 *      rocsparse_create_mat_info(&info);
 *
 *      // Obtain required buffer size
 *      size_t buffer_size_M;
 *      size_t buffer_size_L;
 *      size_t buffer_size_U;
 *      rocsparse_dbsrilu0_buffer_size(handle,
 *                                     rocsparse_direction_row,
 *                                     mb,
 *                                     nnzb,
 *                                     descr_M,
 *                                     bsr_val,
 *                                     bsr_row_ptr,
 *                                     bsr_col_ind,
 *                                     block_dim,
 *                                     info,
 *                                     &buffer_size_M);
 *      rocsparse_dbsrsv_buffer_size(handle,
 *                                   rocsparse_direction_row,
 *                                   rocsparse_operation_none,
 *                                   mb,
 *                                   nnzb,
 *                                   descr_L,
 *                                   bsr_val,
 *                                   bsr_row_ptr,
 *                                   bsr_col_ind,
 *                                   block_dim,
 *                                   info,
 *                                   &buffer_size_L);
 *      rocsparse_dbsrsv_buffer_size(handle,
 *                                   rocsparse_direction_row,
 *                                   rocsparse_operation_none,
 *                                   mb,
 *                                   nnzb,
 *                                   descr_U,
 *                                   bsr_val,
 *                                   bsr_row_ptr,
 *                                   bsr_col_ind,
 *                                   block_dim,
 *                                   info,
 *                                   &buffer_size_U);
 *
 *      size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_U));
 *
 *      // Allocate temporary buffer
 *      void* temp_buffer;
 *      hipMalloc(&temp_buffer, buffer_size);
 *
 *      // Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
 *      // computation performance
 *      rocsparse_dbsrilu0_analysis(handle,
 *                                  rocsparse_direction_row,
 *                                  mb,
 *                                  nnzb,
 *                                  descr_M,
 *                                  bsr_val,
 *                                  bsr_row_ptr,
 *                                  bsr_col_ind,
 *                                  block_dim,
 *                                  info,
 *                                  rocsparse_analysis_policy_reuse,
 *                                  rocsparse_solve_policy_auto,
 *                                  temp_buffer);
 *      rocsparse_dbsrsv_analysis(handle,
 *                                rocsparse_direction_row,
 *                                rocsparse_operation_none,
 *                                mb,
 *                                nnzb,
 *                                descr_L,
 *                                bsr_val,
 *                                bsr_row_ptr,
 *                                bsr_col_ind,
 *                                block_dim,
 *                                info,
 *                                rocsparse_analysis_policy_reuse,
 *                                rocsparse_solve_policy_auto,
 *                                temp_buffer);
 *      rocsparse_dbsrsv_analysis(handle,
 *                                rocsparse_direction_row,
 *                                rocsparse_operation_none,
 *                                mb,
 *                                nnzb,
 *                                descr_U,
 *                                bsr_val,
 *                                bsr_row_ptr,
 *                                bsr_col_ind,
 *                                block_dim,
 *                                info,
 *                                rocsparse_analysis_policy_reuse,
 *                                rocsparse_solve_policy_auto,
 *                                temp_buffer);
 *
 *      // Check for zero pivot
 *      rocsparse_int position;
 *      if(rocsparse_status_zero_pivot == rocsparse_bsrilu0_zero_pivot(handle,
 *                                                                    info,
 *                                                                    &position))
 *      {
 *          printf("A has structural zero at A(%d,%d)\n", position, position);
 *      }
 *
 *      // Compute incomplete LU factorization M = LU
 *      rocsparse_dbsrilu0(handle,
 *                         rocsparse_direction_row,
 *                         mb,
 *                         nnzb,
 *                         descr_M,
 *                         bsr_val,
 *                         bsr_row_ptr,
 *                         bsr_col_ind,
 *                         block_dim,
 *                         info,
 *                         rocsparse_solve_policy_auto,
 *                         temp_buffer);
 *
 *      // Check for zero pivot
 *      if(rocsparse_status_zero_pivot == rocsparse_bsrilu0_zero_pivot(handle,
 *                                                                     info,
 *                                                                     &position))
 *      {
 *          printf("L has structural and/or numerical zero at L(%d,%d)\n",
 *                 position,
 *                 position);
 *      }
 *
 *      // Solve Lz = x
 *      rocsparse_dbsrsv_solve(handle,
 *                             rocsparse_direction_row,
 *                             rocsparse_operation_none,
 *                             mb,
 *                             nnzb,
 *                             &alpha,
 *                             descr_L,
 *                             bsr_val,
 *                             bsr_row_ptr,
 *                             bsr_col_ind,
 *                             block_dim,
 *                             info,
 *                             x,
 *                             z,
 *                             rocsparse_solve_policy_auto,
 *                             temp_buffer);
 *
 *      // Solve Uy = z
 *      rocsparse_dbsrsv_solve(handle,
 *                             rocsparse_direction_row,
 *                             rocsparse_operation_none,
 *                             mb,
 *                             nnzb,
 *                             &alpha,
 *                             descr_U,
 *                             bsr_val,
 *                             bsr_row_ptr,
 *                             bsr_col_ind,
 *                             block_dim,
 *                             info,
 *                             z,
 *                             y,
 *                             rocsparse_solve_policy_auto,
 *                             temp_buffer);
 *
 *      // Clean up
 *      hipFree(temp_buffer);
 *      rocsparse_destroy_mat_info(info);
 *      rocsparse_destroy_mat_descr(descr_M);
 *      rocsparse_destroy_mat_descr(descr_L);
 *      rocsparse_destroy_mat_descr(descr_U);
 *      rocsparse_destroy_handle(handle);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrilu0(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nnzb,
                                    const rocsparse_mat_descr descr,
                                    float*                    bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrilu0(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nnzb,
                                    const rocsparse_mat_descr descr,
                                    double*                   bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrilu0(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nnzb,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_float_complex*  bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrilu0(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nnzb,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_double_complex* bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSRILU0_H */
