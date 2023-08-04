/*! \file */
/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_CSRILU0_H
#define ROCSPARSE_CSRILU0_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csrilu0_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
*  structural or numerical zero has been found during rocsparse_scsrilu0(),
*  rocsparse_dcsrilu0(), rocsparse_ccsrilu0() or rocsparse_zcsrilu0() computation. The
*  first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same index
*  base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_csrilu0_zero_pivot is a blocking function. It might influence
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
rocsparse_status rocsparse_csrilu0_zero_pivot(rocsparse_handle   handle,
                                              rocsparse_mat_info info,
                                              rocsparse_int*     position);

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_csrilu0_numeric_boost enables the user to replace a numerical value in
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
rocsparse_status rocsparse_scsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const float*       boost_tol,
                                                  const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const double*      boost_tol,
                                                  const double*      boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0_numeric_boost(rocsparse_handle               handle,
                                                  rocsparse_mat_info             info,
                                                  int                            enable_boost,
                                                  const float*                   boost_tol,
                                                  const rocsparse_float_complex* boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0_numeric_boost(rocsparse_handle                handle,
                                                  rocsparse_mat_info              info,
                                                  int                             enable_boost,
                                                  const double*                   boost_tol,
                                                  const rocsparse_double_complex* boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dscsrilu0_numeric_boost(rocsparse_handle   handle,
                                                   rocsparse_mat_info info,
                                                   int                enable_boost,
                                                   const double*      boost_tol,
                                                   const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dccsrilu0_numeric_boost(rocsparse_handle               handle,
                                                   rocsparse_mat_info             info,
                                                   int                            enable_boost,
                                                   const double*                  boost_tol,
                                                   const rocsparse_float_complex* boost_val);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csrilu0_buffer_size returns the size of the temporary storage buffer
*  that is required by rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
*  rocsparse_ccsrilu0_analysis(), rocsparse_zcsrilu0_analysis(), rocsparse_scsrilu0(),
*  rocsparse_dcsrilu0(), rocsparse_ccsrilu0() and rocsparse_zcsrilu0(). The temporary
*  storage buffer must be allocated by the user. The size of the temporary storage
*  buffer is identical to the size returned by rocsparse_scsrsv_buffer_size(),
*  rocsparse_dcsrsv_buffer_size(), rocsparse_ccsrsv_buffer_size() and
*  rocsparse_zcsrsv_buffer_size() if the matrix sparsity pattern is identical. The user
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
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
*              rocsparse_ccsrilu0_analysis(), rocsparse_zcsrilu0_analysis(),
*              rocsparse_scsrilu0(), rocsparse_dcsrilu0(), rocsparse_ccsrilu0() and
*              rocsparse_zcsrilu0().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind, \p info or \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const float*              csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const double*             csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0_buffer_size(rocsparse_handle               handle,
                                                rocsparse_int                  m,
                                                rocsparse_int                  nnz,
                                                const rocsparse_mat_descr      descr,
                                                const rocsparse_float_complex* csr_val,
                                                const rocsparse_int*           csr_row_ptr,
                                                const rocsparse_int*           csr_col_ind,
                                                rocsparse_mat_info             info,
                                                size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0_buffer_size(rocsparse_handle                handle,
                                                rocsparse_int                   m,
                                                rocsparse_int                   nnz,
                                                const rocsparse_mat_descr       descr,
                                                const rocsparse_double_complex* csr_val,
                                                const rocsparse_int*            csr_row_ptr,
                                                const rocsparse_int*            csr_col_ind,
                                                rocsparse_mat_info              info,
                                                size_t*                         buffer_size);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csrilu0_analysis performs the analysis step for rocsparse_scsrilu0(),
*  rocsparse_dcsrilu0(), rocsparse_ccsrilu0() and rocsparse_zcsrilu0(). It is expected
*  that this function will be executed only once for a given matrix and particular
*  operation type. The analysis meta data can be cleared by rocsparse_csrilu0_clear().
*
*  \p rocsparse_csrilu0_analysis can share its meta data with
*  rocsparse_scsric0_analysis(), rocsparse_dcsric0_analysis(),
*  rocsparse_ccsric0_analysis(), rocsparse_zcsric0_analysis(),
*  rocsparse_scsrsv_analysis(), rocsparse_dcsrsv_analysis(),
*  rocsparse_ccsrsv_analysis(), rocsparse_zcsrsv_analysis(),
*  rocsparse_scsrsm_analysis(), rocsparse_dcsrsm_analysis(),
*  rocsparse_scsrsm_analysis() and rocsparse_dcsrsm_analysis(). Selecting
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
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
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
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind, \p info or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrilu0_analysis(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             const float*              csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0_analysis(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             const double*             csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0_analysis(rocsparse_handle               handle,
                                             rocsparse_int                  m,
                                             rocsparse_int                  nnz,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* csr_val,
                                             const rocsparse_int*           csr_row_ptr,
                                             const rocsparse_int*           csr_col_ind,
                                             rocsparse_mat_info             info,
                                             rocsparse_analysis_policy      analysis,
                                             rocsparse_solve_policy         solve,
                                             void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0_analysis(rocsparse_handle                handle,
                                             rocsparse_int                   m,
                                             rocsparse_int                   nnz,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* csr_val,
                                             const rocsparse_int*            csr_row_ptr,
                                             const rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info              info,
                                             rocsparse_analysis_policy       analysis,
                                             rocsparse_solve_policy          solve,
                                             void*                           temp_buffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csrilu0_clear deallocates all memory that was allocated by
*  rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
*  rocsparse_ccsrilu0_analysis() or rocsparse_zcsrilu0_analysis(). This is especially
*  useful, if memory is an issue and the analysis data is not required for further
*  computation.
*
*  \note
*  Calling \p rocsparse_csrilu0_clear is optional. All allocated resources will be
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
rocsparse_status rocsparse_csrilu0_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csrilu0 computes the incomplete LU factorization with 0 fill-ins and no
*  pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LU
*  \f]
*
*  \p rocsparse_csrilu0 requires a user allocated temporary buffer. Its size is returned
*  by rocsparse_scsrilu0_buffer_size(), rocsparse_dcsrilu0_buffer_size(),
*  rocsparse_ccsrilu0_buffer_size() or rocsparse_zcsrilu0_buffer_size(). Furthermore,
*  analysis meta data is required. It can be obtained by rocsparse_scsrilu0_analysis(),
*  rocsparse_dcsrilu0_analysis(), rocsparse_ccsrilu0_analysis() or
*  rocsparse_zcsrilu0_analysis(). \p rocsparse_csrilu0 reports the first zero pivot
*  (either numerical or structural zero). The zero pivot status can be obtained by
*  calling rocsparse_csrilu0_zero_pivot().
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
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
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
*  @param[inout]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  policy      \ref rocsparse_solve_policy_auto.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr
*              or \p csr_col_ind pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  Consider the sparse \f$m \times m\f$ matrix \f$A\f$, stored in CSR
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
*      rocsparse_dcsrilu0_buffer_size(handle,
*                                    m,
*                                    nnz,
*                                    descr_M,
*                                    csr_val,
*                                    csr_row_ptr,
*                                    csr_col_ind,
*                                    info,
*                                    &buffer_size_M);
*      rocsparse_dcsrsv_buffer_size(handle,
*                                   rocsparse_operation_none,
*                                   m,
*                                   nnz,
*                                   descr_L,
*                                   csr_val,
*                                   csr_row_ptr,
*                                   csr_col_ind,
*                                   info,
*                                   &buffer_size_L);
*      rocsparse_dcsrsv_buffer_size(handle,
*                                   rocsparse_operation_none,
*                                   m,
*                                   nnz,
*                                   descr_U,
*                                   csr_val,
*                                   csr_row_ptr,
*                                   csr_col_ind,
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
*      rocsparse_dcsrilu0_analysis(handle,
*                                  m,
*                                  nnz,
*                                  descr_M,
*                                  csr_val,
*                                  csr_row_ptr,
*                                  csr_col_ind,
*                                  info,
*                                  rocsparse_analysis_policy_reuse,
*                                  rocsparse_solve_policy_auto,
*                                  temp_buffer);
*      rocsparse_dcsrsv_analysis(handle,
*                                rocsparse_operation_none,
*                                m,
*                                nnz,
*                                descr_L,
*                                csr_val,
*                                csr_row_ptr,
*                                csr_col_ind,
*                                info,
*                                rocsparse_analysis_policy_reuse,
*                                rocsparse_solve_policy_auto,
*                                temp_buffer);
*      rocsparse_dcsrsv_analysis(handle,
*                                rocsparse_operation_none,
*                                m,
*                                nnz,
*                                descr_U,
*                                csr_val,
*                                csr_row_ptr,
*                                csr_col_ind,
*                                info,
*                                rocsparse_analysis_policy_reuse,
*                                rocsparse_solve_policy_auto,
*                                temp_buffer);
*
*      // Check for zero pivot
*      rocsparse_int position;
*      if(rocsparse_status_zero_pivot == rocsparse_csrilu0_zero_pivot(handle,
*                                                                     info,
*                                                                     &position))
*      {
*          printf("A has structural zero at A(%d,%d)\n", position, position);
*      }
*
*      // Compute incomplete LU factorization
*      rocsparse_dcsrilu0(handle,
*                         m,
*                         nnz,
*                         descr_M,
*                         csr_val,
*                         csr_row_ptr,
*                         csr_col_ind,
*                         info,
*                         rocsparse_solve_policy_auto,
*                         temp_buffer);
*
*      // Check for zero pivot
*      if(rocsparse_status_zero_pivot == rocsparse_csrilu0_zero_pivot(handle,
*                                                                     info,
*                                                                     &position))
*      {
*          printf("U has structural and/or numerical zero at U(%d,%d)\n",
*                 position,
*                 position);
*      }
*
*      // Solve Lz = x
*      rocsparse_dcsrsv_solve(handle,
*                             rocsparse_operation_none,
*                             m,
*                             nnz,
*                             &alpha,
*                             descr_L,
*                             csr_val,
*                             csr_row_ptr,
*                             csr_col_ind,
*                             info,
*                             x,
*                             z,
*                             rocsparse_solve_policy_auto,
*                             temp_buffer);
*
*      // Solve Uy = z
*      rocsparse_dcsrsv_solve(handle,
*                             rocsparse_operation_none,
*                             m,
*                             nnz,
*                             &alpha,
*                             descr_U,
*                             csr_val,
*                             csr_row_ptr,
*                             csr_col_ind,
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
rocsparse_status rocsparse_scsrilu0(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             nnz,
                                    const rocsparse_mat_descr descr,
                                    float*                    csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             nnz,
                                    const rocsparse_mat_descr descr,
                                    double*                   csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             nnz,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_float_complex*  csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             nnz,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_double_complex* csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRILU0_H */
