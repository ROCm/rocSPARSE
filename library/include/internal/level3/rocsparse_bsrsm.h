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

#ifndef ROCSPARSE_BSRSM_H
#define ROCSPARSE_BSRSM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsm_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
*  structural or numerical zero has been found during rocsparse_sbsrsm_solve(),
*  rocsparse_dbsrsm_solve(), rocsparse_cbsrsm_solve() or rocsparse_zbsrsm_solve()
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
*  using same index base as the BSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_bsrsm_zero_pivot is a blocking function. It might influence
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
rocsparse_status rocsparse_bsrsm_zero_pivot(rocsparse_handle   handle,
                                            rocsparse_mat_info info,
                                            rocsparse_int*     position);

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsm_buffer_size returns the size of the temporary storage buffer that
*  is required by rocsparse_sbsrsm_analysis(), rocsparse_dbsrsm_analysis(),
*  rocsparse_cbsrsm_analysis(), rocsparse_zbsrsm_analysis(), rocsparse_sbsrsm_solve(),
*  rocsparse_dbsrsm_solve(), rocsparse_cbsrsm_solve() and rocsparse_zbsrsm_solve(). The
*  temporary storage buffer must be allocated by the user.
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
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans_A     matrix A operation type.
*  @param[in]
*  trans_X     matrix X operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix A.
*  @param[in]
*  nrhs        number of columns of the column-oriented dense matrix op(X).
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix A.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix A.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  block_dim   block dimension of the sparse BSR matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_sbsrsm_analysis(), rocsparse_dbsrsm_analysis(),
*              rocsparse_cbsrsm_analysis(), rocsparse_zbsrsm_analysis(),
*              rocsparse_sbsrsm_solve(), rocsparse_dbsrsm_solve(),
*              rocsparse_cbsrsm_solve() and rocsparse_zbsrsm_solve().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nrhs, \p nnzb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p info or \p buffer_size pointer
*              is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans_A == \ref rocsparse_operation_conjugate_transpose,
*              \p trans_X == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrsm_buffer_size(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_X,
                                              rocsparse_int             mb,
                                              rocsparse_int             nrhs,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              const float*              bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrsm_buffer_size(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_X,
                                              rocsparse_int             mb,
                                              rocsparse_int             nrhs,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              const double*             bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrsm_buffer_size(rocsparse_handle               handle,
                                              rocsparse_direction            dir,
                                              rocsparse_operation            trans_A,
                                              rocsparse_operation            trans_X,
                                              rocsparse_int                  mb,
                                              rocsparse_int                  nrhs,
                                              rocsparse_int                  nnzb,
                                              const rocsparse_mat_descr      descr,
                                              const rocsparse_float_complex* bsr_val,
                                              const rocsparse_int*           bsr_row_ptr,
                                              const rocsparse_int*           bsr_col_ind,
                                              rocsparse_int                  block_dim,
                                              rocsparse_mat_info             info,
                                              size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrsm_buffer_size(rocsparse_handle                handle,
                                              rocsparse_direction             dir,
                                              rocsparse_operation             trans_A,
                                              rocsparse_operation             trans_X,
                                              rocsparse_int                   mb,
                                              rocsparse_int                   nrhs,
                                              rocsparse_int                   nnzb,
                                              const rocsparse_mat_descr       descr,
                                              const rocsparse_double_complex* bsr_val,
                                              const rocsparse_int*            bsr_row_ptr,
                                              const rocsparse_int*            bsr_col_ind,
                                              rocsparse_int                   block_dim,
                                              rocsparse_mat_info              info,
                                              size_t*                         buffer_size);
/**@}*/

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsm_analysis performs the analysis step for rocsparse_sbsrsm_solve(),
*  rocsparse_dbsrsm_solve(), rocsparse_cbsrsm_solve() and rocsparse_zbsrsm_solve(). It
*  is expected that this function will be executed only once for a given matrix and
*  particular operation type. The analysis meta data can be cleared by
*  rocsparse_bsrsm_clear().
*
*  \p rocsparse_bsrsm_analysis can share its meta data with
*  rocsparse_sbsrilu0_analysis(), rocsparse_dbsrilu0_analysis(),
*  rocsparse_cbsrilu0_analysis(), rocsparse_zbsrilu0_analysis(),
*  rocsparse_sbsric0_analysis(), rocsparse_dbsric0_analysis(),
*  rocsparse_cbsric0_analysis(), rocsparse_zbsric0_analysis(),
*  rocsparse_sbsrsv_analysis(), rocsparse_dbsrsv_analysis(),
*  rocsparse_cbsrsv_analysis() and rocsparse_zbsrsv_analysis(). Selecting
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
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans_A     matrix A operation type.
*  @param[in]
*  trans_X     matrix X operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix A.
*  @param[in]
*  nrhs        number of columns of the column-oriented dense matrix op(X).
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix A.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix A.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix A.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix A.
*  @param[in]
*  bsr_col_ind array of \p nnzb containing the block column indices of the sparse
*              BSR matrix A.
*  @param[in]
*  block_dim   block dimension of the sparse BSR matrix A.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
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
*  \retval     rocsparse_status_invalid_size \p mb, \p nrhs, \p nnzb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
*              \p bsr_col_ind, \p info or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans_A == \ref rocsparse_operation_conjugate_transpose,
*              \p trans_X == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrsm_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_X,
                                           rocsparse_int             mb,
                                           rocsparse_int             nrhs,
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
rocsparse_status rocsparse_dbsrsm_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_X,
                                           rocsparse_int             mb,
                                           rocsparse_int             nrhs,
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
rocsparse_status rocsparse_cbsrsm_analysis(rocsparse_handle               handle,
                                           rocsparse_direction            dir,
                                           rocsparse_operation            trans_A,
                                           rocsparse_operation            trans_X,
                                           rocsparse_int                  mb,
                                           rocsparse_int                  nrhs,
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
rocsparse_status rocsparse_zbsrsm_analysis(rocsparse_handle                handle,
                                           rocsparse_direction             dir,
                                           rocsparse_operation             trans_A,
                                           rocsparse_operation             trans_X,
                                           rocsparse_int                   mb,
                                           rocsparse_int                   nrhs,
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

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsm_clear deallocates all memory that was allocated by
*  rocsparse_sbsrsm_analysis(), rocsparse_dbsrsm_analysis(), rocsparse_cbsrsm_analysis()
*  or rocsparse_zbsrsm_analysis(). This is especially useful, if memory is an issue and
*  the analysis data is not required for further computation, e.g. when switching to
*  another sparse matrix format. Calling \p rocsparse_bsrsm_clear is optional. All
*  allocated resources will be cleared, when the opaque \ref rocsparse_mat_info struct
*  is destroyed using rocsparse_destroy_mat_info().
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
rocsparse_status rocsparse_bsrsm_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsm_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix (where \f$m = mb \times block\_dim\f$), defined in BSR storage format, a column-oriented dense solution matrix
*  \f$X\f$ and the column-oriented dense right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot op(X) = \alpha \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == rocsparse_operation_none} \\
*        A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  ,
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_X == rocsparse_operation_none} \\
*        B^T, & \text{if trans_X == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_X == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        X,   & \text{if trans_X == rocsparse_operation_none} \\
*        X^T, & \text{if trans_X == rocsparse_operation_transpose} \\
*        X^H, & \text{if trans_X == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  Note that as indicated above, the operation type of both \f$op(B)\f$ and \f$op(X)\f$ is specified by the
*  \p trans_X parameter and that the operation type of B and X must match. For example, if \f$op(B)=B\f$ then
*  \f$op(X)=X\f$. Likewise, if \f$op(B)=B^T\f$ then \f$op(X)=X^T\f$.
*
*  Given that the sparse matrix A is a square matrix, its size is \f$m \times m\f$ regardless of
*  whether A is transposed or not. The size of the column-oriented dense matrices B and X have
*  size that depends on the value of \p trans_X:
*
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        ldb \times nrhs, \text{  } ldb \ge m, & \text{if trans_X == rocsparse_operation_none} \\
*        ldb \times m, \text{  } ldb \ge nrhs,  & \text{if trans_X == rocsparse_operation_transpose} \\
*        ldb \times m, \text{  } ldb \ge nrhs, & \text{if trans_X == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        ldb \times nrhs, \text{  } ldb \ge m, & \text{if trans_X == rocsparse_operation_none} \\
*        ldb \times m, \text{  } ldb \ge nrhs,  & \text{if trans_X == rocsparse_operation_transpose} \\
*        ldb \times m, \text{  } ldb \ge nrhs, & \text{if trans_X == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \p rocsparse_bsrsm_solve requires a user allocated temporary buffer. Its size is returned by
*  rocsparse_sbsrsm_buffer_size(), rocsparse_dbsrsm_buffer_size(), rocsparse_cbsrsm_buffer_size()
*  or rocsparse_zbsrsm_buffer_size(). The size of the required buffer is larger when  \p trans_A equals
*  \ref rocsparse_operation_transpose or \ref rocsparse_operation_conjugate_transpose and when \p trans_X
*  is \ref rocsparse_operation_none. The subsequent solve will also be faster when \f$A\f$ is non-transposed
*  and \f$B\f$ is transposed (or conjugate transposed). For example, instead of solving:
*
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c}
*       a_{00} & a_{01} \\
*       a_{10} & a_{11}
*      \end{array} &
*      \begin{array}{c c}
*       0 & 0 \\
*       0 & 0
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       a_{20} & a_{21} \\
*       a_{30} & a_{31}
*      \end{array} &
*      \begin{array}{c c}
*       a_{22} & a_{23} \\
*       a_{32} & a_{33}
*      \end{array} \\
*    \end{array}
*   \right]
*    \cdot
*    \begin{bmatrix}
*    x_{00} & x_{01} \\
*    x_{10} & x_{11} \\
*    x_{20} & x_{21} \\
*    x_{30} & x_{31} \\
*    \end{bmatrix}
*    =
*    \begin{bmatrix}
*    b_{00} & b_{01} \\
*    b_{10} & b_{11} \\
*    b_{20} & b_{21} \\
*    b_{30} & b_{31} \\
*    \end{bmatrix}
*  \f]
*
*  Consider solving:
*
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c}
*       a_{00} & a_{01} \\
*       a_{10} & a_{11}
*      \end{array} &
*      \begin{array}{c c}
*       0 & 0 \\
*       0 & 0
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       a_{20} & a_{21} \\
*       a_{30} & a_{31}
*      \end{array} &
*      \begin{array}{c c}
*       a_{22} & a_{23} \\
*       a_{32} & a_{33}
*      \end{array} \\
*    \end{array}
*   \right]
*    \cdot
*    \begin{bmatrix}
*    x_{00} & x_{10} & x_{20} & x_{30} \\
*    x_{01} & x_{11} & x_{21} & x_{31}
*    \end{bmatrix}^{T}
*    =
*    \begin{bmatrix}
*    b_{00} & b_{10} & b_{20} & b_{30} \\
*    b_{01} & b_{11} & b_{21} & b_{31}
*    \end{bmatrix}^{T}
*  \f]
*
*  Once the temporary storage buffer has been allocated, analysis meta data is required. It can be obtained
*  by rocsparse_sbsrsm_analysis(), rocsparse_dbsrsm_analysis(), rocsparse_cbsrsm_analysis() or
*  rocsparse_zbsrsm_analysis().
*
*  \p rocsparse_bsrsm_solve reports the first zero pivot (either numerical or structural zero). The zero pivot
*  status can be checked calling rocsparse_bsrsm_zero_pivot(). If \ref rocsparse_diag_type == \ref rocsparse_diag_type_unit,
*  no zero pivot will be reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
*
*  \note
*  The sparse BSR matrix has to be sorted.
*
*  \note
*  Operation type of B and X must match, for example if \f$op(B)=B\f$ then \f$op(X)=X\f$.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans_A != \ref rocsparse_operation_conjugate_transpose and
*  \p trans_X != \ref rocsparse_operation_conjugate_transpose is supported.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans_A     matrix A operation type.
*  @param[in]
*  trans_X     matrix X operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix A.
*  @param[in]
*  nrhs        number of columns of the column-oriented dense matrix op(X).
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix A.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix A.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  block_dim   block dimension of the sparse BSR matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  B           column-oriented dense matrix B with leading dimension \p ldb.
*  @param[in]
*  ldb         leading dimension of rhs matrix B.
*  @param[out]
*  X           column-oriented dense solution matrix X with leading dimension \p ldx.
*  @param[in]
*  ldx         leading dimension of solution matrix X.
*  @param[in]
*  policy      \ref rocsparse_solve_policy_auto.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nrhs, \p nnzb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p alpha, \p descr, \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p B, \p X \p info or \p temp_buffer pointer
*              is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans_A == \ref rocsparse_operation_conjugate_transpose,
*              \p trans_X == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  Consider the lower triangular \f$m \times m\f$ matrix \f$L\f$, stored in BSR
*  storage format with non-unit diagonal. The following example solves \f$L \cdot X = B\f$.
*  \code{.c}
*      // rocSPARSE handle
*      rocsparse_handle handle;
*      rocsparse_create_handle(&handle);
*
*      // A = ( 1.0  0.0  0.0  0.0 )
*      //     ( 2.0  3.0  0.0  0.0 )
*      //     ( 4.0  5.0  6.0  0.0 )
*      //     ( 7.0  0.0  8.0  9.0 )
*      //
*      // with bsr_dim = 2
*      //
*      //      -------------------
*      //   = | 1.0 0.0 | 0.0 0.0 |
*      //     | 2.0 3.0 | 0.0 0.0 |
*      //      -------------------
*      //     | 4.0 5.0 | 6.0 0.0 |
*      //     | 7.0 0.0 | 8.0 9.0 |
*      //      -------------------
*
*      // Number of rows and columns
*      rocsparse_int m = 4;
*
*      // Number of block rows and block columns
*      rocsparse_int mb = 2;
*      rocsparse_int nb = 2;
*
*      // BSR block dimension
*      rocsparse_int bsr_dim = 2;
*
*      // Number of right-hand-sides
*      rocsparse_int nrhs = 4;
*
*      // Number of non-zero blocks
*      rocsparse_int nnzb = 3;
*
*      // BSR row pointers
*      rocsparse_int hbsr_row_ptr[3] = {0, 1, 3};
*
*      // BSR column indices
*      rocsparse_int hbsr_col_ind[3] = {0, 0, 1};
*
*      // BSR values
*      double hbsr_val[12] = {1.0, 2.0, 0.0, 3.0, 4.0, 7.0, 5.0, 0.0, 6.0, 8.0, 0.0, 9.0};
*
*      // Storage scheme of the BSR blocks
*      rocsparse_direction dir = rocsparse_direction_column;
*
*      // Transposition of the matrix and rhs matrix
*      rocsparse_operation transA = rocsparse_operation_none;
*      rocsparse_operation transX = rocsparse_operation_none;
*
*      // Analysis policy
*      rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;
*
*      // Solve policy
*      rocsparse_solve_policy solve_policy = rocsparse_solve_policy_auto;
*
*      // Scalar alpha and beta
*      double alpha = 3.7;
*
*      // rhs and solution matrix
*      rocsparse_int ldb = nb * bsr_dim;
*      rocsparse_int ldx = mb * bsr_dim;
*
*      double hB[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
*      double hX[16];
*
*      // Offload data to device
*      rocsparse_int* dbsr_row_ptr;
*      rocsparse_int* dbsr_col_ind;
*      double*        dbsr_val;
*      double*        dB;
*      double*        dX;
*
*      hipMalloc((void**)&dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1));
*      hipMalloc((void**)&dbsr_col_ind, sizeof(rocsparse_int) * nnzb);
*      hipMalloc((void**)&dbsr_val, sizeof(double) * nnzb * bsr_dim * bsr_dim);
*      hipMalloc((void**)&dB, sizeof(double) * nb * bsr_dim * nrhs);
*      hipMalloc((void**)&dX, sizeof(double) * mb * bsr_dim * nrhs);
*
*      hipMemcpy(dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice);
*      hipMemcpy(dbsr_val, hbsr_val, sizeof(double) * nnzb * bsr_dim * bsr_dim, hipMemcpyHostToDevice);
*      hipMemcpy(dB, hB, sizeof(double) * nb * bsr_dim * nrhs, hipMemcpyHostToDevice);
*
*      // Matrix descriptor
*      rocsparse_mat_descr descr;
*      rocsparse_create_mat_descr(&descr);
*
*      // Matrix fill mode
*      rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower);
*
*      // Matrix diagonal type
*      rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit);
*
*      // Matrix info structure
*      rocsparse_mat_info info;
*      rocsparse_create_mat_info(&info);
*
*      // Obtain required buffer size
*      size_t buffer_size;
*      rocsparse_dbsrsm_buffer_size(handle,
*                                dir,
*                                transA,
*                                transX,
*                                mb,
*                                nrhs,
*                                nnzb,
*                                descr,
*                                dbsr_val,
*                                dbsr_row_ptr,
*                                dbsr_col_ind,
*                                bsr_dim,
*                                info,
*                                &buffer_size);
*
*      // Allocate temporary buffer
*      void* temp_buffer;
*      hipMalloc(&temp_buffer, buffer_size);
*
*      // Perform analysis step
*      rocsparse_dbsrsm_analysis(handle,
*                            dir,
*                            transA,
*                            transX,
*                            mb,
*                            nrhs,
*                            nnzb,
*                            descr,
*                            dbsr_val,
*                            dbsr_row_ptr,
*                            dbsr_col_ind,
*                            bsr_dim,
*                            info,
*                            analysis_policy,
*                            solve_policy,
*                            temp_buffer);
*
*      // Call dbsrsm to perform lower triangular solve LX = B
*      rocsparse_dbsrsm_solve(handle,
*                            dir,
*                            transA,
*                            transX,
*                            mb,
*                            nrhs,
*                            nnzb,
*                            &alpha,
*                            descr,
*                            dbsr_val,
*                            dbsr_row_ptr,
*                            dbsr_col_ind,
*                            bsr_dim,
*                            info,
*                            dB,
*                            ldb,
*                            dX,
*                            ldx,
*                            solve_policy,
*                            temp_buffer);
*
*      // Check for zero pivots
*      rocsparse_int    pivot;
*      rocsparse_status status = rocsparse_bsrsm_zero_pivot(handle, info, &pivot);
*
*      if(status == rocsparse_status_zero_pivot)
*      {
*          std::cout << "Found zero pivot in matrix row " << pivot << std::endl;
*      }
*
*      // Copy result back to host
*      hipMemcpy(hX, dX, sizeof(double) * mb * bsr_dim * nrhs, hipMemcpyDeviceToHost);
*
*      // Clear rocSPARSE
*      rocsparse_destroy_mat_info(info);
*      rocsparse_destroy_mat_descr(descr);
*      rocsparse_destroy_handle(handle);
*
*      // Clear device memory
*      hipFree(dbsr_row_ptr);
*      hipFree(dbsr_col_ind);
*      hipFree(dbsr_val);
*      hipFree(dB);
*      hipFree(dX);
*      hipFree(temp_buffer);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrsm_solve(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_X,
                                        rocsparse_int             mb,
                                        rocsparse_int             nrhs,
                                        rocsparse_int             nnzb,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             block_dim,
                                        rocsparse_mat_info        info,
                                        const float*              B,
                                        rocsparse_int             ldb,
                                        float*                    X,
                                        rocsparse_int             ldx,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrsm_solve(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_X,
                                        rocsparse_int             mb,
                                        rocsparse_int             nrhs,
                                        rocsparse_int             nnzb,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             block_dim,
                                        rocsparse_mat_info        info,
                                        const double*             B,
                                        rocsparse_int             ldb,
                                        double*                   X,
                                        rocsparse_int             ldx,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrsm_solve(rocsparse_handle               handle,
                                        rocsparse_direction            dir,
                                        rocsparse_operation            trans_A,
                                        rocsparse_operation            trans_X,
                                        rocsparse_int                  mb,
                                        rocsparse_int                  nrhs,
                                        rocsparse_int                  nnzb,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* bsr_val,
                                        const rocsparse_int*           bsr_row_ptr,
                                        const rocsparse_int*           bsr_col_ind,
                                        rocsparse_int                  block_dim,
                                        rocsparse_mat_info             info,
                                        const rocsparse_float_complex* B,
                                        rocsparse_int                  ldb,
                                        rocsparse_float_complex*       X,
                                        rocsparse_int                  ldx,
                                        rocsparse_solve_policy         policy,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrsm_solve(rocsparse_handle                handle,
                                        rocsparse_direction             dir,
                                        rocsparse_operation             trans_A,
                                        rocsparse_operation             trans_X,
                                        rocsparse_int                   mb,
                                        rocsparse_int                   nrhs,
                                        rocsparse_int                   nnzb,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* bsr_val,
                                        const rocsparse_int*            bsr_row_ptr,
                                        const rocsparse_int*            bsr_col_ind,
                                        rocsparse_int                   block_dim,
                                        rocsparse_mat_info              info,
                                        const rocsparse_double_complex* B,
                                        rocsparse_int                   ldb,
                                        rocsparse_double_complex*       X,
                                        rocsparse_int                   ldx,
                                        rocsparse_solve_policy          policy,
                                        void*                           temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSRSM_H */
