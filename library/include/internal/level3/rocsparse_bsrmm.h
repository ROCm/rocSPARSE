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

#ifndef ROCSPARSE_BSRMM_H
#define ROCSPARSE_BSRMM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level3_module
 *  \brief Sparse matrix dense matrix multiplication using BSR storage format
 *
 *  \details
 *  \p rocsparse_bsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$mb \times kb\f$
 *  matrix \f$A\f$, defined in BSR storage format, and the column-oriented dense \f$k \times n\f$
 *  matrix \f$B\f$ (where \f$k = block\_dim \times kb\f$) and adds the result to the column-oriented dense
 *  \f$m \times n\f$ matrix \f$C\f$ (where \f$m = block\_dim \times mb\f$) that
 *  is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans_A == rocsparse_operation_none} \\
 *    \end{array}
 *    \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if trans_B == rocsparse_operation_none} \\
 *        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
 *
 *  \note
 *  This routine supports execution in a hipGraph context.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  dir         the storage format of the blocks. Can be \ref rocsparse_direction_row or \ref rocsparse_direction_column.
 *  @param[in]
 *  trans_A     matrix \f$A\f$ operation type. Currently, only \ref rocsparse_operation_none is supported.
 *  @param[in]
 *  trans_B     matrix \f$B\f$ operation type. Currently, only \ref rocsparse_operation_none and rocsparse_operation_transpose
 *              are supported.
 *  @param[in]
 *  mb          number of block rows of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  n           number of columns of the column-oriented dense matrix \f$op(B)\f$ and \f$C\f$.
 *  @param[in]
 *  kb          number of block columns of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  nnzb        number of non-zero blocks of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix \f$A\f$. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  bsr_val     array of \p nnzb*block_dim*block_dim elements of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse
 *              BSR matrix \f$A\f$.
 *  @param[in]
 *  block_dim   size of the blocks in the sparse BSR matrix.
 *  @param[in]
 *  B           column-oriented dense matrix of dimension \f$ldb \times n\f$ (\f$op(B) == B\f$),
 *              \f$ldb \times k\f$ otherwise.
 *  @param[in]
 *  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$ (\f$ op(B) == B\f$) where \f$k = block\_dim \times kb\f$,
 *  \f$\max{(1, n)}\f$ otherwise.
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  C           column-oriented dense matrix of  dimension \f$ldc \times n\f$.
 *  @param[in]
 *  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$ (\f$ op(A) == A\f$) where \f$m = block\_dim \times mb\f$,
 *  \f$\max{(1, k)}\f$ where \f$k = block\_dim \times kb\f$ otherwise.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p n, \p kb, \p nnzb, \p ldb or \p ldc
 *              is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
 *              \p bsr_row_ptr, \p bsr_col_ind, \p B, \p beta or \p C pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans_A != \ref rocsparse_operation_none or
 *              \p trans_B == \ref rocsparse_operation_conjugate_transpose or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  This example multiplies a BSR matrix with a column-oriented dense matrix.
 *  \code{.c}
 *      //     1 2 0 3 0 0
 *      // A = 0 4 5 0 0 0
 *      //     0 0 0 7 8 0
 *      //     0 0 1 2 4 1
 *
 *      rocsparse_int block_dim = 2;
 *      rocsparse_int mb   = 2;
 *      rocsparse_int kb   = 3;
 *      rocsparse_int nnzb = 4;
 *      rocsparse_direction dir = rocsparse_direction_row;
 *
 *      bsr_row_ptr[mb+1]                 = {0, 2, 4};                                        // device memory
 *      bsr_col_ind[nnzb]                 = {0, 1, 1, 2};                                     // device memory
 *      bsr_val[nnzb*block_dim*block_dim] = {1, 2, 0, 4, 0, 3, 5, 0, 0, 7, 1, 2, 8, 0, 4, 1}; // device memory
 *
 *      // Set dimension n of B
 *      rocsparse_int n = 64;
 *      rocsparse_int m = mb * block_dim;
 *      rocsparse_int k = kb * block_dim;
 *
 *      // Allocate and generate column-oriented dense matrix B
 *      std::vector<float> hB(k * n);
 *      for(rocsparse_int i = 0; i < k * n; ++i)
 *      {
 *          hB[i] = static_cast<float>(rand()) / RAND_MAX;
 *      }
 *
 *      // Copy B to the device
 *      float* B;
 *      hipMalloc((void**)&B, sizeof(float) * k * n);
 *      hipMemcpy(B, hB.data(), sizeof(float) * k * n, hipMemcpyHostToDevice);
 *
 *      // alpha and beta
 *      float alpha = 1.0f;
 *      float beta  = 0.0f;
 *
 *      // Allocate memory for the resulting matrix C
 *      float* C;
 *      hipMalloc((void**)&C, sizeof(float) * m * n);
 *
 *      // Perform the matrix multiplication
 *      rocsparse_sbsrmm(handle,
 *                       dir,
 *                       rocsparse_operation_none,
 *                       rocsparse_operation_none,
 *                       mb,
 *                       n,
 *                       kb,
 *                       nnzb,
 *                       &alpha,
 *                       descr,
 *                       bsr_val,
 *                       bsr_row_ptr,
 *                       bsr_col_ind,
 *                       block_dim,
 *                       B,
 *                       k,
 *                       &beta,
 *                       C,
 *                       m);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrmm(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             mb,
                                  rocsparse_int             n,
                                  rocsparse_int             kb,
                                  rocsparse_int             nnzb,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  const float*              B,
                                  rocsparse_int             ldb,
                                  const float*              beta,
                                  float*                    C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrmm(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             mb,
                                  rocsparse_int             n,
                                  rocsparse_int             kb,
                                  rocsparse_int             nnzb,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  const double*             B,
                                  rocsparse_int             ldb,
                                  const double*             beta,
                                  double*                   C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrmm(rocsparse_handle               handle,
                                  rocsparse_direction            dir,
                                  rocsparse_operation            trans_A,
                                  rocsparse_operation            trans_B,
                                  rocsparse_int                  mb,
                                  rocsparse_int                  n,
                                  rocsparse_int                  kb,
                                  rocsparse_int                  nnzb,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* bsr_val,
                                  const rocsparse_int*           bsr_row_ptr,
                                  const rocsparse_int*           bsr_col_ind,
                                  rocsparse_int                  block_dim,
                                  const rocsparse_float_complex* B,
                                  rocsparse_int                  ldb,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       C,
                                  rocsparse_int                  ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrmm(rocsparse_handle                handle,
                                  rocsparse_direction             dir,
                                  rocsparse_operation             trans_A,
                                  rocsparse_operation             trans_B,
                                  rocsparse_int                   mb,
                                  rocsparse_int                   n,
                                  rocsparse_int                   kb,
                                  rocsparse_int                   nnzb,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* bsr_val,
                                  const rocsparse_int*            bsr_row_ptr,
                                  const rocsparse_int*            bsr_col_ind,
                                  rocsparse_int                   block_dim,
                                  const rocsparse_double_complex* B,
                                  rocsparse_int                   ldb,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       C,
                                  rocsparse_int                   ldc);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSRMM_H */
