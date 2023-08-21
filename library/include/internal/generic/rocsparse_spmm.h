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

#ifndef ROCSPARSE_SPMM_H
#define ROCSPARSE_SPMM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix dense matrix multiplication, extension routine.
*
*  \details
*  \p rocsparse_spmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
*  matrix \f$A\f$, defined in CSR or COO or Blocked ELL storage format, and the dense \f$k \times n\f$
*  matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
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
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  Only the \ref rocsparse_spmm_stage_buffer_size stage and the \ref rocsparse_spmm_stage_compute stage are non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The \ref rocsparse_spmm_stage_preprocess stage is blocking with respect to the host.
*
*  \note
*  Currently, only \p trans_A == \ref rocsparse_operation_none is supported for COO and Blocked ELL formats.
*
*  \note
*  Only the \ref rocsparse_spmm_stage_buffer_size stage and the \ref rocsparse_spmm_stage_compute stage
*  support execution in a hipGraph context. The \ref rocsparse_spmm_stage_preprocess stage does not support hipGraph.
*
*  \note
*  Currently, only CSR, COO and Blocked ELL sparse formats are supported.
*
*  \note
*  Different algorithms are available which can provide better performance for different matrices.
*  Currently, the available algorithms are rocsparse_spmm_alg_csr, rocsparse_spmm_alg_csr_row_split
*  or rocsparse_spmm_alg_csr_merge for CSR matrices, rocsparse_spmm_alg_bell for Blocked ELL matrices and
*  rocsparse_spmm_alg_coo_segmented or rocsparse_spmm_alg_coo_atomic for COO matrices. Additionally,
*  one can specify the algorithm to be rocsparse_spmm_alg_default. In the case of CSR matrices this will
*  set the algorithm to be rocsparse_spmm_alg_csr, in the case of Blocked ELL matrices this will set the
*  algorithm to be rocsparse_spmm_alg_bell and for COO matrices it will set the algorithm to be
*  rocsparse_spmm_alg_coo_atomic. When A is transposed, rocsparse_spmm will revert to using
*  rocsparse_spmm_alg_csr for CSR format and rocsparse_spmm_alg_coo_atomic for COO format regardless
*  of algorithm selected.
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the SpMM operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note SpMM requires three stages to complete. The first stage
*  \ref rocsparse_spmm_stage_buffer_size will return the size of the temporary storage buffer
*  that is required for subsequent calls to \ref rocsparse_spmm. The second stage
*  \ref rocsparse_spmm_stage_preprocess will preprocess data that would be saved in the temporary storage buffer.
*  In the final stage \ref rocsparse_spmm_stage_compute, the actual computation is performed.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans_A      matrix operation type.
*  @param[in]
*  trans_B      matrix operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat_A        matrix descriptor.
*  @param[in]
*  mat_B        matrix descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[in]
*  mat_C        matrix descriptor.
*  @param[in]
*  compute_type floating point precision for the SpMM computation.
*  @param[in]
*  alg          SpMM algorithm for the SpMM computation.
*  @param[in]
*  stage        SpMM stage for the SpMM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpMM operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p mat_A, \p mat_B, \p mat_C, \p beta, or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_not_implemented \p trans_A, \p trans_B, \p compute_type or \p alg is
*               currently not supported.
*  \par Example
*  This example performs sparse matrix-dense matrix multiplication, C = alpha * A * B + beta * C
*  \code{.c}
*      //     1 4 0 0 0 0
*      // A = 0 2 3 0 0 0
*      //     5 0 0 7 8 0
*      //     0 0 9 0 6 0
*
*      //     1 4 2
*      //     1 2 3
*      // B = 5 4 0
*      //     3 1 9
*      //     1 2 2
*      //     0 3 0
*
*      //     1 1 5
*      // C = 1 2 1
*      //     1 3 1
*      //     6 2 4
*
*      rocsparse_int m   = 4;
*      rocsparse_int k   = 6;
*      rocsparse_int n   = 3;
*
*      csr_row_ptr[m + 1] = {0, 1, 3};                                              // device memory
*      csr_col_ind[nnz]   = {0, 0, 1};                                              // device memory
*      csr_val[nnz]       = {1, 0, 4, 2, 0, 3, 5, 0, 0, 0, 0, 9, 7, 0, 8, 6, 0, 0}; // device memory
*
*      B[k * n]       = {1, 1, 5, 3, 1, 0, 4, 2, 4, 1, 2, 3, 2, 3, 0, 9, 2, 0};     // device memory
*      C[m * n]       = {1, 1, 1, 6, 1, 2, 3, 2, 5, 1, 1, 4};                       // device memory
*
*      rocsparse_int nnz = csr_row_ptr[m] - csr_row_ptr[0];
*
*      float alpha = 1.0f;
*      float beta  = 0.0f;
*
*      // Create CSR arrays on device
*      rocsparse_int* csr_row_ptr;
*      rocsparse_int* csr_col_ind;
*      float* csr_val;
*      float* B;
*      float* C;
*      hipMalloc((void**)&csr_row_ptr, sizeof(rocsparse_int) * (m + 1));
*      hipMalloc((void**)&csr_col_ind, sizeof(rocsparse_int) * nnz);
*      hipMalloc((void**)&csr_val, sizeof(float) * nnz);
*      hipMalloc((void**)&B, sizeof(float) * k * n);
*      hipMalloc((void**)&C, sizeof(float) * m * n);
*
*      // Create rocsparse handle
*      rocsparse_local_handle handle;
*
*      // Types
*      rocsparse_indextype itype = rocsparse_indextype_i32;
*      rocsparse_indextype jtype = rocsparse_indextype_i32;
*      rocsparse_datatype  ttype = rocsparse_datatype_f32_r;
*
*      // Create descriptors
*      rocsparse_spmat_descr mat_A;
*      rocsparse_dnmat_descr mat_B;
*      rocsparse_dnmat_descr mat_C;
*
*      rocsparse_create_csr_descr(&mat_A, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val, itype, jtype, rocsparse_index_base_zero, ttype);
*      rocsparse_create_dnmat_descr(&mat_B, k, n, k, B, ttype, rocsparse_order_column);
*      rocsparse_create_dnmat_descr(&mat_C, m, n, m, C, ttype, rocsparse_order_column);
*
*      // Query SpMM buffer
*      size_t buffer_size;
*      rocsparse_spmm(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     &alpha,
*                     mat_A,
*                     mat_B,
*                     &beta,
*                     mat_C,
*                     ttype,
*                     rocsparse_spmm_alg_default,
*                     rocsparse_spmm_stage_buffer_size,
*                     &buffer_size,
*                     nullptr));
*
*      // Allocate buffer
*      void* buffer;
*      hipMalloc(&buffer, buffer_size);
*
*      rocsparse_spmm(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     &alpha,
*                     mat_A,
*                     mat_B,
*                     &beta,
*                     mat_C,
*                     ttype,
*                     rocsparse_spmm_alg_default,
*                     rocsparse_spmm_stage_preprocess,
*                     &buffer_size,
*                     buffer));
*
*      // Pointer mode host
*      rocsparse_spmm(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     &alpha,
*                     mat_A,
*                     mat_B,
*                     &beta,
*                     mat_C,
*                     ttype,
*                     rocsparse_spmm_alg_default,
*                     rocsparse_spmm_stage_compute,
*                     &buffer_size,
*                     buffer));
*
*      // Clear up on device
*      hipFree(csr_row_ptr);
*      hipFree(csr_col_ind);
*      hipFree(csr_val);
*      hipFree(B);
*      hipFree(C);
*      hipFree(temp_buffer);
*
*      rocsparse_destroy_spmat_descr(mat_A);
*      rocsparse_destroy_dnmat_descr(mat_B);
*      rocsparse_destroy_dnmat_descr(mat_C);
*  \endcode
*
*  \par Example
*  SpMM also supports batched computation for CSR and COO matrices. There are three supported batch modes:
*      C_i = A * B_i
*      C_i = A_i * B
*      C_i = A_i * B_i
*  The batch mode is determined by the batch count and stride passed for each matrix. For example
*  to use the first batch mode (C_i = A * B_i) with 100 batches for non-transposed A, B, and C, one passes:
*      batch_count_A = 1
*      batch_count_B = 100
*      batch_count_C = 100
*      offsets_batch_stride_A        = 0
*      columns_values_batch_stride_A = 0
*      batch_stride_B                = k * n
*      batch_stride_C                = m * n
*  To use the second batch mode (C_i = A_i * B) one could use:
*      batch_count_A = 100
*      batch_count_B = 1
*      batch_count_C = 100
*      offsets_batch_stride_A        = m + 1
*      columns_values_batch_stride_A = nnz
*      batch_stride_B                = 0
*      batch_stride_C                = m * n
*  And to use the third batch mode (C_i = A_i * B_i) one could use:
*      batch_count_A = 100
*      batch_count_B = 100
*      batch_count_C = 100
*      offsets_batch_stride_A        = m + 1
*      columns_values_batch_stride_A = nnz
*      batch_stride_B                = k * n
*      batch_stride_C                = m * n
*  An example of the first batch mode (C_i = A * B_i) is provided below.
*  \code{.c}
*      //     1 4 0 0 0 0
*      // A = 0 2 3 0 0 0
*      //     5 0 0 7 8 0
*      //     0 0 9 0 6 0
*
*      rocsparse_int m   = 4;
*      rocsparse_int k   = 6;
*      rocsparse_int n   = 3;
*
*      csr_row_ptr[m + 1] = {0, 1, 3};                                              // device memory
*      csr_col_ind[nnz]   = {0, 0, 1};                                              // device memory
*      csr_val[nnz]       = {1, 0, 4, 2, 0, 3, 5, 0, 0, 0, 0, 9, 7, 0, 8, 6, 0, 0}; // device memory
*
*      B[batch_count_B * k * n]       = {...};     // device memory
*      C[batch_count_C * m * n]       = {...};     // device memory
*
*      rocsparse_int nnz = csr_row_ptr[m] - csr_row_ptr[0];
*
*      rocsparse_int batch_count_A = 1;
*      rocsparse_int batch_count_B = 100;
*      rocsparse_int batch_count_C = 100;
*
*      rocsparse_int offsets_batch_stride_A        = 0;
*      rocsparse_int columns_values_batch_stride_A = 0;
*      rocsparse_int batch_stride_B                = k * n;
*      rocsparse_int batch_stride_C                = m * n;
*
*      float alpha = 1.0f;
*      float beta  = 0.0f;
*
*      // Create CSR arrays on device
*      rocsparse_int* csr_row_ptr;
*      rocsparse_int* csr_col_ind;
*      float* csr_val;
*      float* B;
*      float* C;
*      hipMalloc((void**)&csr_row_ptr, sizeof(rocsparse_int) * (m + 1));
*      hipMalloc((void**)&csr_col_ind, sizeof(rocsparse_int) * nnz);
*      hipMalloc((void**)&csr_val, sizeof(float) * nnz);
*      hipMalloc((void**)&B, sizeof(float) * batch_count_B * k * n);
*      hipMalloc((void**)&C, sizeof(float) * batch_count_C * m * n);
*
*      // Create rocsparse handle
*      rocsparse_local_handle handle;
*
*      // Types
*      rocsparse_indextype itype = rocsparse_indextype_i32;
*      rocsparse_indextype jtype = rocsparse_indextype_i32;
*      rocsparse_datatype  ttype = rocsparse_datatype_f32_r;
*
*      // Create descriptors
*      rocsparse_spmat_descr mat_A;
*      rocsparse_dnmat_descr mat_B;
*      rocsparse_dnmat_descr mat_C;
*
*      rocsparse_create_csr_descr(&mat_A, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val, itype, jtype, rocsparse_index_base_zero, ttype);
*      rocsparse_create_dnmat_descr(&mat_B, k, n, k, B, ttype, rocsparse_order_column);
*      rocsparse_create_dnmat_descr(&mat_C, m, n, m, C, ttype, rocsparse_order_column);
*
*      rocsparse_csr_set_strided_batch(mat_A, batch_count_A, offsets_batch_stride_A, columns_values_batch_stride_A);
*      rocsparse_dnmat_set_strided_batch(B, batch_count_B, batch_stride_B);
*      rocsparse_dnmat_set_strided_batch(C, batch_count_C, batch_stride_C);
*
*      // Query SpMM buffer
*      size_t buffer_size;
*      rocsparse_spmm(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     &alpha,
*                     mat_A,
*                     mat_B,
*                     &beta,
*                     mat_C,
*                     ttype,
*                     rocsparse_spmm_alg_default,
*                     rocsparse_spmm_stage_buffer_size,
*                     &buffer_size,
*                     nullptr));
*
*      // Allocate buffer
*      void* buffer;
*      hipMalloc(&buffer, buffer_size);
*
*      rocsparse_spmm(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     &alpha,
*                     mat_A,
*                     mat_B,
*                     &beta,
*                     mat_C,
*                     ttype,
*                     rocsparse_spmm_alg_default,
*                     rocsparse_spmm_stage_preprocess,
*                     &buffer_size,
*                     buffer));
*
*      // Pointer mode host
*      rocsparse_spmm(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     &alpha,
*                     mat_A,
*                     mat_B,
*                     &beta,
*                     mat_C,
*                     ttype,
*                     rocsparse_spmm_alg_default,
*                     rocsparse_spmm_stage_compute,
*                     &buffer_size,
*                     buffer));
*
*      // Clear up on device
*      hipFree(csr_row_ptr);
*      hipFree(csr_col_ind);
*      hipFree(csr_val);
*      hipFree(B);
*      hipFree(C);
*      hipFree(temp_buffer);
*
*      rocsparse_destroy_spmat_descr(mat_A);
*      rocsparse_destroy_dnmat_descr(mat_B);
*      rocsparse_destroy_dnmat_descr(mat_C);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmm(rocsparse_handle            handle,
                                rocsparse_operation         trans_A,
                                rocsparse_operation         trans_B,
                                const void*                 alpha,
                                rocsparse_const_spmat_descr mat_A,
                                rocsparse_const_dnmat_descr mat_B,
                                const void*                 beta,
                                const rocsparse_dnmat_descr mat_C,
                                rocsparse_datatype          compute_type,
                                rocsparse_spmm_alg          alg,
                                rocsparse_spmm_stage        stage,
                                size_t*                     buffer_size,
                                void*                       temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPMM_H */
