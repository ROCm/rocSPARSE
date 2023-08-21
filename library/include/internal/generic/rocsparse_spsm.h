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

#ifndef ROCSPARSE_SPSM_H
#define ROCSPARSE_SPSM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse triangular system solve
*
*  \details
*  \p rocsparse_spsm_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR or COO storage format, a dense solution matrix
*  \f$C\f$ and the right-hand side \f$B\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot C = \alpha \cdot op(B),
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
*  \note SpSM requires three stages to complete. The first stage
*  \ref rocsparse_spsm_stage_buffer_size will return the size of the temporary storage buffer
*  that is required for subsequent calls. The second stage
*  \ref rocsparse_spsm_stage_preprocess will preprocess data that would be saved in the temporary storage buffer.
*  In the final stage \ref rocsparse_spsm_stage_compute, the actual computation is performed.
*
*  \note
*  Only the \ref rocsparse_spsm_stage_buffer_size stage and the \ref rocsparse_spsm_stage_compute stage are non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The \ref rocsparse_spsm_stage_preprocess stage is blocking with respect to the host.
*
*  \note
*  Currently, only \p trans_A == \ref rocsparse_operation_none and \p trans_A == \ref rocsparse_operation_transpose is supported.
*  Currently, only \p trans_B == \ref rocsparse_operation_none and \p trans_B == \ref rocsparse_operation_transpose is supported.
*
*  \note
*  Only the \ref rocsparse_spsm_stage_buffer_size stage and the \ref rocsparse_spsm_stage_compute stage
*  support execution in a hipGraph context. The \ref rocsparse_spsm_stage_preprocess stage does not support hipGraph.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans_A      matrix operation type for the sparse matrix A.
*  @param[in]
*  trans_B      matrix operation type for the dense matrix B.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  matA          sparse matrix descriptor.
*  @param[in]
*  matB          dense matrix descriptor.
*  @param[inout]
*  matC          dense matrix descriptor.
*  @param[in]
*  compute_type floating point precision for the SpSM computation.
*  @param[in]
*  alg          SpSM algorithm for the SpSM computation.
*  @param[in]
*  stage        SpSM stage for the SpSM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpSM operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p matA, \p matB, \p matC, \p descr or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_not_implemented \p trans_A, \p trans_B, \p compute_type, \p stage or \p alg is
*               currently not supported.
*
*  \par Example
*  \code{.c}
*   //     1 0 0 0
*   // A = 4 2 0 0
*   //     0 3 7 0
*   //     0 0 0 1
*   rocsparse_int m   = 4;
*   rocsparse_int n   = 2;
*
*   std::vector<int> hcsr_row_ptr = {0, 1, 3, 5, 6};
*   std::vector<int> hcsr_col_ind = {0, 0, 1, 1, 2, 3};
*   std::vector<float> hcsr_val   = {1, 4, 2, 3, 7, 1};
*   std::vector<float> hB(m * n);
*   std::vector<float> hC(m * n);
*
*   for(int i = 0; i < n; i++)
*   {
*       for(int j = 0; j < m; j++)
*       {
*           hB[m * i + j] = static_cast<float>(i + 1);
*       }
*   }
*
*   // Scalar alpha
*   float alpha = 1.0f;
*
*   rocsparse_int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];
*
*   // Offload data to device
*   int* dcsr_row_ptr;
*   int* dcsr_col_ind;
*   float* dcsr_val;
*   float* dB;
*   float* dC;
*   hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1));
*   hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz);
*   hipMalloc((void**)&dcsr_val, sizeof(float) * nnz);
*   hipMalloc((void**)&dB, sizeof(float) * m * n);
*   hipMalloc((void**)&dC, sizeof(float) * m * n);
*
*   hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dB, hB.data(), sizeof(float) * m * n, hipMemcpyHostToDevice);
*
*   rocsparse_handle     handle;
*   rocsparse_spmat_descr matA;
*   rocsparse_dnmat_descr matB;
*   rocsparse_dnmat_descr matC;
*
*   rocsparse_indextype row_idx_type = rocsparse_indextype_i32;
*   rocsparse_indextype col_idx_type = rocsparse_indextype_i32;
*   rocsparse_datatype  data_type = rocsparse_datatype_f32_r;
*   rocsparse_datatype  compute_type = rocsparse_datatype_f32_r;
*   rocsparse_index_base idx_base = rocsparse_index_base_zero;
*   rocsparse_operation trans_A = rocsparse_operation_none;
*   rocsparse_operation trans_B = rocsparse_operation_none;
*
*   rocsparse_create_handle(&handle);
*
*   // Create sparse matrix A
*   rocsparse_create_csr_descr(&matA,
*                              m,
*                              m,
*                              nnz,
*                              dcsr_row_ptr,
*                              dcsr_col_ind,
*                              dcsr_val,
*                              row_idx_type,
*                              col_idx_type,
*                              idx_base,
*                              data_type);
*
*   // Create dense matrix B
*   rocsparse_create_dnmat_descr(&matB,
*                                m,
*                                n,
*                                m,
*                                dB,
*                                data_type,
*                                rocsparse_order_column);
*
*   // Create dense matrix C
*   rocsparse_create_dnmat_descr(&matC,
*                                m,
*                                n,
*                                m,
*                                dC,
*                                data_type,
*                                rocsparse_order_column);
*
*   // Call spsv to get buffer size
*   size_t buffer_size;
*   rocsparse_spsm(handle,
*                  trans_A,
*                  trans_B,
*                  &alpha,
*                  matA,
*                  matB,
*                  matC,
*                  compute_type,
*                  rocsparse_spsm_alg_default,
*                  rocsparse_spsm_stage_buffer_size,
*                  &buffer_size,
*                  nullptr);
*
*   void* temp_buffer;
*   hipMalloc((void**)&temp_buffer, buffer_size);
*
*   // Call spsv to perform analysis
*   rocsparse_spsm(handle,
*                  trans_A,
*                  trans_B,
*                  &alpha,
*                  matA,
*                  matB,
*                  matC,
*                  compute_type,
*                  rocsparse_spsm_alg_default,
*                  rocsparse_spsm_stage_preprocess,
*                  &buffer_size,
*                  temp_buffer);
*
*   // Call spsv to perform computation
*   rocsparse_spsm(handle,
*                  trans_A,
*                  trans_B,
*                  &alpha,
*                  matA,
*                  matB,
*                  matC,
*                  compute_type,
*                  rocsparse_spsm_alg_default,
*                  rocsparse_spsm_stage_compute,
*                  &buffer_size,
*                  temp_buffer);
*
*   // Copy result back to host
*   hipMemcpy(hC.data(), dC, sizeof(float) * m * n, hipMemcpyDeviceToHost);
*
*   std::cout << "hC" << std::endl;
*   for(size_t i = 0; i < hC.size(); ++i)
*   {
*       std::cout << hC[i] << " ";
*   }
*   std::cout << std::endl;
*
*   // Clear rocSPARSE
*   rocsparse_destroy_spmat_descr(matA);
*   rocsparse_destroy_dnmat_descr(matB);
*   rocsparse_destroy_dnmat_descr(matC);
*   rocsparse_destroy_handle(handle);
*
*   // Clear device memory
*   hipFree(dcsr_row_ptr);
*   hipFree(dcsr_col_ind);
*   hipFree(dcsr_val);
*   hipFree(dB);
*   hipFree(dC);
*   hipFree(temp_buffer);
*  \endcode
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spsm(rocsparse_handle            handle,
                                rocsparse_operation         trans_A,
                                rocsparse_operation         trans_B,
                                const void*                 alpha,
                                rocsparse_const_spmat_descr matA,
                                rocsparse_const_dnmat_descr matB,
                                const rocsparse_dnmat_descr matC,
                                rocsparse_datatype          compute_type,
                                rocsparse_spsm_alg          alg,
                                rocsparse_spsm_stage        stage,
                                size_t*                     buffer_size,
                                void*                       temp_buffer);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPSM_H */
