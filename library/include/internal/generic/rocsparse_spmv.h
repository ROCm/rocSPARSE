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

#ifndef ROCSPARSE_SPMV_H
#define ROCSPARSE_SPMV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix vector multiplication
*
*  \details
*  \ref rocsparse_spmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix and the dense vector \f$x\f$ and adds the result to the dense vector \f$y\f$
*  that is multiplied by the scalar \f$\beta\f$, such that
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
*  \details
*  \ref rocsparse_spmv supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix A and the dense vectors X and Y and the compute
*  type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on memory bandwidth and storage
*  when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmv_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmv_mixed">Mixed Precisions</caption>
*  <tr><th>A / X                   <th>Y                        <th>compute_type
*  <tr><td>rocsparse_datatype_i8_r <td>rocsparse_datatype_i32_r <td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_i8_r <td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_i8_r <td>rocsparse_datatype_i32_r <td>rocsparse_datatype_i32_r
*  </table>
*
*  \par Mixed-regular real precisions
*  <table>
*  <caption id="spmv_mixed_regular_real">Mixed-regular real precisions</caption>
*  <tr><th>A                        <th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c <td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed-regular Complex precisions
*  <table>
*  <caption id="spmv_mixed_regular_complex">Mixed-regular Complex precisions</caption>
*  <tr><th>A                        <th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f64_c
*  </table>
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the SpMV operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note
*  Only the \ref rocsparse_spmv_stage_buffer_size stage and the \ref rocsparse_spmv_stage_compute stage are non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The \ref rocsparse_spmv_stage_preprocess stage is blocking with respect to the host.
*
*  \note
*  Only the \ref rocsparse_spmv_stage_buffer_size stage and the \ref rocsparse_spmv_stage_compute stage
*  support execution in a hipGraph context. The \ref rocsparse_spmv_stage_preprocess stage does not support hipGraph.
*
*  \note
*  The sparse matrix formats currently supported are: rocsparse_format_bsr, rocsparse_format_coo,
*  rocsparse_format_coo_aos, rocsparse_format_csr, rocsparse_format_csc and rocsparse_format_ell.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans        matrix operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat          matrix descriptor.
*  @param[in]
*  x            vector descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  y            vector descriptor.
*  @param[in]
*  compute_type floating point precision for the SpMV computation.
*  @param[in]
*  alg          SpMV algorithm for the SpMV computation.
*  @param[in]
*  stage        SpMV stage for the SpMV computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpMV operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context \p handle was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p mat, \p x, \p beta, \p y or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_invalid_value the value of \p trans, \p compute_type, \p alg, or \p stage is incorrect.
*  \retval      rocsparse_status_not_implemented \p compute_type or \p alg is
*               currently not supported.
*
*  \par Example
*  \code{.c}
*   //     1 4 0 0 0 0
*   // A = 0 2 3 0 0 0
*   //     5 0 0 7 8 0
*   //     0 0 9 0 6 0
*   rocsparse_int m   = 4;
*   rocsparse_int n   = 6;
*
*   std::vector<int> hcsr_row_ptr = {0, 2, 4, 7, 9};
*   std::vector<int> hcsr_col_ind = {0, 1, 1, 2, 0, 3, 4, 2, 4};
*   std::vector<float> hcsr_val   = {1, 4, 2, 3, 5, 7, 8, 9, 6};
*   std::vector<float> hx(n, 1.0f);
*   std::vector<float> hy(m, 0.0f);
*
*   // Scalar alpha
*   float alpha = 3.7f;
*
*   // Scalar beta
*   float beta = 0.0f;
*
*   rocsparse_int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];
*
*   // Offload data to device
*   int* dcsr_row_ptr;
*   int* dcsr_col_ind;
*   float* dcsr_val;
*   float* dx;
*   float* dy;
*   hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1));
*   hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz);
*   hipMalloc((void**)&dcsr_val, sizeof(float) * nnz);
*   hipMalloc((void**)&dx, sizeof(float) * n);
*   hipMalloc((void**)&dy, sizeof(float) * m);
*
*   hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dx, hx.data(), sizeof(float) * n, hipMemcpyHostToDevice);
*
*   rocsparse_handle     handle;
*   rocsparse_spmat_descr matA;
*   rocsparse_dnvec_descr vecX;
*   rocsparse_dnvec_descr vecY;
*
*   rocsparse_indextype row_idx_type = rocsparse_indextype_i32;
*   rocsparse_indextype col_idx_type = rocsparse_indextype_i32;
*   rocsparse_datatype  data_type = rocsparse_datatype_f32_r;
*   rocsparse_datatype  compute_type = rocsparse_datatype_f32_r;
*   rocsparse_index_base idx_base = rocsparse_index_base_zero;
*   rocsparse_operation trans = rocsparse_operation_none;
*
*   rocsparse_create_handle(&handle);
*
*   // Create sparse matrix A
*   rocsparse_create_csr_descr(&matA,
*                              m,
*                              n,
*                              nnz,
*                              dcsr_row_ptr,
*                              dcsr_col_ind,
*                              dcsr_val,
*                              row_idx_type,
*                              col_idx_type,
*                              idx_base,
*                              data_type);
*
*   // Create dense vector X
*   rocsparse_create_dnvec_descr(&vecX,
*                                n,
*                                dx,
*                                data_type);
*
*   // Create dense vector Y
*   rocsparse_create_dnvec_descr(&vecY,
*                                m,
*                                dy,
*                                data_type);
*
*   // Call spmv to get buffer size
*   size_t buffer_size;
*   rocsparse_spmv(handle,
*                  trans,
*                  &alpha,
*                  matA,
*                  vecX,
*                  &beta,
*                  vecY,
*                  compute_type,
*                  rocsparse_spmv_alg_csr_adaptive,
*                  rocsparse_spmv_stage_buffer_size,
*                  &buffer_size,
*                  nullptr);
*
*   void* temp_buffer;
*   hipMalloc((void**)&temp_buffer, buffer_size);
*
*   // Call spmv to perform analysis
*   rocsparse_spmv(handle,
*                  trans,
*                  &alpha,
*                  matA,
*                  vecX,
*                  &beta,
*                  vecY,
*                  compute_type,
*                  rocsparse_spmv_alg_csr_adaptive,
*                  rocsparse_spmv_stage_preprocess,
*                  &buffer_size,
*                  temp_buffer);
*
*   // Call spmv to perform computation
*   rocsparse_spmv(handle,
*                  trans,
*                  &alpha,
*                  matA,
*                  vecX,
*                  &beta,
*                  vecY,
*                  compute_type,
*                  rocsparse_spmv_alg_csr_adaptive,
*                  rocsparse_spmv_stage_compute,
*                  &buffer_size,
*                  temp_buffer);
*
*   // Copy result back to host
*   hipMemcpy(hy.data(), dy, sizeof(float) * m, hipMemcpyDeviceToHost);
*
*   std::cout << "hy" << std::endl;
*   for(size_t i = 0; i < hy.size(); ++i)
*   {
*       std::cout << hy[i] << " ";
*   }
*   std::cout << std::endl;
*
*   // Clear rocSPARSE
*   rocsparse_destroy_spmat_descr(matA);
*   rocsparse_destroy_dnvec_descr(vecX);
*   rocsparse_destroy_dnvec_descr(vecY);
*   rocsparse_destroy_handle(handle);
*
*   // Clear device memory
*   hipFree(dcsr_row_ptr);
*   hipFree(dcsr_col_ind);
*   hipFree(dcsr_val);
*   hipFree(dx);
*   hipFree(dy);
*   hipFree(temp_buffer);
*  \endcode
*/
ROCSPARSE_EXPORT rocsparse_status rocsparse_spmv(rocsparse_handle            handle,
                                                 rocsparse_operation         trans,
                                                 const void*                 alpha,
                                                 rocsparse_const_spmat_descr mat,
                                                 rocsparse_const_dnvec_descr x,
                                                 const void*                 beta,
                                                 const rocsparse_dnvec_descr y,
                                                 rocsparse_datatype          compute_type,
                                                 rocsparse_spmv_alg          alg,
                                                 rocsparse_spmv_stage        stage,
                                                 size_t*                     buffer_size,
                                                 void*                       temp_buffer);

/*! \ingroup generic_module
*  \brief Sparse matrix vector multiplication
*
*  \details
*  \ref rocsparse_spmv_ex multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix and the dense vector \f$x\f$ and adds the result to the dense vector \f$y\f$
*  that is multiplied by the scalar \f$\beta\f$, such that
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
*  \details
*  \ref rocsparse_spmv_ex supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix A and the dense vectors X and Y and the compute
*  type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on memory bandwidth and storage
*  when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmv_uniform_ex">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmv_mixed_ex">Mixed Precisions</caption>
*  <tr><th>A / X                   <th>Y                        <th>compute_type
*  <tr><td>rocsparse_datatype_i8_r <td>rocsparse_datatype_i32_r <td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_i8_r <td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_i8_r <td>rocsparse_datatype_i32_r <td>rocsparse_datatype_i32_r
*  </table>
*
*  \par Mixed-regular real precisions
*  <table>
*  <caption id="spmv_mixed_regular_real_ex">Mixed-regular real precisions</caption>
*  <tr><th>A                        <th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c <td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed-regular Complex precisions
*  <table>
*  <caption id="spmv_mixed_regular_complex_ex">Mixed-regular Complex precisions</caption>
*  <tr><th>A                        <th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f64_c
*  </table>
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the SpMV operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note
*  The sparse matrix formats currently supported are: rocsparse_format_bsr, rocsparse_format_coo,
*  rocsparse_format_coo_aos, rocsparse_format_csr, rocsparse_format_csc and rocsparse_format_ell.
*
*  \note SpMV_ex requires three stages to complete. The first stage
*  \ref rocsparse_spmv_stage_buffer_size will return the size of the temporary storage buffer
*  that is required for subsequent calls to \ref rocsparse_spmv_ex. The second stage
*  \ref rocsparse_spmv_stage_preprocess will preprocess data that would be saved in the temporary storage buffer.
*  In the final stage \ref rocsparse_spmv_stage_compute, the actual computation is performed.
*
*  \note
*  Only the \ref rocsparse_spmv_stage_buffer_size stage and the \ref rocsparse_spmv_stage_compute stage are non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The \ref rocsparse_spmv_stage_preprocess stage is blocking with respect to the host.
*
*  \note
*  Only the \ref rocsparse_spmv_stage_buffer_size stage and the \ref rocsparse_spmv_stage_compute stage
*  support execution in a hipGraph context. The \ref rocsparse_spmv_stage_preprocess stage does not support hipGraph.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans        matrix operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat          matrix descriptor.
*  @param[in]
*  x            vector descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  y            vector descriptor.
*  @param[in]
*  compute_type floating point precision for the SpMV computation.
*  @param[in]
*  alg          SpMV algorithm for the SpMV computation.
*  @param[in]
*  stage        SpMV stage for the SpMV computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpMV operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context \p handle was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p mat, \p x, \p beta, \p y or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_invalid_value the value of \p trans, \p compute_type, \p alg or \p stage is incorrect.
*  \retval      rocsparse_status_not_implemented \p compute_type or \p alg is
*               currently not supported.
*/
__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_spmv instead."))) ROCSPARSE_EXPORT rocsparse_status
    rocsparse_spmv_ex(rocsparse_handle            handle,
                      rocsparse_operation         trans,
                      const void*                 alpha,
                      const rocsparse_spmat_descr mat,
                      const rocsparse_dnvec_descr x,
                      const void*                 beta,
                      const rocsparse_dnvec_descr y,
                      rocsparse_datatype          compute_type,
                      rocsparse_spmv_alg          alg,
                      rocsparse_spmv_stage        stage,
                      size_t*                     buffer_size,
                      void*                       temp_buffer);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPMV_H */
