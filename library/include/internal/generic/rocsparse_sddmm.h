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

#ifndef ROCSPARSE_SDDMM_H
#define ROCSPARSE_SDDMM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Calculate the size in bytes of the required buffer for the use of \ref rocsparse_sddmm and \ref rocsparse_sddmm_preprocess
*
*  \details
*  \p rocsparse_sddmm_buffer_size returns the size of the required buffer to execute the SDDMM operation from a given configuration.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  opA      dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB      dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  A            dense matrix \f$A\f$ descriptor.
*  @param[in]
*  B            dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  C            sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  compute_type floating point precision for the SDDMM computation.
*  @param[in]
*  alg specification of the algorithm to use.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_value the value of \p opA or \p opB is incorrect.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p A, \p B, \p D, \p C or \p buffer_size pointer is invalid.
*  \retval rocsparse_status_not_implemented
*          \p opA == \ref rocsparse_operation_conjugate_transpose or
*          \p opB == \ref rocsparse_operation_conjugate_transpose.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sddmm_buffer_size(rocsparse_handle            handle,
                                             rocsparse_operation         opA,
                                             rocsparse_operation         opB,
                                             const void*                 alpha,
                                             rocsparse_const_dnmat_descr A,
                                             rocsparse_const_dnmat_descr B,
                                             const void*                 beta,
                                             rocsparse_spmat_descr       C,
                                             rocsparse_datatype          compute_type,
                                             rocsparse_sddmm_alg         alg,
                                             size_t*                     buffer_size);

/*! \ingroup generic_module
*  \brief Preprocess data before the use of \ref rocsparse_sddmm.
*
*  \details
*  \p rocsparse_sddmm_preprocess executes a part of the algorithm that can be calculated once in the context of multiple
*  calls of the \ref rocsparse_sddmm with the same sparsity pattern.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  opA      dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB      dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  A            dense matrix \f$A\f$ descriptor.
*  @param[in]
*  B            dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  C            sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  compute_type floating point precision for the SDDMM computation.
*  @param[in]
*  alg specification of the algorithm to use.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*  The size must be greater or equal to the size obtained with \ref rocsparse_sddmm_buffer_size.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_value the value of \p opA or \p opB is incorrect.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p A, \p B, \p D, \p C or \p temp_buffer pointer is invalid.
*  \retval rocsparse_status_not_implemented
*          \p opA == \ref rocsparse_operation_conjugate_transpose or
*          \p opB == \ref rocsparse_operation_conjugate_transpose.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sddmm_preprocess(rocsparse_handle            handle,
                                            rocsparse_operation         opA,
                                            rocsparse_operation         opB,
                                            const void*                 alpha,
                                            rocsparse_const_dnmat_descr A,
                                            rocsparse_const_dnmat_descr B,
                                            const void*                 beta,
                                            rocsparse_spmat_descr       C,
                                            rocsparse_datatype          compute_type,
                                            rocsparse_sddmm_alg         alg,
                                            void*                       temp_buffer);

/*! \ingroup generic_module
*  \brief  Sampled Dense-Dense Matrix Multiplication.
*
*  \details
*  \p rocsparse_sddmm multiplies the scalar \f$\alpha\f$ with the dense
*  \f$m \times k\f$ matrix \f$A\f$, the dense \f$k \times n\f$ matrix \f$B\f$, filtered by the sparsity pattern of the \f$m \times n\f$ sparse matrix \f$C\f$ and
*  adds the result to \f$C\f$ scaled by
*  \f$\beta\f$. The final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha ( op(A) \cdot op(B) ) \circ spy(C) + \beta C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if op(A) == rocsparse_operation_none} \\
*        A^T,   & \text{if op(A) == rocsparse_operation_transpose} \\
*    \end{array}
*    \right.
*  \f],
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if op(B) == rocsparse_operation_none} \\
*        B^T,   & \text{if op(B) == rocsparse_operation_transpose} \\
*    \end{array}
*    \right.
*  \f]
*   and
*  \f[
*    spy(C)_ij = \left\{
*    \begin{array}{ll}
*        1 \text{ if C_ij != 0},   & 0 \text{ otherwise} \\
*    \end{array}
*    \right.
*  \f]
*
*  \note \p opA == \ref rocsparse_operation_conjugate_transpose is not supported.
*  \note \p opB == \ref rocsparse_operation_conjugate_transpose is not supported.
*  \note
*  This routine supports execution in a hipGraph context only when \p alg == \ref rocsparse_sddmm_alg_default.
*
*  \note
*  Different algorithms are available which can provide better performance for different matrices.
*  Currently, the available algorithms are rocsparse_sddmm_alg_default or rocsparse_sddmm_alg_dense.
*  The algorithm rocsparse_sddmm_alg_default uses the sparsity pattern of matrix C to perform a limited set of dot products.
*  On the other hand, rocsparse_sddmm_alg_dense explicitly converts the matrix C into a dense matrix to perform a dense matrix multiply and add.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  opA      dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB      dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  A            dense matrix \f$A\f$ descriptor.
*  @param[in]
*  B            dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  C            sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  compute_type floating point precision for the SDDMM computation.
*  @param[in]
*  alg specification of the algorithm to use.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*  The size must be greater or equal to the size obtained with \ref rocsparse_sddmm_buffer_size.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_value the value of \p opA, \p opB, \p compute\_type or alg is incorrect.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p A, \p B, \p D, \p C or \p temp_buffer pointer is invalid.
*  \retval rocsparse_status_not_implemented
*          \p opA == \ref rocsparse_operation_conjugate_transpose or
*          \p opB == \ref rocsparse_operation_conjugate_transpose.
*
*  \par Example
*  This example performs sampled dense-dense matrix product, C = alpha * (op(A) * op(B)) o spy(C) + beta * C where o is the hadamard product
*  \code{.c}
*    // rocSPARSE handle
*    rocsparse_handle handle;
*    rocsparse_create_handle(&handle);
*
*    float halpha = 1.0f;
*    float hbeta = -1.0f;
*
*    // A, B, and C are mxk, kxn, and mxn
*    int m = 4;
*    int k = 3;
*    int n = 2;
*    int nnzC = 5;
*
*    //     2  3  -1
*    // A = 0  2   1
*    //     0  0   5
*    //     0 -2 0.5
*
*    //      0  4
*    // B =  1  0
*    //     -2  0.5
*
*    //      1 0            1 0
*    // C =  2 3   spy(C) = 1 1
*    //      0 0            0 0
*    //      4 5            1 1
*
*    std::vector<float> hA = {2.0f, 3.0f, -1.0f, 0.0, 2.0f, 1.0f, 0.0f, 0.0f, 5.0f, 0.0f, -2.0f, 0.5f};
*    std::vector<float> hB = {0.0f, 4.0f, 1.0f, 0.0, -2.0f, 0.5f};
*
*    std::vector<int> hcsr_row_ptrC = {0, 1, 3, 3, 5};
*    std::vector<int> hcsr_col_indC = {0, 0, 1, 0, 1};
*    std::vector<float> hcsr_valC = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
*
*    float* dA = nullptr;
*    float* dB = nullptr;
*    hipMalloc((void**)&dA, sizeof(float) * m * k);
*    hipMalloc((void**)&dB, sizeof(float) * k * n);
*
*    int* dcsr_row_ptrC = nullptr;
*    int* dcsr_col_indC = nullptr;
*    float* dcsr_valC = nullptr;
*    hipMalloc((void**)&dcsr_row_ptrC, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsr_col_indC, sizeof(int) * nnzC);
*    hipMalloc((void**)&dcsr_valC, sizeof(float) * nnzC);
*
*    hipMemcpy(dA, hA.data(), sizeof(float) * m * k, hipMemcpyHostToDevice);
*    hipMemcpy(dB, hB.data(), sizeof(float) * k * n, hipMemcpyHostToDevice);
*
*    hipMemcpy(dcsr_row_ptrC, hcsr_row_ptrC.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_col_indC, hcsr_col_indC.data(), sizeof(int) * nnzC, hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_valC, hcsr_valC.data(), sizeof(float) * nnzC, hipMemcpyHostToDevice);
*
*    rocsparse_dnmat_descr matA;
*    rocsparse_create_dnmat_descr(&matA, m, k, k, dA, rocsparse_datatype_f32_r, rocsparse_order_row);
*
*    rocsparse_dnmat_descr matB;
*    rocsparse_create_dnmat_descr(&matB, k, n, n, dB, rocsparse_datatype_f32_r, rocsparse_order_row);
*
*    rocsparse_spmat_descr matC;
*    rocsparse_create_csr_descr(&matC,
*                               m,
*                               n,
*                               nnzC,
*                               dcsr_row_ptrC,
*                               dcsr_col_indC,
*                               dcsr_valC,
*                               rocsparse_indextype_i32,
*                               rocsparse_indextype_i32,
*                               rocsparse_index_base_zero,
*                               rocsparse_datatype_f32_r);
*
*    size_t buffer_size = 0;
*    rocsparse_sddmm_buffer_size(handle,
*                                rocsparse_operation_none,
*                                rocsparse_operation_none,
*                                &halpha,
*                                matA,
*                                matB,
*                                &hbeta,
*                                matC,
*                                rocsparse_datatype_f32_r,
*                                rocsparse_sddmm_alg_default,
*                                &buffer_size);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**) &dbuffer, buffer_size);
*
*    rocsparse_sddmm_preprocess(handle,
*                               rocsparse_operation_none,
*                               rocsparse_operation_none,
*                               &halpha,
*                               matA,
*                               matB,
*                               &hbeta,
*                               matC,
*                               rocsparse_datatype_f32_r,
*                               rocsparse_sddmm_alg_default,
*                               dbuffer);
*
*    rocsparse_sddmm(handle,
*                    rocsparse_operation_none,
*                    rocsparse_operation_none,
*                    &halpha,
*                    matA,
*                    matB,
*                    &hbeta,
*                    matC,
*                    rocsparse_datatype_f32_r,
*                    rocsparse_sddmm_alg_default,
*                    dbuffer);
*
*    hipMemcpy(hcsr_row_ptrC.data(), dcsr_row_ptrC, sizeof(int) * (m + 1), hipMemcpyDeviceToHost);
*    hipMemcpy(hcsr_col_indC.data(), dcsr_col_indC, sizeof(int) * nnzC, hipMemcpyDeviceToHost);
*    hipMemcpy(hcsr_valC.data(), dcsr_valC, sizeof(float) * nnzC, hipMemcpyDeviceToHost);
*
*    rocsparse_destroy_dnmat_descr(matA);
*    rocsparse_destroy_dnmat_descr(matB);
*    rocsparse_destroy_spmat_descr(matC);
*    rocsparse_destroy_handle(handle);
*
*    hipFree(dA);
*    hipFree(dB);
*    hipFree(dcsr_row_ptrC);
*    hipFree(dcsr_col_indC);
*    hipFree(dcsr_valC);
*    hipFree(dbuffer);
*  \endcode
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sddmm(rocsparse_handle            handle,
                                 rocsparse_operation         opA,
                                 rocsparse_operation         opB,
                                 const void*                 alpha,
                                 rocsparse_const_dnmat_descr A,
                                 rocsparse_const_dnmat_descr B,
                                 const void*                 beta,
                                 rocsparse_spmat_descr       C,
                                 rocsparse_datatype          compute_type,
                                 rocsparse_sddmm_alg         alg,
                                 void*                       temp_buffer);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SDDMM_H */
