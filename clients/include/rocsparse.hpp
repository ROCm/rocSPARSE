/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCSPARSE_HPP_
#define _ROCSPARSE_HPP_

#include <rocsparse.h>

namespace rocsparse {

template <typename T>
rocsparse_status rocsparse_axpyi(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const T* alpha,
                                 const T* x_val,
                                 const rocsparse_int* x_ind,
                                 T* y,
                                 rocsparse_index_base idx_base);

template <typename T>
rocsparse_status rocsparse_doti(rocsparse_handle handle,
                                rocsparse_int nnz,
                                const T* x_val,
                                const rocsparse_int* x_ind,
                                const T* y,
                                T* result,
                                rocsparse_index_base idx_base);

template <typename T>
rocsparse_status rocsparse_gthr(rocsparse_handle handle,
                                rocsparse_int nnz,
                                const T* y,
                                T* x_val,
                                const rocsparse_int* x_ind,
                                rocsparse_index_base idx_base);

template <typename T>
rocsparse_status rocsparse_gthrz(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 T* y,
                                 T* x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);

template <typename T>
rocsparse_status rocsparse_roti(rocsparse_handle handle,
                                rocsparse_int nnz,
                                T* x_val,
                                const rocsparse_int* x_ind,
                                T* y,
                                const T* c,
                                const T* s,
                                rocsparse_index_base idx_base);

template <typename T>
rocsparse_status rocsparse_sctr(rocsparse_handle handle,
                                rocsparse_int nnz,
                                const T* x_val,
                                const rocsparse_int* x_ind,
                                T* y,
                                rocsparse_index_base idx_base);

template <typename T>
rocsparse_status rocsparse_coomv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int nnz,
                                 const T* alpha,
                                 const rocsparse_mat_descr descr,
                                 const T* coo_val,
                                 const rocsparse_int* coo_row_ind,
                                 const rocsparse_int* coo_col_ind,
                                 const T* x,
                                 const T* beta,
                                 T* y);

template <typename T>
rocsparse_status rocsparse_csrmv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int nnz,
                                          const rocsparse_mat_descr descr,
                                          const T* csr_val,
                                          const rocsparse_int* csr_row_ptr,
                                          const rocsparse_int* csr_col_ind,
                                          rocsparse_mat_info info);

template <typename T>
rocsparse_status rocsparse_csrmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int nnz,
                                 const T* alpha,
                                 const rocsparse_mat_descr descr,
                                 const T* csr_val,
                                 const rocsparse_int* csr_row_ptr,
                                 const rocsparse_int* csr_col_ind,
                                 rocsparse_mat_info info,
                                 const T* x,
                                 const T* beta,
                                 T* y);

template <typename T>
rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int nnz,
                                             const rocsparse_mat_descr descr,
                                             const T* csr_val,
                                             const rocsparse_int* csr_row_ptr,
                                             const rocsparse_int* csr_col_ind,
                                             rocsparse_mat_info info,
                                             size_t* buffer_size);

template <typename T>
rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          rocsparse_int m,
                                          rocsparse_int nnz,
                                          const rocsparse_mat_descr descr,
                                          const T* csr_val,
                                          const rocsparse_int* csr_row_ptr,
                                          const rocsparse_int* csr_col_ind,
                                          rocsparse_mat_info info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy solve,
                                          void* temp_buffer);

template <typename T>
rocsparse_status rocsparse_csrsv_solve(rocsparse_handle handle,
                                       rocsparse_operation trans,
                                       rocsparse_int m,
                                       rocsparse_int nnz,
                                       const T* alpha,
                                       const rocsparse_mat_descr descr,
                                       const T* csr_val,
                                       const rocsparse_int* csr_row_ptr,
                                       const rocsparse_int* csr_col_ind,
                                       rocsparse_mat_info info,
                                       const T* x,
                                       T* y,
                                       rocsparse_solve_policy policy,
                                       void* temp_buffer);

template <typename T>
rocsparse_status rocsparse_ellmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 const T* alpha,
                                 const rocsparse_mat_descr descr,
                                 const T* ell_val,
                                 const rocsparse_int* ell_col_ind,
                                 rocsparse_int ell_width,
                                 const T* x,
                                 const T* beta,
                                 T* y);

template <typename T>
rocsparse_status rocsparse_hybmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 const T* alpha,
                                 const rocsparse_mat_descr descr,
                                 const rocsparse_hyb_mat hyb,
                                 const T* x,
                                 const T* beta,
                                 T* y);

template <typename T>
rocsparse_status rocsparse_csrmm(rocsparse_handle handle,
                                 rocsparse_operation trans_A,
                                 rocsparse_operation trans_B,
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int k,
                                 rocsparse_int nnz,
                                 const T* alpha,
                                 const rocsparse_mat_descr descr,
                                 const T* csr_val,
                                 const rocsparse_int* csr_row_ptr,
                                 const rocsparse_int* csr_col_ind,
                                 const T* B,
                                 rocsparse_int ldb,
                                 const T* beta,
                                 T* C,
                                 rocsparse_int ldc);

template <typename T>
rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int nnz,
                                               const rocsparse_mat_descr descr,
                                               const T* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_mat_info info,
                                               size_t* buffer_size);

template <typename T>
rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle handle,
                                            rocsparse_int m,
                                            rocsparse_int nnz,
                                            const rocsparse_mat_descr descr,
                                            const T* csr_val,
                                            const rocsparse_int* csr_row_ptr,
                                            const rocsparse_int* csr_col_ind,
                                            rocsparse_mat_info info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy solve,
                                            void* temp_buffer);

template <typename T>
rocsparse_status rocsparse_csrilu0(rocsparse_handle handle,
                                   rocsparse_int m,
                                   rocsparse_int nnz,
                                   const rocsparse_mat_descr descr,
                                   T* csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   const rocsparse_int* csr_col_ind,
                                   rocsparse_mat_info info,
                                   rocsparse_solve_policy policy,
                                   void* temp_buffer);

template <typename T>
rocsparse_status rocsparse_csr2csc(rocsparse_handle handle,
                                   rocsparse_int m,
                                   rocsparse_int n,
                                   rocsparse_int nnz,
                                   const T* csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   const rocsparse_int* csr_col_ind,
                                   T* csc_val,
                                   rocsparse_int* csc_row_ind,
                                   rocsparse_int* csc_col_ptr,
                                   rocsparse_action copy_values,
                                   rocsparse_index_base idx_base,
                                   void* temp_buffer);

template <typename T>
rocsparse_status rocsparse_csr2ell(rocsparse_handle handle,
                                   rocsparse_int m,
                                   const rocsparse_mat_descr csr_descr,
                                   const T* csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   const rocsparse_int* csr_col_ind,
                                   const rocsparse_mat_descr ell_descr,
                                   rocsparse_int ell_width,
                                   T* ell_val,
                                   rocsparse_int* ell_col_ind);

template <typename T>
rocsparse_status rocsparse_csr2hyb(rocsparse_handle handle,
                                   rocsparse_int m,
                                   rocsparse_int n,
                                   const rocsparse_mat_descr descr,
                                   const T* csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   const rocsparse_int* csr_col_ind,
                                   rocsparse_hyb_mat hyb,
                                   rocsparse_int user_ell_width,
                                   rocsparse_hyb_partition partition_type);

template <typename T>
rocsparse_status rocsparse_ell2csr(rocsparse_handle handle,
                                   rocsparse_int m,
                                   rocsparse_int n,
                                   const rocsparse_mat_descr ell_descr,
                                   rocsparse_int ell_width,
                                   const T* ell_val,
                                   const rocsparse_int* ell_col_ind,
                                   const rocsparse_mat_descr csr_descr,
                                   T* csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   rocsparse_int* csr_col_ind);

} // namespace rocsparse

#endif // _ROCSPARSE_HPP_
