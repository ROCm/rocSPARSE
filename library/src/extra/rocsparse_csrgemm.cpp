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

#include "rocsparse.h"
#include "rocsparse_csrgemm.hpp"

extern "C" rocsparse_status rocsparse_csrgemm_nnz(rocsparse_handle handle,
                                                  rocsparse_int m,
                                                  rocsparse_int n,
                                                  rocsparse_int k,
                                                  const rocsparse_mat_descr descr_A,
                                                  rocsparse_int nnz_A,
                                                  const rocsparse_int* csr_row_ptr_A,
                                                  const rocsparse_int* csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  rocsparse_int nnz_B,
                                                  const rocsparse_int* csr_row_ptr_B,
                                                  const rocsparse_int* csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_C,
                                                  rocsparse_int* csr_row_ptr_C,
                                                  rocsparse_int* nnz_C,
                                                  const rocsparse_mat_info info,
                                                  void* temp_buffer)
{
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrgemm_buffer_size(rocsparse_handle handle,
                                                           rocsparse_int m,
                                                           rocsparse_int n,
                                                           rocsparse_int k,
                                                           const float* alpha,
                                                           const rocsparse_mat_descr descr_A,
                                                           rocsparse_int nnz_A,
                                                           const rocsparse_int* csr_row_ptr_A,
                                                           const rocsparse_int* csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           rocsparse_int nnz_B,
                                                           const rocsparse_int* csr_row_ptr_B,
                                                           const rocsparse_int* csr_col_ind_B,
                                                           rocsparse_mat_info info,
                                                           size_t* buffer_size)
{
    return rocsparse_csrgemm_buffer_size_template<float>(handle,
                                                         m,
                                                         n,
                                                         k,
                                                         alpha,
                                                         descr_A,
                                                         nnz_A,
                                                         csr_row_ptr_A,
                                                         csr_col_ind_A,
                                                         descr_B,
                                                         nnz_B,
                                                         csr_row_ptr_B,
                                                         csr_col_ind_B,
                                                         info,
                                                         buffer_size);
}

extern "C" rocsparse_status rocsparse_dcsrgemm_buffer_size(rocsparse_handle handle,
                                                           rocsparse_int m,
                                                           rocsparse_int n,
                                                           rocsparse_int k,
                                                           const double* alpha,
                                                           const rocsparse_mat_descr descr_A,
                                                           rocsparse_int nnz_A,
                                                           const rocsparse_int* csr_row_ptr_A,
                                                           const rocsparse_int* csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           rocsparse_int nnz_B,
                                                           const rocsparse_int* csr_row_ptr_B,
                                                           const rocsparse_int* csr_col_ind_B,
                                                           rocsparse_mat_info info,
                                                           size_t* buffer_size)
{
    return rocsparse_csrgemm_buffer_size_template<double>(handle,
                                                          m,
                                                          n,
                                                          k,
                                                          alpha,
                                                          descr_A,
                                                          nnz_A,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          descr_B,
                                                          nnz_B,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          info,
                                                          buffer_size);
}

extern "C" rocsparse_status rocsparse_scsrgemm(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int k,
                                               const float* alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int nnz_A,
                                               const float* csr_val_A,
                                               const rocsparse_int* csr_row_ptr_A,
                                               const rocsparse_int* csr_col_ind_A,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int nnz_B,
                                               const float* csr_val_B,
                                               const rocsparse_int* csr_row_ptr_B,
                                               const rocsparse_int* csr_col_ind_B,
                                               const rocsparse_mat_descr descr_C,
                                               float* csr_val_C,
                                               const rocsparse_int* csr_row_ptr_C,
                                               rocsparse_int* csr_col_ind_C,
                                               const rocsparse_mat_info info,
                                               void* temp_buffer)
{
    return rocsparse_csrgemm_template<float>(handle,
                                             m,
                                             n,
                                             k,
                                             alpha,
                                             descr_A,
                                             nnz_A,
                                             csr_val_A,
                                             csr_row_ptr_A,
                                             csr_col_ind_A,
                                             descr_B,
                                             nnz_B,
                                             csr_val_B,
                                             csr_row_ptr_B,
                                             csr_col_ind_B,
                                             descr_C,
                                             csr_val_C,
                                             csr_row_ptr_C,
                                             csr_col_ind_C,
                                             info,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsrgemm(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int k,
                                               const double* alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int nnz_A,
                                               const double* csr_val_A,
                                               const rocsparse_int* csr_row_ptr_A,
                                               const rocsparse_int* csr_col_ind_A,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int nnz_B,
                                               const double* csr_val_B,
                                               const rocsparse_int* csr_row_ptr_B,
                                               const rocsparse_int* csr_col_ind_B,
                                               const rocsparse_mat_descr descr_C,
                                               double* csr_val_C,
                                               const rocsparse_int* csr_row_ptr_C,
                                               rocsparse_int* csr_col_ind_C,
                                               const rocsparse_mat_info info,
                                               void* temp_buffer)
{
    return rocsparse_csrgemm_template<double>(handle,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_val_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_val_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              descr_C,
                                              csr_val_C,
                                              csr_row_ptr_C,
                                              csr_col_ind_C,
                                              info,
                                              temp_buffer);
}
