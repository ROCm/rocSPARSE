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
#ifndef ROCSPARSE_CSRGEMM_HPP
#define ROCSPARSE_CSRGEMM_HPP

#include "rocsparse.h"

template <typename T>
rocsparse_status rocsparse_csrgemm_buffer_size_template(rocsparse_handle handle,
                                                        rocsparse_int m,
                                                        rocsparse_int n,
                                                        rocsparse_int k,
                                                        const T* alpha,
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
    return rocsparse_status_not_implemented;
}

template <typename T>
rocsparse_status rocsparse_csrgemm_template(rocsparse_handle handle,
                                            rocsparse_int m,
                                            rocsparse_int n,
                                            rocsparse_int k,
                                            const T* alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int nnz_A,
                                            const T* csr_val_A,
                                            const rocsparse_int* csr_row_ptr_A,
                                            const rocsparse_int* csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int nnz_B,
                                            const T* csr_val_B,
                                            const rocsparse_int* csr_row_ptr_B,
                                            const rocsparse_int* csr_col_ind_B,
                                            const rocsparse_mat_descr descr_C,
                                            T* csr_val_C,
                                            const rocsparse_int* csr_row_ptr_C,
                                            rocsparse_int* csr_col_ind_C,
                                            const rocsparse_mat_info info,
                                            void* temp_buffer)
{
    return rocsparse_status_not_implemented;
}

#endif // ROCSPARSE_CSRGEMM_HPP
