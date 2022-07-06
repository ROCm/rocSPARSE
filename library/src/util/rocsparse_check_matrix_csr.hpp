/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_CHECK_MATRIX_CSR_HPP
#define ROCSPARSE_CHECK_MATRIX_CSR_HPP

#include "handle.h"

template <typename T, typename I, typename J>
rocsparse_status rocsparse_check_matrix_csr_buffer_size_template(rocsparse_handle       handle,
                                                                 J                      m,
                                                                 J                      n,
                                                                 I                      nnz,
                                                                 const T*               csr_val,
                                                                 const I*               csr_row_ptr,
                                                                 const J*               csr_col_ind,
                                                                 rocsparse_index_base   idx_base,
                                                                 rocsparse_matrix_type  matrix_type,
                                                                 rocsparse_fill_mode    uplo,
                                                                 rocsparse_storage_mode storage,
                                                                 size_t* buffer_size);

template <typename T, typename I, typename J>
rocsparse_status rocsparse_check_matrix_csr_template(rocsparse_handle       handle,
                                                     J                      m,
                                                     J                      n,
                                                     I                      nnz,
                                                     const T*               csr_val,
                                                     const I*               csr_row_ptr,
                                                     const J*               csr_col_ind,
                                                     rocsparse_index_base   idx_base,
                                                     rocsparse_matrix_type  matrix_type,
                                                     rocsparse_fill_mode    uplo,
                                                     rocsparse_storage_mode storage,
                                                     rocsparse_data_status* data_status,
                                                     void*                  temp_buffer);

#endif // ROCSPARSE_CHECK_MATRIX_CSR_HPP
