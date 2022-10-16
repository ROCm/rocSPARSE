/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "handle.h"

template <typename T>
rocsparse_status rocsparse_csr2csc_core(rocsparse_handle     handle,
                                        rocsparse_int        m,
                                        rocsparse_int        n,
                                        rocsparse_int        nnz,
                                        const T*             csr_val,
                                        const rocsparse_int* csr_row_ptr_begin,
                                        const rocsparse_int* csr_row_ptr_end,
                                        const rocsparse_int* csr_col_ind,
                                        T*                   csc_val,
                                        rocsparse_int*       csc_row_ind,
                                        rocsparse_int*       csc_col_ptr,
                                        rocsparse_action     copy_values,
                                        rocsparse_index_base idx_base,
                                        void*                temp_buffer);

template <typename T>
rocsparse_status rocsparse_csr2csc_template(rocsparse_handle     handle,
                                            rocsparse_int        m,
                                            rocsparse_int        n,
                                            rocsparse_int        nnz,
                                            const T*             csr_val,
                                            const rocsparse_int* csr_row_ptr,
                                            const rocsparse_int* csr_col_ind,
                                            T*                   csc_val,
                                            rocsparse_int*       csc_row_ind,
                                            rocsparse_int*       csc_col_ptr,
                                            rocsparse_action     copy_values,
                                            rocsparse_index_base idx_base,
                                            void*                temp_buffer);

template <typename T>
rocsparse_status rocsparse_csr2csc_impl(rocsparse_handle     handle,
                                        rocsparse_int        m,
                                        rocsparse_int        n,
                                        rocsparse_int        nnz,
                                        const T*             csr_val,
                                        const rocsparse_int* csr_row_ptr,
                                        const rocsparse_int* csr_col_ind,
                                        T*                   csc_val,
                                        rocsparse_int*       csc_row_ind,
                                        rocsparse_int*       csc_col_ptr,
                                        rocsparse_action     copy_values,
                                        rocsparse_index_base idx_base,
                                        void*                temp_buffer);

rocsparse_status rocsparse_csr2csc_buffer_size_template(rocsparse_handle     handle,
                                                        rocsparse_int        m,
                                                        rocsparse_int        n,
                                                        rocsparse_int        nnz,
                                                        const rocsparse_int* csr_row_ptr,
                                                        const rocsparse_int* csr_col_ind,
                                                        rocsparse_action     copy_values,
                                                        size_t*              buffer_size);

rocsparse_status rocsparse_csr2csc_buffer_size_core(rocsparse_handle     handle,
                                                    rocsparse_int        m,
                                                    rocsparse_int        n,
                                                    rocsparse_int        nnz,
                                                    const rocsparse_int* csr_row_ptr_begin,
                                                    const rocsparse_int* csr_row_ptr_end,
                                                    const rocsparse_int* csr_col_ind,
                                                    rocsparse_action     copy_values,
                                                    size_t*              buffer_size);
