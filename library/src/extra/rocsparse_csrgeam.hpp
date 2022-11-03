/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename T, typename U>
rocsparse_status rocsparse_csrgeam_dispatch(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            U                         alpha_device_host,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const T*                  csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            U                         beta_device_host,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const T*                  csr_val_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        csr_val_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C);

template <typename T>
rocsparse_status rocsparse_csrgeam_template(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const T*                  csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const T*                  csr_val_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        csr_val_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C);
