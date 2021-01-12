/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_CSRMM_HPP
#define ROCSPARSE_CSRMM_HPP

#include "handle.h"

template <typename T, typename U>
rocsparse_status rocsparse_csrmm_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_int             m,
                                                   rocsparse_int             n,
                                                   rocsparse_int             k,
                                                   rocsparse_int             nnz,
                                                   U                         alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   const T*                  B,
                                                   rocsparse_int             ldb,
                                                   U                         beta_device_host,
                                                   T*                        C,
                                                   rocsparse_int             ldc);

template <typename T>
rocsparse_status rocsparse_csrmm_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          rocsparse_int             k,
                                          rocsparse_int             nnz,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          const T*                  B,
                                          rocsparse_int             ldb,
                                          const T*                  beta,
                                          T*                        C,
                                          rocsparse_int             ldc);

#endif // ROCSPARSE_CSRMM_HPP
