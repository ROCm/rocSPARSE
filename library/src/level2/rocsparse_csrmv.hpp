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
#ifndef ROCSPARSE_CSRMV_HPP
#define ROCSPARSE_CSRMV_HPP

#include "handle.h"

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrmv_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   J                         m,
                                                   J                         n,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   rocsparse_mat_info        info);

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmv_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   J                         m,
                                                   J                         n,
                                                   I                         nnz,
                                                   U                         alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   const T*                  x,
                                                   U                         beta_device_host,
                                                   T*                        y,
                                                   bool                      force_conj);

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrmv_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          J                         m,
                                          J                         n,
                                          I                         nnz,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          rocsparse_mat_info        info,
                                          const T*                  x,
                                          const T*                  beta,
                                          T*                        y,
                                          bool                      force_conj);

#endif // ROCSPARSE_CSRMV_HPP
