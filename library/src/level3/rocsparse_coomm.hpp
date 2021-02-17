/*! \file */
/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_COOMM_HPP
#define ROCSPARSE_COOMM_HPP

#include "handle.h"

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_order           order,
                                                   rocsparse_spmm_alg        alg,
                                                   I                         m,
                                                   I                         n,
                                                   I                         k,
                                                   I                         nnz,
                                                   U                         alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  coo_val,
                                                   const I*                  coo_row_ind,
                                                   const I*                  coo_col_ind,
                                                   const T*                  B,
                                                   I                         ldb,
                                                   U                         beta_device_host,
                                                   T*                        C,
                                                   I                         ldc);

template <typename I, typename T>
rocsparse_status rocsparse_coomm_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_order           order_B,
                                          rocsparse_order           order_C,
                                          rocsparse_spmm_alg        alg,
                                          I                         m,
                                          I                         n,
                                          I                         k,
                                          I                         nnz,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  coo_val,
                                          const I*                  coo_row_ind,
                                          const I*                  coo_col_ind,
                                          const T*                  B,
                                          I                         ldb,
                                          const T*                  beta,
                                          T*                        C,
                                          I                         ldc);

#endif // ROCSPARSE_COOMM_HPP
