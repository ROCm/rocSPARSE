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

#pragma once
#ifndef ROCSPARSE_CSCMM_HPP
#define ROCSPARSE_CSCMM_HPP

#include "handle.h"

#include "rocsparse_csrmm.hpp"

template <typename I, typename J, typename T>
rocsparse_status rocsparse_cscmm_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_csrmm_alg       alg,
                                                      J                         m,
                                                      J                         n,
                                                      J                         k,
                                                      I                         nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  csc_val,
                                                      const I*                  csc_col_ptr,
                                                      const J*                  csc_row_ind,
                                                      size_t*                   buffer_size);

template <typename I, typename J, typename T>
rocsparse_status rocsparse_cscmm_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_csrmm_alg       alg,
                                                   J                         m,
                                                   J                         n,
                                                   J                         k,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csc_val,
                                                   const I*                  csc_col_ptr,
                                                   const J*                  csc_row_ind,
                                                   void*                     temp_buffer);

template <typename I, typename J, typename T>
rocsparse_status rocsparse_cscmm_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_order           order_B,
                                          rocsparse_order           order_C,
                                          rocsparse_csrmm_alg       alg,
                                          J                         m,
                                          J                         n,
                                          J                         k,
                                          I                         nnz,
                                          J                         batch_count_A,
                                          I                         offsets_batch_stride_A,
                                          I                         rows_values_batch_stride_A,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csc_val,
                                          const I*                  csc_col_ptr,
                                          const J*                  csc_row_ind,
                                          const T*                  B,
                                          J                         ldb,
                                          J                         batch_count_B,
                                          I                         batch_stride_B,
                                          const T*                  beta,
                                          T*                        C,
                                          J                         ldc,
                                          J                         batch_count_C,
                                          I                         batch_stride_C,
                                          void*                     temp_buffer);

#endif // ROCSPARSE_CSCMM_HPP
