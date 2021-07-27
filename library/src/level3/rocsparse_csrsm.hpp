/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_CSRSM_HPP
#define ROCSPARSE_CSRSM_HPP

#include "handle.h"

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrsm_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_B,
                                                      J                         m,
                                                      J                         nrhs,
                                                      I                         nnz,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  csr_val,
                                                      const I*                  csr_row_ptr,
                                                      const J*                  csr_col_ind,
                                                      const T*                  B,
                                                      J                         ldb,
                                                      rocsparse_mat_info        info,
                                                      rocsparse_solve_policy    policy,
                                                      size_t*                   buffer_size);

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrsm_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   J                         m,
                                                   J                         nrhs,
                                                   I                         nnz,
                                                   const T*                  alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   const T*                  B,
                                                   J                         ldb,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_analysis_policy analysis,
                                                   rocsparse_solve_policy    solve,
                                                   void*                     temp_buffer);

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrsm_solve_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                J                         m,
                                                J                         nrhs,
                                                I                         nnz,
                                                const T*                  alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                T*                        B,
                                                J                         ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_solve_policy    policy,
                                                void*                     temp_buffer);

#endif // ROCSPARSE_CSRSM_HPP
