/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
rocsparse_status rocsparse_bsrsm_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_X,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nrhs,
                                                      rocsparse_int             nnzb,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             block_dim,
                                                      rocsparse_mat_info        info,
                                                      size_t*                   buffer_size);

template <typename T>
rocsparse_status rocsparse_bsrsm_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_X,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nrhs,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  bsr_val,
                                                   const rocsparse_int*      bsr_row_ptr,
                                                   const rocsparse_int*      bsr_col_ind,
                                                   rocsparse_int             block_dim,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_analysis_policy analysis,
                                                   rocsparse_solve_policy    solve,
                                                   void*                     temp_buffer);

template <typename T>
rocsparse_status rocsparse_bsrsm_solve_template(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_X,
                                                rocsparse_int             mb,
                                                rocsparse_int             nrhs,
                                                rocsparse_int             nnzb,
                                                const T*                  alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                const T*                  B,
                                                int64_t                   ldb,
                                                T*                        x,
                                                int64_t                   ldx,
                                                rocsparse_solve_policy    policy,
                                                void*                     temp_buffer);
