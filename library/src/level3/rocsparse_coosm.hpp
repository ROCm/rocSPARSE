/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

namespace rocsparse
{
    template <typename I, typename T>
    rocsparse_status coosm_buffer_size_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                I                         m,
                                                I                         nrhs,
                                                int64_t                   nnz,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  coo_val,
                                                const I*                  coo_row_ind,
                                                const I*                  coo_col_ind,
                                                const T*                  B,
                                                int64_t                   ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_solve_policy    policy,
                                                size_t*                   buffer_size);

    template <typename I, typename T>
    rocsparse_status coosm_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             I                         m,
                                             I                         nrhs,
                                             int64_t                   nnz,
                                             const T*                  alpha,
                                             const rocsparse_mat_descr descr,
                                             const T*                  coo_val,
                                             const I*                  coo_row_ind,
                                             const I*                  coo_col_ind,
                                             const T*                  B,
                                             int64_t                   ldb,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             void*                     temp_buffer);

    template <typename I, typename T>
    rocsparse_status coosm_solve_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          I                         m,
                                          I                         nrhs,
                                          int64_t                   nnz,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  coo_val,
                                          const I*                  coo_row_ind,
                                          const I*                  coo_col_ind,
                                          T*                        B,
                                          int64_t                   ldb,
                                          rocsparse_mat_info        info,
                                          rocsparse_solve_policy    policy,
                                          void*                     temp_buffer);
}
