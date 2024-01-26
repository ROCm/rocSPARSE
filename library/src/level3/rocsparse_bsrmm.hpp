/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "control.h"
#include "handle.h"

namespace rocsparse
{
    template <typename T, typename U>
    rocsparse_status bsrmm_template_dispatch(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             rocsparse_int             mb,
                                             rocsparse_int             n,
                                             rocsparse_int             kb,
                                             rocsparse_int             nnzb,
                                             U                         alpha,
                                             const rocsparse_mat_descr descr,
                                             const T*                  bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             block_dim,
                                             const T*                  B,
                                             int64_t                   ldb,
                                             U                         beta,
                                             T*                        C,
                                             int64_t                   ldc);
    template <typename T>
    rocsparse_status bsrmm_quickreturn(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             mb,
                                       rocsparse_int             n,
                                       rocsparse_int             kb,
                                       rocsparse_int             nnzb,
                                       const T*                  alpha,
                                       const rocsparse_mat_descr descr,
                                       const T*                  bsr_val,
                                       const rocsparse_int*      bsr_row_ptr,
                                       const rocsparse_int*      bsr_col_ind,
                                       rocsparse_int             block_dim,
                                       const T*                  B,
                                       int64_t                   ldb,
                                       const T*                  beta,
                                       T*                        C,
                                       int64_t                   ldc);
    template <typename T>
    rocsparse_status bsrmm_core(rocsparse_handle          handle,
                                rocsparse_direction       dir,
                                rocsparse_operation       trans_A,
                                rocsparse_operation       trans_B,
                                rocsparse_int             mb,
                                rocsparse_int             n,
                                rocsparse_int             kb,
                                rocsparse_int             nnzb,
                                const T*                  alpha,
                                const rocsparse_mat_descr descr,
                                const T*                  bsr_val,
                                const rocsparse_int*      bsr_row_ptr,
                                const rocsparse_int*      bsr_col_ind,
                                rocsparse_int             block_dim,
                                const T*                  B,
                                int64_t                   ldb,
                                const T*                  beta,
                                T*                        C,
                                int64_t                   ldc);

    template <typename T, typename... P>
    rocsparse_status bsrmm_template(P&&... p)
    {
        const rocsparse_status status = rocsparse::bsrmm_quickreturn<T>(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_core(p...));
        return rocsparse_status_success;
    }
}
