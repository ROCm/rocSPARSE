/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "utility.h"

template <typename T, typename I, typename J>
rocsparse_status rocsparse_ell2csr_core(rocsparse_handle          handle,
                                        J                         m,
                                        J                         n,
                                        const rocsparse_mat_descr ell_descr,
                                        J                         ell_width,
                                        const T*                  ell_val,
                                        const J*                  ell_col_ind,
                                        const rocsparse_mat_descr csr_descr,
                                        T*                        csr_val,
                                        const I*                  csr_row_ptr,
                                        J*                        csr_col_ind);

rocsparse_status rocsparse_ell2csr_quickreturn(rocsparse_handle          handle,
                                               int64_t                   m,
                                               int64_t                   n,
                                               const rocsparse_mat_descr ell_descr,
                                               int64_t                   ell_width,
                                               const void*               ell_val,
                                               const void*               ell_col_ind,
                                               const rocsparse_mat_descr csr_descr,
                                               void*                     csr_val,
                                               const void*               csr_row_ptr,
                                               void*                     csr_col_ind);

template <typename... P>
rocsparse_status rocsparse_ell2csr_template(P... p)
{
    const rocsparse_status status = rocsparse_ell2csr_quickreturn(p...);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ell2csr_core(p...));
    return rocsparse_status_success;
}

rocsparse_status rocsparse_ell2csr_checkarg(rocsparse_handle          handle,
                                            int64_t                   m,
                                            int64_t                   n,
                                            const rocsparse_mat_descr ell_descr,
                                            int64_t                   ell_width,
                                            const void*               ell_val,
                                            const void*               ell_col_ind,
                                            const rocsparse_mat_descr csr_descr,
                                            void*                     csr_val,
                                            const void*               csr_row_ptr,
                                            void*                     csr_col_ind);

template <typename... P>
rocsparse_status rocsparse_ell2csr_impl(P... p)
{
    log_trace("rocsparse_ell2csr_impl", p...);

    const rocsparse_status status = rocsparse_ell2csr_checkarg(p...);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ell2csr_core(p...));
    return rocsparse_status_success;
}

rocsparse_status rocsparse_ell2csr_nnz_quickreturn(rocsparse_handle          handle,
                                                   int64_t                   m,
                                                   int64_t                   n,
                                                   const rocsparse_mat_descr ell_descr,
                                                   int64_t                   ell_width,
                                                   const void*               ell_col_ind,
                                                   const rocsparse_mat_descr csr_descr,
                                                   void*                     csr_row_ptr,
                                                   void*                     csr_nnz);

rocsparse_status rocsparse_ell2csr_nnz_checkarg(rocsparse_handle          handle,
                                                int64_t                   m,
                                                int64_t                   n,
                                                const rocsparse_mat_descr ell_descr,
                                                int64_t                   ell_width,
                                                const void*               ell_col_ind,
                                                const rocsparse_mat_descr csr_descr,
                                                void*                     csr_row_ptr,
                                                void*                     csr_nnz);

template <typename I, typename J>
rocsparse_status rocsparse_ell2csr_nnz_core(rocsparse_handle          handle,
                                            J                         m,
                                            J                         n,
                                            const rocsparse_mat_descr ell_descr,
                                            J                         ell_width,
                                            const J*                  ell_col_ind,
                                            const rocsparse_mat_descr csr_descr,
                                            I*                        csr_row_ptr,
                                            I*                        csr_nnz);

template <typename... P>
rocsparse_status rocsparse_ell2csr_nnz_template(P... p)
{
    log_trace("rocsparse_ell2csr_nnz", p...);
    const rocsparse_status status = rocsparse_ell2csr_nnz_quickreturn(p...);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ell2csr_nnz_core(p...));
    return rocsparse_status_success;
}

template <typename... P>
rocsparse_status rocsparse_ell2csr_nnz_impl(P... p)
{
    log_trace("rocsparse_ell2csr_nnz", p...);
    const rocsparse_status status = rocsparse_ell2csr_nnz_checkarg(p...);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ell2csr_nnz_core(p...));
    return rocsparse_status_success;
}
