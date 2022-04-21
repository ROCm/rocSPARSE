/*! \file */
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include "rocsparse_cscmv.hpp"
#include "rocsparse_csrmv.hpp"

#include "definitions.h"
#include "utility.h"

template <typename I, typename J, typename T>
rocsparse_status rocsparse_cscmv_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   J                         m,
                                                   J                         n,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csc_val,
                                                   const I*                  csc_col_ptr,
                                                   const J*                  csc_row_ind,
                                                   rocsparse_mat_info        info)
{
    switch(trans)
    {
    case rocsparse_operation_none:
    {
        return rocsparse_csrmv_analysis_template(handle,
                                                 rocsparse_operation_transpose,
                                                 n,
                                                 m,
                                                 nnz,
                                                 descr,
                                                 csc_val,
                                                 csc_col_ptr,
                                                 csc_row_ind,
                                                 info);
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        return rocsparse_csrmv_analysis_template(handle,
                                                 rocsparse_operation_none,
                                                 n,
                                                 m,
                                                 nnz,
                                                 descr,
                                                 csc_val,
                                                 csc_col_ptr,
                                                 csc_row_ind,
                                                 info);
    }
    }

    return rocsparse_status_not_implemented;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_cscmv_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          J                         m,
                                          J                         n,
                                          I                         nnz,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csc_val,
                                          const I*                  csc_col_ptr,
                                          const J*                  csc_row_ind,
                                          rocsparse_mat_info        info,
                                          const T*                  x,
                                          const T*                  beta,
                                          T*                        y)
{
    switch(trans)
    {
    case rocsparse_operation_none:
    {
        return rocsparse_csrmv_template(handle,
                                        rocsparse_operation_transpose,
                                        n,
                                        m,
                                        nnz,
                                        alpha,
                                        descr,
                                        csc_val,
                                        csc_col_ptr,
                                        csc_row_ind,
                                        info,
                                        x,
                                        beta,
                                        y,
                                        false);
    }
    case rocsparse_operation_transpose:
    {
        return rocsparse_csrmv_template(handle,
                                        rocsparse_operation_none,
                                        n,
                                        m,
                                        nnz,
                                        alpha,
                                        descr,
                                        csc_val,
                                        csc_col_ptr,
                                        csc_row_ind,
                                        info,
                                        x,
                                        beta,
                                        y,
                                        false);
    }
    case rocsparse_operation_conjugate_transpose:
    {
        return rocsparse_csrmv_template(handle,
                                        rocsparse_operation_none,
                                        n,
                                        m,
                                        nnz,
                                        alpha,
                                        descr,
                                        csc_val,
                                        csc_col_ptr,
                                        csc_row_ind,
                                        info,
                                        x,
                                        beta,
                                        y,
                                        true);
    }
    }

    return rocsparse_status_not_implemented;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                              \
    template rocsparse_status rocsparse_cscmv_analysis_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans,                                              \
        JTYPE                     m,                                                  \
        JTYPE                     n,                                                  \
        ITYPE                     nnz,                                                \
        const rocsparse_mat_descr descr,                                              \
        const TTYPE*              csc_val,                                            \
        const ITYPE*              csc_col_ptr,                                        \
        const JTYPE*              csc_row_ind,                                        \
        rocsparse_mat_info        info);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                     \
    template rocsparse_status rocsparse_cscmv_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                    \
        rocsparse_operation       trans,                                     \
        JTYPE                     m,                                         \
        JTYPE                     n,                                         \
        ITYPE                     nnz,                                       \
        const TTYPE*              alpha,                                     \
        const rocsparse_mat_descr descr,                                     \
        const TTYPE*              csc_val,                                   \
        const ITYPE*              csc_col_ptr,                               \
        const JTYPE*              csc_row_ind,                               \
        rocsparse_mat_info        info,                                      \
        const TTYPE*              x,                                         \
        const TTYPE*              beta,                                      \
        TTYPE*                    y);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE
