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

#include "rocsparse_cscmm.hpp"

#include "definitions.h"
#include "utility.h"

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
                                                      size_t*                   buffer_size)
{
    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        return rocsparse_csrmm_buffer_size_template(handle,
                                                    rocsparse_operation_transpose,
                                                    alg,
                                                    k,
                                                    n,
                                                    m,
                                                    nnz,
                                                    descr,
                                                    csc_val,
                                                    csc_col_ptr,
                                                    csc_row_ind,
                                                    buffer_size);
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        return rocsparse_csrmm_buffer_size_template(handle,
                                                    rocsparse_operation_none,
                                                    alg,
                                                    k,
                                                    n,
                                                    m,
                                                    nnz,
                                                    descr,
                                                    csc_val,
                                                    csc_col_ptr,
                                                    csc_row_ind,
                                                    buffer_size);
    }
    }

    return rocsparse_status_not_implemented;
}

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
                                                   void*                     temp_buffer)
{
    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        return rocsparse_csrmm_analysis_template(handle,
                                                 rocsparse_operation_transpose,
                                                 alg,
                                                 k,
                                                 n,
                                                 m,
                                                 nnz,
                                                 descr,
                                                 csc_val,
                                                 csc_col_ptr,
                                                 csc_row_ind,
                                                 temp_buffer);
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        return rocsparse_csrmm_analysis_template(handle,
                                                 rocsparse_operation_none,
                                                 alg,
                                                 k,
                                                 n,
                                                 m,
                                                 nnz,
                                                 descr,
                                                 csc_val,
                                                 csc_col_ptr,
                                                 csc_row_ind,
                                                 temp_buffer);
    }
    }

    return rocsparse_status_not_implemented;
}

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
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csc_val,
                                          const I*                  csc_col_ptr,
                                          const J*                  csc_row_ind,
                                          const T*                  B,
                                          J                         ldb,
                                          J                         batch_count_B,
                                          I                         batch_stride_B,
                                          const T*                  beta_device_host,
                                          T*                        C,
                                          J                         ldc,
                                          J                         batch_count_C,
                                          I                         batch_stride_C,
                                          void*                     temp_buffer)
{
    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        return rocsparse_csrmm_template(handle,
                                        rocsparse_operation_transpose,
                                        trans_B,
                                        order_B,
                                        order_C,
                                        alg,
                                        k,
                                        n,
                                        m,
                                        nnz,
                                        batch_count_A,
                                        offsets_batch_stride_A,
                                        rows_values_batch_stride_A,
                                        alpha_device_host,
                                        descr,
                                        csc_val,
                                        csc_col_ptr,
                                        csc_row_ind,
                                        B,
                                        ldb,
                                        batch_count_B,
                                        batch_stride_B,
                                        beta_device_host,
                                        C,
                                        ldc,
                                        batch_count_C,
                                        batch_stride_C,
                                        temp_buffer,
                                        false);
    }
    case rocsparse_operation_transpose:
    {
        return rocsparse_csrmm_template(handle,
                                        rocsparse_operation_none,
                                        trans_B,
                                        order_B,
                                        order_C,
                                        alg,
                                        k,
                                        n,
                                        m,
                                        nnz,
                                        batch_count_A,
                                        offsets_batch_stride_A,
                                        rows_values_batch_stride_A,
                                        alpha_device_host,
                                        descr,
                                        csc_val,
                                        csc_col_ptr,
                                        csc_row_ind,
                                        B,
                                        ldb,
                                        batch_count_B,
                                        batch_stride_B,
                                        beta_device_host,
                                        C,
                                        ldc,
                                        batch_count_C,
                                        batch_stride_C,
                                        temp_buffer,
                                        false);
    }
    case rocsparse_operation_conjugate_transpose:
    {
        return rocsparse_csrmm_template(handle,
                                        rocsparse_operation_none,
                                        trans_B,
                                        order_B,
                                        order_C,
                                        alg,
                                        k,
                                        n,
                                        m,
                                        nnz,
                                        batch_count_A,
                                        offsets_batch_stride_A,
                                        rows_values_batch_stride_A,
                                        alpha_device_host,
                                        descr,
                                        csc_val,
                                        csc_col_ptr,
                                        csc_row_ind,
                                        B,
                                        ldb,
                                        batch_count_B,
                                        batch_stride_B,
                                        beta_device_host,
                                        C,
                                        ldc,
                                        batch_count_C,
                                        batch_stride_C,
                                        temp_buffer,
                                        true);
    }
    }

    return rocsparse_status_not_implemented;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template rocsparse_status rocsparse_cscmm_buffer_size_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                                \
        rocsparse_operation       trans_A,                                               \
        rocsparse_csrmm_alg       alg,                                                   \
        JTYPE                     m,                                                     \
        JTYPE                     n,                                                     \
        JTYPE                     k,                                                     \
        ITYPE                     nnz,                                                   \
        const rocsparse_mat_descr descr,                                                 \
        const TTYPE*              csc_val,                                               \
        const ITYPE*              csc_col_ptr,                                           \
        const JTYPE*              csc_row_ind,                                           \
        size_t*                   buffer_size);

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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                              \
    template rocsparse_status rocsparse_cscmm_analysis_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans_A,                                            \
        rocsparse_csrmm_alg       alg,                                                \
        JTYPE                     m,                                                  \
        JTYPE                     n,                                                  \
        JTYPE                     k,                                                  \
        ITYPE                     nnz,                                                \
        const rocsparse_mat_descr descr,                                              \
        const TTYPE*              csc_val,                                            \
        const ITYPE*              csc_col_ptr,                                        \
        const JTYPE*              csc_row_ind,                                        \
        void*                     temp_buffer);

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
    template rocsparse_status rocsparse_cscmm_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                    \
        rocsparse_operation       trans_A,                                   \
        rocsparse_operation       trans_B,                                   \
        rocsparse_order           order_B,                                   \
        rocsparse_order           order_C,                                   \
        rocsparse_csrmm_alg       alg,                                       \
        JTYPE                     m,                                         \
        JTYPE                     n,                                         \
        JTYPE                     k,                                         \
        ITYPE                     nnz,                                       \
        JTYPE                     batch_count_A,                             \
        ITYPE                     offsets_batch_stride_A,                    \
        ITYPE                     rows_values_batch_stride_A,                \
        const TTYPE*              alpha_device_host,                         \
        const rocsparse_mat_descr descr,                                     \
        const TTYPE*              csc_val,                                   \
        const ITYPE*              csc_col_ptr,                               \
        const JTYPE*              csc_row_ind,                               \
        const TTYPE*              B,                                         \
        JTYPE                     ldb,                                       \
        JTYPE                     batch_count_B,                             \
        ITYPE                     batch_stride_B,                            \
        const TTYPE*              beta_device_host,                          \
        TTYPE*                    C,                                         \
        JTYPE                     ldc,                                       \
        JTYPE                     batch_count_C,                             \
        ITYPE                     batch_stride_C,                            \
        void*                     temp_buffer);

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
