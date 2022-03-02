/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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

#include "rocsparse_bellmm.hpp"
#include "definitions.h"
#include "utility.h"

template <typename T, typename U, typename I>
rocsparse_status rocsparse_bellmm_template_general(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_order           order_B,
                                                   rocsparse_order           order_C,
                                                   rocsparse_direction       dir_A,
                                                   I                         mb,
                                                   I                         n,
                                                   I                         kb,
                                                   I                         bell_cols,
                                                   I                         bell_block_dim,
                                                   U                         alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const I*                  bell_col_ind,
                                                   const T*                  bell_val,
                                                   const T*                  B,
                                                   I                         ldb,
                                                   U                         beta,
                                                   T*                        C,
                                                   I                         ldc);

template <typename T, typename U, typename I>
rocsparse_status rocsparse_bellmm_template_dispatch(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    rocsparse_order           order_B,
                                                    rocsparse_order           order_C,
                                                    rocsparse_direction       dir_A,
                                                    I                         mb,
                                                    I                         n,
                                                    I                         kb,
                                                    I                         bell_cols,
                                                    I                         bell_block_dim,
                                                    U                         alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const I*                  bell_col_ind,
                                                    const T*                  bell_val,
                                                    const T*                  B,
                                                    I                         ldb,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    I                         ldc)
{

    return rocsparse_bellmm_template_general(handle,
                                             trans_A,
                                             trans_B,
                                             order_B,
                                             order_C,
                                             dir_A,
                                             mb,
                                             n,
                                             kb,
                                             bell_cols,
                                             bell_block_dim,
                                             alpha_device_host,
                                             descr,
                                             bell_col_ind,
                                             bell_val,
                                             B,
                                             ldb,
                                             beta_device_host,
                                             C,
                                             ldc);
}

template <typename T, typename I>
rocsparse_status rocsparse_bellmm_template_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_operation       trans_A,
                                                       rocsparse_operation       trans_B,
                                                       rocsparse_order           order_B,
                                                       rocsparse_order           order_C,
                                                       rocsparse_direction       dir_A,
                                                       I                         mb,
                                                       I                         n,
                                                       I                         kb,
                                                       I                         bell_cols,
                                                       I                         bell_block_dim,
                                                       const T*                  alpha,
                                                       const rocsparse_mat_descr descr,
                                                       const I*                  bell_col_ind,
                                                       const T*                  bell_val,
                                                       const T*                  B,
                                                       I                         ldb,
                                                       const T*                  beta,
                                                       T*                        C,
                                                       I                         ldc,
                                                       size_t*                   buffer_size)
{
    *buffer_size = 4;
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_bellmm_template_preprocess(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_B,
                                                      rocsparse_order           order_B,
                                                      rocsparse_order           order_C,
                                                      rocsparse_direction       dir_A,
                                                      I                         mb,
                                                      I                         n,
                                                      I                         kb,
                                                      I                         bell_cols,
                                                      I                         bell_block_dim,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const I*                  bell_col_ind,
                                                      const T*                  bell_val,
                                                      const T*                  B,
                                                      I                         ldb,
                                                      const T*                  beta,
                                                      T*                        C,
                                                      I                         ldc,
                                                      void*                     temp_buffer)
{
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_bellmm_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           rocsparse_order           order_B,
                                           rocsparse_order           order_C,
                                           rocsparse_direction       dir_A,
                                           I                         mb,
                                           I                         n,
                                           I                         kb,
                                           I                         bell_cols,
                                           I                         block_dim,
                                           I                         batch_count_A,
                                           I                         batch_stride_A,
                                           const T*                  alpha,
                                           const rocsparse_mat_descr descr,
                                           const I*                  bell_col_ind,
                                           const T*                  bell_val,
                                           const T*                  B,
                                           I                         ldb,
                                           I                         batch_count_B,
                                           I                         batch_stride_B,
                                           const T*                  beta,
                                           T*                        C,
                                           I                         ldc,
                                           I                         batch_count_C,
                                           I                         batch_stride_C,
                                           void*                     temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging TODO bench logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbellmm"),
              trans_A,
              trans_B,
              order_B,
              order_C,
              dir_A,
              mb,
              n,
              kb,
              bell_cols,
              block_dim,
              batch_count_A,
              batch_stride_A,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr,
              (const void*&)bell_col_ind,
              (const void*&)bell_val,
              (const void*&)B,
              ldb,
              batch_count_B,
              batch_stride_B,
              LOG_TRACE_SCALAR_VALUE(handle, beta),
              (const void*&)C,
              ldc,
              batch_count_C,
              batch_stride_C);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(order_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(order_C))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(dir_A))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0 || n < 0 || kb < 0 || bell_cols < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || n == 0 || kb == 0 || bell_cols == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bell_val == nullptr || bell_col_ind == nullptr || B == nullptr || C == nullptr
       || alpha == nullptr || beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }

    // Check leading dimension of B
    if((trans_B == rocsparse_operation_none && order_B == rocsparse_order_column)
       || (trans_B != rocsparse_operation_none && order_B != rocsparse_order_column))
    {
        if(ldb < kb * block_dim)
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(ldb < n)
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check leading dimension of C
    if(ldc < mb * block_dim && order_C == rocsparse_order_column)
    {
        return rocsparse_status_invalid_size;
    }
    else if(ldc < n && order_C == rocsparse_order_row)
    {
        return rocsparse_status_invalid_size;
    }

    // Check batch parameters of matrices
    if(batch_count_A != 1 || batch_count_B != 1 || batch_count_C != 1)
    {
        return rocsparse_status_invalid_value;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_bellmm_template_dispatch(handle,
                                                  trans_A,
                                                  trans_B,
                                                  order_B,
                                                  order_C,
                                                  dir_A,
                                                  mb,
                                                  n,
                                                  kb,
                                                  bell_cols,
                                                  block_dim,
                                                  alpha,
                                                  descr,
                                                  bell_col_ind,
                                                  bell_val,
                                                  B,
                                                  ldb,
                                                  beta,
                                                  C,
                                                  ldc);
    }
    else
    {
        return rocsparse_bellmm_template_dispatch(handle,
                                                  trans_A,
                                                  trans_B,
                                                  order_B,
                                                  order_C,
                                                  dir_A,
                                                  mb,
                                                  n,
                                                  kb,
                                                  bell_cols,
                                                  block_dim,
                                                  *alpha,
                                                  descr,
                                                  bell_col_ind,
                                                  bell_val,
                                                  B,
                                                  ldb,
                                                  *beta,
                                                  C,
                                                  ldc);
    }
}

#define INSTANTIATE(TTYPE, ITYPE)                                                                 \
    template rocsparse_status rocsparse_bellmm_template_buffer_size(                              \
        rocsparse_handle          handle,                                                         \
        rocsparse_operation       trans_A,                                                        \
        rocsparse_operation       trans_B,                                                        \
        rocsparse_order           order_B,                                                        \
        rocsparse_order           order_C,                                                        \
        rocsparse_direction       dir_A,                                                          \
        ITYPE                     mb,                                                             \
        ITYPE                     n,                                                              \
        ITYPE                     kb,                                                             \
        ITYPE                     bell_cols,                                                      \
        ITYPE                     bell_block_dim,                                                 \
        const TTYPE*              alpha,                                                          \
        const rocsparse_mat_descr descr,                                                          \
        const ITYPE*              bell_col_ind,                                                   \
        const TTYPE*              bell_val,                                                       \
        const TTYPE*              B,                                                              \
        ITYPE                     ldb,                                                            \
        const TTYPE*              beta,                                                           \
        TTYPE*                    C,                                                              \
        ITYPE                     ldc,                                                            \
        size_t*                   buffer_size);                                                                     \
    template rocsparse_status rocsparse_bellmm_template_preprocess(                               \
        rocsparse_handle          handle,                                                         \
        rocsparse_operation       trans_A,                                                        \
        rocsparse_operation       trans_B,                                                        \
        rocsparse_order           order_B,                                                        \
        rocsparse_order           order_C,                                                        \
        rocsparse_direction       dir_A,                                                          \
        ITYPE                     mb,                                                             \
        ITYPE                     n,                                                              \
        ITYPE                     kb,                                                             \
        ITYPE                     bell_cols,                                                      \
        ITYPE                     bell_block_dim,                                                 \
        const TTYPE*              alpha,                                                          \
        const rocsparse_mat_descr descr,                                                          \
        const ITYPE*              bell_col_ind,                                                   \
        const TTYPE*              bell_val,                                                       \
        const TTYPE*              B,                                                              \
        ITYPE                     ldb,                                                            \
        const TTYPE*              beta,                                                           \
        TTYPE*                    C,                                                              \
        ITYPE                     ldc,                                                            \
        void*                     temp_buffer);                                                                       \
    template rocsparse_status rocsparse_bellmm_template(rocsparse_handle          handle,         \
                                                        rocsparse_operation       trans_A,        \
                                                        rocsparse_operation       trans_B,        \
                                                        rocsparse_order           order_B,        \
                                                        rocsparse_order           order_C,        \
                                                        rocsparse_direction       dir_A,          \
                                                        ITYPE                     mb,             \
                                                        ITYPE                     n,              \
                                                        ITYPE                     kb,             \
                                                        ITYPE                     bell_cols,      \
                                                        ITYPE                     bell_block_dim, \
                                                        ITYPE                     batch_count_A,  \
                                                        ITYPE                     batch_stride_A, \
                                                        const TTYPE*              alpha,          \
                                                        const rocsparse_mat_descr descr,          \
                                                        const ITYPE*              bell_col_ind,   \
                                                        const TTYPE*              bell_val,       \
                                                        const TTYPE*              B,              \
                                                        ITYPE                     ldb,            \
                                                        ITYPE                     batch_count_B,  \
                                                        ITYPE                     batch_stride_B, \
                                                        const TTYPE*              beta,           \
                                                        TTYPE*                    C,              \
                                                        ITYPE                     ldc,            \
                                                        ITYPE                     batch_count_C,  \
                                                        ITYPE                     batch_stride_C, \
                                                        void*                     temp_buffer)

INSTANTIATE(float, int32_t);
INSTANTIATE(double, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t);

INSTANTIATE(float, int64_t);
INSTANTIATE(double, int64_t);
INSTANTIATE(rocsparse_float_complex, int64_t);
INSTANTIATE(rocsparse_double_complex, int64_t);

#undef INSTANTIATE
