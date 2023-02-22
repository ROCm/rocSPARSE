/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../conversion/rocsparse_identity.hpp"
#include "definitions.h"
#include "rocsparse_csrgemm.hpp"
#include "utility.h"

#include <rocprim/rocprim.hpp>

template <typename I, typename J, typename T>
rocsparse_status rocsparse_bsrgemm_buffer_size_template(rocsparse_handle          handle,
                                                        rocsparse_direction       dir,
                                                        rocsparse_operation       trans_A,
                                                        rocsparse_operation       trans_B,
                                                        J                         mb,
                                                        J                         nb,
                                                        J                         kb,
                                                        J                         block_dim,
                                                        const T*                  alpha,
                                                        const rocsparse_mat_descr descr_A,
                                                        I                         nnzb_A,
                                                        const I*                  bsr_row_ptr_A,
                                                        const J*                  bsr_col_ind_A,
                                                        const rocsparse_mat_descr descr_B,
                                                        I                         nnzb_B,
                                                        const I*                  bsr_row_ptr_B,
                                                        const J*                  bsr_col_ind_B,
                                                        const T*                  beta,
                                                        const rocsparse_mat_descr descr_D,
                                                        I                         nnzb_D,
                                                        const I*                  bsr_row_ptr_D,
                                                        const J*                  bsr_col_ind_D,
                                                        rocsparse_mat_info        info_C,
                                                        size_t*                   buffer_size)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check for valid rocsparse_mat_info
    if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrgemm_buffer_size"),
              dir,
              trans_A,
              trans_B,
              mb,
              nb,
              kb,
              block_dim,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr_A,
              nnzb_A,
              (const void*&)bsr_row_ptr_A,
              (const void*&)bsr_col_ind_A,
              (const void*&)descr_B,
              nnzb_B,
              (const void*&)bsr_row_ptr_B,
              (const void*&)bsr_col_ind_B,
              LOG_TRACE_SCALAR_VALUE(handle, beta),
              (const void*&)descr_D,
              nnzb_D,
              (const void*&)bsr_row_ptr_D,
              (const void*&)bsr_col_ind_D,
              (const void*&)info_C,
              (const void*&)buffer_size);

    // Check direction
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    // Check operation
    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(mb < 0 || nb < 0 || kb < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrgemm_buffer_size_template(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     mb,
                                                                     nb,
                                                                     kb,
                                                                     alpha,
                                                                     descr_A,
                                                                     nnzb_A,
                                                                     bsr_row_ptr_A,
                                                                     bsr_col_ind_A,
                                                                     descr_B,
                                                                     nnzb_B,
                                                                     bsr_row_ptr_B,
                                                                     bsr_col_ind_B,
                                                                     beta,
                                                                     descr_D,
                                                                     nnzb_D,
                                                                     bsr_row_ptr_D,
                                                                     bsr_col_ind_D,
                                                                     info_C,
                                                                     buffer_size));

    *buffer_size += ((sizeof(I) * nnzb_A - 1) / 256 + 1) * 256;

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                   \
    template rocsparse_status rocsparse_bsrgemm_buffer_size_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                                  \
        rocsparse_direction       dir,                                                     \
        rocsparse_operation       trans_A,                                                 \
        rocsparse_operation       trans_B,                                                 \
        JTYPE                     mb,                                                      \
        JTYPE                     nb,                                                      \
        JTYPE                     kb,                                                      \
        JTYPE                     block_dim,                                               \
        const TTYPE*              alpha,                                                   \
        const rocsparse_mat_descr descr_A,                                                 \
        ITYPE                     nnzb_A,                                                  \
        const ITYPE*              bsr_row_ptr_A,                                           \
        const JTYPE*              bsr_col_ind_A,                                           \
        const rocsparse_mat_descr descr_B,                                                 \
        ITYPE                     nnzb_B,                                                  \
        const ITYPE*              bsr_row_ptr_B,                                           \
        const JTYPE*              bsr_col_ind_B,                                           \
        const TTYPE*              beta,                                                    \
        const rocsparse_mat_descr descr_D,                                                 \
        ITYPE                     nnzb_D,                                                  \
        const ITYPE*              bsr_row_ptr_D,                                           \
        const JTYPE*              bsr_col_ind_D,                                           \
        rocsparse_mat_info        info_C,                                                  \
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

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

//
// rocsparse_xbsrgemm_buffer_size
//
#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_direction       dir,           \
                                     rocsparse_operation       trans_A,       \
                                     rocsparse_operation       trans_B,       \
                                     rocsparse_int             mb,            \
                                     rocsparse_int             nb,            \
                                     rocsparse_int             kb,            \
                                     rocsparse_int             block_dim,     \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr_A,       \
                                     rocsparse_int             nnzb_A,        \
                                     const rocsparse_int*      bsr_row_ptr_A, \
                                     const rocsparse_int*      bsr_col_ind_A, \
                                     const rocsparse_mat_descr descr_B,       \
                                     rocsparse_int             nnzb_B,        \
                                     const rocsparse_int*      bsr_row_ptr_B, \
                                     const rocsparse_int*      bsr_col_ind_B, \
                                     const TYPE*               beta,          \
                                     const rocsparse_mat_descr descr_D,       \
                                     rocsparse_int             nnzb_D,        \
                                     const rocsparse_int*      bsr_row_ptr_D, \
                                     const rocsparse_int*      bsr_col_ind_D, \
                                     rocsparse_mat_info        info_C,        \
                                     size_t*                   buffer_size)   \
    try                                                                       \
    {                                                                         \
        return rocsparse_bsrgemm_buffer_size_template(handle,                 \
                                                      dir,                    \
                                                      trans_A,                \
                                                      trans_B,                \
                                                      mb,                     \
                                                      nb,                     \
                                                      kb,                     \
                                                      block_dim,              \
                                                      alpha,                  \
                                                      descr_A,                \
                                                      nnzb_A,                 \
                                                      bsr_row_ptr_A,          \
                                                      bsr_col_ind_A,          \
                                                      descr_B,                \
                                                      nnzb_B,                 \
                                                      bsr_row_ptr_B,          \
                                                      bsr_col_ind_B,          \
                                                      beta,                   \
                                                      descr_D,                \
                                                      nnzb_D,                 \
                                                      bsr_row_ptr_D,          \
                                                      bsr_col_ind_D,          \
                                                      info_C,                 \
                                                      buffer_size);           \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        return exception_to_rocsparse_status();                               \
    }

C_IMPL(rocsparse_sbsrgemm_buffer_size, float);
C_IMPL(rocsparse_dbsrgemm_buffer_size, double);
C_IMPL(rocsparse_cbsrgemm_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrgemm_buffer_size, rocsparse_double_complex);

#undef C_IMPL
