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

#include "definitions.h"
#include "rocsparse_csritsv.hpp"
#include "utility.h"

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csritsv_buffer_size_template(rocsparse_handle          handle,
                                                        rocsparse_operation       trans,
                                                        J                         m,
                                                        I                         nnz,
                                                        const rocsparse_mat_descr descr,
                                                        const T*                  csr_val,
                                                        const I*                  csr_row_ptr,
                                                        const J*                  csr_col_ind,
                                                        rocsparse_mat_info        info,
                                                        size_t*                   buffer_size)
{

    // Quick return if possible
    if(m == 0 || nnz == 0)
    {
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    if(rocsparse_diag_type_non_unit == descr->diag_type)
    {
        *buffer_size = sizeof(T) * m * 2 + sizeof(T) * 4;
    }
    else
    {
        *buffer_size = sizeof(T) * m + sizeof(T) * 4;
    }
    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csritsv_buffer_size_impl(rocsparse_handle          handle,
                                                    rocsparse_operation       trans,
                                                    J                         m,
                                                    I                         nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  csr_val,
                                                    const I*                  csr_row_ptr,
                                                    const J*                  csr_col_ind,
                                                    rocsparse_mat_info        info,
                                                    size_t*                   buffer_size)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsritsv_buffer_size"),
              trans,
              m,
              nnz,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              (const void*&)buffer_size);

    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general
       && descr->type != rocsparse_matrix_type_triangular)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid buffer_size pointer
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    // Check pointer arguments
    if(m > 0 && csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(nnz > 0 && csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(nnz > 0 && csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // need
    return rocsparse_csritsv_buffer_size_template(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                              \
    template rocsparse_status rocsparse_csritsv_buffer_size_template( \
        rocsparse_handle          handle,                             \
        rocsparse_operation       trans,                              \
        JTYPE                     m,                                  \
        ITYPE                     nnz,                                \
        const rocsparse_mat_descr descr,                              \
        const TTYPE*              csr_val,                            \
        const ITYPE*              csr_row_ptr,                        \
        const JTYPE*              csr_col_ind,                        \
        rocsparse_mat_info        info,                               \
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
#define C_IMPL(NAME, TYPE)                                                                       \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                           \
                                     rocsparse_operation       trans,                            \
                                     rocsparse_int             m,                                \
                                     rocsparse_int             nnz,                              \
                                     const rocsparse_mat_descr descr,                            \
                                     const TYPE*               csr_val,                          \
                                     const rocsparse_int*      csr_row_ptr,                      \
                                     const rocsparse_int*      csr_col_ind,                      \
                                     rocsparse_mat_info        info,                             \
                                     size_t*                   buffer_size)                      \
    try                                                                                          \
    {                                                                                            \
        return rocsparse_csritsv_buffer_size_impl(                                               \
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size); \
    }                                                                                            \
    catch(...)                                                                                   \
    {                                                                                            \
        return exception_to_rocsparse_status();                                                  \
    }

C_IMPL(rocsparse_scsritsv_buffer_size, float);
C_IMPL(rocsparse_dcsritsv_buffer_size, double);
C_IMPL(rocsparse_ccsritsv_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcsritsv_buffer_size, rocsparse_double_complex);

#undef C_IMPL
