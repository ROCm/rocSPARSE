/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "control.h"
#include "internal/level2/rocsparse_csrsv.h"
#include "rocsparse_csrsv.hpp"
#include "utility.h"
#include <rocprim/rocprim.hpp>

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrsv_buffer_size_template(rocsparse_handle          handle,
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
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_POINTER(8, info);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsrsv_buffer_size"),
                         trans,
                         m,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)info,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_ENUM(1, trans);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        4, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check matrix sorting mode

    ROCSPARSE_CHECKARG(4,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);

    // Check for valid buffer_size pointer
    ROCSPARSE_CHECKARG_POINTER(9, buffer_size);

    // Quick return if possible
    if(m == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_col_ind);

    // Stream
    hipStream_t stream = handle->stream;

    // rocsparse_int max_nnz
    *buffer_size = 256;

    // rocsparse_int done_array[m]
    *buffer_size += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    // rocsparse_int workspace
    *buffer_size += ((sizeof(J) * m - 1) / 256 + 1) * 256;

    // rocsparse_int workspace2
    *buffer_size += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    size_t rocprim_size = 0;
    int*   ptr1         = reinterpret_cast<int*>(buffer_size);
    I*     ptr2         = reinterpret_cast<I*>(buffer_size);
    J*     ptr3         = reinterpret_cast<J*>(buffer_size);

    rocprim::double_buffer<int> dummy1(ptr1, ptr1);
    rocprim::double_buffer<I>   dummy2(ptr2, ptr2);
    rocprim::double_buffer<J>   dummy3(ptr3, ptr3);

    RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
        nullptr, rocprim_size, dummy1, dummy3, m, 0, rocsparse::clz(m), stream));

    // rocprim buffer
    *buffer_size += rocprim_size;

    // On transposed case, we might need more temporary storage for transposing
    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
    {
        size_t transpose_size;

        // Determine rocprim buffer size
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            nullptr, transpose_size, dummy3, dummy2, nnz, 0, rocsparse::clz(m), stream));

        // rocPRIM does not support in-place sorting, so we need an additional buffer
        transpose_size += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;
        transpose_size += ((rocsparse::max(sizeof(I), sizeof(T)) * nnz - 1) / 256 + 1) * 256;

        *buffer_size = rocsparse::max(*buffer_size, transpose_size);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                             \
    template rocsparse_status rocsparse::csrsv_buffer_size_template( \
        rocsparse_handle          handle,                            \
        rocsparse_operation       trans,                             \
        JTYPE                     m,                                 \
        ITYPE                     nnz,                               \
        const rocsparse_mat_descr descr,                             \
        const TTYPE*              csr_val,                           \
        const ITYPE*              csr_row_ptr,                       \
        const JTYPE*              csr_col_ind,                       \
        rocsparse_mat_info        info,                              \
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
#define C_IMPL(NAME, TYPE)                                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                            \
                                     rocsparse_operation       trans,                             \
                                     rocsparse_int             m,                                 \
                                     rocsparse_int             nnz,                               \
                                     const rocsparse_mat_descr descr,                             \
                                     const TYPE*               csr_val,                           \
                                     const rocsparse_int*      csr_row_ptr,                       \
                                     const rocsparse_int*      csr_col_ind,                       \
                                     rocsparse_mat_info        info,                              \
                                     size_t*                   buffer_size)                       \
    try                                                                                           \
    {                                                                                             \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_buffer_size_template(                          \
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size)); \
        return rocsparse_status_success;                                                          \
    }                                                                                             \
    catch(...)                                                                                    \
    {                                                                                             \
        RETURN_ROCSPARSE_EXCEPTION();                                                             \
    }

C_IMPL(rocsparse_scsrsv_buffer_size, float);
C_IMPL(rocsparse_dcsrsv_buffer_size, double);
C_IMPL(rocsparse_ccsrsv_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsv_buffer_size, rocsparse_double_complex);

#undef C_IMPL
