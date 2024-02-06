/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/util/rocsparse_check_matrix_csc.h"

#include "rocsparse_check_matrix_csc.hpp"
#include "rocsparse_check_matrix_csr.hpp"
#include "to_string.hpp"
#include "utility.h"

template <typename T, typename I, typename J>
rocsparse_status rocsparse::check_matrix_csc_buffer_size_core(rocsparse_handle       handle,
                                                              J                      m,
                                                              J                      n,
                                                              I                      nnz,
                                                              const T*               csc_val,
                                                              const I*               csc_col_ptr,
                                                              const J*               csc_row_ind,
                                                              rocsparse_index_base   idx_base,
                                                              rocsparse_matrix_type  matrix_type,
                                                              rocsparse_fill_mode    uplo,
                                                              rocsparse_storage_mode storage,
                                                              size_t*                buffer_size)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::check_matrix_csr_buffer_size_core(handle,
                                                                           n,
                                                                           m,
                                                                           nnz,
                                                                           csc_val,
                                                                           csc_col_ptr,
                                                                           csc_row_ind,
                                                                           idx_base,
                                                                           matrix_type,
                                                                           uplo,
                                                                           storage,
                                                                           buffer_size));
    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T, typename I, typename J>
    static rocsparse_status
        check_matrix_csc_buffer_size_quickreturn(rocsparse_handle       handle,
                                                 J                      m,
                                                 J                      n,
                                                 I                      nnz,
                                                 const T*               csc_val,
                                                 const I*               csc_col_ptr,
                                                 const J*               csc_row_ind,
                                                 rocsparse_index_base   idx_base,
                                                 rocsparse_matrix_type  matrix_type,
                                                 rocsparse_fill_mode    uplo,
                                                 rocsparse_storage_mode storage,
                                                 size_t*                buffer_size)
    {
        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J>
rocsparse_status
    rocsparse::check_matrix_csc_buffer_size_checkarg(rocsparse_handle       handle, //0
                                                     J                      m, //1
                                                     J                      n, //2
                                                     I                      nnz, //3
                                                     const T*               csc_val, //4
                                                     const I*               csc_col_ptr, //5
                                                     const J*               csc_row_ind, //6
                                                     rocsparse_index_base   idx_base, //7
                                                     rocsparse_matrix_type  matrix_type, //8
                                                     rocsparse_fill_mode    uplo, //9
                                                     rocsparse_storage_mode storage, //10
                                                     size_t*                buffer_size) //11
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, csc_val);
    ROCSPARSE_CHECKARG_ARRAY(5, n, csc_col_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csc_row_ind);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);
    ROCSPARSE_CHECKARG_ENUM(8, matrix_type);
    ROCSPARSE_CHECKARG_ENUM(9, uplo);
    ROCSPARSE_CHECKARG_ENUM(10, storage);
    ROCSPARSE_CHECKARG_POINTER(11, buffer_size);

    if(matrix_type != rocsparse_matrix_type_general)
    {
        if(m != n)
        {
            rocsparse::log_debug(handle,
                                 ("Matrix was specified to be "
                                  + std::string(rocsparse::to_string(matrix_type))
                                  + " but m != n"));
        }
    }
    ROCSPARSE_CHECKARG(2,
                       n,
                       ((matrix_type != rocsparse_matrix_type_general) && (n != m)),
                       rocsparse_status_invalid_size);

    const rocsparse_status status
        = rocsparse::check_matrix_csc_buffer_size_quickreturn(handle,
                                                              m,
                                                              n,
                                                              nnz,
                                                              csc_val,
                                                              csc_col_ptr,
                                                              csc_row_ind,
                                                              idx_base,
                                                              matrix_type,
                                                              uplo,
                                                              storage,
                                                              buffer_size);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

#define INSTANTIATE(I, J, T)                                                             \
    template rocsparse_status rocsparse::check_matrix_csc_buffer_size_core<T, I, J>(     \
        rocsparse_handle       handle,                                                   \
        J                      m,                                                        \
        J                      n,                                                        \
        I                      nnz,                                                      \
        const T*               csc_val,                                                  \
        const I*               csc_col_ptr,                                              \
        const J*               csc_row_ind,                                              \
        rocsparse_index_base   idx_base,                                                 \
        rocsparse_matrix_type  matrix_type,                                              \
        rocsparse_fill_mode    uplo,                                                     \
        rocsparse_storage_mode storage,                                                  \
        size_t*                buffer_size);                                                            \
    template rocsparse_status rocsparse::check_matrix_csc_buffer_size_checkarg<T, I, J>( \
        rocsparse_handle       handle,                                                   \
        J                      m,                                                        \
        J                      n,                                                        \
        I                      nnz,                                                      \
        const T*               csc_val,                                                  \
        const I*               csc_col_ptr,                                              \
        const J*               csc_row_ind,                                              \
        rocsparse_index_base   idx_base,                                                 \
        rocsparse_matrix_type  matrix_type,                                              \
        rocsparse_fill_mode    uplo,                                                     \
        rocsparse_storage_mode storage,                                                  \
        size_t*                buffer_size)

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
#define C_IMPL(NAME, TYPE)                                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,                            \
                                     rocsparse_int          m,                                 \
                                     rocsparse_int          n,                                 \
                                     rocsparse_int          nnz,                               \
                                     const TYPE*            csc_val,                           \
                                     const rocsparse_int*   csc_col_ptr,                       \
                                     const rocsparse_int*   csc_row_ind,                       \
                                     rocsparse_index_base   idx_base,                          \
                                     rocsparse_matrix_type  matrix_type,                       \
                                     rocsparse_fill_mode    uplo,                              \
                                     rocsparse_storage_mode storage,                           \
                                     size_t*                buffer_size)                       \
    try                                                                                        \
    {                                                                                          \
        RETURN_IF_ROCSPARSE_ERROR(                                                             \
            (rocsparse::check_matrix_csc_buffer_size_impl<TYPE, rocsparse_int, rocsparse_int>( \
                handle,                                                                        \
                m,                                                                             \
                n,                                                                             \
                nnz,                                                                           \
                csc_val,                                                                       \
                csc_col_ptr,                                                                   \
                csc_row_ind,                                                                   \
                idx_base,                                                                      \
                matrix_type,                                                                   \
                uplo,                                                                          \
                storage,                                                                       \
                buffer_size)));                                                                \
        return rocsparse_status_success;                                                       \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        RETURN_ROCSPARSE_EXCEPTION();                                                          \
    }

C_IMPL(rocsparse_scheck_matrix_csc_buffer_size, float);
C_IMPL(rocsparse_dcheck_matrix_csc_buffer_size, double);
C_IMPL(rocsparse_ccheck_matrix_csc_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_csc_buffer_size, rocsparse_double_complex);
#undef C_IMPL
