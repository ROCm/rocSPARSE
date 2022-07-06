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
#include "rocsparse_check_matrix_csc.hpp"
#include "definitions.h"
#include "rocsparse_check_matrix_csr.hpp"
#include "utility.h"

template <typename T, typename I, typename J>
rocsparse_status rocsparse_check_matrix_csc_buffer_size_template(rocsparse_handle       handle,
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
    return rocsparse_check_matrix_csr_buffer_size_template(handle,
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
                                                           buffer_size);
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_check_matrix_csc_template(rocsparse_handle       handle,
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
                                                     rocsparse_data_status* data_status,
                                                     void*                  temp_buffer)
{
    return rocsparse_check_matrix_csr_template(handle,
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
                                               data_status,
                                               temp_buffer);
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                      \
    template rocsparse_status                                                 \
        rocsparse_check_matrix_csc_buffer_size_template<TTYPE, ITYPE, JTYPE>( \
            rocsparse_handle       handle,                                    \
            JTYPE                  m,                                         \
            JTYPE                  n,                                         \
            ITYPE                  nnz,                                       \
            const TTYPE*           csc_val,                                   \
            const ITYPE*           csc_col_ptr,                               \
            const JTYPE*           csc_row_ind,                               \
            rocsparse_index_base   idx_base,                                  \
            rocsparse_matrix_type  matrix_type,                               \
            rocsparse_fill_mode    uplo,                                      \
            rocsparse_storage_mode storage,                                   \
            size_t*                buffer_size);

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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                \
    template rocsparse_status rocsparse_check_matrix_csc_template<TTYPE, ITYPE, JTYPE>( \
        rocsparse_handle       handle,                                                  \
        JTYPE                  m,                                                       \
        JTYPE                  n,                                                       \
        ITYPE                  nnz,                                                     \
        const TTYPE*           csc_val,                                                 \
        const ITYPE*           csc_col_ptr,                                             \
        const JTYPE*           csc_row_ind,                                             \
        rocsparse_index_base   idx_base,                                                \
        rocsparse_matrix_type  matrix_type,                                             \
        rocsparse_fill_mode    uplo,                                                    \
        rocsparse_storage_mode storage,                                                 \
        rocsparse_data_status* data_status,                                             \
        void*                  temp_buffer);

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
#define C_IMPL(NAME, TYPE)                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,          \
                                     rocsparse_int          m,               \
                                     rocsparse_int          n,               \
                                     rocsparse_int          nnz,             \
                                     const TYPE*            csc_val,         \
                                     const rocsparse_int*   csc_col_ptr,     \
                                     const rocsparse_int*   csc_row_ind,     \
                                     rocsparse_index_base   idx_base,        \
                                     rocsparse_matrix_type  matrix_type,     \
                                     rocsparse_fill_mode    uplo,            \
                                     rocsparse_storage_mode storage,         \
                                     size_t*                buffer_size)     \
    {                                                                        \
        return rocsparse_check_matrix_csc_buffer_size_template(handle,       \
                                                               m,            \
                                                               n,            \
                                                               nnz,          \
                                                               csc_val,      \
                                                               csc_col_ptr,  \
                                                               csc_row_ind,  \
                                                               idx_base,     \
                                                               matrix_type,  \
                                                               uplo,         \
                                                               storage,      \
                                                               buffer_size); \
    }

C_IMPL(rocsparse_scheck_matrix_csc_buffer_size, float);
C_IMPL(rocsparse_dcheck_matrix_csc_buffer_size, double);
C_IMPL(rocsparse_ccheck_matrix_csc_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_csc_buffer_size, rocsparse_double_complex);
#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                               \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,      \
                                     rocsparse_int          m,           \
                                     rocsparse_int          n,           \
                                     rocsparse_int          nnz,         \
                                     const TYPE*            csc_val,     \
                                     const rocsparse_int*   csc_col_ptr, \
                                     const rocsparse_int*   csc_row_ind, \
                                     rocsparse_index_base   idx_base,    \
                                     rocsparse_matrix_type  matrix_type, \
                                     rocsparse_fill_mode    uplo,        \
                                     rocsparse_storage_mode storage,     \
                                     rocsparse_data_status* data_status, \
                                     void*                  temp_buffer) \
    {                                                                    \
        return rocsparse_check_matrix_csc_template(handle,               \
                                                   m,                    \
                                                   n,                    \
                                                   nnz,                  \
                                                   csc_val,              \
                                                   csc_col_ptr,          \
                                                   csc_row_ind,          \
                                                   idx_base,             \
                                                   matrix_type,          \
                                                   uplo,                 \
                                                   storage,              \
                                                   data_status,          \
                                                   temp_buffer);         \
    }

C_IMPL(rocsparse_scheck_matrix_csc, float);
C_IMPL(rocsparse_dcheck_matrix_csc, double);
C_IMPL(rocsparse_ccheck_matrix_csc, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_csc, rocsparse_double_complex);
#undef C_IMPL
