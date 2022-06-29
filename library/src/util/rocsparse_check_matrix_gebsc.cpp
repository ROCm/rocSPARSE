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
#include "rocsparse_check_matrix_gebsc.hpp"
#include "definitions.h"
#include "rocsparse_check_matrix_gebsr.hpp"
#include "utility.h"

template <typename T, typename I, typename J>
rocsparse_status
    rocsparse_check_matrix_gebsc_buffer_size_template(rocsparse_handle       handle,
                                                      rocsparse_direction    dir,
                                                      J                      mb,
                                                      J                      nb,
                                                      I                      nnzb,
                                                      J                      row_block_dim,
                                                      J                      col_block_dim,
                                                      const T*               bsc_val,
                                                      const I*               bsc_col_ptr,
                                                      const J*               bsc_row_ind,
                                                      rocsparse_index_base   idx_base,
                                                      rocsparse_matrix_type  matrix_type,
                                                      rocsparse_fill_mode    uplo,
                                                      rocsparse_storage_mode storage,
                                                      size_t*                buffer_size)
{
    return rocsparse_check_matrix_gebsr_buffer_size_template(handle,
                                                             dir,
                                                             nb,
                                                             mb,
                                                             nnzb,
                                                             row_block_dim,
                                                             col_block_dim,
                                                             bsc_val,
                                                             bsc_col_ptr,
                                                             bsc_row_ind,
                                                             idx_base,
                                                             matrix_type,
                                                             uplo,
                                                             storage,
                                                             buffer_size);
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_check_matrix_gebsc_template(rocsparse_handle       handle,
                                                       rocsparse_direction    dir,
                                                       J                      mb,
                                                       J                      nb,
                                                       I                      nnzb,
                                                       J                      row_block_dim,
                                                       J                      col_block_dim,
                                                       const T*               bsc_val,
                                                       const I*               bsc_col_ptr,
                                                       const J*               bsc_row_ind,
                                                       rocsparse_index_base   idx_base,
                                                       rocsparse_matrix_type  matrix_type,
                                                       rocsparse_fill_mode    uplo,
                                                       rocsparse_storage_mode storage,
                                                       rocsparse_data_status* data_status,
                                                       void*                  temp_buffer)
{
    return rocsparse_check_matrix_gebsr_template(handle,
                                                 dir,
                                                 nb,
                                                 mb,
                                                 nnzb,
                                                 row_block_dim,
                                                 col_block_dim,
                                                 bsc_val,
                                                 bsc_col_ptr,
                                                 bsc_row_ind,
                                                 idx_base,
                                                 matrix_type,
                                                 uplo,
                                                 storage,
                                                 data_status,
                                                 temp_buffer);
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                        \
    template rocsparse_status                                                   \
        rocsparse_check_matrix_gebsc_buffer_size_template<TTYPE, ITYPE, JTYPE>( \
            rocsparse_handle       handle,                                      \
            rocsparse_direction    dir,                                         \
            JTYPE                  mb,                                          \
            JTYPE                  nb,                                          \
            ITYPE                  nnzb,                                        \
            JTYPE                  row_block_dim,                               \
            JTYPE                  col_block_dim,                               \
            const TTYPE*           bsc_val,                                     \
            const ITYPE*           bsc_col_ptr,                                 \
            const JTYPE*           bsc_row_ind,                                 \
            rocsparse_index_base   idx_base,                                    \
            rocsparse_matrix_type  matrix_type,                                 \
            rocsparse_fill_mode    uplo,                                        \
            rocsparse_storage_mode storage,                                     \
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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                  \
    template rocsparse_status rocsparse_check_matrix_gebsc_template<TTYPE, ITYPE, JTYPE>( \
        rocsparse_handle       handle,                                                    \
        rocsparse_direction    dir,                                                       \
        JTYPE                  mb,                                                        \
        JTYPE                  nb,                                                        \
        ITYPE                  nnzb,                                                      \
        JTYPE                  row_block_dim,                                             \
        JTYPE                  col_block_dim,                                             \
        const TTYPE*           bsc_val,                                                   \
        const ITYPE*           bsc_col_ptr,                                               \
        const JTYPE*           bsc_row_ind,                                               \
        rocsparse_index_base   idx_base,                                                  \
        rocsparse_matrix_type  matrix_type,                                               \
        rocsparse_fill_mode    uplo,                                                      \
        rocsparse_storage_mode storage,                                                   \
        rocsparse_data_status* data_status,                                               \
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
#define C_IMPL(NAME, TYPE)                                                      \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,             \
                                     rocsparse_direction    dir,                \
                                     rocsparse_int          mb,                 \
                                     rocsparse_int          nb,                 \
                                     rocsparse_int          nnzb,               \
                                     rocsparse_int          row_block_dim,      \
                                     rocsparse_int          col_block_dim,      \
                                     const TYPE*            bsc_val,            \
                                     const rocsparse_int*   bsc_col_ptr,        \
                                     const rocsparse_int*   bsc_row_ind,        \
                                     rocsparse_index_base   idx_base,           \
                                     rocsparse_matrix_type  matrix_type,        \
                                     rocsparse_fill_mode    uplo,               \
                                     rocsparse_storage_mode storage,            \
                                     size_t*                buffer_size)        \
    {                                                                           \
        return rocsparse_check_matrix_gebsc_buffer_size_template(handle,        \
                                                                 dir,           \
                                                                 mb,            \
                                                                 nb,            \
                                                                 nnzb,          \
                                                                 row_block_dim, \
                                                                 col_block_dim, \
                                                                 bsc_val,       \
                                                                 bsc_col_ptr,   \
                                                                 bsc_row_ind,   \
                                                                 idx_base,      \
                                                                 matrix_type,   \
                                                                 uplo,          \
                                                                 storage,       \
                                                                 buffer_size);  \
    }

C_IMPL(rocsparse_scheck_matrix_gebsc_buffer_size, float);
C_IMPL(rocsparse_dcheck_matrix_gebsc_buffer_size, double);
C_IMPL(rocsparse_ccheck_matrix_gebsc_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_gebsc_buffer_size, rocsparse_double_complex);
#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                 \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,        \
                                     rocsparse_direction    dir,           \
                                     rocsparse_int          mb,            \
                                     rocsparse_int          nb,            \
                                     rocsparse_int          nnzb,          \
                                     rocsparse_int          row_block_dim, \
                                     rocsparse_int          col_block_dim, \
                                     const TYPE*            bsc_val,       \
                                     const rocsparse_int*   bsc_col_ptr,   \
                                     const rocsparse_int*   bsc_row_ind,   \
                                     rocsparse_index_base   idx_base,      \
                                     rocsparse_matrix_type  matrix_type,   \
                                     rocsparse_fill_mode    uplo,          \
                                     rocsparse_storage_mode storage,       \
                                     rocsparse_data_status* data_status,   \
                                     void*                  temp_buffer)   \
    {                                                                      \
        return rocsparse_check_matrix_gebsc_template(handle,               \
                                                     dir,                  \
                                                     mb,                   \
                                                     nb,                   \
                                                     nnzb,                 \
                                                     row_block_dim,        \
                                                     col_block_dim,        \
                                                     bsc_val,              \
                                                     bsc_col_ptr,          \
                                                     bsc_row_ind,          \
                                                     idx_base,             \
                                                     matrix_type,          \
                                                     uplo,                 \
                                                     storage,              \
                                                     data_status,          \
                                                     temp_buffer);         \
    }

C_IMPL(rocsparse_scheck_matrix_gebsc, float);
C_IMPL(rocsparse_dcheck_matrix_gebsc, double);
C_IMPL(rocsparse_ccheck_matrix_gebsc, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_gebsc, rocsparse_double_complex);
#undef C_IMPL
