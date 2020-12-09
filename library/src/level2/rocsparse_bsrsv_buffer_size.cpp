/*! \file */
/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "rocsparse_bsrsv.hpp"
#include "templates.h"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

// bsrsv_buffer_size
#define C_IMPL(NAME, TYPE)                                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                             \
                                     rocsparse_direction       dir,                                \
                                     rocsparse_operation       trans,                              \
                                     rocsparse_int             mb,                                 \
                                     rocsparse_int             nnzb,                               \
                                     const rocsparse_mat_descr descr,                              \
                                     const TYPE*               bsr_val,                            \
                                     const rocsparse_int*      bsr_row_ptr,                        \
                                     const rocsparse_int*      bsr_col_ind,                        \
                                     rocsparse_int             bsr_dim,                            \
                                     rocsparse_mat_info        info,                               \
                                     size_t*                   buffer_size)                        \
    {                                                                                              \
        /* Check direction */                                                                      \
        if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)                    \
        {                                                                                          \
            return rocsparse_status_invalid_value;                                                 \
        }                                                                                          \
                                                                                                   \
        /* Check sizes that are not checked by csrsv */                                            \
        if(bsr_dim < 0)                                                                            \
        {                                                                                          \
            return rocsparse_status_invalid_size;                                                  \
        }                                                                                          \
                                                                                                   \
        rocsparse_status stat = rocsparse_csrsv_buffer_size(                                       \
            handle, trans, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, info, buffer_size); \
                                                                                                   \
        /* Need additional buffer when using transposed */                                         \
        if(trans == rocsparse_operation_transpose)                                                 \
        {                                                                                          \
            /* Remove additional CSR buffer */                                                     \
            *buffer_size -= sizeof(TYPE) * ((nnzb - 1) / 256 + 1) * 256;                           \
                                                                                                   \
            /* Add BSR buffer instead */                                                           \
            *buffer_size += sizeof(TYPE) * ((nnzb * bsr_dim * bsr_dim - 1) / 256 + 1) * 256;       \
        }                                                                                          \
                                                                                                   \
        return stat;                                                                               \
    }

C_IMPL(rocsparse_sbsrsv_buffer_size, float);
C_IMPL(rocsparse_dbsrsv_buffer_size, double);
C_IMPL(rocsparse_cbsrsv_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsv_buffer_size, rocsparse_double_complex);
#undef C_IMPL
