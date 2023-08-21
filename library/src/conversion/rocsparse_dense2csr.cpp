/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/conversion/rocsparse_dense2csr.h"
#include "rocsparse_dense2csx_impl.hpp"

#define INSTANTIATE(DIRA, ITYPE, JTYPE, TTYPE)                                     \
    template rocsparse_status rocsparse_dense2csx_impl<DIRA, ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                          \
        rocsparse_order           order,                                           \
        JTYPE                     m,                                               \
        JTYPE                     n,                                               \
        const rocsparse_mat_descr descr_A,                                         \
        const TTYPE*              A,                                               \
        ITYPE                     lda,                                             \
        const ITYPE*              nnz_per_row_column,                              \
        TTYPE*                    csx_val_A,                                       \
        ITYPE*                    csx_row_col_ptr_A,                               \
        JTYPE*                    csx_col_row_ind_A);

INSTANTIATE(rocsparse_direction_row, int32_t, int32_t, float);
INSTANTIATE(rocsparse_direction_row, int32_t, int32_t, double);
INSTANTIATE(rocsparse_direction_row, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(rocsparse_direction_row, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(rocsparse_direction_row, int64_t, int32_t, float);
INSTANTIATE(rocsparse_direction_row, int64_t, int32_t, double);
INSTANTIATE(rocsparse_direction_row, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(rocsparse_direction_row, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(rocsparse_direction_row, int64_t, int64_t, float);
INSTANTIATE(rocsparse_direction_row, int64_t, int64_t, double);
INSTANTIATE(rocsparse_direction_row, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(rocsparse_direction_row, int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

extern "C" {

//
// Check if the macro CAPI_IMPL already exists.
//
#ifdef CAPI_IMPL
#error macro CAPI_IMPL is already defined.
#endif

//
// Definition of the C-implementation.
//
#define CAPI_IMPL(name_, type_)                                                           \
    rocsparse_status name_(rocsparse_handle          handle,                              \
                           rocsparse_int             m,                                   \
                           rocsparse_int             n,                                   \
                           const rocsparse_mat_descr descr,                               \
                           const type_*              A,                                   \
                           rocsparse_int             ld,                                  \
                           const rocsparse_int*      nnz_per_rows,                        \
                           type_*                    csr_val,                             \
                           rocsparse_int*            csr_row_ptr,                         \
                           rocsparse_int*            csr_col_ind)                         \
    {                                                                                     \
        try                                                                               \
        {                                                                                 \
            RETURN_IF_ROCSPARSE_ERROR(                                                    \
                rocsparse_dense2csx_impl<rocsparse_direction_row>(handle,                 \
                                                                  rocsparse_order_column, \
                                                                  m,                      \
                                                                  n,                      \
                                                                  descr,                  \
                                                                  A,                      \
                                                                  ld,                     \
                                                                  nnz_per_rows,           \
                                                                  csr_val,                \
                                                                  csr_row_ptr,            \
                                                                  csr_col_ind));          \
            return rocsparse_status_success;                                              \
        }                                                                                 \
        catch(...)                                                                        \
        {                                                                                 \
            RETURN_ROCSPARSE_EXCEPTION();                                                 \
        }                                                                                 \
    }

//
// C-implementations.
//
CAPI_IMPL(rocsparse_sdense2csr, float);
CAPI_IMPL(rocsparse_ddense2csr, double);
CAPI_IMPL(rocsparse_cdense2csr, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zdense2csr, rocsparse_double_complex);

//
// Undefine the macro CAPI_IMPL.
//
#undef CAPI_IMPL
}
