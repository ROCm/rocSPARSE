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

#include "rocsparse_csr2ell_strided_batched.hpp"
#include "control.h"
#include "internal/conversion/rocsparse_csr2ell.h"
#include "rocsparse_csr2ell.hpp"
#include "utility.h"

#include "csr2ell_device.h"

rocsparse_status rocsparse::csr2ell_strided_batched_quickreturn(rocsparse_handle handle,
                                                                int64_t          batch_count,
                                                                int64_t          m,
                                                                const rocsparse_mat_descr csr_descr,
                                                                const void*               csr_val,
                                                                int64_t     csr_val_stride,
                                                                const void* csr_row_ptr,
                                                                const void* csr_col_ind,
                                                                const rocsparse_mat_descr ell_descr,
                                                                int64_t                   ell_width,
                                                                void*                     ell_val,
                                                                int64_t ell_val_stride,
                                                                void*   ell_col_ind)
{
    if(m == 0 || ell_width == 0 || batch_count == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

namespace rocsparse
{
    static rocsparse_status
        csr2ell_strided_batched_checkarg(rocsparse_handle          handle, //0
                                         int64_t                   batch_count, // 1
                                         int64_t                   m, // 2
                                         const rocsparse_mat_descr csr_descr, // 3
                                         const void*               csr_val, // 4
                                         int64_t                   csr_val_stride, //  5
                                         const void*               csr_row_ptr, // 6
                                         const void*               csr_col_ind, // 7
                                         const rocsparse_mat_descr ell_descr, // 8
                                         int64_t                   ell_width, // 9
                                         void*                     ell_val, // 10
                                         int64_t                   ell_val_stride, //  11
                                         void*                     ell_col_ind) //  12
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(9, ell_width);
        ROCSPARSE_CHECKARG_SIZE(1, batch_count);

        const rocsparse_status status
            = rocsparse::csr2ell_strided_batched_quickreturn(handle,
                                                             batch_count,
                                                             m,
                                                             csr_descr,
                                                             csr_val,
                                                             csr_val_stride,
                                                             csr_row_ptr,
                                                             csr_col_ind,
                                                             ell_descr,
                                                             ell_width,
                                                             ell_val,
                                                             ell_val_stride,
                                                             ell_col_ind);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(3, csr_descr);
        ROCSPARSE_CHECKARG(3,
                           csr_descr,
                           (csr_descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3,
                           csr_descr,
                           (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_POINTER(8, ell_descr);
        ROCSPARSE_CHECKARG(8,
                           ell_descr,
                           (ell_descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(8,
                           ell_descr,
                           (ell_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_POINTER(4, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_POINTER(7, csr_col_ind);

        ROCSPARSE_CHECKARG_POINTER(10, ell_val);
        ROCSPARSE_CHECKARG_POINTER(12, ell_col_ind);

        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::csr2ell_strided_batched_core(rocsparse_handle          handle,
                                                         int64_t                   batch_count,
                                                         J                         m,
                                                         const rocsparse_mat_descr csr_descr,
                                                         const T*                  csr_val,
                                                         int64_t                   csr_val_stride,
                                                         const I*                  csr_row_ptr,
                                                         const J*                  csr_col_ind,
                                                         const rocsparse_mat_descr ell_descr,
                                                         J                         ell_width,
                                                         T*                        ell_val,
                                                         int64_t                   ell_val_stride,
                                                         J*                        ell_col_ind)
{
    // Stream
    hipStream_t stream = handle->stream;

#define CSR2ELL_STRIDED_BATCHED_DIM 512
    dim3 csr2ell_strided_batched_blocks((m - 1) / CSR2ELL_STRIDED_BATCHED_DIM + 1, batch_count);
    dim3 csr2ell_strided_batched_threads(CSR2ELL_STRIDED_BATCHED_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::csr2ell_strided_batched_kernel<CSR2ELL_STRIDED_BATCHED_DIM>),
        csr2ell_strided_batched_blocks,
        csr2ell_strided_batched_threads,
        0,
        stream,
        m,
        csr_val,
        csr_val_stride,
        csr_row_ptr,
        csr_col_ind,
        csr_descr->base,
        ell_width,
        ell_col_ind,
        ell_val,
        ell_val_stride,
        ell_descr->base);
#undef CSR2ELL_STRIDED_BATCHED_DIM
    return rocsparse_status_success;
}

#define INSTANTIATE(T, I, J)                                                    \
    template rocsparse_status rocsparse::csr2ell_strided_batched_core<T, I, J>( \
        rocsparse_handle          handle,                                       \
        int64_t                   batch_count,                                  \
        J                         m,                                            \
        const rocsparse_mat_descr csr_descr,                                    \
        const T*                  csr_val,                                      \
        int64_t                   csr_val_stride,                               \
        const I*                  csr_row_ptr,                                  \
        const J*                  csr_col_ind,                                  \
        const rocsparse_mat_descr ell_descr,                                    \
        J                         ell_width,                                    \
        T*                        ell_val,                                      \
        int64_t                   ell_val_stride,                               \
        J*                        ell_col_ind)

INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t);
INSTANTIATE(int32_t, int32_t, int64_t);
INSTANTIATE(int32_t, int64_t, int64_t);

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(float, int32_t, int64_t);
INSTANTIATE(float, int64_t, int64_t);

INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(double, int32_t, int64_t);
INSTANTIATE(double, int64_t, int64_t);

INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);

INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);

namespace rocsparse
{
    template <typename... P>
    static rocsparse_status csr2ell_strided_batched_impl(P&&... p)
    {
        log_trace("rocsparse_Xcsr2ell_strided_batched", p...);
        const rocsparse_status status = rocsparse::csr2ell_strided_batched_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_strided_batched_core(p...));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsr2ell_strided_batched(rocsparse_handle handle,
                                                               rocsparse_int    batch_count,
                                                               rocsparse_int    m,
                                                               const rocsparse_mat_descr csr_descr,
                                                               const float*              csr_val,
                                                               rocsparse_int        csr_val_stride,
                                                               const rocsparse_int* csr_row_ptr,
                                                               const rocsparse_int* csr_col_ind,
                                                               const rocsparse_mat_descr ell_descr,
                                                               rocsparse_int             ell_width,
                                                               float*                    ell_val,
                                                               rocsparse_int  ell_val_stride,
                                                               rocsparse_int* ell_col_ind)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_strided_batched_impl(handle,
                                                                      batch_count,
                                                                      m,
                                                                      csr_descr,
                                                                      csr_val,
                                                                      csr_val_stride,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      ell_descr,
                                                                      ell_width,
                                                                      ell_val,
                                                                      ell_val_stride,
                                                                      ell_col_ind));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_dcsr2ell_strided_batched(rocsparse_handle handle,
                                                               rocsparse_int    batch_count,
                                                               rocsparse_int    m,
                                                               const rocsparse_mat_descr csr_descr,
                                                               const double*             csr_val,
                                                               rocsparse_int        csr_val_stride,
                                                               const rocsparse_int* csr_row_ptr,
                                                               const rocsparse_int* csr_col_ind,
                                                               const rocsparse_mat_descr ell_descr,
                                                               rocsparse_int             ell_width,
                                                               double*                   ell_val,
                                                               rocsparse_int  ell_val_stride,
                                                               rocsparse_int* ell_col_ind)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_strided_batched_impl(handle,
                                                                      batch_count,
                                                                      m,
                                                                      csr_descr,
                                                                      csr_val,
                                                                      csr_val_stride,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      ell_descr,
                                                                      ell_width,
                                                                      ell_val,
                                                                      ell_val_stride,
                                                                      ell_col_ind));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status
    rocsparse_ccsr2ell_strided_batched(rocsparse_handle               handle,
                                       rocsparse_int                  batch_count,
                                       rocsparse_int                  m,
                                       const rocsparse_mat_descr      csr_descr,
                                       const rocsparse_float_complex* csr_val,
                                       rocsparse_int                  csr_val_stride,
                                       const rocsparse_int*           csr_row_ptr,
                                       const rocsparse_int*           csr_col_ind,
                                       const rocsparse_mat_descr      ell_descr,
                                       rocsparse_int                  ell_width,
                                       rocsparse_float_complex*       ell_val,
                                       rocsparse_int                  ell_val_stride,
                                       rocsparse_int*                 ell_col_ind)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_strided_batched_impl(handle,
                                                                      batch_count,
                                                                      m,
                                                                      csr_descr,
                                                                      csr_val,
                                                                      csr_val_stride,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      ell_descr,
                                                                      ell_width,
                                                                      ell_val,
                                                                      ell_val_stride,
                                                                      ell_col_ind));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status
    rocsparse_zcsr2ell_strided_batched(rocsparse_handle                handle,
                                       rocsparse_int                   batch_count,
                                       rocsparse_int                   m,
                                       const rocsparse_mat_descr       csr_descr,
                                       const rocsparse_double_complex* csr_val,
                                       rocsparse_int                   csr_val_stride,
                                       const rocsparse_int*            csr_row_ptr,
                                       const rocsparse_int*            csr_col_ind,
                                       const rocsparse_mat_descr       ell_descr,
                                       rocsparse_int                   ell_width,
                                       rocsparse_double_complex*       ell_val,
                                       rocsparse_int                   ell_val_stride,
                                       rocsparse_int*                  ell_col_ind)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_strided_batched_impl(handle,
                                                                      batch_count,
                                                                      m,
                                                                      csr_descr,
                                                                      csr_val,
                                                                      csr_val_stride,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      ell_descr,
                                                                      ell_width,
                                                                      ell_val,
                                                                      ell_val_stride,
                                                                      ell_col_ind));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
