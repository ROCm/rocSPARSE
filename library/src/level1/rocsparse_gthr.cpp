/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_gthr.hpp"

#include "utility.h"

#include "gthr_device.h"
template <typename I, typename T>
rocsparse_status rocsparse_gthr_template(rocsparse_handle     handle,
                                         I                    nnz,
                                         const T*             y,
                                         T*                   x_val,
                                         const I*             x_ind,
                                         rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgthr"),
              nnz,
              (const void*&)y,
              (const void*&)x_val,
              (const void*&)x_ind,
              idx_base);

    log_bench(handle, "./rocsparse-bench -f gthr -r", replaceX<T>("X"), "--mtx <vector.mtx> ");

    // Check index base
    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    // Check size
    if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(x_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(x_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define GTHR_DIM 512
    dim3 gthr_blocks((nnz - 1) / GTHR_DIM + 1);
    dim3 gthr_threads(GTHR_DIM);

    hipLaunchKernelGGL((gthr_kernel<GTHR_DIM>),
                       gthr_blocks,
                       gthr_threads,
                       0,
                       stream,
                       nnz,
                       y,
                       x_val,
                       x_ind,
                       idx_base);
#undef GTHR_DIM
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                    \
    template rocsparse_status rocsparse_gthr_template<ITYPE, TTYPE>( \
        rocsparse_handle     handle,                                 \
        ITYPE                nnz,                                    \
        const TTYPE*         y,                                      \
        TTYPE*               x_val,                                  \
        const ITYPE*         x_ind,                                  \
        rocsparse_index_base idx_base);

INSTANTIATE(int32_t, int32_t)
INSTANTIATE(int64_t, int32_t)
INSTANTIATE(int64_t, int64_t)
INSTANTIATE(int32_t, float)
INSTANTIATE(int32_t, double)
INSTANTIATE(int32_t, rocsparse_float_complex)
INSTANTIATE(int32_t, rocsparse_double_complex)
INSTANTIATE(int64_t, float)
INSTANTIATE(int64_t, double)
INSTANTIATE(int64_t, rocsparse_float_complex)
INSTANTIATE(int64_t, rocsparse_double_complex)
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                      \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle,               \
                                     rocsparse_int        nnz,                  \
                                     const TYPE*          y,                    \
                                     TYPE*                x_val,                \
                                     const rocsparse_int* x_ind,                \
                                     rocsparse_index_base idx_base)             \
    {                                                                           \
        return rocsparse_gthr_template(handle, nnz, y, x_val, x_ind, idx_base); \
    }

C_IMPL(rocsparse_sgthr, float);
C_IMPL(rocsparse_dgthr, double);
C_IMPL(rocsparse_cgthr, rocsparse_float_complex);
C_IMPL(rocsparse_zgthr, rocsparse_double_complex);
#undef C_IMPL
