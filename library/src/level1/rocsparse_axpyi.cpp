/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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

#include "rocsparse_axpyi.hpp"
#include "axpyi_device.h"
#include "utility.h"

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) __global__ void axpyi_kernel(
    I nnz, U alpha_device_host, const T* x_val, const I* x_ind, T* y, rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != static_cast<T>(0))
    {
        axpyi_device<BLOCKSIZE>(nnz, alpha, x_val, x_ind, y, idx_base);
    }
}

template <typename I, typename T>
rocsparse_status rocsparse_axpyi_template(rocsparse_handle     handle,
                                          I                    nnz,
                                          const T*             alpha,
                                          const T*             x_val,
                                          const I*             x_ind,
                                          T*                   y,
                                          rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xaxpyi"),
                  nnz,
                  *alpha,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  (const void*&)y);

        log_bench(handle,
                  "./rocsparse-bench -f axpyi -r",
                  replaceX<T>("X"),
                  "--mtx <vector.mtx> ",
                  "--alpha",
                  *alpha);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xaxpyi"),
                  nnz,
                  (const void*&)alpha,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  (const void*&)y);
    }

    // Check index base
    if(idx_base != rocsparse_index_base_zero && idx_base != rocsparse_index_base_one)
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
    if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_host && *alpha == static_cast<T>(0))
    {
        return rocsparse_status_success;
    }

    if(x_val == nullptr || x_ind == nullptr || y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define AXPYI_DIM 256
    dim3 axpyi_blocks((nnz - 1) / AXPYI_DIM + 1);
    dim3 axpyi_threads(AXPYI_DIM);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((axpyi_kernel<AXPYI_DIM>),
                           axpyi_blocks,
                           axpyi_threads,
                           0,
                           stream,
                           nnz,
                           alpha,
                           x_val,
                           x_ind,
                           y,
                           idx_base);
    }
    else
    {
        if(*alpha == static_cast<T>(0))
        {
            return rocsparse_status_success;
        }

        hipLaunchKernelGGL((axpyi_kernel<AXPYI_DIM>),
                           axpyi_blocks,
                           axpyi_threads,
                           0,
                           stream,
                           nnz,
                           *alpha,
                           x_val,
                           x_ind,
                           y,
                           idx_base);
    }
#undef AXPYI_DIM
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                     \
    template rocsparse_status rocsparse_axpyi_template<ITYPE, TTYPE>( \
        rocsparse_handle     handle,                                  \
        ITYPE                nnz,                                     \
        const TTYPE*         alpha,                                   \
        const TTYPE*         x_val,                                   \
        const ITYPE*         x_ind,                                   \
        TTYPE*               y,                                       \
        rocsparse_index_base idx_base);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_saxpyi(rocsparse_handle     handle,
                                             rocsparse_int        nnz,
                                             const float*         alpha,
                                             const float*         x_val,
                                             const rocsparse_int* x_ind,
                                             float*               y,
                                             rocsparse_index_base idx_base)
{
    return rocsparse_axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

extern "C" rocsparse_status rocsparse_daxpyi(rocsparse_handle     handle,
                                             rocsparse_int        nnz,
                                             const double*        alpha,
                                             const double*        x_val,
                                             const rocsparse_int* x_ind,
                                             double*              y,
                                             rocsparse_index_base idx_base)
{
    return rocsparse_axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

extern "C" rocsparse_status rocsparse_caxpyi(rocsparse_handle               handle,
                                             rocsparse_int                  nnz,
                                             const rocsparse_float_complex* alpha,
                                             const rocsparse_float_complex* x_val,
                                             const rocsparse_int*           x_ind,
                                             rocsparse_float_complex*       y,
                                             rocsparse_index_base           idx_base)
{
    return rocsparse_axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

extern "C" rocsparse_status rocsparse_zaxpyi(rocsparse_handle                handle,
                                             rocsparse_int                   nnz,
                                             const rocsparse_double_complex* alpha,
                                             const rocsparse_double_complex* x_val,
                                             const rocsparse_int*            x_ind,
                                             rocsparse_double_complex*       y,
                                             rocsparse_index_base            idx_base)
{
    return rocsparse_axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}
