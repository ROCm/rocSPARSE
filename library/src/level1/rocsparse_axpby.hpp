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

#pragma once
#ifndef ROCSPARSE_AXPBY_HPP
#define ROCSPARSE_AXPBY_HPP

#include "axpby_device.h"
#include "definitions.h"
#include "utility.h"

template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) __global__ void axpby_kernel_host_scalar(
    I nnz, T alpha, const T* x_val, const I* x_ind, T beta, T* y, rocsparse_index_base idx_base)
{
    axpby_device<BLOCKSIZE>(nnz, alpha, x_val, x_ind, beta, y, idx_base);
}

template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void axpby_kernel_device_scalar(I                    nnz,
                                    const T*             alpha,
                                    const T*             x_val,
                                    const I*             x_ind,
                                    const T*             beta,
                                    T*                   y,
                                    rocsparse_index_base idx_base)
{
    if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
    {
        return;
    }

    axpby_device<BLOCKSIZE>(nnz, *alpha, x_val, x_ind, *beta, y, idx_base);
}

template <typename I, typename T>
rocsparse_status rocsparse_axpby_template(rocsparse_handle     handle,
                                          I                    nnz,
                                          const T*             alpha,
                                          const T*             x_val,
                                          const I*             x_ind,
                                          const T*             beta,
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
                  "rocsparse_axpby",
                  nnz,
                  *alpha,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  *beta,
                  (const void*&)y);

        log_bench(handle,
                  "./rocsparse-bench -f axpby -r",
                  replaceX<T>("X"),
                  "--mtx <vector.mtx> ",
                  "--alpha ",
                  *alpha);
    }
    else
    {
        log_trace(handle,
                  "rocsparse_axpby",
                  nnz,
                  (const void*&)alpha,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  (const void*&)beta,
                  (const void*&)y);
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
    RETURN_IF_NULLPTR(alpha);
    RETURN_IF_NULLPTR(beta);
    RETURN_IF_NULLPTR(x_val);
    RETURN_IF_NULLPTR(x_ind);
    RETURN_IF_NULLPTR(y);

#define AXPBY_DIM 256
    dim3 axpby_blocks((nnz - 1) / AXPBY_DIM + 1);
    dim3 axpby_threads(AXPBY_DIM);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((axpby_kernel_device_scalar<AXPBY_DIM>),
                           axpby_blocks,
                           axpby_threads,
                           0,
                           handle->stream,
                           nnz,
                           alpha,
                           x_val,
                           x_ind,
                           beta,
                           y,
                           idx_base);
    }
    else
    {
        if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
        {
            return rocsparse_status_success;
        }

        hipLaunchKernelGGL((axpby_kernel_host_scalar<AXPBY_DIM>),
                           axpby_blocks,
                           axpby_threads,
                           0,
                           handle->stream,
                           nnz,
                           *alpha,
                           x_val,
                           x_ind,
                           *beta,
                           y,
                           idx_base);
    }
#undef AXPBY_DIM

    return rocsparse_status_success;
}

#endif // ROCSPARSE_AXPBY_HPP
