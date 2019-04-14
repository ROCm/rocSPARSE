/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_AXPYI_HPP
#define ROCSPARSE_AXPYI_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "axpyi_device.h"

#include <hip/hip_runtime.h>

template <typename T>
__global__ void axpyi_kernel_host_scalar(rocsparse_int nnz,
                                         T alpha,
                                         const T* x_val,
                                         const rocsparse_int* x_ind,
                                         T* y,
                                         rocsparse_index_base idx_base)
{
    axpyi_device(nnz, alpha, x_val, x_ind, y, idx_base);
}

template <typename T>
__global__ void axpyi_kernel_device_scalar(rocsparse_int nnz,
                                           const T* alpha,
                                           const T* x_val,
                                           const rocsparse_int* x_ind,
                                           T* y,
                                           rocsparse_index_base idx_base)
{
    if(*alpha == static_cast<T>(0))
    {
        return;
    }

    axpyi_device(nnz, *alpha, x_val, x_ind, y, idx_base);
}

template <typename T>
rocsparse_status rocsparse_axpyi_template(rocsparse_handle handle,
                                          rocsparse_int nnz,
                                          const T* alpha,
                                          const T* x_val,
                                          const rocsparse_int* x_ind,
                                          T* y,
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
    else if(x_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(x_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
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
        hipLaunchKernelGGL((axpyi_kernel_device_scalar<T>),
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
        if(*alpha == 0.0)
        {
            return rocsparse_status_success;
        }

        hipLaunchKernelGGL((axpyi_kernel_host_scalar<T>),
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

#endif // ROCSPARSE_AXPYI_HPP
