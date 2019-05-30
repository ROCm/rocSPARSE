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
#ifndef ROCSPARSE_ELLMV_HPP
#define ROCSPARSE_ELLMV_HPP

#include "definitions.h"
#include "ellmv_device.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>

template <typename T>
__global__ void ellmvn_kernel_host_pointer(rocsparse_int m,
                                           rocsparse_int n,
                                           rocsparse_int ell_width,
                                           T             alpha,
                                           const rocsparse_int* __restrict__ ell_col_ind,
                                           const T* __restrict__ ell_val,
                                           const T* __restrict__ x,
                                           T beta,
                                           T* __restrict__ y,
                                           rocsparse_index_base idx_base)
{
    ellmvn_device(m, n, ell_width, alpha, ell_col_ind, ell_val, x, beta, y, idx_base);
}

template <typename T>
__global__ void ellmvn_kernel_device_pointer(rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int ell_width,
                                             const T*      alpha,
                                             const rocsparse_int* __restrict__ ell_col_ind,
                                             const T* __restrict__ ell_val,
                                             const T* __restrict__ x,
                                             const T* beta,
                                             T* __restrict__ y,
                                             rocsparse_index_base idx_base)
{
    ellmvn_device(m, n, ell_width, *alpha, ell_col_ind, ell_val, x, *beta, y, idx_base);
}

template <typename T>
rocsparse_status rocsparse_ellmv_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  ell_val,
                                          const rocsparse_int*      ell_col_ind,
                                          rocsparse_int             ell_width,
                                          const T*                  x,
                                          const T*                  beta,
                                          T*                        y)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xellmv"),
                  trans,
                  m,
                  n,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)ell_val,
                  (const void*&)ell_col_ind,
                  ell_width,
                  (const void*&)x,
                  *beta,
                  (const void*&)y);

        log_bench(handle,
                  "./rocsparse-bench -f ellmv -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx> "
                  "--alpha",
                  *alpha,
                  "--beta",
                  *beta);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xellmv"),
                  trans,
                  m,
                  n,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)ell_val,
                  (const void*&)ell_col_ind,
                  ell_width,
                  (const void*&)x,
                  (const void*&)beta,
                  (const void*&)y);
    }

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(ell_width < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Sanity check
    if((m == 0 || n == 0) && ell_width != 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(ell_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(x == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different ellmv kernels
    if(trans == rocsparse_operation_none)
    {
#define ELLMVN_DIM 512
        dim3 ellmvn_blocks((m - 1) / ELLMVN_DIM + 1);
        dim3 ellmvn_threads(ELLMVN_DIM);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL((ellmvn_kernel_device_pointer<T>),
                               ellmvn_blocks,
                               ellmvn_threads,
                               0,
                               stream,
                               m,
                               n,
                               ell_width,
                               alpha,
                               ell_col_ind,
                               ell_val,
                               x,
                               beta,
                               y,
                               descr->base);
        }
        else
        {
            if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }

            hipLaunchKernelGGL((ellmvn_kernel_host_pointer<T>),
                               ellmvn_blocks,
                               ellmvn_threads,
                               0,
                               stream,
                               m,
                               n,
                               ell_width,
                               *alpha,
                               ell_col_ind,
                               ell_val,
                               x,
                               *beta,
                               y,
                               descr->base);
        }
#undef ELLMVN_DIM
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_ELLMV_HPP
