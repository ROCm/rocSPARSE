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

#pragma once
#include "utility.h"

#include "roti_device.h"

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void roti_kernel(I                    nnz,
                 T*                   x_val,
                 const I*             x_ind,
                 T*                   y,
                 U                    c_device_host,
                 U                    s_device_host,
                 rocsparse_index_base idx_base)
{
    auto c = load_scalar_device_host(c_device_host);
    auto s = load_scalar_device_host(s_device_host);
    if(c == static_cast<T>(1) && s == static_cast<T>(0))
    {
        return;
    }
    roti_device<BLOCKSIZE>(nnz, x_val, x_ind, y, c, s, idx_base);
}

template <typename I, typename T>
rocsparse_status rocsparse_roti_template(rocsparse_handle     handle,
                                         I                    nnz,
                                         T*                   x_val,
                                         const I*             x_ind,
                                         T*                   y,
                                         const T*             c,
                                         const T*             s,
                                         rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging // TODO bench logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xroti"),
              nnz,
              (const void*&)x_val,
              (const void*&)x_ind,
              (const void*&)y,
              LOG_TRACE_SCALAR_VALUE(handle, c),
              LOG_TRACE_SCALAR_VALUE(handle, s),
              idx_base);

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
    if(c == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(s == nullptr)
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

#define ROTI_DIM 512
    dim3 roti_blocks((nnz - 1) / ROTI_DIM + 1);
    dim3 roti_threads(ROTI_DIM);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((roti_kernel<ROTI_DIM>),
                           roti_blocks,
                           roti_threads,
                           0,
                           stream,
                           nnz,
                           x_val,
                           x_ind,
                           y,
                           c,
                           s,
                           idx_base);
    }
    else
    {
        if(*c == static_cast<T>(1) && *s == static_cast<T>(0))
        {
            return rocsparse_status_success;
        }

        hipLaunchKernelGGL((roti_kernel<ROTI_DIM>),
                           roti_blocks,
                           roti_threads,
                           0,
                           stream,
                           nnz,
                           x_val,
                           x_ind,
                           y,
                           *c,
                           *s,
                           idx_base);
    }
#undef ROTI_DIM
    return rocsparse_status_success;
}
