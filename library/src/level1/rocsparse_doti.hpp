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
#ifndef ROCSPARSE_DOTI_HPP
#define ROCSPARSE_DOTI_HPP
#include "utility.h"

#include "definitions.h"
#include "doti_device.h"

template <typename I, typename T>
rocsparse_status rocsparse_doti_template(rocsparse_handle     handle,
                                         I                    nnz,
                                         const T*             x_val,
                                         const I*             x_ind,
                                         const T*             y,
                                         T*                   result,
                                         rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xdoti"),
              nnz,
              (const void*&)x_val,
              (const void*&)x_ind,
              (const void*&)y,
              LOG_TRACE_SCALAR_VALUE(handle, result),
              idx_base);

    log_bench(handle, "./rocsparse-bench -f doti -r", replaceX<T>("X"), "--mtx <vector.mtx> ");

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
    if(x_val == nullptr)
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
    else if(result == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define DOTI_DIM 256
    // Get workspace from handle device buffer
    T* workspace = reinterpret_cast<T*>(handle->buffer);

    hipLaunchKernelGGL((doti_kernel_part1<DOTI_DIM>),
                       dim3(DOTI_DIM),
                       dim3(DOTI_DIM),
                       0,
                       stream,
                       nnz,
                       x_val,
                       x_ind,
                       y,
                       workspace,
                       idx_base);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL(
            (doti_kernel_part2<DOTI_DIM>), dim3(1), dim3(DOTI_DIM), 0, stream, workspace, result);
    }
    else
    {
        hipLaunchKernelGGL((doti_kernel_part2<DOTI_DIM>),
                           dim3(1),
                           dim3(DOTI_DIM),
                           0,
                           stream,
                           workspace,
                           (T*)nullptr);

        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(result, workspace, sizeof(T), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    }
#undef DOTI_DIM

    return rocsparse_status_success;
}

#endif // ROCSPARSE_DOTI_HPP
