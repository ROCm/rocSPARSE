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
#ifndef ROCSPARSE_SCTR_HPP
#define ROCSPARSE_SCTR_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "sctr_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_sctr_template(rocsparse_handle handle,
                                         rocsparse_int nnz,
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
    log_trace(handle,
              replaceX<T>("rocsparse_Xsctr"),
              nnz,
              (const void*&)x_val,
              (const void*&)x_ind,
              (const void*&)y,
              idx_base);

    log_bench(handle, "./rocsparse-bench -f sctr -r", replaceX<T>("X"), "--mtx <vector.mtx> ");

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

    // Stream
    hipStream_t stream = handle->stream;

#define SCTR_DIM 512
    dim3 sctr_blocks((nnz - 1) / SCTR_DIM + 1);
    dim3 sctr_threads(SCTR_DIM);

    hipLaunchKernelGGL(
        (sctr_kernel<T>), sctr_blocks, sctr_threads, 0, stream, nnz, x_val, x_ind, y, idx_base);
#undef SCTR_DIM
    return rocsparse_status_success;
}

#endif // ROCSPARSE_SCTR_HPP
