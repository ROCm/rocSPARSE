/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_GTHRZ_HPP
#define ROCSPARSE_GTHRZ_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "gthrz_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_gthrz_template(rocsparse_handle handle,
                                          rocsparse_int nnz,
                                          T* y,
                                          T* x_val,
                                          const rocsparse_int* x_ind,
                                          rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgthrz"),
              nnz,
              (const void*&)y,
              (const void*&)x_val,
              (const void*&)x_ind,
              idx_base);

    log_bench(handle, "./rocsparse-bench -f gthrz -r", replaceX<T>("X"), "--mtx <vector.mtx> ");

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

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define GTHRZ_DIM 512
    dim3 gthrz_blocks((nnz - 1) / GTHRZ_DIM + 1);
    dim3 gthrz_threads(GTHRZ_DIM);

    hipLaunchKernelGGL(
        (gthrz_kernel<T>), gthrz_blocks, gthrz_threads, 0, stream, nnz, y, x_val, x_ind, idx_base);
#undef GTHRZ_DIM
    return rocsparse_status_success;
}

#endif // ROCSPARSE_GTHRZ_HPP
