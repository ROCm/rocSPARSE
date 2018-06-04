/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_GTHR_HPP
#define ROCSPARSE_GTHR_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "gthr_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_gthr_template(rocsparse_handle handle,
                                         rocsparse_int nnz,
                                         const T* y,
                                         T* x_val,
                                         const rocsparse_int* x_ind,
                                         rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging // TODO bench logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgthr"),
              nnz,
              (const void*&)y,
              (const void*&)x_val,
              (const void*&)x_ind,
              idx_base);

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

#define GTHR_DIM 512
    dim3 gthr_blocks((nnz - 1) / GTHR_DIM + 1);
    dim3 gthr_threads(GTHR_DIM);

    hipLaunchKernelGGL(
        (gthr_kernel<T>), gthr_blocks, gthr_threads, 0, stream, nnz, y, x_val, x_ind, idx_base);
#undef GTHR_DIM
    return rocsparse_status_success;
}

#endif // ROCSPARSE_GTHR_HPP
