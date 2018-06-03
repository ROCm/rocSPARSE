/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
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

    // Logging // TODO bench logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xsctr"),
              nnz,
              (const void*&)x_val,
              (const void*&)x_ind,
              (const void*&)y,
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

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define SCTR_DIM 512
    dim3 sctr_blocks((nnz - 1) / SCTR_DIM + 1);
    dim3 sctr_threads(SCTR_DIM);

    hipLaunchKernelGGL((sctr_kernel<T>),
                       sctr_blocks,
                       sctr_threads,
                       0,
                       stream,
                       nnz,
                       x_val,
                       x_ind,
                       y,
                       idx_base);
#undef SCTR_DIM
    return rocsparse_status_success;
}

#endif // ROCSPARSE_SCTR_HPP
