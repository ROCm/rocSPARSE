/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_DOTI_HPP
#define ROCSPARSE_DOTI_HPP

#include "rocsparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"
#include "doti_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_doti_template(rocsparse_handle handle,
                                         rocsparse_int nnz,
                                         const T* x_val,
                                         const rocsparse_int* x_ind,
                                         const T* y,
                                         T* result,
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
                  replaceX<T>("rocsparse_Xdoti"),
                  nnz,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  (const void*&)y,
                  *result,
                  idx_base);

        log_bench(handle,
                  "./rocsparse-bench -f doti -r",
                  replaceX<T>("X"),
                  "--mtx <vector.mtx> ");
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xdoti"),
                  nnz,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  (const void*&)y,
                  (const void*&)result,
                  idx_base);
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

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define DOTI_DIM 512
    rocsparse_int nblocks = (nnz - 1) / DOTI_DIM + 1;

    // Allocate workspace
    T* workspace = NULL;
    RETURN_IF_HIP_ERROR(hipMalloc((void**)&workspace, sizeof(T) * nblocks));

    dim3 doti_blocks(nblocks);
    dim3 doti_threads(DOTI_DIM);

    hipLaunchKernelGGL((doti_kernel_part1<T, DOTI_DIM>),
                       doti_blocks,
                       doti_threads,
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
        hipLaunchKernelGGL((doti_kernel_part2<T, DOTI_DIM, 1>),
                           dim3(1),
                           doti_threads,
                           0,
                           stream,
                           nblocks,
                           workspace,
                           result);
    }
    else
    {
        if(nblocks > 1)
        {
            hipLaunchKernelGGL((doti_kernel_part2<T, DOTI_DIM, 0>),
                               dim3(1),
                               doti_threads,
                               0,
                               stream,
                               nblocks,
                               workspace,
                               result);
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(result, workspace, sizeof(T), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(hipFree(workspace));
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_DOTI_HPP
