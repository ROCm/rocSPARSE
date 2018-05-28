/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
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
    axpyi_device(nnz, *alpha, x_val, x_ind, y, idx_base);
}

/*! \brief SPARSE Level 1 API

    \details
    axpyi  compute y := alpha * x + y

    @param[in]
    handle    rocsparse_handle.
              handle to the rocsparse library context queue.
    @param[in]
    nnz       number of non-zero entries in x
              if nnz <= 0 quick return with rocsparse_status_success
    @param[in]
    alpha     scalar alpha.
    @param[in]
    x_val     pointer storing vector x non-zero values on the GPU.
    @param[in]
    x_ind     pointer storing vector x non-zero value indices on the GPU.
    @param[inout]
    y         pointer storing y on the GPU.
    @param[in]
    idx_base  specifies the index base.

    ********************************************************************/
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

    // Logging // TODO bench logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xaxpyi"),
                  nnz,
                  *alpha,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  (const void*&)y);
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

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
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
