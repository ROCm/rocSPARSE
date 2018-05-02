/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"

#include <hip/hip_runtime.h>

template <typename T>
__device__
void axpyi_device(rocsparse_int nnz,
                  T alpha,
                  const T *xVal,
                  const rocsparse_int *xInd,
                  T *y,
                  rocsparse_index_base idxBase)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (tid >= nnz)
    {
        return;
    }

    y[xInd[tid]-idxBase] += alpha * xVal[tid];
}

template <typename T>
__global__
void axpyi_kernel_host_scalar(rocsparse_int nnz,
                              T alpha,
                              const T *xVal,
                              const rocsparse_int *xInd,
                              T *y,
                              rocsparse_index_base idxBase)
{
    axpyi_device<T>(nnz, alpha, xVal, xInd, y, idxBase);
}

template <typename T>
__global__
void axpyi_kernel_device_scalar(rocsparse_int nnz,
                                const T *alpha,
                                const T *xVal,
                                const rocsparse_int *xInd,
                                T *y,
                                rocsparse_index_base idxBase)
{
    axpyi_device<T>(nnz, *alpha, xVal, xInd, y, idxBase);
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
    xVal      pointer storing vector x non-zero values on the GPU.
    @param[in]
    xInd      pointer storing vector x non-zero value indices on the GPU.
    @param[inout]
    y         pointer storing y on the GPU.
    @param[in]
    idxBase   specifies the index base.

    ********************************************************************/
template <typename T>
rocsparse_status rocsparse_axpyi_template(rocsparse_handle handle,
                                          rocsparse_int nnz,
                                          const T *alpha,
                                          const T *xVal,
                                          const rocsparse_int *xInd,
                                          T *y,
                                          rocsparse_index_base idxBase)
{
    // Check for valid handle
    if (handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging // TODO bench logging
    if (handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_axpyi"),
                  nnz,
                  *alpha,
                  (const void*&) xVal,
                  (const void*&) xInd,
                  (const void*&) y);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_axpyi"),
                  nnz,
                  (const void*&) alpha,
                  (const void*&) xVal,
                  (const void*&) xInd,
                  (const void*&) y);
    }

    // Check index base
    if (idxBase != rocsparse_index_base_zero &&
        idxBase != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check size
    if (nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if (alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (xVal == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (xInd == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if (nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define AXPYI_DIM 256
    dim3 axpyi_blocks((nnz-1)/AXPYI_DIM+1);
    dim3 axpyi_threads(AXPYI_DIM);

    if (handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((axpyi_kernel_device_scalar<T>),
                           axpyi_blocks, axpyi_threads, 0, stream,
                           nnz, alpha, xVal, xInd, y, idxBase);
    }
    else
    {
        if (*alpha == 0.0)
        {
            return rocsparse_status_success;
        }

        hipLaunchKernelGGL((axpyi_kernel_host_scalar<T>),
                           axpyi_blocks, axpyi_threads, 0, stream,
                           nnz, *alpha, xVal, xInd, y, idxBase);
    }
#undef AXPYI_DIM
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C"
rocsparse_status rocsparse_saxpyi(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  const float *alpha,
                                  const float *xVal,
                                  const rocsparse_int *xInd,
                                  float *y,
                                  rocsparse_index_base idxBase)
{
    return rocsparse_axpyi_template<float>(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

extern "C"
rocsparse_status rocsparse_daxpyi(rocsparse_handle handle,
                                  rocsparse_int nnz,
                                  const double *alpha,
                                  const double *xVal,
                                  const rocsparse_int *xInd,
                                  double *y,
                                  rocsparse_index_base idxBase)
{
    return rocsparse_axpyi_template<double>(handle, nnz, alpha, xVal, xInd, y, idxBase);
}
