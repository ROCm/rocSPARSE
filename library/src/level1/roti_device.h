/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef ROTI_DEVICE_H
#define ROTI_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T>
__device__ void roti_device(rocsparse_int nnz,
                            T* x_val,
                            const rocsparse_int* x_ind,
                            T* y,
                            T c,
                            T s,
                            rocsparse_index_base idx_base)
{
    rocsparse_int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(idx >= nnz)
    {
        return;
    }

    rocsparse_int i = x_ind[idx] - idx_base;

    T xr = x_val[idx];
    T yr = y[i];

    x_val[idx] = c * xr + s * yr;
    y[i]       = c * yr - s * xr;
}

#endif // ROTI_DEVICE_H
