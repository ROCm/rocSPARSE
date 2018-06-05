/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef GTHRZ_DEVICE_H
#define GTHRZ_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T>
__global__ void gthrz_kernel(
    rocsparse_int nnz, T* y, T* x_val, const rocsparse_int* x_ind, rocsparse_index_base idx_base)
{
    rocsparse_int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(idx >= nnz)
    {
        return;
    }

    rocsparse_int i = x_ind[idx] - idx_base;

    x_val[idx] = y[i];
    y[i]       = static_cast<T>(0);
}

#endif // GTHRZ_DEVICE_H
