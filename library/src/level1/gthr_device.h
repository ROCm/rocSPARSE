/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef GTHR_DEVICE_H
#define GTHR_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T>
__global__ void gthr_kernel(rocsparse_int nnz,
                            const T* y,
                            T* x_val,
                            const rocsparse_int* x_ind,
                            rocsparse_index_base idx_base)
{
    rocsparse_int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(idx >= nnz)
    {
        return;
    }

    x_val[idx] = y[x_ind[idx] - idx_base];
}

#endif // GTHR_DEVICE_H
