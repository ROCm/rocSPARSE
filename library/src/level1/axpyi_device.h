/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef AXPYI_DEVICE_H
#define AXPYI_DEVICE_H

#include <hip/hip_runtime.h>

// y = a * x + y kernel for sparse x and dense y
template <typename T>
__device__ void axpyi_device(rocsparse_int nnz,
                             T alpha,
                             const T* x_val,
                             const rocsparse_int* x_ind,
                             T* y,
                             rocsparse_index_base idx_base)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid >= nnz)
    {
        return;
    }

    y[x_ind[tid] - idx_base] += alpha * x_val[tid];
}

#endif // AXPYI_DEVICE_H
