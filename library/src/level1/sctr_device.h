/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef SCTR_DEVICE_H
#define SCTR_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T>
__global__ void sctr_kernel(rocsparse_int nnz,
                            const T* x_val,
                            const rocsparse_int* x_ind,
                            T* y,
                            rocsparse_index_base idx_base)
{
    rocsparse_int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(idx >= nnz)
    {
        return;
    }

    y[x_ind[idx] - idx_base] = x_val[idx];
}

#endif // SCTR_DEVICE_H
