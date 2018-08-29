/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef COOSORT_DEVICE_H
#define COOSORT_DEVICE_H

#include <hip/hip_runtime.h>

// COO to CSR matrix conversion kernel
__global__ void coosort_permute_kernel(rocsparse_int nnz,
                                       const rocsparse_int* in,
                                       const rocsparse_int* perm,
                                       rocsparse_int* out)
{
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= nnz)
    {
        return;
    }

    out[gid] = in[perm[gid]];
}

#endif // COOSORT_DEVICE_H
