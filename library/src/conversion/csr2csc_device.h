/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef CSR2CSC_DEVICE_H
#define CSR2CSC_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T>
__global__ void csr2csc_permute_kernel(rocsparse_int nnz,
                                       const rocsparse_int* in1,
                                       const T* in2,
                                       const rocsparse_int* map,
                                       rocsparse_int* out1,
                                       T* out2)
{
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= nnz)
    {
        return;
    }

    out1[gid] = in1[map[gid]];
    out2[gid] = in2[map[gid]];
}

#endif // CSR2CSC_DEVICE_H
