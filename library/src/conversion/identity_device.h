/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef IDENTITY_DEVICE_H
#define IDENTITY_DEVICE_H

#include <hip/hip_runtime.h>

// Create identity permutation
__global__ void identity_kernel(rocsparse_int n, rocsparse_int* p)
{
    rocsparse_int gid  = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= n)
    {
        return;
    }

    p[gid] = gid;
}

#endif // IDENTITY_DEVICE_H
