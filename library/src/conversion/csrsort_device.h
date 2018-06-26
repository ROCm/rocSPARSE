/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef CSRSORT_DEVICE_H
#define CSRSORT_DEVICE_H

#include <hip/hip_runtime.h>

// Shift CSR offsets
__global__ void
csrsort_shift_kernel(rocsparse_int size, const rocsparse_int* in, rocsparse_int* out)
{
    int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= size)
    {
        return;
    }

    out[gid] = in[gid] - 1;
}

#endif // CSRSORT_DEVICE_H
