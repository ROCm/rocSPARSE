/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef COO2CSR_DEVICE_H
#define COO2CSR_DEVICE_H

#include <hip/hip_runtime.h>

// Compute lower bound by binary search
static inline __device__ rocsparse_int lower_bound(const rocsparse_int* arr,
                                                   rocsparse_int key,
                                                   rocsparse_int low,
                                                   rocsparse_int high)
{
    while(low < high)
    {
        rocsparse_int mid = low + ((high - low) >> 1);

        if(arr[mid] < key)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }

    return low;
}

// COO to CSR matrix conversion kernel
__global__ void coo2csr_kernel(rocsparse_int m,
                               rocsparse_int nnz,
                               const rocsparse_int* coo_row_ind,
                               rocsparse_int* csr_row_ptr,
                               rocsparse_index_base idx_base)
{
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= m)
    {
        return;
    }

    if(gid == 0)
    {
        csr_row_ptr[0] = idx_base;
        csr_row_ptr[m] = nnz + idx_base;
        return;
    }

    // Binary search
    csr_row_ptr[gid] = lower_bound(coo_row_ind, gid + idx_base, 0, nnz - 1) + idx_base;
}

#endif // COO2CSR_DEVICE_H
