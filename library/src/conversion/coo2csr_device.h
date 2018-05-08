/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef COO2CSR_DEVICE_H
#define COO2CSR_DEVICE_H

#include <hip/hip_runtime.h>

__device__
rocsparse_int lower_bound(const rocsparse_int *arr,
                          rocsparse_int key,
                          rocsparse_int low,
                          rocsparse_int high)
{
    if (low > high)
    {
        return low;
    }

    rocsparse_int mid = low + ((high - low) >> 1);

    if (arr[mid] >= key)
    {
        high = mid - 1;
    }
    else
    {
        low = mid + 1;
    }
    return lower_bound(arr, key, low, high);
}

__global__
void coo2csr_kernel(rocsparse_int m,
                    rocsparse_int nnz,
                    const rocsparse_int *coo_row_ind,
                    rocsparse_int *csr_row_ptr)
{
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (gid >= m)
    {
        return;
    }

    if (gid == 0)
    {
        csr_row_ptr[0] = 0;
        csr_row_ptr[m] = nnz;
        return;
    }

    // Binary search
    csr_row_ptr[gid] = lower_bound(coo_row_ind, gid, 0, nnz-1);
}

#endif // COO2CSR_DEVICE_H
