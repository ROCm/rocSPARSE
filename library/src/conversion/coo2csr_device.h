/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef COO2CSR_DEVICE_H
#define COO2CSR_DEVICE_H

#include <hip/hip_runtime.h>

// Compute lower bound by binary search
static inline __device__ rocsparse_int lower_bound(const rocsparse_int* arr,
                                                   rocsparse_int        key,
                                                   rocsparse_int        low,
                                                   rocsparse_int        high)
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
__global__ void coo2csr_kernel(rocsparse_int        m,
                               rocsparse_int        nnz,
                               const rocsparse_int* coo_row_ind,
                               rocsparse_int*       csr_row_ptr,
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
    csr_row_ptr[gid] = lower_bound(coo_row_ind, gid + idx_base, 0, nnz) + idx_base;
}

#endif // COO2CSR_DEVICE_H
