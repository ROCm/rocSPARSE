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
#ifndef DOTI_DEVICE_H
#define DOTI_DEVICE_H

#include "common.h"

#include <hip/hip_runtime.h>

template <typename T, rocsparse_int NB>
__global__ void doti_kernel_part1(rocsparse_int        nnz,
                                  const T*             x_val,
                                  const rocsparse_int* x_ind,
                                  const T*             y,
                                  T*                   workspace,
                                  rocsparse_index_base idx_base)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int gid = hipBlockDim_x * hipBlockIdx_x + tid;

    __shared__ T sdata[NB];
    sdata[tid] = static_cast<T>(0);

    for(rocsparse_int idx = gid; idx < nnz; idx += hipGridDim_x * hipBlockDim_x)
    {
        sdata[tid] += y[x_ind[idx] - idx_base] * x_val[idx];
    }

    __syncthreads();

    rocsparse_blockreduce_sum<T, NB>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <typename T, rocsparse_int NB>
__global__ void doti_kernel_part2(rocsparse_int n, T* workspace, T* result)
{
    rocsparse_int tid = hipThreadIdx_x;

    __shared__ T sdata[NB];
    sdata[tid] = static_cast<T>(0);

    for(rocsparse_int i = tid; i < n; i += NB)
    {
        sdata[tid] += workspace[i];
    }

    __syncthreads();

    rocsparse_blockreduce_sum<T, NB>(tid, sdata);

    if(tid == 0)
    {
        if(result)
        {
            *result = sdata[0];
        }
        else
        {
            workspace[0] = sdata[0];
        }
    }
}

#endif // DOTI_DEVICE_H
