/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"

template <unsigned int BLOCKSIZE, typename I, typename T>
ROCSPARSE_KERNEL(BLOCKSIZE)
void doti_kernel_part1(
    I nnz, const T* x_val, const I* x_ind, const T* y, T* workspace, rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    I   gid = BLOCKSIZE * hipBlockIdx_x + tid;

    T dot = static_cast<T>(0);

    for(I idx = gid; idx < nnz; idx += hipGridDim_x * BLOCKSIZE)
    {
        dot = rocsparse_fma(y[x_ind[idx] - idx_base], x_val[idx], dot);
    }

    __shared__ T sdata[BLOCKSIZE];
    sdata[tid] = dot;

    __syncthreads();

    rocsparse_blockreduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE, typename T>
ROCSPARSE_KERNEL(BLOCKSIZE)
void doti_kernel_part2(T* workspace, T* result)
{
    int tid = hipThreadIdx_x;

    __shared__ T sdata[BLOCKSIZE];

    sdata[tid] = workspace[tid];
    __syncthreads();

    rocsparse_blockreduce_sum<BLOCKSIZE>(tid, sdata);

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
