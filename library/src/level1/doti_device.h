/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef DOTI_DEVICE_H
#define DOTI_DEVICE_H

#include <hip/hip_runtime.h>

template <rocsparse_int n, typename T>
__device__ void rocsparse_sum_reduce(rocsparse_int tid, T* x)
{
    // clang-format off
    __syncthreads();
    if(n > 512) { if(tid < 512 && tid + 512 < n) { x[tid] += x[tid + 512]; } __syncthreads(); }
    if(n > 256) { if(tid < 256 && tid + 256 < n) { x[tid] += x[tid + 256]; } __syncthreads(); }
    if(n > 128) { if(tid < 128 && tid + 128 < n) { x[tid] += x[tid + 128]; } __syncthreads(); } 
    if(n >  64) { if(tid <  64 && tid +  64 < n) { x[tid] += x[tid +  64]; } __syncthreads(); }
    if(n >  32) { if(tid <  32 && tid +  32 < n) { x[tid] += x[tid +  32]; } __syncthreads(); }
    if(n >  16) { if(tid <  16 && tid +  16 < n) { x[tid] += x[tid +  16]; } __syncthreads(); }
    if(n >   8) { if(tid <   8 && tid +   8 < n) { x[tid] += x[tid +   8]; } __syncthreads(); }
    if(n >   4) { if(tid <   4 && tid +   4 < n) { x[tid] += x[tid +   4]; } __syncthreads(); }
    if(n >   2) { if(tid <   2 && tid +   2 < n) { x[tid] += x[tid +   2]; } __syncthreads(); }
    if(n >   1) { if(tid <   1 && tid +   1 < n) { x[tid] += x[tid +   1]; } __syncthreads(); }
    // clang-format on
}

template <typename T, rocsparse_int NB>
__global__ void doti_kernel_part1(rocsparse_int nnz,
                                  const T* x_val,
                                  const rocsparse_int* x_ind,
                                  const T* y,
                                  T* workspace,
                                  rocsparse_index_base idx_base)
{
    rocsparse_int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocsparse_int tid = hipThreadIdx_x;

    __shared__ T sdata[NB];

    if(idx < nnz)
    {
        sdata[tid] = y[x_ind[idx] - idx_base] * x_val[idx];
    }
    else
    {
        sdata[tid] = static_cast<T>(0);
    }

    rocsparse_sum_reduce<NB, T>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <typename T, rocsparse_int NB, rocsparse_int flag>
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

    if(n < 32)
    {
        if(tid == 0)
        {
            for(rocsparse_int i = 1; i < n; ++i)
            {
                sdata[0] += sdata[i];
            }
        }
    }
    else
    {
        rocsparse_sum_reduce<NB, T>(tid, sdata);
    }

    if(tid == 0)
    {
        if(flag)
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
