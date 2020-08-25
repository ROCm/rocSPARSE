/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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
#ifndef NNZ_DEVICE_H
#define NNZ_DEVICE_H

#include "handle.h"
#include <hip/hip_runtime.h>

//!
//! @brief Recursive Compile Time Device Reduction.
//! @param tx        The local thread id.
//! @param sdata     The array of data.
//!
template <rocsparse_int n>
__forceinline__ __device__ void nnz_device_reduce(rocsparse_int tx, rocsparse_int* sdata)
{
    __syncthreads();
    if(tx < n / 2)
    {
        sdata[tx] += sdata[tx + n / 2];
    }
    nnz_device_reduce<n / 2>(tx, sdata);
}

template <>
__forceinline__ __device__ void nnz_device_reduce<0>(rocsparse_int tx, rocsparse_int* sdata)
{
}

//!
//! @brief Kernel for counting the number of non-zeros per column.
//! @param m         		The number of rows.
//! @param n         		The number of columns.
//! @param A       		The pointer to the values.
//! @param lda       		The leading dimension.
//! @param nnzPerColumn         The array storing the results of the nnz per column.
//!
template <rocsparse_int NB_X, typename T>
__launch_bounds__(NB_X) __global__ void nnz_kernel_col(rocsparse_int m,
                                                       rocsparse_int n,
                                                       const T* __restrict__ A,
                                                       rocsparse_int lda,
                                                       rocsparse_int* __restrict__ nnzPerColumn)
{
    static constexpr T s_zero = {};
    rocsparse_int tx = hipThreadIdx_x, col = hipBlockIdx_x, m_full = (m / NB_X) * NB_X, res = 0;

    __shared__ rocsparse_int sdata[NB_X];

    A += col * lda + ((tx < m) ? tx : 0);

    for(rocsparse_int i = 0; i < m_full; i += NB_X)
        res += (A[i] != s_zero) ? 1 : 0;

    if(tx + m_full < m)
        res += (A[m_full] != s_zero) ? 1 : 0;

    sdata[tx] = res;

    if(NB_X > 16 && m >= NB_X)
    {
        nnz_device_reduce<NB_X>(tx, sdata);
    }
    else
    {
        __syncthreads();

        if(tx == 0)
        {
            for(rocsparse_int i = 1; i < m && i < NB_X; i++)
                sdata[0] += sdata[i];
        }

        __syncthreads();
    }

    if(tx == 0)
    {
        nnzPerColumn[col] = sdata[0];
    }
}

//!
//! @brief Kernel for counting the number of non-zeros per row.
//! @param m         		The number of rows.
//! @param n         		The number of columns.
//! @param A       		The pointer to the values.
//! @param lda       		The leading dimension.
//! @param nnzPerRow            The array storing the results of the nnz per row.
//!
template <rocsparse_int DIM_X, rocsparse_int DIM_Y, typename T>
__launch_bounds__(DIM_X* DIM_Y) __global__
    void nnz_kernel_row(rocsparse_int m,
                        rocsparse_int n,
                        const T* __restrict__ A,
                        rocsparse_int lda,
                        rocsparse_int* __restrict__ nnzPerRow)
{
    static constexpr T s_zero = {};

    rocsparse_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    rocsparse_int tx        = thread_id % DIM_X;
    rocsparse_int ty        = thread_id / DIM_X;
    rocsparse_int ind       = hipBlockIdx_x * DIM_X * 4 + tx;
    rocsparse_int n_tail    = n % (4 * DIM_Y);
    rocsparse_int col       = ty * 4;
    rocsparse_int res_A[4];

    __shared__ rocsparse_int sdata[DIM_X * 4 * DIM_Y];

    for(rocsparse_int k = 0; k < 4; ++k)
    {
        res_A[k] = 0;
    }

    for(col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y)
    {
        for(rocsparse_int k = 0; k < 4; ++k)
        {
            if(ind + k * DIM_X < m)
            {
                for(rocsparse_int j = 0; j < 4; ++j)
                {
                    if(A[ind + k * DIM_X + (col + j) * lda] != s_zero)
                        res_A[k] += 1;
                }
            }
        }
    }

    if(n_tail > 0)
    {
        for(rocsparse_int k = 0; k < 4; ++k)
        {
            if(ind + k * DIM_X < m)
            {
                for(rocsparse_int j = 0; j < 4; ++j)
                {
                    if(col + j < n)
                    {
                        res_A[k] += (A[ind + k * DIM_X + (col + j) * lda] != s_zero) ? 1 : 0;
                    }
                }
            }
        }
    }

    for(rocsparse_int k = 0; k < 4; ++k)
    {
        sdata[tx + k * DIM_X + ty * DIM_X * 4] = res_A[k];
    }

    __syncthreads();

    ind = hipBlockIdx_x * DIM_X * 4 + thread_id;
    if(thread_id < DIM_X * 4)
    {
        for(rocsparse_int j = 1; j < DIM_Y; j++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * 4 * j];
        }

        if(ind < m)
        {
            nnzPerRow[ind] = sdata[thread_id];
        }
    }
}

#endif
