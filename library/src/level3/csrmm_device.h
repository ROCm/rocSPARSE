/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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
#ifndef CSRMM_DEVICE_H
#define CSRMM_DEVICE_H

#include "common.h"

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename J, typename T>
static __device__ void csrmmnn_general_device(J M,
                                              J N,
                                              J K,
                                              I nnz,
                                              T alpha,
                                              const I* __restrict__ csr_row_ptr,
                                              const J* __restrict__ csr_col_ind,
                                              const T* __restrict__ csr_val,
                                              const T* __restrict__ B,
                                              J ldb,
                                              T beta,
                                              T* __restrict__ C,
                                              J                    ldc,
                                              rocsparse_order      order,
                                              rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
    int lid = gid & (WF_SIZE - 1);
    int wid = tid / WF_SIZE;
    J   nwf = hipGridDim_x * BLOCKSIZE / WF_SIZE;
    J   col = lid + hipBlockIdx_y * WF_SIZE;

    J colB = col * ldb;

    __shared__ J shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

    for(J row = gid / WF_SIZE; row < M; row += nwf)
    {
        I row_start = csr_row_ptr[row] - idx_base;
        I row_end   = csr_row_ptr[row + 1] - idx_base;

        T sum = static_cast<T>(0);

        for(I j = row_start; j < row_end; j += WF_SIZE)
        {
            I k = j + lid;

            __syncthreads();

            shared_col[wid][lid] = (k < row_end) ? csr_col_ind[k] - idx_base : 0;
            shared_val[wid][lid] = (k < row_end) ? csr_val[k] : static_cast<T>(0);

            __syncthreads();

            for(J i = 0; i < WF_SIZE && col < N; ++i)
            {
                sum = rocsparse_fma(shared_val[wid][i], B[shared_col[wid][i] + colB], sum);
            }
        }

        if(col < N)
        {
            if(beta == static_cast<T>(0))
            {
                if(order == rocsparse_order_column)
                {
                    C[row + col * ldc] = alpha * sum;
                }
                else
                {
                    C[row * ldc + col] = alpha * sum;
                }
            }
            else
            {
                if(order == rocsparse_order_column)
                {
                    C[row + col * ldc] = rocsparse_fma(beta, C[row + col * ldc], alpha * sum);
                }
                else
                {
                    C[row * ldc + col] = rocsparse_fma(beta, C[row * ldc + col], alpha * sum);
                }
            }
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename J, typename T>
static __device__ void csrmmnt_general_device(J offset,
                                              J ncol,
                                              J M,
                                              J N,
                                              J K,
                                              I nnz,
                                              T alpha,
                                              const I* __restrict__ csr_row_ptr,
                                              const J* __restrict__ csr_col_ind,
                                              const T* __restrict__ csr_val,
                                              const T* __restrict__ B,
                                              J ldb,
                                              T beta,
                                              T* __restrict__ C,
                                              J                    ldc,
                                              rocsparse_order      order,
                                              rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
    J   row = gid / WF_SIZE;
    int lid = tid & (WF_SIZE - 1);
    int wid = tid / WF_SIZE;

    if(row >= M)
    {
        return;
    }

    __shared__ J shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

    I row_start = csr_row_ptr[row] - idx_base;
    I row_end   = csr_row_ptr[row + 1] - idx_base;

    for(J l = offset; l < ncol; l += WF_SIZE)
    {
        J col = l + lid;
        T sum = static_cast<T>(0);

        for(I j = row_start; j < row_end; j += WF_SIZE)
        {
            I k = j + lid;

            __syncthreads();

            shared_col[wid][lid] = (k < row_end) ? ldb * (csr_col_ind[k] - idx_base) : 0;
            shared_val[wid][lid] = (k < row_end) ? csr_val[k] : static_cast<T>(0);

            __syncthreads();

            for(J i = 0; i < WF_SIZE; ++i)
            {
                T val_B = (col < ncol) ? rocsparse_ldg(B + col + shared_col[wid][i])
                                       : static_cast<T>(0);
                sum     = rocsparse_fma(shared_val[wid][i], val_B, sum);
            }
        }

        if(col < ncol)
        {
            if(beta == static_cast<T>(0))
            {
                if(order == rocsparse_order_column)
                {
                    C[row + col * ldc] = alpha * sum;
                }
                else
                {
                    C[row * ldc + col] = alpha * sum;
                }
            }
            else
            {
                if(order == rocsparse_order_column)
                {
                    C[row + col * ldc] = rocsparse_fma(beta, C[row + col * ldc], alpha * sum);
                }
                else
                {
                    C[row * ldc + col] = rocsparse_fma(beta, C[row * ldc + col], alpha * sum);
                }
            }
        }
    }
}

#endif // CSRMM_DEVICE_H
