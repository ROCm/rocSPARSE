/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef BSRMM_DEVICE_H
#define BSRMM_DEVICE_H

#include "common.h"

#include <hip/hip_runtime.h>

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE, rocsparse_int BSR_BLOCK_DIM>
static __device__ void bsrmmnn_small_blockdim_device(rocsparse_direction direction,
                                                     rocsparse_int       Mb,
                                                     rocsparse_int       N,
                                                     T                   alpha,
                                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                                     const rocsparse_int* __restrict__ bsr_col_ind,
                                                     const T* __restrict__ bsr_val,
                                                     const T* __restrict__ B,
                                                     rocsparse_int ldb,
                                                     T             beta,
                                                     T* __restrict__ C,
                                                     rocsparse_int        ldc,
                                                     rocsparse_index_base idx_base)
{
    constexpr rocsparse_int PADDED_BSR_BLOCK_DIM = (BSR_BLOCK_DIM + 1);

    rocsparse_int tid  = hipThreadIdx_x;
    rocsparse_int gid  = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int lid  = gid & (WF_SIZE - 1);
    rocsparse_int wid  = tid / WF_SIZE;
    rocsparse_int nwfb = hipGridDim_x * hipBlockDim_x / (WF_SIZE * BSR_BLOCK_DIM);
    rocsparse_int col  = lid + hipBlockIdx_y * WF_SIZE;

    rocsparse_int colB = col * ldb;
    rocsparse_int colC = col * ldc;

    // global row
    rocsparse_int global_row = (gid / WF_SIZE);

    // local row within block row
    rocsparse_int local_row = (gid / WF_SIZE) % BSR_BLOCK_DIM;

    __shared__ rocsparse_int shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ T             shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE * PADDED_BSR_BLOCK_DIM];

    for(rocsparse_int block_row = gid / (WF_SIZE * BSR_BLOCK_DIM); block_row < Mb;
        block_row += nwfb)
    {
        rocsparse_int block_row_start = bsr_row_ptr[block_row] - idx_base;
        rocsparse_int block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;

        T sum = static_cast<T>(0);

        for(rocsparse_int j = block_row_start; j < block_row_end; j += WF_SIZE)
        {
            rocsparse_int k = j + lid;

            shared_col[wid][lid]
                = (k < block_row_end) ? BSR_BLOCK_DIM * (bsr_col_ind[k] - idx_base) : 0;

            if(direction == rocsparse_direction_row)
            {
                // Perform:
                // for(rocsparse_int l = 0; l < BSR_BLOCK_DIM; l++)
                // {
                //     shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + l]
                //         = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                //                                               + BSR_BLOCK_DIM * local_row + l]
                //                               : static_cast<T>(0);
                // }
                // as unrolled loop.
                shared_val[wid][PADDED_BSR_BLOCK_DIM * lid]
                    = (k < block_row_end)
                          ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + BSR_BLOCK_DIM * local_row]
                          : static_cast<T>(0);
                if(BSR_BLOCK_DIM >= 2)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 1]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 1]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 3)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 2]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 2]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 4)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 3]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 3]
                                              : static_cast<T>(0);
                }
            }
            else
            {
                // Perform:
                // for(rocsparse_int l = 0; l < BSR_BLOCK_DIM; l++)
                // {
                //     shared_val[wid][BSR_BLOCK_DIM * lid + l]
                //         = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                //                                               + BSR_BLOCK_DIM * l + local_row]
                //                               : static_cast<T>(0);
                // }
                // as unrolled loop.
                shared_val[wid][PADDED_BSR_BLOCK_DIM * lid]
                    = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + local_row]
                                          : static_cast<T>(0);
                if(BSR_BLOCK_DIM >= 2)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 1]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 1 + local_row]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 3)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 2]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 2 + local_row]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 4)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 3]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 3 + local_row]
                                              : static_cast<T>(0);
                }
            }

            __syncthreads();

            if(col < N)
            {
                for(rocsparse_int i = 0; i < WF_SIZE; ++i)
                {
                    // Perform:
                    // for(rocsparse_int l = 0; l < BSR_BLOCK_DIM; l++)
                    // {
                    //     sum = rocsparse_fma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + l],
                    //                         B[shared_col[wid][i] + l],
                    //                         sum);
                    // }
                    // as unrolled loop.
                    sum = rocsparse_fma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i],
                                        B[shared_col[wid][i] + colB],
                                        sum);
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        sum = rocsparse_fma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 1],
                                            B[shared_col[wid][i] + 1 + colB],
                                            sum);
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        sum = rocsparse_fma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 2],
                                            B[shared_col[wid][i] + 2 + colB],
                                            sum);
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        sum = rocsparse_fma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 3],
                                            B[shared_col[wid][i] + 3 + colB],
                                            sum);
                    }
                }
            }
        }

        if(col < N)
        {
            if(beta == static_cast<T>(0))
            {
                C[global_row + colC] = alpha * sum;
            }
            else
            {
                C[global_row + colC] = rocsparse_fma(beta, C[global_row + colC], alpha * sum);
            }
        }
    }
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE, rocsparse_int BSR_BLOCK_DIM>
static __device__ void bsrmmnt_small_blockdim_device(rocsparse_direction direction,
                                                     rocsparse_int       Mb,
                                                     rocsparse_int       N,
                                                     T                   alpha,
                                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                                     const rocsparse_int* __restrict__ bsr_col_ind,
                                                     const T* __restrict__ bsr_val,
                                                     const T* __restrict__ B,
                                                     rocsparse_int ldb,
                                                     T             beta,
                                                     T* __restrict__ C,
                                                     rocsparse_int        ldc,
                                                     rocsparse_index_base idx_base)
{
    constexpr rocsparse_int PADDED_BSR_BLOCK_DIM = (BSR_BLOCK_DIM + 1);

    rocsparse_int tid        = hipThreadIdx_x;
    rocsparse_int gid        = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int block_row  = gid / (WF_SIZE * BSR_BLOCK_DIM);
    rocsparse_int global_row = gid / WF_SIZE;
    rocsparse_int local_row  = (gid / WF_SIZE) % BSR_BLOCK_DIM;
    rocsparse_int lid        = tid & (WF_SIZE - 1);
    rocsparse_int wid        = tid / WF_SIZE;

    if(block_row >= Mb)
    {
        return;
    }

    __shared__ rocsparse_int shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ T             shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE * PADDED_BSR_BLOCK_DIM];

    rocsparse_int block_row_start = bsr_row_ptr[block_row] - idx_base;
    rocsparse_int block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;

    for(rocsparse_int l = 0; l < N; l += WF_SIZE)
    {
        rocsparse_int col = l + lid;
        T             sum = static_cast<T>(0);

        for(rocsparse_int j = block_row_start; j < block_row_end; j += WF_SIZE)
        {
            rocsparse_int k = j + lid;

            shared_col[wid][lid]
                = (k < block_row_end) ? N * BSR_BLOCK_DIM * (bsr_col_ind[k] - idx_base) : 0;

            if(direction == rocsparse_direction_row)
            {
                // Perform:
                // for(rocsparse_int p = 0; p < BSR_BLOCK_DIM; p++)
                // {
                //     shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + p]
                //         = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                //                                               + BSR_BLOCK_DIM * local_row + p]
                //                               : static_cast<T>(0);
                // }
                // as unrolled loop.
                shared_val[wid][PADDED_BSR_BLOCK_DIM * lid]
                    = (k < block_row_end)
                          ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + BSR_BLOCK_DIM * local_row]
                          : static_cast<T>(0);
                if(BSR_BLOCK_DIM >= 2)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 1]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 1]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 3)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 2]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 2]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 4)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 3]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 3]
                                              : static_cast<T>(0);
                }
            }
            else
            {
                // Perform:
                // for(rocsparse_int p = 0; p < BSR_BLOCK_DIM; p++)
                // {
                //     shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + p]
                //         = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                //                                               + BSR_BLOCK_DIM * p + local_row]
                //                               : static_cast<T>(0);
                // }
                // as unrolled loop.
                shared_val[wid][PADDED_BSR_BLOCK_DIM * lid]
                    = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + local_row]
                                          : static_cast<T>(0);
                if(BSR_BLOCK_DIM >= 2)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 1]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 1 + local_row]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 3)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 2]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 2 + local_row]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 4)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 3]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 3 + local_row]
                                              : static_cast<T>(0);
                }
            }

            __syncthreads();

            if(col < N)
            {
                for(rocsparse_int i = 0; i < WF_SIZE; ++i)
                {
                    // Perform:
                    // for(rocsparse_int p = 0; p < BSR_BLOCK_DIM; p++)
                    // {
                    //     T val_B = rocsparse_ldg(B + col + N * p + shared_col[wid][i]);
                    //     sum = rocsparse_fma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + p], val_B, sum);
                    // }
                    // as unrolled loop.
                    T val_B = rocsparse_ldg(B + col + shared_col[wid][i]);
                    sum     = rocsparse_fma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i], val_B, sum);
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        val_B = rocsparse_ldg(B + col + N * 1 + shared_col[wid][i]);
                        sum   = rocsparse_fma(
                            shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 1], val_B, sum);
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        val_B = rocsparse_ldg(B + col + N * 2 + shared_col[wid][i]);
                        sum   = rocsparse_fma(
                            shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 2], val_B, sum);
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        val_B = rocsparse_ldg(B + col + N * 3 + shared_col[wid][i]);
                        sum   = rocsparse_fma(
                            shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 3], val_B, sum);
                    }
                }
            }
        }

        if(col < N)
        {
            if(beta == static_cast<T>(0))
            {
                C[global_row + col * ldc] = alpha * sum;
            }
            else
            {
                C[global_row + col * ldc]
                    = rocsparse_fma(beta, C[global_row + col * ldc], alpha * sum);
            }
        }
    }
}

template <typename T, rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y>
static __device__ void bsrmm_large_blockdim_device(rocsparse_direction direction,
                                                   rocsparse_operation trans_B,
                                                   rocsparse_int       Mb,
                                                   rocsparse_int       N,
                                                   T                   alpha,
                                                   const rocsparse_int* __restrict__ bsr_row_ptr,
                                                   const rocsparse_int* __restrict__ bsr_col_ind,
                                                   const T* __restrict__ bsr_val,
                                                   rocsparse_int block_dim,
                                                   const T* __restrict__ B,
                                                   rocsparse_int ldb,
                                                   T             beta,
                                                   T* __restrict__ C,
                                                   rocsparse_int        ldc,
                                                   rocsparse_index_base idx_base)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int tidy = hipThreadIdx_y;

    rocsparse_int global_row = tidx + hipBlockIdx_x * block_dim;
    rocsparse_int global_col = tidy + hipBlockIdx_y * BLK_SIZE_Y;

    rocsparse_int block_row = hipBlockIdx_x;

    rocsparse_int block_row_start = 0;
    rocsparse_int block_row_end   = 0;
    if(block_row < Mb)
    {
        block_row_start = bsr_row_ptr[block_row] - idx_base;
        block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;
    }

    rocsparse_int colB = global_col * ldb;
    rocsparse_int colC = global_col * ldc;

    __shared__ T shared_B[BSR_BLOCK_DIM * BLK_SIZE_Y];
    __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

    T sum = static_cast<T>(0);

    rocsparse_int index         = BSR_BLOCK_DIM * tidy + tidx;
    rocsparse_int block_dim_sqr = block_dim * block_dim;

    for(rocsparse_int k = block_row_start; k < block_row_end; k++)
    {
        rocsparse_int block_col = (bsr_col_ind[k] - idx_base);

        if(trans_B == rocsparse_operation_none)
        {
            shared_B[index] = (global_col < N && tidx < block_dim)
                                  ? B[block_dim * block_col + tidx + colB]
                                  : static_cast<T>(0);
        }
        else
        {
            shared_B[index] = (global_col < N && tidx < block_dim)
                                  ? B[global_col + ldb * (block_dim * block_col + tidx)]
                                  : static_cast<T>(0);
        }

        if(direction == rocsparse_direction_row)
        {
            if(tidx < block_dim && tidy < block_dim)
            {
                shared_A[index] = bsr_val[block_dim_sqr * k + block_dim * tidx + tidy];
            }
        }
        else
        {
            if(tidx < block_dim && tidy < block_dim)
            {
                shared_A[index] = bsr_val[block_dim_sqr * k + block_dim * tidy + tidx];
            }
        }

        __syncthreads();

        for(rocsparse_int j = 0; j < block_dim; j++)
        {
            sum = rocsparse_fma(
                shared_A[BSR_BLOCK_DIM * j + tidx], shared_B[BSR_BLOCK_DIM * tidy + j], sum);
        }

        __syncthreads();
    }

    if(block_row < Mb && global_col < N && tidx < block_dim)
    {
        if(beta == static_cast<T>(0))
        {
            C[global_row + colC] = alpha * sum;
        }
        else
        {
            C[global_row + colC] = rocsparse_fma(beta, C[global_row + colC], alpha * sum);
        }
    }
}

template <typename T, rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y>
static __device__ void bsrmm_general_blockdim_device(rocsparse_direction direction,
                                                     rocsparse_operation trans_B,
                                                     rocsparse_int       Mb,
                                                     rocsparse_int       N,
                                                     T                   alpha,
                                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                                     const rocsparse_int* __restrict__ bsr_col_ind,
                                                     const T* __restrict__ bsr_val,
                                                     rocsparse_int block_dim,
                                                     const T* __restrict__ B,
                                                     rocsparse_int ldb,
                                                     T             beta,
                                                     T* __restrict__ C,
                                                     rocsparse_int        ldc,
                                                     rocsparse_index_base idx_base)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int tidy = hipThreadIdx_y;

    rocsparse_int block_row = hipBlockIdx_x;

    rocsparse_int block_row_start = 0;
    rocsparse_int block_row_end   = 0;
    if(block_row < Mb)
    {
        block_row_start = bsr_row_ptr[block_row] - idx_base;
        block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;
    }

    __shared__ T shared_B[BSR_BLOCK_DIM * BLK_SIZE_Y];
    __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

    rocsparse_int global_col = tidy + hipBlockIdx_y * BLK_SIZE_Y;

    rocsparse_int colB = global_col * ldb;
    rocsparse_int colC = global_col * ldc;

    for(rocsparse_int x = 0; x < block_dim; x += BSR_BLOCK_DIM)
    {
        rocsparse_int global_row = tidx + x + hipBlockIdx_x * block_dim;

        T sum = static_cast<T>(0);

        for(rocsparse_int k = block_row_start; k < block_row_end; k++)
        {
            rocsparse_int block_col = (bsr_col_ind[k] - idx_base);

            for(rocsparse_int y = 0; y < block_dim; y += BLK_SIZE_Y)
            {
                if(trans_B == rocsparse_operation_none)
                {
                    shared_B[BSR_BLOCK_DIM * tidy + tidx]
                        = (global_col < N && (tidx + y) < block_dim)
                              ? B[block_dim * block_col + (tidx + y) + colB]
                              : static_cast<T>(0);
                }
                else
                {
                    shared_B[BSR_BLOCK_DIM * tidy + tidx]
                        = (global_col < N && (tidx + y) < block_dim)
                              ? B[global_col + ldb * (block_dim * block_col + (tidx + y))]
                              : static_cast<T>(0);
                }

                if(direction == rocsparse_direction_row)
                {
                    shared_A[BSR_BLOCK_DIM * tidy + tidx]
                        = ((tidx + x) < block_dim && (tidy + y) < block_dim)
                              ? bsr_val[block_dim * block_dim * k + block_dim * (tidx + x)
                                        + (tidy + y)]
                              : static_cast<T>(0);
                }
                else
                {
                    shared_A[BSR_BLOCK_DIM * tidy + tidx]
                        = ((tidx + x) < block_dim && (tidy + y) < block_dim)
                              ? bsr_val[block_dim * block_dim * k + block_dim * (tidy + y)
                                        + (tidx + x)]
                              : static_cast<T>(0);
                }

                __syncthreads();

                for(rocsparse_int j = 0; j < BSR_BLOCK_DIM; j++)
                {
                    sum = rocsparse_fma(shared_A[BSR_BLOCK_DIM * j + tidx],
                                        shared_B[BSR_BLOCK_DIM * tidy + j],
                                        sum);
                }

                __syncthreads();
            }
        }

        if(block_row < Mb && global_col < N && (tidx + x) < block_dim)
        {
            if(beta == static_cast<T>(0))
            {
                C[global_row + colC] = alpha * sum;
            }
            else
            {
                C[global_row + colC] = rocsparse_fma(beta, C[global_row + colC], alpha * sum);
            }
        }
    }
}

#endif // BSRMM_DEVICE_H