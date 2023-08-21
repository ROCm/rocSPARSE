/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

template <rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE, rocsparse_int BSR_BLOCK_DIM, typename T>
ROCSPARSE_DEVICE_ILF void
    bsrmmnn_small_blockdim_device(rocsparse_direction direction,
                                  rocsparse_int       Mb,
                                  rocsparse_int       N,
                                  T                   alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  const T* __restrict__ B,
                                  int64_t ldb,
                                  T       beta,
                                  T* __restrict__ C,
                                  int64_t              ldc,
                                  rocsparse_index_base idx_base)
{
    constexpr rocsparse_int PADDED_BSR_BLOCK_DIM = (BSR_BLOCK_DIM + 1);

    rocsparse_int tid  = hipThreadIdx_x;
    rocsparse_int gid  = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int lid  = gid & (WF_SIZE - 1);
    rocsparse_int wid  = tid / WF_SIZE;
    rocsparse_int nwfb = hipGridDim_x * hipBlockDim_x / (WF_SIZE * BSR_BLOCK_DIM);
    rocsparse_int col  = lid + hipBlockIdx_y * WF_SIZE;

    int64_t colB = col * ldb;
    int64_t colC = col * ldc;

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

template <rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE, rocsparse_int BSR_BLOCK_DIM, typename T>
ROCSPARSE_DEVICE_ILF void
    bsrmmnt_small_blockdim_device(rocsparse_direction direction,
                                  rocsparse_int       Mb,
                                  rocsparse_int       N,
                                  T                   alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  const T* __restrict__ B,
                                  int64_t ldb,
                                  T       beta,
                                  T* __restrict__ C,
                                  int64_t              ldc,
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
                = (k < block_row_end) ? BSR_BLOCK_DIM * (bsr_col_ind[k] - idx_base) : 0;

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
                    //     T val_B = rocsparse_ldg(B + col + ldb * (p + shared_col[wid][i]));
                    //     sum = rocsparse_fma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + p], val_B, sum);
                    // }
                    // as unrolled loop.
                    T val_B = rocsparse_ldg(B + col + ldb * shared_col[wid][i]);
                    sum     = rocsparse_fma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i], val_B, sum);
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        val_B = rocsparse_ldg(B + col + ldb * (1 + shared_col[wid][i]));
                        sum   = rocsparse_fma(
                            shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 1], val_B, sum);
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        val_B = rocsparse_ldg(B + col + ldb * (2 + shared_col[wid][i]));
                        sum   = rocsparse_fma(
                            shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 2], val_B, sum);
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        val_B = rocsparse_ldg(B + col + ldb * (3 + shared_col[wid][i]));
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
