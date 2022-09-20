/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <hip/hip_runtime.h>

#include "common.h"

template <rocsparse_direction DIRECTION,
          rocsparse_int       BLOCK_SIZE,
          rocsparse_int       BLOCK_DIM,
          typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void bsr2csr_block_per_row_2_7_kernel(rocsparse_int        mb,
                                          rocsparse_int        nb,
                                          rocsparse_index_base bsr_base,
                                          const T* __restrict__ bsr_val,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          rocsparse_int        block_dim,
                                          rocsparse_index_base csr_base,
                                          T* __restrict__ csr_val,
                                          rocsparse_int* __restrict__ csr_row_ptr,
                                          rocsparse_int* __restrict__ csr_col_ind)
{
    // Find next largest power of 2
    unsigned int BLOCK_DIM2 = fnp2(BLOCK_DIM);

    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int bid = hipBlockIdx_x;

    rocsparse_int start = bsr_row_ptr[bid] - bsr_base;
    rocsparse_int end   = bsr_row_ptr[bid + 1] - bsr_base;

    if(bid == 0 && tid == 0)
    {
        csr_row_ptr[0] = csr_base;
    }

    rocsparse_int lid = tid & (BLOCK_DIM2 - 1);
    rocsparse_int wid = tid / BLOCK_DIM2;

    rocsparse_int r = lid;

    if(r >= BLOCK_DIM)
    {
        return;
    }

    rocsparse_int prev    = BLOCK_DIM * BLOCK_DIM * start + BLOCK_DIM * (end - start) * r;
    rocsparse_int current = BLOCK_DIM * (end - start);

    csr_row_ptr[BLOCK_DIM * bid + r + 1] = prev + current + csr_base;

    for(rocsparse_int i = start + wid; i < end; i += (BLOCK_SIZE / BLOCK_DIM2))
    {
        rocsparse_int col    = bsr_col_ind[i] - bsr_base;
        rocsparse_int offset = prev + BLOCK_DIM * (i - start);

        for(rocsparse_int j = 0; j < BLOCK_DIM; j++)
        {
            csr_col_ind[offset + j] = BLOCK_DIM * col + j + csr_base;

            if(DIRECTION == rocsparse_direction_row)
            {
                csr_val[offset + j] = bsr_val[BLOCK_DIM * BLOCK_DIM * i + r * BLOCK_DIM + j];
            }
            else
            {
                csr_val[offset + j] = bsr_val[BLOCK_DIM * BLOCK_DIM * i + r + BLOCK_DIM * j];
            }
        }
    }
}

template <rocsparse_direction DIRECTION,
          rocsparse_int       BLOCK_SIZE,
          rocsparse_int       BLOCK_DIM,
          typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void bsr2csr_block_per_row_8_32_kernel(rocsparse_int        mb,
                                           rocsparse_int        nb,
                                           rocsparse_index_base bsr_base,
                                           const T* __restrict__ bsr_val,
                                           const rocsparse_int* __restrict__ bsr_row_ptr,
                                           const rocsparse_int* __restrict__ bsr_col_ind,
                                           rocsparse_int        block_dim,
                                           rocsparse_index_base csr_base,
                                           T* __restrict__ csr_val,
                                           rocsparse_int* __restrict__ csr_row_ptr,
                                           rocsparse_int* __restrict__ csr_col_ind)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int bid = hipBlockIdx_x;

    rocsparse_int start = bsr_row_ptr[bid] - bsr_base;
    rocsparse_int end   = bsr_row_ptr[bid + 1] - bsr_base;

    if(bid == 0 && tid == 0)
    {
        csr_row_ptr[0] = csr_base;
    }

    rocsparse_int lid = tid & (BLOCK_DIM * BLOCK_DIM - 1);
    rocsparse_int wid = tid / (BLOCK_DIM * BLOCK_DIM);

    rocsparse_int c = lid & (BLOCK_DIM - 1);
    rocsparse_int r = lid / BLOCK_DIM;

    if(r >= block_dim || c >= block_dim)
    {
        return;
    }

    rocsparse_int prev    = block_dim * block_dim * start + block_dim * (end - start) * r;
    rocsparse_int current = block_dim * (end - start);

    csr_row_ptr[block_dim * bid + r + 1] = prev + current + csr_base;

    for(rocsparse_int i = start + wid; i < end; i += (BLOCK_SIZE / (BLOCK_DIM * BLOCK_DIM)))
    {
        rocsparse_int col    = bsr_col_ind[i] - bsr_base;
        rocsparse_int offset = prev + block_dim * (i - start) + c;

        csr_col_ind[offset] = block_dim * col + c + csr_base;

        if(DIRECTION == rocsparse_direction_row)
        {
            csr_val[offset] = bsr_val[block_dim * block_dim * i + block_dim * r + c];
        }
        else
        {
            csr_val[offset] = bsr_val[block_dim * block_dim * i + block_dim * c + r];
        }
    }
}

template <rocsparse_direction DIRECTION,
          rocsparse_int       BLOCK_SIZE,
          rocsparse_int       BLOCK_DIM,
          rocsparse_int       SUB_BLOCK_DIM,
          typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void bsr2csr_block_per_row_33_128_kernel(rocsparse_int        mb,
                                             rocsparse_int        nb,
                                             rocsparse_index_base bsr_base,
                                             const T* __restrict__ bsr_val,
                                             const rocsparse_int* __restrict__ bsr_row_ptr,
                                             const rocsparse_int* __restrict__ bsr_col_ind,
                                             rocsparse_int        block_dim,
                                             rocsparse_index_base csr_base,
                                             T* __restrict__ csr_val,
                                             rocsparse_int* __restrict__ csr_row_ptr,
                                             rocsparse_int* __restrict__ csr_col_ind)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int bid = hipBlockIdx_x;

    rocsparse_int start = bsr_row_ptr[bid] - bsr_base;
    rocsparse_int end   = bsr_row_ptr[bid + 1] - bsr_base;

    if(bid == 0 && tid == 0)
    {
        csr_row_ptr[0] = csr_base;
    }

    for(rocsparse_int y = 0; y < (BLOCK_DIM / SUB_BLOCK_DIM); y++)
    {
        rocsparse_int r = (tid / SUB_BLOCK_DIM) + SUB_BLOCK_DIM * y;

        if(r < block_dim)
        {
            rocsparse_int prev    = block_dim * block_dim * start + block_dim * (end - start) * r;
            rocsparse_int current = block_dim * (end - start);

            csr_row_ptr[block_dim * bid + r + 1] = prev + current + csr_base;
        }
    }

    for(rocsparse_int i = start; i < end; i++)
    {
        rocsparse_int col = bsr_col_ind[i] - bsr_base;

        for(rocsparse_int y = 0; y < (BLOCK_DIM / SUB_BLOCK_DIM); y++)
        {
            for(rocsparse_int x = 0; x < (BLOCK_DIM / SUB_BLOCK_DIM); x++)
            {
                rocsparse_int c = (tid & (SUB_BLOCK_DIM - 1)) + SUB_BLOCK_DIM * x;
                rocsparse_int r = (tid / SUB_BLOCK_DIM) + SUB_BLOCK_DIM * y;

                if(r < block_dim && c < block_dim)
                {
                    rocsparse_int prev
                        = block_dim * block_dim * start + block_dim * (end - start) * r;

                    rocsparse_int offset = prev + block_dim * (i - start) + c;

                    csr_col_ind[offset] = block_dim * col + c + csr_base;

                    if(DIRECTION == rocsparse_direction_row)
                    {
                        csr_val[offset] = bsr_val[block_dim * block_dim * i + block_dim * r + c];
                    }
                    else
                    {
                        csr_val[offset] = bsr_val[block_dim * block_dim * i + block_dim * c + r];
                    }
                }
            }
        }
    }
}

template <rocsparse_int BLOCK_SIZE, typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void bsr2csr_block_dim_equals_one_kernel(rocsparse_int        mb,
                                             rocsparse_int        nb,
                                             rocsparse_index_base bsr_base,
                                             const T* __restrict__ bsr_val,
                                             const rocsparse_int* __restrict__ bsr_row_ptr,
                                             const rocsparse_int* __restrict__ bsr_col_ind,
                                             rocsparse_index_base csr_base,
                                             T* __restrict__ csr_val,
                                             rocsparse_int* __restrict__ csr_row_ptr,
                                             rocsparse_int* __restrict__ csr_col_ind)
{
    rocsparse_int tid = hipThreadIdx_x + BLOCK_SIZE * hipBlockIdx_x;

    if(tid < mb)
    {
        if(tid == 0)
        {
            csr_row_ptr[0] = (bsr_row_ptr[0] - bsr_base) + csr_base;
        }

        csr_row_ptr[tid + 1] = (bsr_row_ptr[tid + 1] - bsr_base) + csr_base;
    }

    rocsparse_int nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];

    rocsparse_int index = tid;
    while(index < nnzb)
    {
        csr_col_ind[index] = (bsr_col_ind[index] - bsr_base) + csr_base;
        csr_val[index]     = bsr_val[index];

        index += BLOCK_SIZE * hipGridDim_x;
    }
}
