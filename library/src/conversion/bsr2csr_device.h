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

template <rocsparse_direction DIRECTION, rocsparse_int BLOCK_SIZE, typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void bsr2csr_kernel(rocsparse_int        mb,
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
    rocsparse_int entries_in_block = block_dim * block_dim;

    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
    rocsparse_int warp_id   = thread_id / 64;
    rocsparse_int lane_id   = thread_id % 64;

    if(warp_id >= mb * block_dim)
    { // one warp per row in matrix
        return;
    }

    rocsparse_int block_row    = warp_id / block_dim; // block row in bsr matrix
    rocsparse_int row_in_block = warp_id % block_dim; // local row in bsr row block

    rocsparse_int bsr_row_start = bsr_row_ptr[block_row] - bsr_base;
    rocsparse_int bsr_row_end   = bsr_row_ptr[block_row + 1] - bsr_base;

    rocsparse_int entries_in_row = (bsr_row_end - bsr_row_start) * block_dim;
    rocsparse_int number_of_entries_in_prev_rows
        = bsr_row_start * entries_in_block + row_in_block * entries_in_row;

    if(warp_id == 0)
    {
        csr_row_ptr[0] = csr_base;
    }

    csr_row_ptr[warp_id + 1] = number_of_entries_in_prev_rows + entries_in_row + csr_base;

    for(rocsparse_int i = bsr_row_start + lane_id; i < bsr_row_end; i += 64)
    {

        rocsparse_int col    = bsr_col_ind[i] - bsr_base;
        rocsparse_int offset = number_of_entries_in_prev_rows + block_dim * (i - bsr_row_start);

        for(rocsparse_int j = 0; j < block_dim; j++)
        {
            csr_col_ind[offset + j] = block_dim * col + j + csr_base;

            if(DIRECTION == rocsparse_direction_row)
            {
                csr_val[offset + j] = bsr_val[i * entries_in_block + row_in_block * block_dim + j];
            }
            else
            {
                csr_val[offset + j] = bsr_val[i * entries_in_block + row_in_block + block_dim * j];
            }
        }
    }
}

template <rocsparse_direction DIRECTION,
          rocsparse_int       BLOCK_SIZE,
          rocsparse_int       BSR_BLOCK_DIM,
          typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void bsr2csr_unroll_kernel(rocsparse_int        mb,
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
    static constexpr rocsparse_int entries_in_block = BSR_BLOCK_DIM * BSR_BLOCK_DIM;

    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
    rocsparse_int warp_id   = thread_id / 64;
    rocsparse_int lane_id   = thread_id % 64;

    if(warp_id >= mb * BSR_BLOCK_DIM)
    { // one warp per row in matrix
        return;
    }

    rocsparse_int block_row    = warp_id / BSR_BLOCK_DIM; // block row in bsr matrix
    rocsparse_int row_in_block = warp_id % BSR_BLOCK_DIM; // local row in bsr row block

    rocsparse_int bsr_row_start = bsr_row_ptr[block_row] - bsr_base;
    rocsparse_int bsr_row_end   = bsr_row_ptr[block_row + 1] - bsr_base;

    rocsparse_int entries_in_row = (bsr_row_end - bsr_row_start) * BSR_BLOCK_DIM;
    rocsparse_int number_of_entries_in_prev_rows
        = bsr_row_start * entries_in_block + row_in_block * entries_in_row;

    if(warp_id == 0)
    {
        csr_row_ptr[0] = csr_base;
    }

    csr_row_ptr[warp_id + 1] = number_of_entries_in_prev_rows + entries_in_row + csr_base;

    for(rocsparse_int i = bsr_row_start + lane_id; i < bsr_row_end; i += 64)
    {
        rocsparse_int col    = bsr_col_ind[i] - bsr_base;
        rocsparse_int offset = number_of_entries_in_prev_rows + BSR_BLOCK_DIM * (i - bsr_row_start);

        if(BSR_BLOCK_DIM >= 1)
        {
            csr_col_ind[offset] = BSR_BLOCK_DIM * col + csr_base;
        }
        if(BSR_BLOCK_DIM >= 2)
        {
            csr_col_ind[offset + 1] = BSR_BLOCK_DIM * col + 1 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 3)
        {
            csr_col_ind[offset + 2] = BSR_BLOCK_DIM * col + 2 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 4)
        {
            csr_col_ind[offset + 3] = BSR_BLOCK_DIM * col + 3 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 5)
        {
            csr_col_ind[offset + 4] = BSR_BLOCK_DIM * col + 4 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 6)
        {
            csr_col_ind[offset + 5] = BSR_BLOCK_DIM * col + 5 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 7)
        {
            csr_col_ind[offset + 6] = BSR_BLOCK_DIM * col + 6 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 8)
        {
            csr_col_ind[offset + 7] = BSR_BLOCK_DIM * col + 7 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 9)
        {
            csr_col_ind[offset + 8] = BSR_BLOCK_DIM * col + 8 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 10)
        {
            csr_col_ind[offset + 9] = BSR_BLOCK_DIM * col + 9 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 11)
        {
            csr_col_ind[offset + 10] = BSR_BLOCK_DIM * col + 10 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 12)
        {
            csr_col_ind[offset + 11] = BSR_BLOCK_DIM * col + 11 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 13)
        {
            csr_col_ind[offset + 12] = BSR_BLOCK_DIM * col + 12 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 14)
        {
            csr_col_ind[offset + 13] = BSR_BLOCK_DIM * col + 13 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 15)
        {
            csr_col_ind[offset + 14] = BSR_BLOCK_DIM * col + 14 + csr_base;
        }
        if(BSR_BLOCK_DIM >= 16)
        {
            csr_col_ind[offset + 15] = BSR_BLOCK_DIM * col + 15 + csr_base;
        }

        if(DIRECTION == rocsparse_direction_row)
        {
            if(BSR_BLOCK_DIM >= 1)
            {
                csr_val[offset] = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM];
            }
            if(BSR_BLOCK_DIM >= 2)
            {
                csr_val[offset + 1]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 1];
            }
            if(BSR_BLOCK_DIM >= 3)
            {
                csr_val[offset + 2]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 2];
            }
            if(BSR_BLOCK_DIM >= 4)
            {
                csr_val[offset + 3]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 3];
            }
            if(BSR_BLOCK_DIM >= 5)
            {
                csr_val[offset + 4]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 4];
            }
            if(BSR_BLOCK_DIM >= 6)
            {
                csr_val[offset + 5]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 5];
            }
            if(BSR_BLOCK_DIM >= 7)
            {
                csr_val[offset + 6]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 6];
            }
            if(BSR_BLOCK_DIM >= 8)
            {
                csr_val[offset + 7]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 7];
            }
            if(BSR_BLOCK_DIM >= 9)
            {
                csr_val[offset + 8]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 8];
            }
            if(BSR_BLOCK_DIM >= 10)
            {
                csr_val[offset + 9]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 9];
            }
            if(BSR_BLOCK_DIM >= 11)
            {
                csr_val[offset + 10]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 10];
            }
            if(BSR_BLOCK_DIM >= 12)
            {
                csr_val[offset + 11]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 11];
            }
            if(BSR_BLOCK_DIM >= 13)
            {
                csr_val[offset + 12]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 12];
            }
            if(BSR_BLOCK_DIM >= 14)
            {
                csr_val[offset + 13]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 13];
            }
            if(BSR_BLOCK_DIM >= 15)
            {
                csr_val[offset + 14]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 14];
            }
            if(BSR_BLOCK_DIM >= 16)
            {
                csr_val[offset + 15]
                    = bsr_val[i * entries_in_block + row_in_block * BSR_BLOCK_DIM + 15];
            }
        }
        else
        {
            if(BSR_BLOCK_DIM >= 1)
            {
                csr_val[offset] = bsr_val[i * entries_in_block + row_in_block];
            }
            if(BSR_BLOCK_DIM >= 2)
            {
                csr_val[offset + 1]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 1];
            }
            if(BSR_BLOCK_DIM >= 3)
            {
                csr_val[offset + 2]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 2];
            }
            if(BSR_BLOCK_DIM >= 4)
            {
                csr_val[offset + 3]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 3];
            }
            if(BSR_BLOCK_DIM >= 5)
            {
                csr_val[offset + 4]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 4];
            }
            if(BSR_BLOCK_DIM >= 6)
            {
                csr_val[offset + 5]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 5];
            }
            if(BSR_BLOCK_DIM >= 7)
            {
                csr_val[offset + 6]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 6];
            }
            if(BSR_BLOCK_DIM >= 8)
            {
                csr_val[offset + 7]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 7];
            }
            if(BSR_BLOCK_DIM >= 9)
            {
                csr_val[offset + 8]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 8];
            }
            if(BSR_BLOCK_DIM >= 10)
            {
                csr_val[offset + 9]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 9];
            }
            if(BSR_BLOCK_DIM >= 11)
            {
                csr_val[offset + 10]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 10];
            }
            if(BSR_BLOCK_DIM >= 12)
            {
                csr_val[offset + 11]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 11];
            }
            if(BSR_BLOCK_DIM >= 13)
            {
                csr_val[offset + 12]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 12];
            }
            if(BSR_BLOCK_DIM >= 14)
            {
                csr_val[offset + 13]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 13];
            }
            if(BSR_BLOCK_DIM >= 15)
            {
                csr_val[offset + 14]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 14];
            }
            if(BSR_BLOCK_DIM >= 16)
            {
                csr_val[offset + 15]
                    = bsr_val[i * entries_in_block + row_in_block + BSR_BLOCK_DIM * 15];
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
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    if(thread_id < mb)
    {
        if(thread_id == 0)
        {
            csr_row_ptr[0] = (bsr_row_ptr[0] - bsr_base) + csr_base;
        }

        csr_row_ptr[thread_id + 1] = (bsr_row_ptr[thread_id + 1] - bsr_base) + csr_base;
    }

    rocsparse_int nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];

    rocsparse_int index = thread_id;
    while(index < nnzb)
    {
        csr_col_ind[index] = (bsr_col_ind[index] - bsr_base) + csr_base;
        csr_val[index]     = bsr_val[index];

        index += hipBlockDim_x * hipGridDim_x;
    }
}
