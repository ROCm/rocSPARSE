/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef GEBSR2CSR_DEVICE_H
#define GEBSR2CSR_DEVICE_H

#include <hip/hip_runtime.h>

template <rocsparse_direction DIRECTION,
          rocsparse_int       BLOCK_SIZE,
          rocsparse_int       WF_SIZE,
          typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void gebsr2csr_kernel(rocsparse_int        mb,
                          rocsparse_int        nb,
                          rocsparse_index_base bsr_base,
                          const T* __restrict__ bsr_val,
                          const rocsparse_int* __restrict__ bsr_row_ptr,
                          const rocsparse_int* __restrict__ bsr_col_ind,
                          rocsparse_int        row_block_dim,
                          rocsparse_int        col_block_dim,
                          rocsparse_index_base csr_base,
                          T* __restrict__ csr_val,
                          rocsparse_int* __restrict__ csr_row_ptr,
                          rocsparse_int* __restrict__ csr_col_ind)
{
    rocsparse_int entries_in_block = row_block_dim * col_block_dim;

    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
    rocsparse_int warp_id   = thread_id / WF_SIZE;
    rocsparse_int lane_id   = thread_id % WF_SIZE;

    if(warp_id >= mb * row_block_dim)
    { // one warp per row in matrix
        return;
    }

    rocsparse_int block_row    = warp_id / row_block_dim; // block row in bsr matrix
    rocsparse_int row_in_block = warp_id % row_block_dim; // local row in bsr row block

    rocsparse_int bsr_row_start = bsr_row_ptr[block_row] - bsr_base;
    rocsparse_int bsr_row_end   = bsr_row_ptr[block_row + 1] - bsr_base;

    rocsparse_int entries_in_row = (bsr_row_end - bsr_row_start) * col_block_dim;
    rocsparse_int number_of_entries_in_prev_rows
        = bsr_row_start * entries_in_block + row_in_block * entries_in_row;

    if(warp_id == 0)
    {
        csr_row_ptr[0] = csr_base;
    }

    csr_row_ptr[warp_id + 1] = number_of_entries_in_prev_rows + entries_in_row + csr_base;

    for(rocsparse_int i = bsr_row_start + lane_id; i < bsr_row_end; i += WF_SIZE)
    {

        rocsparse_int col    = bsr_col_ind[i] - bsr_base;
        rocsparse_int offset = number_of_entries_in_prev_rows + col_block_dim * (i - bsr_row_start);

        for(rocsparse_int j = 0; j < col_block_dim; j++)
        {
            csr_col_ind[offset + j] = col_block_dim * col + j + csr_base;

            if(DIRECTION == rocsparse_direction_row)
            {
                csr_val[offset + j]
                    = bsr_val[i * entries_in_block + row_in_block * col_block_dim + j];
            }
            else
            {
                csr_val[offset + j]
                    = bsr_val[i * entries_in_block + row_in_block + row_block_dim * j];
            }
        }
    }
}

template <rocsparse_direction DIRECTION, rocsparse_int BLOCK_SIZE, rocsparse_int WF_SIZE>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void gebsr2csr_nnz_kernel(rocsparse_int        mb,
                              rocsparse_int        nb,
                              rocsparse_index_base bsr_base,
                              const rocsparse_int* __restrict__ bsr_row_ptr,
                              const rocsparse_int* __restrict__ bsr_col_ind,
                              rocsparse_int        row_block_dim,
                              rocsparse_int        col_block_dim,
                              rocsparse_index_base csr_base,
                              rocsparse_int* __restrict__ csr_row_ptr,
                              rocsparse_int* __restrict__ csr_col_ind)
{
    rocsparse_int entries_in_block = row_block_dim * col_block_dim;

    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
    rocsparse_int warp_id   = thread_id / WF_SIZE;
    rocsparse_int lane_id   = thread_id % WF_SIZE;

    if(warp_id >= mb * row_block_dim)
    { // one warp per row in matrix
        return;
    }

    rocsparse_int block_row    = warp_id / row_block_dim; // block row in bsr matrix
    rocsparse_int row_in_block = warp_id % row_block_dim; // local row in bsr row block

    rocsparse_int bsr_row_start = bsr_row_ptr[block_row] - bsr_base;
    rocsparse_int bsr_row_end   = bsr_row_ptr[block_row + 1] - bsr_base;

    rocsparse_int entries_in_row = (bsr_row_end - bsr_row_start) * col_block_dim;
    rocsparse_int number_of_entries_in_prev_rows
        = bsr_row_start * entries_in_block + row_in_block * entries_in_row;

    if(warp_id == 0)
    {
        csr_row_ptr[0] = csr_base;
    }

    csr_row_ptr[warp_id + 1] = number_of_entries_in_prev_rows + entries_in_row + csr_base;

    for(rocsparse_int i = bsr_row_start + lane_id; i < bsr_row_end; i += WF_SIZE)
    {

        rocsparse_int col    = bsr_col_ind[i] - bsr_base;
        rocsparse_int offset = number_of_entries_in_prev_rows + col_block_dim * (i - bsr_row_start);

        for(rocsparse_int j = 0; j < col_block_dim; j++)
        {
            csr_col_ind[offset + j] = col_block_dim * col + j + csr_base;
        }
    }
}

#endif // GEBSR2CSR_DEVICE_H
