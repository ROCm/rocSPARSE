/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef BSRPAD_VALUE_DEVICE_H
#define BSRPAD_VALUE_DEVICE_H

#include "common.h"

template <rocsparse_int BLOCK_SIZE, typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void bsrpad_value_kernel_sorted(rocsparse_int        m,
                                    rocsparse_int        mb,
                                    rocsparse_int        block_dim,
                                    T                    value,
                                    rocsparse_index_base bsr_base,
                                    T* __restrict__ bsr_val,
                                    const rocsparse_int* __restrict__ bsr_row_ptr,
                                    const rocsparse_int* __restrict__ bsr_col_ind)
{
    rocsparse_int block_id = hipBlockIdx_x;
    rocsparse_int local_id = hipThreadIdx_x;

    rocsparse_int i = block_id * BLOCK_SIZE + local_id;

    if(i < block_dim)
    {
        rocsparse_int start_local_index = m % block_dim;
        if((start_local_index > 0)
           && ((bsr_col_ind[(bsr_row_ptr[mb] - bsr_base) - 1] - bsr_base) == (mb - 1)))
        {
            if(i >= start_local_index)
            {
                bsr_val[((bsr_row_ptr[mb] - bsr_base) - 1) * block_dim * block_dim + i * block_dim
                        + i]
                    = value;
            }
        }
    }
}

template <rocsparse_int BLOCK_SIZE, typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void bsrpad_value_kernel_unsorted(rocsparse_int        m,
                                      rocsparse_int        mb,
                                      rocsparse_int        block_dim,
                                      T                    value,
                                      rocsparse_index_base bsr_base,
                                      T* __restrict__ bsr_val,
                                      const rocsparse_int* __restrict__ bsr_row_ptr,
                                      const rocsparse_int* __restrict__ bsr_col_ind)
{
    rocsparse_int block_id = hipBlockIdx_x;
    rocsparse_int local_id = hipThreadIdx_x;

    __shared__ rocsparse_int block_index;

    rocsparse_int i = block_id * BLOCK_SIZE + local_id;

    if(local_id == 0)
    {
        block_index = -1;
    }

    __syncthreads();

    //find block
    for(rocsparse_int index = local_id; bsr_row_ptr[mb - 1] + index < bsr_row_ptr[mb];
        index += hipBlockDim_x)
    {
        if(bsr_col_ind[(bsr_row_ptr[mb - 1] - bsr_base) + index] - bsr_base == mb - 1)
        {
            block_index = (bsr_row_ptr[mb - 1] - bsr_base) + index;
        }
    }

    __syncthreads();

    if(block_index >= 0)
    {
        if(i < block_dim)
        {
            rocsparse_int start_local_index = m % block_dim;
            if(start_local_index > 0)
            {
                if(i >= start_local_index)
                {
                    bsr_val[block_index * block_dim * block_dim + i * block_dim + i] = value;
                }
            }
        }
    }
}

#endif // BSRPAD_VALUE_DEVICE_H
