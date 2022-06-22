/*! \file */
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef BSRPAD_IDENTITY_DEVICE_H
#define BSRPAD_IDENTITY_DEVICE_H

#include "common.h"

template <rocsparse_int BLOCK_SIZE, typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void bsrpad_identity_kernel(rocsparse_int        m,
                                rocsparse_int        n,
                                rocsparse_int        mb,
                                rocsparse_int        nb,
                                rocsparse_int        block_dim,
                                rocsparse_index_base bsr_base,
                                T* __restrict__ bsr_val,
                                const rocsparse_int* __restrict__ bsr_row_ptr,
                                const rocsparse_int* __restrict__ bsr_col_ind)
{
    rocsparse_int block_id = hipBlockIdx_x;
    rocsparse_int local_id = hipThreadIdx_x;

    rocsparse_int row = m / block_dim + block_id * BLOCK_SIZE + local_id;

    if(row < mb)
    {
        //search for diagonal block
        for(rocsparse_int i = bsr_row_ptr[row] - bsr_base; i < bsr_row_ptr[row + 1] - bsr_base; i++)
        {
            if(bsr_col_ind[i] - bsr_base == row)
            {
                for(rocsparse_int k = 0; k < block_dim; k++)
                {
                    if(row * block_dim + k >= m)
                    {
                        rocsparse_int j = i * block_dim * block_dim + k * block_dim + k;
                        if(bsr_val[j] == static_cast<T>(0))
                        {
                            bsr_val[j] = static_cast<T>(1);
                        }
                    }
                }

                break;
            }
        }
    }
}

#endif // BSRPAD_IDENTITY_DEVICE_H
