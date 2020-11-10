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
#ifndef CSX2DENSE2_DEVICE_H
#define CSX2DENSE2_DEVICE_H

#include "handle.h"
#include <hip/hip_runtime.h>

template <rocsparse_int NUMROWS_PER_BLOCK, rocsparse_int WF_SIZE, typename T>
__launch_bounds__(WF_SIZE* NUMROWS_PER_BLOCK) __global__
    void csr2dense_kernel(rocsparse_int base,
                          rocsparse_int m,
                          rocsparse_int n,
                          const T* __restrict__ csr_val,
                          const rocsparse_int* __restrict__ csr_row_ptr,
                          const rocsparse_int* __restrict__ csr_col_ind,
                          T* __restrict__ dense_val,
                          rocsparse_int ld)
{
    const rocsparse_int wavefront_index = hipThreadIdx_x / WF_SIZE,
                        lane_index      = hipThreadIdx_x % WF_SIZE;
    const rocsparse_int row_index       = NUMROWS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

    if(row_index < m)
    {
        rocsparse_int       shift    = csr_row_ptr[row_index] - base;
        const rocsparse_int size_row = csr_row_ptr[row_index + 1] - base - shift;

        //
        // One wavefront executes one sparse row.
        //
        for(rocsparse_int index = lane_index; index < size_row; index += WF_SIZE)
        {
            __syncthreads();
            const rocsparse_int column_index         = csr_col_ind[shift + index] - base;
            dense_val[column_index * ld + row_index] = csr_val[shift + index];
        }
    }
}

template <rocsparse_int NUMCOLUMNS_PER_BLOCK, rocsparse_int WF_SIZE, typename T>
__launch_bounds__(WF_SIZE* NUMCOLUMNS_PER_BLOCK) __global__
    void csc2dense_kernel(rocsparse_int base,
                          rocsparse_int m,
                          rocsparse_int n,
                          const T* __restrict__ csc_val,
                          const rocsparse_int* __restrict__ csc_col_ptr,
                          const rocsparse_int* __restrict__ csc_row_ind,
                          T* __restrict__ dense_val,
                          rocsparse_int ld)
{
    const rocsparse_int wavefront_index = hipThreadIdx_x / WF_SIZE,
                        lane_index      = hipThreadIdx_x % WF_SIZE;
    const rocsparse_int column_index    = NUMCOLUMNS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

    if(column_index < n)
    {
        const rocsparse_int shift       = csc_col_ptr[column_index] - base;
        const rocsparse_int size_column = csc_col_ptr[column_index + 1] - base - shift;

        //
        // One wavefront executes one sparse column.
        //
        for(rocsparse_int index = lane_index; index < size_column; index += WF_SIZE)
        {
            const rocsparse_int row_index            = csc_row_ind[shift + index] - base;
            dense_val[column_index * ld + row_index] = csc_val[shift + index];
        }
    }
}

#endif
