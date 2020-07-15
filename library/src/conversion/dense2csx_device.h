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
#ifndef DENSE2CSX_DEVICE_H
#define DENSE2CSX_DEVICE_H

#include "handle.h"
#include <hip/hip_runtime.h>
#include <iostream>

template <rocsparse_int NUMROWS_PER_BLOCK, rocsparse_int WF_SIZE, typename T>
__launch_bounds__(WF_SIZE* NUMROWS_PER_BLOCK) static __global__
    void dense2csr_kernel(rocsparse_int base,
                          rocsparse_int m,
                          rocsparse_int n,
                          const T* __restrict__ dense_val,
                          rocsparse_int ld,
                          T* __restrict__ csr_val,
                          rocsparse_int* __restrict__ csr_row_ptr,
                          rocsparse_int* __restrict__ csr_col_ind)
{
    const rocsparse_int wavefront_index = hipThreadIdx_x / WF_SIZE,
                        lane_index      = hipThreadIdx_x % WF_SIZE;
    const uint64_t      filter          = 0xffffffffffffffff >> (63 - lane_index);
    const rocsparse_int row_index       = NUMROWS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

    if(row_index < m)
    {
        rocsparse_int shift = csr_row_ptr[row_index] - base;
        //
        // The warp handles the entire row.
        //
        for(rocsparse_int column_index = lane_index; column_index < n; column_index += WF_SIZE)
        {
            //
            // Synchronize for cache considerations.
            //
            __syncthreads();

            //
            // Get value.
            //
            const T value = dense_val[row_index + column_index * ld];

            //
            // Predicate.
            //
            const bool predicate = value != 0;

            //
            // Mask of the wavefront.
            //
            const uint64_t wavefront_mask = __ballot(predicate);

            //
            // Get the number of previous non-zero in the row.
            //
            const uint64_t count_previous_nnzs = __popcll(wavefront_mask & filter);

            if(predicate)
            {
                //
                // Calculate local index.
                //
                const uint64_t local_index_in_warp = count_previous_nnzs - 1;

                //
                // Populate the sparse matrix.
                //
                csr_val[shift + local_index_in_warp]     = value;
                csr_col_ind[shift + local_index_in_warp] = column_index + base;
            }

            //
            // Broadcast the update of the shift to all 64 threads for the next set of 64 columns.
            // Choose the last lane since that it contains the size of the sparse row (even if its predicate is false).
            //
            shift += __shfl(static_cast<int>(count_previous_nnzs), WF_SIZE - 1);
        }
    }
}
template <rocsparse_int NUMCOLUMNS_PER_BLOCK, rocsparse_int WF_SIZE, typename T>
__launch_bounds__(WF_SIZE* NUMCOLUMNS_PER_BLOCK) static __global__
    void dense2csc_kernel(rocsparse_int base,
                          rocsparse_int m,
                          rocsparse_int n,
                          const T* __restrict__ dense_val,
                          rocsparse_int ld,
                          T* __restrict__ csc_val,
                          rocsparse_int* __restrict__ csc_col_ptr,
                          rocsparse_int* __restrict__ csc_row_ind)
{
    const rocsparse_int wavefront_index = hipThreadIdx_x / WF_SIZE,
                        lane_index      = hipThreadIdx_x % WF_SIZE;
    const uint64_t      filter          = 0xffffffffffffffff >> (63 - lane_index);
    const rocsparse_int column_index    = NUMCOLUMNS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

    if(column_index < n)
    {

        rocsparse_int shift = csc_col_ptr[column_index] - base;
        //
        // The warp handles the entire column.
        //
        dense_val += column_index * ld;
        for(rocsparse_int row_index = lane_index; row_index < m; row_index += WF_SIZE)
        {

            //
            // Get value.
            //
            const T value = dense_val[row_index];

            //
            // Predicate.
            //
            const bool predicate = value != 0;

            //
            // Mask of the wavefront.
            //
            const uint64_t wavefront_mask = __ballot(predicate);

            //
            // Get the number of previous non-zero in the row.
            //
            const uint64_t count_previous_nnzs = __popcll(wavefront_mask & filter);

            //
            // Synchronize for cache considerations.
            //
            __syncthreads();

            if(predicate)
            {
                //
                // Calculate local index.
                //
                const uint64_t local_index = count_previous_nnzs - 1;

                //
                // Populate the sparse matrix.
                //
                csc_val[shift + local_index]     = value;
                csc_row_ind[shift + local_index] = row_index + base;
            }

            //
            // Broadcast the update of the shift to all 64 threads for the next set of 64 columns.
            // Choose the last lane since that it contains the size of the sparse row (even if its predicate is false).
            //
            shift += __shfl(static_cast<int>(count_previous_nnzs), WF_SIZE - 1);
        }
    }
}

#endif
