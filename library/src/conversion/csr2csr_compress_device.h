/*! \file */
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
#ifndef CSR2CSR_COMPRESS_DEVICE_H
#define CSR2CSR_COMPRESS_DEVICE_H

#include <limits>

#include "common.h"

template <rocsparse_int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void fill_row_ptr_device(rocsparse_int        m,
                             rocsparse_index_base idx_base_C,
                             rocsparse_int* __restrict__ csr_row_ptr_C)
{
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(thread_id >= m)
    {
        return;
    }

    csr_row_ptr_C[thread_id + 1] = idx_base_C;

    if(thread_id == 0)
    {
        csr_row_ptr_C[0] = idx_base_C;
    }
}

template <rocsparse_int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void fill_row_ptr_device(rocsparse_int        m,
                             rocsparse_index_base idx_base_C,
                             const rocsparse_int* __restrict__ nnz_per_row,
                             rocsparse_int* __restrict__ csr_row_ptr_C)
{
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(thread_id >= m)
    {
        return;
    }

    csr_row_ptr_C[thread_id + 1] = nnz_per_row[thread_id];

    if(thread_id == 0)
    {
        csr_row_ptr_C[0] = idx_base_C;
    }
}

template <rocsparse_int BLOCK_SIZE,
          rocsparse_int SEGMENTS_PER_BLOCK,
          rocsparse_int SEGMENT_SIZE,
          rocsparse_int WF_SIZE,
          typename T>
__device__ void csr2csr_compress_device(rocsparse_int        m,
                                        rocsparse_int        n,
                                        rocsparse_index_base idx_base_A,
                                        const T* __restrict__ csr_val_A,
                                        const rocsparse_int* __restrict__ csr_row_ptr_A,
                                        const rocsparse_int* __restrict__ csr_col_ind_A,
                                        rocsparse_int        nnz_A,
                                        rocsparse_index_base idx_base_C,
                                        T* __restrict__ csr_val_C,
                                        const rocsparse_int* __restrict__ csr_row_ptr_C,
                                        rocsparse_int* __restrict__ csr_col_ind_C,
                                        T tol)
{
    const rocsparse_int segment_id      = hipThreadIdx_x / SEGMENT_SIZE;
    const rocsparse_int segment_lane_id = hipThreadIdx_x % SEGMENT_SIZE;

    const rocsparse_int id_of_segment_within_warp = segment_id % (WF_SIZE / SEGMENT_SIZE);

    const uint64_t filter_mask         = (0xffffffffffffffff >> (63 - segment_lane_id));
    const uint64_t shifted_filter_mask = filter_mask << (SEGMENT_SIZE * id_of_segment_within_warp);

    const rocsparse_int row_index = SEGMENTS_PER_BLOCK * hipBlockIdx_x + segment_id;

    if(row_index < m)
    {
        const rocsparse_int start_A = csr_row_ptr_A[row_index] - idx_base_A;
        const rocsparse_int end_A   = csr_row_ptr_A[row_index + 1] - idx_base_A;

        rocsparse_int start_C = csr_row_ptr_C[row_index] - idx_base_C;

        // One segment per row
        for(rocsparse_int i = start_A + segment_lane_id; i < end_A; i += SEGMENT_SIZE)
        {
            const T value = csr_val_A[i];

            // Check if value in matrix will be kept
            const bool predicate
                = rocsparse_abs(value) > rocsparse_real(tol)
                          && rocsparse_abs(value) > std::numeric_limits<float>::min()
                      ? true
                      : false;

            // Ballot operates on an entire warp (32 or 64 threads). Therefore the computed
            // wavefront_mask may contain information for multiple rows if the segment size is
            // less then the wavefront size. This is why we shift the filter mask above so that
            // when combined with wave_front_mask and popcll below it will give the correct
            // results for the particular row we are in.
            const uint64_t wavefront_mask = __ballot(predicate);

            // Get the number of retained matrix entries up to this thread in the segment
            const uint64_t count_previous_nnzs = __popcll(wavefront_mask & shifted_filter_mask);

            // If we are keeping the matrix entry, insert it into the compressed CSR matrix
            if(predicate)
            {
                csr_val_C[start_C + count_previous_nnzs - 1] = value;
                csr_col_ind_C[start_C + count_previous_nnzs - 1]
                    = (csr_col_ind_A[i] - idx_base_A) + idx_base_C;
            }

            // Broadcast the update of the start_C to all threads in the seegment. Choose the last
            // segment lane since that it contains the number of entries in the compressed sparse
            // row (even if its predicate is false).
            start_C
                += __shfl(static_cast<int>(count_previous_nnzs), SEGMENT_SIZE - 1, SEGMENT_SIZE);
        }
    }
}

#endif // CSR2CSR_COMPRESS_DEVICE_H
