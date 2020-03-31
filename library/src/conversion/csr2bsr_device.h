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
#ifndef CSR2BSR_DEVICE_H
#define CSR2BSR_DEVICE_H

#include "common.h"

#include <hip/hip_runtime.h>

template <rocsparse_int BLOCK_SIZE, rocsparse_int BSR_BLOCK_DIM, rocsparse_int WF_SEGMENT_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void csr2bsr_nnz_fast_kernel(rocsparse_int        m,
                                 rocsparse_int        n,
                                 rocsparse_int        mb,
                                 rocsparse_int        nb,
                                 rocsparse_index_base csr_base,
                                 const rocsparse_int* __restrict__ csr_row_ptr,
                                 const rocsparse_int* __restrict__ csr_col_ind,
                                 rocsparse_index_base bsr_base,
                                 rocsparse_int* __restrict__ bsr_row_ptr)
{
    constexpr rocsparse_int WF_PER_BLOCK       = (BLOCK_SIZE / 64);
    constexpr rocsparse_int SEGMENTS_PER_WF    = (64 / WF_SEGMENT_SIZE);
    constexpr rocsparse_int SEGMENTS_PER_BLOCK = (BLOCK_SIZE / WF_SEGMENT_SIZE);

    rocsparse_int block_id = hipBlockIdx_x;
    rocsparse_int warp_id  = hipThreadIdx_x / 64;

    rocsparse_int warp_segment_id      = (hipThreadIdx_x / WF_SEGMENT_SIZE) % SEGMENTS_PER_WF;
    rocsparse_int warp_segment_lane_id = hipThreadIdx_x % WF_SEGMENT_SIZE;
    rocsparse_int block_segment_id     = SEGMENTS_PER_WF * warp_id + warp_segment_id;

    rocsparse_int index = SEGMENTS_PER_BLOCK * BSR_BLOCK_DIM * block_id
                          + SEGMENTS_PER_WF * BSR_BLOCK_DIM * warp_id
                          + BSR_BLOCK_DIM * warp_segment_id + warp_segment_lane_id;

    rocsparse_int row_start = 0;
    rocsparse_int row_end   = 0;

    if(index < m && warp_segment_lane_id < BSR_BLOCK_DIM)
    {
        row_start = csr_row_ptr[index] - csr_base;
        row_end   = csr_row_ptr[index + 1] - csr_base;
    }

    rocsparse_int col          = 0;
    rocsparse_int nnzb_per_row = 0;

    while(col < nb)
    {
        // Find minimum column index that is also greater than or equal to column
        rocsparse_int min_col_ind = nb;

        for(rocsparse_int i = row_start; i < row_end; i++)
        {
            rocsparse_int col_ind = (csr_col_ind[i] - csr_base) / BSR_BLOCK_DIM;
            if(col_ind >= col)
            {
                min_col_ind = col_ind;
                row_start   = i;
                break;
            }
        }

        // last thread in segment will contain the min after this call
        rocsparse_wfreduce_min<WF_SEGMENT_SIZE>(&min_col_ind);

        // broadcast min_col_ind from last thread in segment to all threads in segment
        min_col_ind = __shfl(min_col_ind, WF_SEGMENT_SIZE - 1, WF_SEGMENT_SIZE);

        col = min_col_ind + 1;

        if(warp_segment_lane_id == WF_SEGMENT_SIZE - 1)
        {
            if(min_col_ind < nb)
            {
                nnzb_per_row++;
            }
        }
    }

    if(SEGMENTS_PER_BLOCK * block_id + block_segment_id < mb
       && warp_segment_lane_id == WF_SEGMENT_SIZE - 1)
    {
        bsr_row_ptr[0]                                                    = bsr_base;
        bsr_row_ptr[SEGMENTS_PER_BLOCK * block_id + block_segment_id + 1] = nnzb_per_row;
    }
}

// This implementation is only called for large BSR block dimensions. For any realiztic choice of
// block dimension, the fast version above can be called instead but we need this to cover
// exceptional cases like choosing the block dimension to be the same as the full matrix dimension etc.
__global__ void csr2bsr_nnz_slow_kernel(rocsparse_int        m,
                                        rocsparse_int        n,
                                        rocsparse_int        nb,
                                        rocsparse_index_base csr_base,
                                        const rocsparse_int* __restrict__ csr_row_ptr,
                                        const rocsparse_int* __restrict__ csr_col_ind,
                                        rocsparse_int        block_dim,
                                        rocsparse_index_base bsr_base,
                                        rocsparse_int* __restrict__ bsr_row_ptr,
                                        rocsparse_int* __restrict__ temp)
{
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    __shared__ rocsparse_int snnzb_in_row;
    __shared__ rocsparse_int scolumn;

    snnzb_in_row = 0;
    scolumn      = 0;

    rocsparse_int row_start = 0;
    rocsparse_int row_end   = 0;

    if(thread_id < m)
    {
        row_start = csr_row_ptr[thread_id] - csr_base;
        row_end   = csr_row_ptr[thread_id + 1] - csr_base;
    }

    temp += block_dim * hipBlockIdx_x;

    while(scolumn < nb)
    {
        // Find minimum column index that is also greater than or equal to scolumn
        rocsparse_int min_col_ind = nb;
        for(rocsparse_int i = row_start; i < row_end; i++)
        {
            rocsparse_int col_ind = (csr_col_ind[i] - csr_base) / block_dim;
            if(col_ind >= scolumn)
            {
                min_col_ind = col_ind;
                row_start   = i;
                break;
            }
        }

        temp[hipThreadIdx_x] = min_col_ind;

        __syncthreads();

        if(hipThreadIdx_x == 0)
        {
            rocsparse_int min = temp[0];
            for(rocsparse_int i = 1; i < block_dim; i++)
            {
                min = (min < temp[hipThreadIdx_x + i] ? min : temp[hipThreadIdx_x + i]);
            }

            scolumn = min + 1;

            if(min < nb)
            {
                snnzb_in_row++;
            }
        }

        __syncthreads();
    }

    if(hipThreadIdx_x == 0)
    {
        bsr_row_ptr[0]                 = bsr_base;
        bsr_row_ptr[hipBlockIdx_x + 1] = snnzb_in_row;
    }
}

__global__ void
    csr2bsr_nnz_block_dim_equals_one_kernel(rocsparse_int        m,
                                            rocsparse_index_base csr_base,
                                            const rocsparse_int* __restrict__ csr_row_ptr,
                                            rocsparse_index_base bsr_base,
                                            rocsparse_int* __restrict__ bsr_row_ptr,
                                            rocsparse_int* __restrict__ bsr_nnz)
{
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    if(thread_id < m + 1)
    {
        bsr_row_ptr[thread_id] = (csr_row_ptr[thread_id] - csr_base) + bsr_base;
    }

    if(thread_id == 0)
    {
        *bsr_nnz = csr_row_ptr[m] - csr_row_ptr[0];
    }
}

__global__ void
    csr2bsr_nnz_block_dim_equals_one_kernel(rocsparse_int        m,
                                            rocsparse_index_base csr_base,
                                            const rocsparse_int* __restrict__ csr_row_ptr,
                                            rocsparse_index_base bsr_base,
                                            rocsparse_int* __restrict__ bsr_row_ptr)
{
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    if(thread_id < m + 1)
    {
        bsr_row_ptr[thread_id] = (csr_row_ptr[thread_id] - csr_base) + bsr_base;
    }
}

__global__ void csr2bsr_nnz_compute_nnz_total_kernel(rocsparse_int mb,
                                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                                     rocsparse_int* __restrict__ bsr_nnz)
{
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    if(thread_id == 0)
    {
        *bsr_nnz = bsr_row_ptr[mb] - bsr_row_ptr[0];
    }
}

template <typename T,
          rocsparse_direction DIRECTION,
          rocsparse_int       BLOCK_SIZE,
          rocsparse_int       BSR_BLOCK_DIM,
          rocsparse_int       WF_SEGMENT_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void csr2bsr_fast_kernel(rocsparse_int              m,
                             rocsparse_int              n,
                             rocsparse_int              mb,
                             rocsparse_int              nb,
                             const rocsparse_index_base csr_base,
                             const T* __restrict__ csr_val,
                             const rocsparse_int* __restrict__ csr_row_ptr,
                             const rocsparse_int* __restrict__ csr_col_ind,
                             const rocsparse_index_base bsr_base,
                             T* __restrict__ bsr_val,
                             rocsparse_int* __restrict__ bsr_row_ptr,
                             rocsparse_int* __restrict__ bsr_col_ind)
{
    constexpr rocsparse_int WF_PER_BLOCK       = (BLOCK_SIZE / 64);
    constexpr rocsparse_int SEGMENTS_PER_WF    = (64 / WF_SEGMENT_SIZE);
    constexpr rocsparse_int SEGMENTS_PER_BLOCK = (BLOCK_SIZE / WF_SEGMENT_SIZE);

    rocsparse_int block_id = hipBlockIdx_x;
    rocsparse_int warp_id  = hipThreadIdx_x / 64;

    rocsparse_int warp_segment_id      = (hipThreadIdx_x / WF_SEGMENT_SIZE) % SEGMENTS_PER_WF;
    rocsparse_int warp_segment_lane_id = hipThreadIdx_x % WF_SEGMENT_SIZE;
    rocsparse_int block_segment_id     = SEGMENTS_PER_WF * warp_id + warp_segment_id;

    rocsparse_int index = SEGMENTS_PER_BLOCK * BSR_BLOCK_DIM * block_id
                          + SEGMENTS_PER_WF * BSR_BLOCK_DIM * warp_id
                          + BSR_BLOCK_DIM * warp_segment_id + warp_segment_lane_id;

    rocsparse_int row_start = 0;
    rocsparse_int row_end   = 0;

    if(index < m && warp_segment_lane_id < BSR_BLOCK_DIM)
    {
        row_start = csr_row_ptr[index] - csr_base;
        row_end   = csr_row_ptr[index + 1] - csr_base;
    }

    rocsparse_int bsr_row_start = 0;

    if(SEGMENTS_PER_BLOCK * block_id + block_segment_id < mb)
    {
        bsr_row_start = bsr_row_ptr[SEGMENTS_PER_BLOCK * block_id + block_segment_id] - bsr_base;
    }

    rocsparse_int csr_col      = 0;
    rocsparse_int bsr_col      = 0;
    rocsparse_int nnzb_per_row = 0;

    while(csr_col < n)
    {
        // For each CSR row, find minimum CSR column index that is also greater than or equal to csr_col
        T             csr_value     = 0;
        rocsparse_int csr_col_index = n;

        T             min_csr_value     = 0;
        rocsparse_int min_csr_col_index = n;

        for(rocsparse_int i = row_start; i < row_end; i++)
        {
            csr_value     = csr_val[i];
            csr_col_index = csr_col_ind[i] - csr_base;

            if(csr_col_index >= csr_col)
            {
                min_csr_value     = csr_value;
                min_csr_col_index = csr_col_index;
                row_start         = i;
                break;
            }
        }

        // find minimum CSR column index across all threads in this segment and store in last thread of segment
        rocsparse_wfreduce_min<WF_SEGMENT_SIZE>(&min_csr_col_index);

        // have last thread in segment write to CSR column indices array
        if(min_csr_col_index < n && warp_segment_lane_id == WF_SEGMENT_SIZE - 1)
        {
            if((min_csr_col_index / BSR_BLOCK_DIM) >= bsr_col)
            {
                bsr_col_ind[bsr_row_start + nnzb_per_row]
                    = min_csr_col_index / BSR_BLOCK_DIM + bsr_base;

                nnzb_per_row++;
                bsr_col = (min_csr_col_index / BSR_BLOCK_DIM) + 1;
            }
        }

        //__syncthreads();

        // broadcast CSR minimum column index from last thread in segment to all threads in segment
        min_csr_col_index = __shfl(min_csr_col_index, WF_SEGMENT_SIZE - 1, WF_SEGMENT_SIZE);

        // broadcast nnzb_per_row from last thread in segment to all threads in segment
        nnzb_per_row = __shfl(nnzb_per_row, WF_SEGMENT_SIZE - 1, WF_SEGMENT_SIZE);

        if(csr_col_index < n && csr_col_index / BSR_BLOCK_DIM == min_csr_col_index / BSR_BLOCK_DIM)
        {
            if(DIRECTION == rocsparse_direction_row)
            {
                rocsparse_int j = (bsr_row_start + nnzb_per_row - 1) * BSR_BLOCK_DIM * BSR_BLOCK_DIM
                                  + warp_segment_lane_id * BSR_BLOCK_DIM
                                  + csr_col_index % BSR_BLOCK_DIM;

                bsr_val[j] = csr_value;
            }
            else
            {
                rocsparse_int j = (bsr_row_start + nnzb_per_row - 1) * BSR_BLOCK_DIM * BSR_BLOCK_DIM
                                  + (csr_col_index % BSR_BLOCK_DIM) * BSR_BLOCK_DIM
                                  + warp_segment_lane_id;
                bsr_val[j] = csr_value;
            }
        }

        // update csr_col for all threads in segment
        csr_col = min_csr_col_index + 1;
    }
}

// This implementation is only called for large BSR block dimensions. For any realiztic choice of
// block dimension, the fast version above can be called instead but we need this to cover
// exceptional cases like choosing the block dimension to be the same as the full matrix dimension etc.
template <typename T>
__global__ void csr2bsr_slow_kernel(rocsparse_direction        direction,
                                    rocsparse_int              m,
                                    rocsparse_int              n,
                                    rocsparse_int              mb,
                                    rocsparse_int              nb,
                                    const rocsparse_index_base csr_base,
                                    const T* __restrict__ csr_val,
                                    const rocsparse_int* __restrict__ csr_row_ptr,
                                    const rocsparse_int* __restrict__ csr_col_ind,
                                    rocsparse_int              block_dim,
                                    const rocsparse_index_base bsr_base,
                                    T* __restrict__ bsr_val,
                                    rocsparse_int* __restrict__ bsr_row_ptr,
                                    rocsparse_int* __restrict__ bsr_col_ind,
                                    rocsparse_int* __restrict__ temp)
{
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
    rocsparse_int block_id  = hipBlockIdx_x;

    __shared__ rocsparse_int snnzb_per_row;
    __shared__ rocsparse_int scsr_col;
    __shared__ rocsparse_int sbsr_col;

    snnzb_per_row = 0;
    scsr_col      = 0;
    sbsr_col      = 0;

    rocsparse_int row_start = 0;
    rocsparse_int row_end   = 0;

    if(thread_id < m)
    {
        row_start = csr_row_ptr[thread_id] - csr_base;
        row_end   = csr_row_ptr[thread_id + 1] - csr_base;
    }

    rocsparse_int bsr_row_start = 0;

    if(block_id < mb)
    {
        bsr_row_start = bsr_row_ptr[block_id] - bsr_base;
    }

    temp += block_dim * block_id;

    while(scsr_col < n)
    {
        // Find minimum CSR column index that is also greater than or equal to csr_col
        T             csr_value     = 0;
        rocsparse_int csr_col_index = n;

        T             min_csr_value     = 0;
        rocsparse_int min_csr_col_index = n;

        for(rocsparse_int i = row_start; i < row_end; i++)
        {
            csr_value     = csr_val[i];
            csr_col_index = csr_col_ind[i] - csr_base;

            if(csr_col_index >= scsr_col)
            {
                min_csr_value     = csr_value;
                min_csr_col_index = csr_col_index;
                row_start         = i;
                break;
            }
        }

        temp[hipThreadIdx_x] = min_csr_col_index;

        __syncthreads();

        // have first thread find min for block and write to BSR column indices array
        if(hipThreadIdx_x == 0)
        {
            rocsparse_int min = temp[0];
            for(rocsparse_int i = 1; i < block_dim; i++)
            {
                min = (min < temp[hipThreadIdx_x + i] ? min : temp[hipThreadIdx_x + i]);
            }

            if(min < n)
            {
                if((min / block_dim) >= sbsr_col)
                {
                    bsr_col_ind[bsr_row_start + snnzb_per_row] = min / block_dim + bsr_base;

                    snnzb_per_row++;
                    sbsr_col = (min / block_dim) + 1;
                }
            }

            scsr_col = min;
        }

        __syncthreads();

        // update min_csr_col_index for all threads in block
        min_csr_col_index = scsr_col;

        if(csr_col_index < n && csr_col_index / block_dim == min_csr_col_index / block_dim)
        {
            if(direction == rocsparse_direction_row)
            {
                rocsparse_int j = (bsr_row_start + snnzb_per_row - 1) * block_dim * block_dim
                                  + hipThreadIdx_x * block_dim + csr_col_index % block_dim;

                bsr_val[j] = csr_value;
            }
            else
            {
                rocsparse_int j = (bsr_row_start + snnzb_per_row - 1) * block_dim * block_dim
                                  + (csr_col_index % block_dim) * block_dim + hipThreadIdx_x;
                bsr_val[j] = csr_value;
            }
        }

        // update scsr_col for a threads in block
        scsr_col = min_csr_col_index + 1;

        __syncthreads();
    }
}

template <typename T, rocsparse_int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void csr2bsr_block_dim_equals_one_kernel(rocsparse_int              m,
                                             rocsparse_int              n,
                                             rocsparse_int              mb,
                                             rocsparse_int              nb,
                                             const rocsparse_index_base csr_base,
                                             const T*                   csr_val,
                                             const rocsparse_int*       csr_row_ptr,
                                             const rocsparse_int*       csr_col_ind,
                                             const rocsparse_index_base bsr_base,
                                             T*                         bsr_val,
                                             rocsparse_int*             bsr_row_ptr,
                                             rocsparse_int*             bsr_col_ind)
{
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    rocsparse_int nnz = csr_row_ptr[m] - csr_row_ptr[0];

    rocsparse_int index = thread_id;
    while(index < nnz)
    {
        bsr_col_ind[index] = (csr_col_ind[index] - csr_base) + bsr_base;
        bsr_val[index]     = csr_val[index];

        index += hipBlockDim_x * hipGridDim_x;
    }
}

#endif // CSR2BSR_DEVICE_H
