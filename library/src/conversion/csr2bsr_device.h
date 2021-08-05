/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

template <rocsparse_int BLOCK_SIZE, rocsparse_int WF_SEGMENT_SIZE>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void csr2bsr_nnz_2_32_kernel(rocsparse_int        m,
                                 rocsparse_int        n,
                                 rocsparse_int        mb,
                                 rocsparse_int        nb,
                                 rocsparse_int        block_dim,
                                 rocsparse_index_base csr_base,
                                 const rocsparse_int* __restrict__ csr_row_ptr,
                                 const rocsparse_int* __restrict__ csr_col_ind,
                                 rocsparse_index_base bsr_base,
                                 rocsparse_int* __restrict__ bsr_row_ptr)
{
    constexpr rocsparse_int SEGMENTS_PER_BLOCK = (BLOCK_SIZE / WF_SEGMENT_SIZE);

    rocsparse_int block_id = hipBlockIdx_x;

    rocsparse_int wf_segment_id      = (hipThreadIdx_x / WF_SEGMENT_SIZE) % SEGMENTS_PER_BLOCK;
    rocsparse_int wf_segment_lane_id = hipThreadIdx_x % WF_SEGMENT_SIZE;

    rocsparse_int row = SEGMENTS_PER_BLOCK * block_dim * block_id + block_dim * wf_segment_id
                        + wf_segment_lane_id;

    rocsparse_int row_start = 0;
    rocsparse_int row_end   = 0;

    if(row < m && wf_segment_lane_id < block_dim)
    {
        row_start = csr_row_ptr[row] - csr_base;
        row_end   = csr_row_ptr[row + 1] - csr_base;
    }

    rocsparse_int block_col    = 0;
    rocsparse_int nnzb_per_row = 0;

    while(block_col < nb)
    {
        // Find minimum column index that is also greater than or equal to column
        rocsparse_int min_block_col_index = nb;

        for(rocsparse_int i = row_start; i < row_end; i++)
        {
            rocsparse_int col_index = (csr_col_ind[i] - csr_base) / block_dim;
            if(col_index >= block_col)
            {
                min_block_col_index = col_index;
                row_start           = i;
                break;
            }
        }

        // last thread in segment will contain the min after this call
        rocsparse_wfreduce_min<WF_SEGMENT_SIZE>(&min_block_col_index);

        // broadcast min_block_col_index from last thread in segment to all threads in segment
        min_block_col_index = __shfl(min_block_col_index, WF_SEGMENT_SIZE - 1, WF_SEGMENT_SIZE);

        block_col = min_block_col_index + 1;

        if(wf_segment_lane_id == WF_SEGMENT_SIZE - 1)
        {
            if(min_block_col_index < nb)
            {
                nnzb_per_row++;
            }
        }
    }

    if(SEGMENTS_PER_BLOCK * block_id + wf_segment_id < mb
       && wf_segment_lane_id == WF_SEGMENT_SIZE - 1)
    {
        bsr_row_ptr[0]                                                 = bsr_base;
        bsr_row_ptr[SEGMENTS_PER_BLOCK * block_id + wf_segment_id + 1] = nnzb_per_row;
    }
}

template <rocsparse_direction DIRECTION,
          rocsparse_int       BLOCK_SIZE,
          rocsparse_int       WF_SEGMENT_SIZE,
          typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void csr2bsr_2_32_kernel(rocsparse_int              m,
                             rocsparse_int              n,
                             rocsparse_int              mb,
                             rocsparse_int              nb,
                             rocsparse_int              block_dim,
                             const rocsparse_index_base csr_base,
                             const T* __restrict__ csr_val,
                             const rocsparse_int* __restrict__ csr_row_ptr,
                             const rocsparse_int* __restrict__ csr_col_ind,
                             const rocsparse_index_base bsr_base,
                             T* __restrict__ bsr_val,
                             rocsparse_int* __restrict__ bsr_row_ptr,
                             rocsparse_int* __restrict__ bsr_col_ind)
{
    constexpr rocsparse_int SEGMENTS_PER_BLOCK = (BLOCK_SIZE / WF_SEGMENT_SIZE);

    rocsparse_int block_id = hipBlockIdx_x;

    rocsparse_int wf_segment_id      = (hipThreadIdx_x / WF_SEGMENT_SIZE) % SEGMENTS_PER_BLOCK;
    rocsparse_int wf_segment_lane_id = hipThreadIdx_x % WF_SEGMENT_SIZE;

    rocsparse_int row = SEGMENTS_PER_BLOCK * block_dim * block_id + block_dim * wf_segment_id
                        + wf_segment_lane_id;

    rocsparse_int row_start = 0;
    rocsparse_int row_end   = 0;

    if(row < m && wf_segment_lane_id < block_dim)
    {
        row_start = csr_row_ptr[row] - csr_base;
        row_end   = csr_row_ptr[row + 1] - csr_base;
    }

    rocsparse_int bsr_row_start = 0;

    if(SEGMENTS_PER_BLOCK * block_id + wf_segment_id < mb)
    {
        bsr_row_start = bsr_row_ptr[SEGMENTS_PER_BLOCK * block_id + wf_segment_id] - bsr_base;
    }

    rocsparse_int csr_col       = 0;
    rocsparse_int bsr_block_col = 0;
    rocsparse_int nnzb_per_row  = 0;

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
        if(min_csr_col_index < n && wf_segment_lane_id == WF_SEGMENT_SIZE - 1)
        {
            if((min_csr_col_index / block_dim) >= bsr_block_col)
            {
                bsr_col_ind[bsr_row_start + nnzb_per_row]
                    = min_csr_col_index / block_dim + bsr_base;

                nnzb_per_row++;
                bsr_block_col = (min_csr_col_index / block_dim) + 1;
            }
        }

        // broadcast CSR minimum column index from last thread in segment to all threads in segment
        min_csr_col_index = __shfl(min_csr_col_index, WF_SEGMENT_SIZE - 1, WF_SEGMENT_SIZE);

        // broadcast nnzb_per_row from last thread in segment to all threads in segment
        nnzb_per_row = __shfl(nnzb_per_row, WF_SEGMENT_SIZE - 1, WF_SEGMENT_SIZE);

        if(csr_col_index < n && csr_col_index / block_dim == min_csr_col_index / block_dim)
        {
            if(DIRECTION == rocsparse_direction_row)
            {
                rocsparse_int j = (bsr_row_start + nnzb_per_row - 1) * block_dim * block_dim
                                  + wf_segment_lane_id * block_dim + csr_col_index % block_dim;

                bsr_val[j] = csr_value;
            }
            else
            {
                rocsparse_int j = (bsr_row_start + nnzb_per_row - 1) * block_dim * block_dim
                                  + (csr_col_index % block_dim) * block_dim + wf_segment_lane_id;
                bsr_val[j] = csr_value;
            }
        }

        // update csr_col for all threads in segment
        csr_col = min_csr_col_index + 1;
    }
}

template <rocsparse_int BLOCK_SIZE, rocsparse_int ROWS_PER_SEGMENT>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void csr2bsr_nnz_33_64_kernel(rocsparse_int        m,
                                  rocsparse_int        n,
                                  rocsparse_int        mb,
                                  rocsparse_int        nb,
                                  rocsparse_int        block_dim,
                                  rocsparse_index_base csr_base,
                                  const rocsparse_int* __restrict__ csr_row_ptr,
                                  const rocsparse_int* __restrict__ csr_col_ind,
                                  rocsparse_index_base bsr_base,
                                  rocsparse_int* __restrict__ bsr_row_ptr)
{
    __shared__ rocsparse_int row_start[ROWS_PER_SEGMENT * BLOCK_SIZE];
    __shared__ rocsparse_int row_end[ROWS_PER_SEGMENT * BLOCK_SIZE];

    rocsparse_int block_id = hipBlockIdx_x;
    rocsparse_int lane_id  = hipThreadIdx_x;

    rocsparse_int block_col    = 0;
    rocsparse_int nnzb_per_row = 0;

    for(rocsparse_int j = 0; j < ROWS_PER_SEGMENT; j++)
    {
        row_start[BLOCK_SIZE * j + lane_id] = 0;
        row_end[BLOCK_SIZE * j + lane_id]   = 0;

        rocsparse_int row = block_dim * block_id + BLOCK_SIZE * j + lane_id;

        if(row < m && (BLOCK_SIZE * j + lane_id) < block_dim)
        {
            row_start[BLOCK_SIZE * j + lane_id] = csr_row_ptr[row] - csr_base;
            row_end[BLOCK_SIZE * j + lane_id]   = csr_row_ptr[row + 1] - csr_base;
        }
    }

    __threadfence_block();

    while(block_col < nb)
    {
        // Find minimum column index that is also greater than or equal to col
        rocsparse_int min_block_col_index = nb;

        for(rocsparse_int j = 0; j < ROWS_PER_SEGMENT; j++)
        {
            for(rocsparse_int i = row_start[BLOCK_SIZE * j + lane_id];
                i < row_end[BLOCK_SIZE * j + lane_id];
                i++)
            {
                rocsparse_int block_col_index = (csr_col_ind[i] - csr_base) / block_dim;

                if(block_col_index >= block_col)
                {
                    if(block_col_index <= min_block_col_index)
                    {
                        min_block_col_index = block_col_index;
                    }

                    row_start[BLOCK_SIZE * j + lane_id] = i;

                    break;
                }
            }
        }

        __threadfence_block();

        // last thread in segment will contain the min after this call
        rocsparse_wfreduce_min<BLOCK_SIZE>(&min_block_col_index);

        // broadcast min_block_col_index from last thread in segment to all threads in segment
        min_block_col_index = __shfl(min_block_col_index, BLOCK_SIZE - 1, BLOCK_SIZE);

        block_col = min_block_col_index + 1;

        if(lane_id == BLOCK_SIZE - 1)
        {
            if(min_block_col_index < nb)
            {
                nnzb_per_row++;
            }
        }
    }

    if(block_id < mb && lane_id == BLOCK_SIZE - 1)
    {
        bsr_row_ptr[0]            = bsr_base;
        bsr_row_ptr[block_id + 1] = nnzb_per_row;
    }
}

template <rocsparse_direction DIRECTION,
          rocsparse_int       BLOCK_SIZE,
          rocsparse_int       ROWS_PER_SEGMENT,
          typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void csr2bsr_33_64_kernel(rocsparse_int              m,
                              rocsparse_int              n,
                              rocsparse_int              mb,
                              rocsparse_int              nb,
                              rocsparse_int              block_dim,
                              const rocsparse_index_base csr_base,
                              const T* __restrict__ csr_val,
                              const rocsparse_int* __restrict__ csr_row_ptr,
                              const rocsparse_int* __restrict__ csr_col_ind,
                              const rocsparse_index_base bsr_base,
                              T* __restrict__ bsr_val,
                              rocsparse_int* __restrict__ bsr_row_ptr,
                              rocsparse_int* __restrict__ bsr_col_ind)
{
    __shared__ rocsparse_int row_start[ROWS_PER_SEGMENT * BLOCK_SIZE];
    __shared__ rocsparse_int row_end[ROWS_PER_SEGMENT * BLOCK_SIZE];
    __shared__ rocsparse_int csr_col_index[ROWS_PER_SEGMENT * BLOCK_SIZE];
    __shared__ T             csr_value[ROWS_PER_SEGMENT * BLOCK_SIZE];

    rocsparse_int block_id = hipBlockIdx_x;
    rocsparse_int lane_id  = hipThreadIdx_x;

    rocsparse_int bsr_row_start = 0;

    if(block_id < mb)
    {
        bsr_row_start = bsr_row_ptr[block_id] - bsr_base;
    }

    rocsparse_int csr_col       = 0;
    rocsparse_int bsr_block_col = 0;
    rocsparse_int nnzb_per_row  = 0;

    for(rocsparse_int j = 0; j < ROWS_PER_SEGMENT; j++)
    {
        row_start[BLOCK_SIZE * j + lane_id] = 0;
        row_end[BLOCK_SIZE * j + lane_id]   = 0;

        rocsparse_int row_index = block_dim * block_id + BLOCK_SIZE * j + lane_id;

        if(row_index < m && (BLOCK_SIZE * j + lane_id) < block_dim)
        {
            row_start[BLOCK_SIZE * j + lane_id] = csr_row_ptr[row_index] - csr_base;
            row_end[BLOCK_SIZE * j + lane_id]   = csr_row_ptr[row_index + 1] - csr_base;
        }
    }

    __threadfence_block();

    while(csr_col < n)
    {
        T             min_csr_value     = 0;
        rocsparse_int min_csr_col_index = n;

        for(rocsparse_int j = 0; j < ROWS_PER_SEGMENT; j++)
        {
            csr_value[BLOCK_SIZE * j + lane_id]     = 0;
            csr_col_index[BLOCK_SIZE * j + lane_id] = n;

            for(rocsparse_int i = row_start[BLOCK_SIZE * j + lane_id];
                i < row_end[BLOCK_SIZE * j + lane_id];
                i++)
            {
                csr_value[BLOCK_SIZE * j + lane_id]     = csr_val[i];
                csr_col_index[BLOCK_SIZE * j + lane_id] = csr_col_ind[i] - csr_base;

                if(csr_col_index[BLOCK_SIZE * j + lane_id] >= csr_col)
                {
                    if(csr_col_index[BLOCK_SIZE * j + lane_id] <= min_csr_col_index)
                    {
                        min_csr_value     = csr_value[BLOCK_SIZE * j + lane_id];
                        min_csr_col_index = csr_col_index[BLOCK_SIZE * j + lane_id];
                    }

                    row_start[BLOCK_SIZE * j + lane_id] = i;

                    break;
                }
            }
        }

        __threadfence_block();

        // find minimum CSR column index across all threads in this segment and store in last thread of segment
        rocsparse_wfreduce_min<BLOCK_SIZE>(&min_csr_col_index);

        // have last thread in segment write to BSR column indices array
        if(min_csr_col_index < n && lane_id == BLOCK_SIZE - 1)
        {
            if((min_csr_col_index / block_dim) >= bsr_block_col)
            {
                bsr_col_ind[bsr_row_start + nnzb_per_row]
                    = min_csr_col_index / block_dim + bsr_base;

                nnzb_per_row++;
                bsr_block_col = (min_csr_col_index / block_dim) + 1;
            }
        }

        // broadcast CSR minimum column index from last thread in segment to all threads in segment
        min_csr_col_index = __shfl(min_csr_col_index, BLOCK_SIZE - 1, BLOCK_SIZE);

        // broadcast nnzb_per_row from last thread in segment to all threads in segment
        nnzb_per_row = __shfl(nnzb_per_row, BLOCK_SIZE - 1, BLOCK_SIZE);

        // Write BSR values
        for(rocsparse_int j = 0; j < ROWS_PER_SEGMENT; j++)
        {
            if(csr_col_index[BLOCK_SIZE * j + lane_id] < n
               && csr_col_index[BLOCK_SIZE * j + lane_id] / block_dim
                      == min_csr_col_index / block_dim)
            {
                if(DIRECTION == rocsparse_direction_row)
                {
                    rocsparse_int k = (bsr_row_start + nnzb_per_row - 1) * block_dim * block_dim
                                      + (BLOCK_SIZE * j + lane_id) * block_dim
                                      + csr_col_index[BLOCK_SIZE * j + lane_id] % block_dim;

                    bsr_val[k] = csr_value[BLOCK_SIZE * j + lane_id];
                }
                else
                {
                    rocsparse_int k
                        = (bsr_row_start + nnzb_per_row - 1) * block_dim * block_dim
                          + (csr_col_index[BLOCK_SIZE * j + lane_id] % block_dim) * block_dim
                          + (BLOCK_SIZE * j + lane_id);
                    bsr_val[k] = csr_value[BLOCK_SIZE * j + lane_id];
                }
            }
        }

        // update csr_col for all threads in segment
        csr_col = min_csr_col_index + 1;
    }
}

template <rocsparse_int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void csr2bsr_nnz_65_inf_kernel(rocsparse_int        m,
                                   rocsparse_int        n,
                                   rocsparse_int        mb,
                                   rocsparse_int        nb,
                                   rocsparse_int        block_dim,
                                   rocsparse_int        rows_per_segment,
                                   rocsparse_index_base csr_base,
                                   const rocsparse_int* __restrict__ csr_row_ptr,
                                   const rocsparse_int* __restrict__ csr_col_ind,
                                   rocsparse_index_base bsr_base,
                                   rocsparse_int* __restrict__ bsr_row_ptr,
                                   rocsparse_int* __restrict__ temp1)
{
    rocsparse_int block_id = hipBlockIdx_x;
    rocsparse_int lane_id  = hipThreadIdx_x;

    rocsparse_int block_col    = 0;
    rocsparse_int nnzb_per_row = 0;

    // temp array used as global scratch pad
    rocsparse_int* row_start
        = temp1 + (2 * rows_per_segment * BLOCK_SIZE * block_id) + rows_per_segment * lane_id;
    rocsparse_int* row_end = temp1 + (2 * rows_per_segment * BLOCK_SIZE * block_id)
                             + rows_per_segment * BLOCK_SIZE + rows_per_segment * lane_id;

    for(rocsparse_int j = 0; j < rows_per_segment; j++)
    {
        row_start[j] = 0;
        row_end[j]   = 0;

        rocsparse_int row_index = block_dim * block_id + BLOCK_SIZE * j + lane_id;

        if(row_index < m && (BLOCK_SIZE * j + lane_id) < block_dim)
        {
            row_start[j] = csr_row_ptr[row_index] - csr_base;
            row_end[j]   = csr_row_ptr[row_index + 1] - csr_base;
        }
    }

    while(block_col < nb)
    {
        // Find minimum column index that is also greater than or equal to col
        rocsparse_int min_block_col_index = nb;

        for(rocsparse_int j = 0; j < rows_per_segment; j++)
        {
            for(rocsparse_int i = row_start[j]; i < row_end[j]; i++)
            {
                rocsparse_int block_col_index = (csr_col_ind[i] - csr_base) / block_dim;

                if(block_col_index >= block_col)
                {
                    if(block_col_index <= min_block_col_index)
                    {
                        min_block_col_index = block_col_index;
                    }

                    row_start[j] = i;

                    break;
                }
            }
        }

        // last thread in segment will contain the min after this call
        rocsparse_wfreduce_min<BLOCK_SIZE>(&min_block_col_index);

        // broadcast min_block_col_index from last thread in segment to all threads in segment
        min_block_col_index = __shfl(min_block_col_index, BLOCK_SIZE - 1, BLOCK_SIZE);

        block_col = min_block_col_index + 1;

        if(lane_id == BLOCK_SIZE - 1)
        {
            if(min_block_col_index < nb)
            {
                nnzb_per_row++;
            }
        }
    }

    if(block_id < mb && lane_id == BLOCK_SIZE - 1)
    {
        bsr_row_ptr[0]            = bsr_base;
        bsr_row_ptr[block_id + 1] = nnzb_per_row;
    }
}

template <rocsparse_int BLOCK_SIZE, typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void csr2bsr_65_inf_kernel(rocsparse_direction        direction,
                               rocsparse_int              m,
                               rocsparse_int              n,
                               rocsparse_int              mb,
                               rocsparse_int              nb,
                               rocsparse_int              block_dim,
                               rocsparse_int              rows_per_segment,
                               const rocsparse_index_base csr_base,
                               const T* __restrict__ csr_val,
                               const rocsparse_int* __restrict__ csr_row_ptr,
                               const rocsparse_int* __restrict__ csr_col_ind,
                               const rocsparse_index_base bsr_base,
                               T* __restrict__ bsr_val,
                               rocsparse_int* __restrict__ bsr_row_ptr,
                               rocsparse_int* __restrict__ bsr_col_ind,
                               rocsparse_int* __restrict__ temp1,
                               T* __restrict__ temp2)
{
    rocsparse_int block_id = hipBlockIdx_x;
    rocsparse_int lane_id  = hipThreadIdx_x;

    rocsparse_int bsr_row_start = 0;

    if(block_id < mb)
    {
        bsr_row_start = bsr_row_ptr[block_id] - bsr_base;
    }

    rocsparse_int csr_col       = 0;
    rocsparse_int bsr_block_col = 0;
    rocsparse_int nnzb_per_row  = 0;

    // temp arrays used as global scratch pad
    rocsparse_int* row_start
        = temp1 + (3 * rows_per_segment * BLOCK_SIZE * block_id) + rows_per_segment * lane_id;
    rocsparse_int* row_end = temp1 + (3 * rows_per_segment * BLOCK_SIZE * block_id)
                             + rows_per_segment * BLOCK_SIZE + rows_per_segment * lane_id;
    rocsparse_int* csr_col_index = temp1 + (3 * rows_per_segment * BLOCK_SIZE * block_id)
                                   + 2 * rows_per_segment * BLOCK_SIZE + rows_per_segment * lane_id;
    T* csr_value = temp2 + (rows_per_segment * BLOCK_SIZE * block_id) + rows_per_segment * lane_id;

    for(rocsparse_int j = 0; j < rows_per_segment; j++)
    {
        row_start[j] = 0;
        row_end[j]   = 0;

        rocsparse_int row_index = block_dim * block_id + BLOCK_SIZE * j + lane_id;

        if(row_index < m && (BLOCK_SIZE * j + lane_id) < block_dim)
        {
            row_start[j] = csr_row_ptr[row_index] - csr_base;
            row_end[j]   = csr_row_ptr[row_index + 1] - csr_base;
        }
    }

    while(csr_col < n)
    {
        T             min_csr_value     = 0;
        rocsparse_int min_csr_col_index = n;

        for(rocsparse_int j = 0; j < rows_per_segment; j++)
        {
            csr_value[j]     = 0;
            csr_col_index[j] = n;

            for(rocsparse_int i = row_start[j]; i < row_end[j]; i++)
            {
                csr_value[j]     = csr_val[i];
                csr_col_index[j] = csr_col_ind[i] - csr_base;

                if(csr_col_index[j] >= csr_col)
                {
                    if(csr_col_index[j] <= min_csr_col_index)
                    {
                        min_csr_value     = csr_value[j];
                        min_csr_col_index = csr_col_index[j];
                    }

                    row_start[j] = i;

                    break;
                }
            }
        }

        // find minimum CSR column index across all threads in this segment and store in last thread of segment
        rocsparse_wfreduce_min<BLOCK_SIZE>(&min_csr_col_index);

        // have last thread in segment write to BSR column indices array
        if(min_csr_col_index < n && lane_id == BLOCK_SIZE - 1)
        {
            if((min_csr_col_index / block_dim) >= bsr_block_col)
            {
                bsr_col_ind[bsr_row_start + nnzb_per_row]
                    = min_csr_col_index / block_dim + bsr_base;

                nnzb_per_row++;
                bsr_block_col = (min_csr_col_index / block_dim) + 1;
            }
        }

        // broadcast CSR minimum column index from last thread in segment to all threads in segment
        min_csr_col_index = __shfl(min_csr_col_index, BLOCK_SIZE - 1, BLOCK_SIZE);

        // broadcast nnzb_per_row from last thread in segment to all threads in segment
        nnzb_per_row = __shfl(nnzb_per_row, BLOCK_SIZE - 1, BLOCK_SIZE);

        // Write BSR values
        for(rocsparse_int j = 0; j < rows_per_segment; j++)
        {
            if(csr_col_index[j] < n
               && csr_col_index[j] / block_dim == min_csr_col_index / block_dim)
            {
                if(direction == rocsparse_direction_row)
                {
                    rocsparse_int k = (bsr_row_start + nnzb_per_row - 1) * block_dim * block_dim
                                      + (BLOCK_SIZE * j + lane_id) * block_dim
                                      + csr_col_index[j] % block_dim;

                    bsr_val[k] = csr_value[j];
                }
                else
                {
                    rocsparse_int k = (bsr_row_start + nnzb_per_row - 1) * block_dim * block_dim
                                      + (csr_col_index[j] % block_dim) * block_dim
                                      + (BLOCK_SIZE * j + lane_id);
                    bsr_val[k] = csr_value[j];
                }
            }
        }

        // update csr_col for all threads in segment
        csr_col = min_csr_col_index + 1;
    }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csr2bsr_nnz_block_dim_equals_one_kernel(rocsparse_int        m,
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

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csr2bsr_nnz_block_dim_equals_one_kernel(rocsparse_int        m,
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

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csr2bsr_nnz_compute_nnz_total_kernel(rocsparse_int mb,
                                              const rocsparse_int* __restrict__ bsr_row_ptr,
                                              rocsparse_int* __restrict__ bsr_nnz)
{
    rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    if(thread_id == 0)
    {
        *bsr_nnz = bsr_row_ptr[mb] - bsr_row_ptr[0];
    }
}

template <rocsparse_int BLOCK_SIZE, typename T>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
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
