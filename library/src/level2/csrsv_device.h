/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"

extern "C" void __builtin_amdgcn_s_sleep(int);

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, bool SLEEP, typename I, typename J>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrsv_analysis_lower_kernel(J m,
                                     const I* __restrict__ csr_row_ptr,
                                     const J* __restrict__ csr_col_ind,
                                     I* __restrict__ csr_diag_ind,
                                     int* __restrict__ done_array,
                                     I* __restrict__ max_nnz,
                                     J* __restrict__ zero_pivot,
                                     rocsparse_index_base idx_base,
                                     rocsparse_diag_type  diag_type)
{
    int lid = hipThreadIdx_x & (WF_SIZE - 1);
    int wid = hipThreadIdx_x / WF_SIZE;

    // First row in this block
    J first_row = hipBlockIdx_x * (BLOCKSIZE / WF_SIZE);

    // Row that the wavefront will process
    J row = first_row + wid;

    // Shared memory to set done flag for intra-block dependencies
    __shared__ int local_done_array[BLOCKSIZE / WF_SIZE];

    // Initialize local done array
    local_done_array[wid] = 0;

    // Wait for initialization to finish
    __syncthreads();

    // Do not run out of bounds
    if(row >= m)
    {
        return;
    }

    // Initialize matrix diagonal index
    if(lid == 0)
    {
        csr_diag_ind[row] = -1;
    }

    // Local depth
    int local_max = 0;

    I row_begin = csr_row_ptr[row] - idx_base;
    I row_end   = csr_row_ptr[row + 1] - idx_base;

    // This wavefront operates on a single row, from its beginning to end.
    // First, we process all nodes that have dependencies outside the current block.
    J local_col = -1;
    I j;
    for(j = row_begin + lid; j < row_end; j += WF_SIZE)
    {
        // local_col will tell us, for this iteration of the above for loop
        // (i.e. for this entry in this row), which columns contain the
        // non-zero values. We must then ensure that the output from the row
        // associated with the local_col is complete to ensure that we can
        // calculate the right answer.
        local_col = rocsparse_nontemporal_load(csr_col_ind + j) - idx_base;

        // Skip all columns where corresponding row belongs to this block
        if(local_col >= first_row)
        {
            break;
        }

        // While there are threads in this workgroup that have been unable to
        // get their input, loop and wait for the flag to exist.
        int          local_done    = atomicOr(&done_array[local_col], 0);
        unsigned int times_through = 0;
        while(!local_done)
        {
            if(SLEEP)
            {
                for(unsigned int i = 0; i < times_through; ++i)
                {
                    __builtin_amdgcn_s_sleep(1);
                }

                if(times_through < 3907)
                {
                    ++times_through;
                }
            }

            local_done = atomicOr(&done_array[local_col], 0);
        }

        // Local maximum
        local_max = max(local_done, local_max);
    }

    // Process remaining columns
    if(j < row_end)
    {
        // Store diagonal index
        if(local_col == row)
        {
            csr_diag_ind[row] = j;
        }

        // Now, process all nodes that belong to the current block
        if(local_col < row)
        {
            // Index into shared memory to query for done flag
            int local_idx = local_col - first_row;

            int          local_done    = atomicOr(&local_done_array[local_idx], 0);
            unsigned int times_through = 0;
            while(!local_done)
            {
                if(SLEEP)
                {
                    for(unsigned int i = 0; i < times_through; ++i)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }

                    if(times_through < 3907)
                    {
                        ++times_through;
                    }
                }

                local_done = atomicOr(&local_done_array[local_idx], 0);
            }

            local_max = max(local_done, local_max);
        }
    }

    // Determine maximum local depth within the wavefront
    rocsparse_wfreduce_max<WF_SIZE>(&local_max);

    if(lid == WF_SIZE - 1)
    {
        // Write the local "row is done" flag
        atomicOr(&local_done_array[wid], local_max + 1);

        // Write the "row is done" flag
        atomicOr(&done_array[row], local_max + 1);

        // Obtain maximum nnz
        rocsparse_atomic_max(max_nnz, row_end - row_begin);

        if(csr_diag_ind[row] == -1 && diag_type == rocsparse_diag_type_non_unit)
        {
            // We are looking for the first zero pivot
            rocsparse_atomic_min(zero_pivot, row + idx_base);
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, bool SLEEP, typename I, typename J>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrsv_analysis_upper_kernel(J m,
                                     const I* __restrict__ csr_row_ptr,
                                     const J* __restrict__ csr_col_ind,
                                     I* __restrict__ csr_diag_ind,
                                     int* __restrict__ done_array,
                                     I* __restrict__ max_nnz,
                                     J* __restrict__ zero_pivot,
                                     rocsparse_index_base idx_base,
                                     rocsparse_diag_type  diag_type)
{
    int lid = hipThreadIdx_x & (WF_SIZE - 1);
    int wid = hipThreadIdx_x / WF_SIZE;

    // Last row in this block
    J last_row = m - 1 - hipBlockIdx_x * (BLOCKSIZE / WF_SIZE);

    // Row that the wavefront will process
    J row = last_row - wid;

    // Shared memory to set done flag for intra-block dependencies
    __shared__ int local_done_array[BLOCKSIZE / WF_SIZE];

    // Initialize local done array
    local_done_array[wid] = 0;

    // Wait for initialization to finish
    __syncthreads();

    // Do not run out of bounds
    if(row < 0)
    {
        return;
    }

    // Initialize matrix diagonal index
    if(lid == 0)
    {
        csr_diag_ind[row] = -1;
    }

    // Local depth
    int local_max = 0;

    I row_begin = csr_row_ptr[row] - idx_base;
    I row_end   = csr_row_ptr[row + 1] - idx_base;

    // This wavefront operates on a single row, from its end to its begin.
    // First, we process all nodes that have dependencies outside the current block.
    J local_col = -1;
    I j;
    for(j = row_end - 1 - lid; j >= row_begin; j -= WF_SIZE)
    {
        // local_col will tell us, for this iteration of the above for loop
        // (i.e. for this entry in this row), which columns contain the
        // non-zero values. We must then ensure that the output from the row
        // associated with the local_col is complete to ensure that we can
        // calculate the right answer.
        local_col = rocsparse_nontemporal_load(csr_col_ind + j) - idx_base;

        // Skip all columns where corresponding row belongs to this block
        if(local_col <= last_row)
        {
            break;
        }

        // While there are threads in this workgroup that have been unable to
        // get their input, loop and wait for the flag to exist.
        int          local_done    = atomicOr(&done_array[local_col], 0);
        unsigned int times_through = 0;
        while(!local_done)
        {
            if(SLEEP)
            {
                for(unsigned int i = 0; i < times_through; ++i)
                {
                    __builtin_amdgcn_s_sleep(1);
                }

                if(times_through < 3907)
                {
                    ++times_through;
                }
            }

            local_done = atomicOr(&done_array[local_col], 0);
        }

        // Local maximum
        local_max = max(local_done, local_max);
    }

    // Process remaining columns
    if(j >= row_begin)
    {
        // Store diagonal index
        if(local_col == row)
        {
            csr_diag_ind[row] = j;
        }

        // Now, process all nodes that belong to the current block
        if(local_col > row)
        {
            // Index into shared memory to query for done flag
            int local_idx = last_row - local_col;

            int          local_done    = atomicOr(&local_done_array[local_idx], 0);
            unsigned int times_through = 0;
            while(!local_done)
            {
                if(SLEEP)
                {
                    for(unsigned int i = 0; i < times_through; ++i)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }

                    if(times_through < 3907)
                    {
                        ++times_through;
                    }
                }

                local_done = atomicOr(&local_done_array[local_idx], 0);
            }

            local_max = max(local_done, local_max);
        }
    }

    // Determine maximum local depth within the wavefront
    rocsparse_wfreduce_max<WF_SIZE>(&local_max);

    if(lid == WF_SIZE - 1)
    {
        // Write the local "row is done" flag
        atomicOr(&local_done_array[wid], local_max + 1);

        // Write the "row is done" flag
        atomicOr(&done_array[row], local_max + 1);

        // Obtain maximum nnz
        rocsparse_atomic_max(max_nnz, row_end - row_begin);

        if(csr_diag_ind[row] == -1 && diag_type == rocsparse_diag_type_non_unit)
        {
            // We are looking for the first zero pivot
            rocsparse_atomic_min(zero_pivot, row + idx_base);
        }
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          bool         SLEEP,
          typename I,
          typename J,
          typename T>
__device__ void csrsv_device(J m,
                             T alpha,
                             const I* __restrict__ csr_row_ptr,
                             const J* __restrict__ csr_col_ind,
                             const T* __restrict__ csr_val,
                             const T* __restrict__ x,
                             T* __restrict__ y,
                             int* __restrict__ done_array,
                             J* __restrict__ map,
                             int offset,
                             J* __restrict__ zero_pivot,
                             rocsparse_index_base idx_base,
                             rocsparse_fill_mode  fill_mode,
                             rocsparse_diag_type  diag_type)
{
    int lid = hipThreadIdx_x & (WF_SIZE - 1);
    int wid = hipThreadIdx_x / WF_SIZE;

    // Index into the row map
    J idx = hipBlockIdx_x * (BLOCKSIZE / WF_SIZE) + wid;

    // Shared memory to hold diagonal entry
    __shared__ T diagonal[BLOCKSIZE / WF_SIZE];

    // Do not run out of bounds
    if(idx >= m)
    {
        return;
    }

    // Get the row this warp will operate on
    J row = map[idx + offset];

    // Current row entry point and exit point
    I row_begin = csr_row_ptr[row] - idx_base;
    I row_end   = csr_row_ptr[row + 1] - idx_base;

    // Local summation variable.
    T local_sum = static_cast<T>(0);

    if(lid == 0)
    {
        // Lane 0 initializes its local sum with alpha and x
        local_sum = alpha * rocsparse_nontemporal_load(x + row);
    }

    for(I j = row_begin + lid; j < row_end; j += WF_SIZE)
    {
        // Current column this lane operates on
        J local_col = rocsparse_nontemporal_load(csr_col_ind + j) - idx_base;

        // Local value this lane operates with
        T local_val = rocsparse_nontemporal_load(csr_val + j);

        // Check for numerical zero
        if(local_val == static_cast<T>(0) && local_col == row
           && diag_type == rocsparse_diag_type_non_unit)
        {
            // Numerical zero pivot found, avoid division by 0
            // and store index for later use.
            rocsparse_atomic_min(zero_pivot, row + idx_base);
            local_val = static_cast<T>(1);
        }

        // Differentiate upper and lower triangular mode
        if(fill_mode == rocsparse_fill_mode_upper)
        {
            // Processing upper triangular

            // Ignore all entries that are below the diagonal
            if(local_col < row)
            {
                continue;
            }

            // Diagonal entry
            if(local_col == row)
            {
                // If diagonal type is non unit, do division by diagonal entry
                // This is not required for unit diagonal for obvious reasons
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    diagonal[wid] = static_cast<T>(1) / local_val;
                }

                continue;
            }
        }
        else if(fill_mode == rocsparse_fill_mode_lower)
        {
            // Processing lower triangular

            // Ignore all entries that are above the diagonal
            if(local_col > row)
            {
                break;
            }

            // Diagonal entry
            if(local_col == row)
            {
                // If diagonal type is non unit, do division by diagonal entry
                // This is not required for unit diagonal for obvious reasons
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    diagonal[wid] = static_cast<T>(1) / local_val;
                }

                break;
            }
        }

        // Spin loop until dependency has been resolved
        int          local_done    = atomicOr(&done_array[local_col], 0);
        unsigned int times_through = 0;
        while(!local_done)
        {
            if(SLEEP)
            {
                for(unsigned int i = 0; i < times_through; ++i)
                {
                    __builtin_amdgcn_s_sleep(1);
                }

                if(times_through < 3907)
                {
                    ++times_through;
                }
            }

            local_done = atomicOr(&done_array[local_col], 0);
        }

        // Wait for y to be visible globally
        __threadfence();

        // Local sum computation for each lane
        local_sum = rocsparse_fma(-local_val, y[local_col], local_sum);
    }

    // Gather all local sums for each lane
    local_sum = rocsparse_wfreduce_sum<WF_SIZE>(local_sum);

    // If we have non unit diagonal, take the diagonal into account
    // For unit diagonal, this would be multiplication with one
    if(diag_type == rocsparse_diag_type_non_unit)
    {
        __threadfence_block();

        local_sum = local_sum * diagonal[wid];
    }

    if(lid == WF_SIZE - 1)
    {
        // Store the rows result in y
        rocsparse_nontemporal_store(local_sum, &y[row]);

        // Make sure y is written to global memory before setting "row is done" flag
        __threadfence();

        // Mark row as done
        atomicOr(&done_array[row], 1);
    }
}
