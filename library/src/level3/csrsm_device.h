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
#ifndef CSRSM_DEVICE_H
#define CSRSM_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T, unsigned int BLOCKSIZE, unsigned int WF_SIZE, bool SLEEP>
__device__ void csrsm_device(rocsparse_int       m,
                             rocsparse_int       n,
                             rocsparse_operation trans_B,
                             T                   alpha,
                             const rocsparse_int* __restrict__ csr_row_ptr,
                             const rocsparse_int* __restrict__ csr_col_ind,
                             const T* __restrict__ csr_val,
                             T* __restrict__ B,
                             rocsparse_int ldb,
                             int* __restrict__ done_array,
                             rocsparse_int* __restrict__ map,
                             rocsparse_int* __restrict__ zero_pivot,
                             rocsparse_index_base idx_base,
                             rocsparse_fill_mode  fill_mode,
                             rocsparse_diag_type  diag_type)
{
    int lid = hipThreadIdx_x & (WF_SIZE - 1);
    int wid = hipThreadIdx_x / WF_SIZE;

    // Index into the row map
    rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WF_SIZE + wid;

    // Shared memory to hold diagonal entry
    __shared__ T diagonal[BLOCKSIZE / WF_SIZE];

    // Do not run out of bounds
    if(idx >= m)
    {
        return;
    }

    // Get the row this warp will operate on
    rocsparse_int row = map[idx];

    // Current row entry point and exit point
    rocsparse_int row_begin = csr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;

    // Local summation variable.
    T local_sum = static_cast<T>(0);

    // Index into B
    rocsparse_int idx_B = (trans_B == rocsparse_operation_none) ? n * ldb + row : row * ldb + n;

    if(lid == 0)
    {
        // Lane 0 initializes its local sum with alpha and x
        local_sum = alpha * rocsparse_nontemporal_load(B + idx_B);
    }

    for(rocsparse_int j = row_begin + lid; j < row_end; j += WF_SIZE)
    {
        // Current column this lane operates on
        rocsparse_int local_col = rocsparse_nontemporal_load(csr_col_ind + j) - idx_base;

        // Local value this lane operates with
        T local_val = rocsparse_nontemporal_load(csr_val + j);

        // Check for numerical zero
        if(local_val == static_cast<T>(0) && local_col == row
           && diag_type == rocsparse_diag_type_non_unit)
        {
            // Numerical zero pivot found, avoid division by 0
            // and store index for later use.
            atomicMin(zero_pivot, row + idx_base);
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
        int          local_done = rocsparse_atomic_load(&done_array[local_col], __ATOMIC_ACQUIRE);
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

            local_done = rocsparse_atomic_load(&done_array[local_col], __ATOMIC_ACQUIRE);
        }

        // Index into X
        rocsparse_int idx_X
            = (trans_B == rocsparse_operation_none) ? n * ldb + local_col : local_col * ldb + n;

        // Local sum computation for each lane
        local_sum = fma(-local_val, B[idx_X], local_sum);
    }

    // Gather all local sums for each lane
    local_sum = rocsparse_wfreduce_sum<WF_SIZE>(local_sum);

    // If we have non unit diagonal, take the diagonal into account
    // For unit diagonal, this would be multiplication with one
    if(diag_type == rocsparse_diag_type_non_unit)
    {
        local_sum = local_sum * diagonal[wid];
    }

    if(lid == WF_SIZE - 1)
    {
        // Write the "row is done" flag and store the rows result in B
        rocsparse_nontemporal_store(local_sum, &B[idx_B]);
        rocsparse_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE);
    }
}

#endif // CSRSM_DEVICE_H
