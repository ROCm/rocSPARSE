/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int HASH, typename T, typename U>
__device__ void csrilu0_hash_kernel(rocsparse_int m,
                                    const rocsparse_int* __restrict__ csr_row_ptr,
                                    const rocsparse_int* __restrict__ csr_col_ind,
                                    T* __restrict__ csr_val,
                                    const rocsparse_int* __restrict__ csr_diag_ind,
                                    int* __restrict__ done,
                                    const rocsparse_int* __restrict__ map,
                                    rocsparse_int* __restrict__ zero_pivot,
                                    rocsparse_index_base idx_base,
                                    int                  boost,
                                    U                    boost_tol,
                                    T                    boost_val)
{
    int lid = hipThreadIdx_x & (WFSIZE - 1);
    int wid = hipThreadIdx_x / WFSIZE;

    __shared__ rocsparse_int stable[BLOCKSIZE * HASH];
    __shared__ rocsparse_int sdata[BLOCKSIZE * HASH];

    // Pointer to each wavefronts shared data
    rocsparse_int* table = &stable[wid * WFSIZE * HASH];
    rocsparse_int* data  = &sdata[wid * WFSIZE * HASH];

    // Initialize hash table with -1
    for(unsigned int j = lid; j < WFSIZE * HASH; j += WFSIZE)
    {
        table[j] = -1;
    }

    __threadfence_block();

    rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= m)
    {
        return;
    }

    // Current row this wavefront is working on
    rocsparse_int row = map[idx];

    // Diagonal entry point of the current row
    rocsparse_int row_diag = csr_diag_ind[row];

    // Row entry point
    rocsparse_int row_begin = csr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;

    // Fill hash table
    // Loop over columns of current row and fill hash table with row dependencies
    // Each lane processes one entry
    for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
    {
        // Insert key into hash table
        rocsparse_int key = csr_col_ind[j];
        // Compute hash
        rocsparse_int hash = (key * 103) & (WFSIZE * HASH - 1);

        // Hash operation
#pragma unroll 4
        for(unsigned int h = 0; h < WFSIZE * HASH; ++h)
        {
            if(table[hash] == key)
            {
                // key is already inserted, done
                break;
            }
            else if(rocsparse_atomic_cas(&table[hash], -1, key) == -1)
            {
                // inserted key into the table, done
                data[hash] = j;
                break;
            }
            else
            {
                // collision, compute new hash
                hash = (hash + 1) & (WFSIZE * HASH - 1);
            }
        }
    }

    __threadfence_block();

    // Loop over column of current row
    for(rocsparse_int j = row_begin; j < row_diag; ++j)
    {
        // Column index currently being processes
        rocsparse_int local_col = csr_col_ind[j] - idx_base;

        // Corresponding value
        T local_val = csr_val[j];

        // End of the row that corresponds to local_col
        rocsparse_int local_end = csr_row_ptr[local_col + 1] - idx_base;

        // Diagonal entry point of row local_col
        rocsparse_int local_diag = csr_diag_ind[local_col];

        // Structural zero pivot, do not process this row
        if(local_diag == -1)
        {
            local_diag = local_end - 1;
        }

        // Spin loop until dependency has been resolved
        while(!atomicOr(&done[local_col], 0))
            ;

        // Make sure updated csr_val is visible
        __threadfence();

        // Load diagonal entry
        T diag_val = csr_val[local_diag];

        // Numeric boost
        if(boost)
        {
            diag_val = (boost_tol >= rocsparse_abs(diag_val)) ? boost_val : diag_val;

            __threadfence();

            if(lid == 0)
            {
                csr_val[local_diag] = diag_val;
            }
        }
        else
        {
            // Row has numerical zero diagonal
            if(diag_val == static_cast<T>(0))
            {
                if(lid == 0)
                {
                    // We are looking for the first zero pivot
                    rocsparse_atomic_min(zero_pivot, local_col + idx_base);
                }

                // Skip this row if it has a zero pivot
                break;
            }
        }

        csr_val[j] = local_val = local_val / diag_val;

        // Loop over the row the current column index depends on
        // Each lane processes one entry
        for(rocsparse_int k = local_diag + 1 + lid; k < local_end; k += WFSIZE)
        {
            // Get value from hash table
            rocsparse_int key = csr_col_ind[k];

            // Compute hash
            rocsparse_int hash = (key * 103) & (WFSIZE * HASH - 1);

            // Hash operation
#pragma unroll 4
            for(unsigned int h = 0; h < WFSIZE * HASH; ++h)
            {
                if(table[hash] == -1)
                {
                    // No entry for the key, done
                    break;
                }
                else if(table[hash] == key)
                {
                    // Entry found, do ILU computation
                    rocsparse_int idx_data = data[hash];
                    csr_val[idx_data] = rocsparse_fma(-local_val, csr_val[k], csr_val[idx_data]);
                    break;
                }
                else
                {
                    // Collision, compute new hash
                    hash = (hash + 1) & (WFSIZE * HASH - 1);
                }
            }
        }
    }

    // Make sure updated csr_val is written to global memory
    __threadfence();

    if(lid == 0)
    {
        // First lane writes "we are done" flag
        atomicOr(&done[row], 1);
    }
}

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP, typename T, typename U>
__device__ void csrilu0_binsearch_kernel(rocsparse_int m_,
                                         const rocsparse_int* __restrict__ csr_row_ptr,
                                         const rocsparse_int* __restrict__ csr_col_ind,
                                         T* __restrict__ csr_val,
                                         const rocsparse_int* __restrict__ csr_diag_ind,
                                         int* __restrict__ done,
                                         const rocsparse_int* __restrict__ map,
                                         rocsparse_int* __restrict__ zero_pivot,
                                         rocsparse_index_base idx_base,
                                         int                  boost,
                                         U                    boost_tol,
                                         T                    boost_val)
{
    int lid = hipThreadIdx_x & (WFSIZE - 1);
    int wid = hipThreadIdx_x / WFSIZE;

    rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= m_)
    {
        return;
    }

    // Current row this wavefront is working on
    rocsparse_int row = map[idx];

    // Diagonal entry point of the current row
    rocsparse_int row_diag = csr_diag_ind[row];

    // Row entry point
    rocsparse_int row_begin = csr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;

    // Loop over column of current row
    for(rocsparse_int j = row_begin; j < row_diag; ++j)
    {
        // Column index currently being processes
        rocsparse_int local_col = csr_col_ind[j] - idx_base;

        // Corresponding value
        T local_val = csr_val[j];

        // End of the row that corresponds to local_col
        rocsparse_int local_end = csr_row_ptr[local_col + 1] - idx_base;

        // Diagonal entry point of row local_col
        rocsparse_int local_diag = csr_diag_ind[local_col];

        // Structural zero pivot, do not process this row
        if(local_diag == -1)
        {
            local_diag = local_end - 1;
        }

        // Spin loop until dependency has been resolved
        int          local_done    = atomicOr(&done[local_col], 0);
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

            local_done = atomicOr(&done[local_col], 0);
        }

        // Make sure updated csr_val is visible
        __threadfence();

        // Load diagonal entry
        T diag_val = csr_val[local_diag];

        // Numeric boost
        if(boost)
        {
            diag_val = (boost_tol >= rocsparse_abs(diag_val)) ? boost_val : diag_val;

            __threadfence();

            if(lid == 0)
            {
                csr_val[local_diag] = diag_val;
            }
        }
        else
        {
            // Row has numerical zero diagonal
            if(diag_val == static_cast<T>(0))
            {
                if(lid == 0)
                {
                    // We are looking for the first zero pivot
                    rocsparse_atomic_min(zero_pivot, local_col + idx_base);
                }

                // Skip this row if it has a zero pivot
                break;
            }
        }

        csr_val[j] = local_val = local_val / diag_val;

        // Loop over the row the current column index depends on
        // Each lane processes one entry
        rocsparse_int l = j + 1;
        for(rocsparse_int k = local_diag + 1 + lid; k < local_end; k += WFSIZE)
        {
            // Perform a binary search to find matching columns
            rocsparse_int r     = row_end - 1;
            rocsparse_int m     = (r + l) >> 1;
            rocsparse_int col_j = csr_col_ind[m];

            rocsparse_int col_k = csr_col_ind[k];

            // Binary search
            while(l < r)
            {
                if(col_j < col_k)
                {
                    l = m + 1;
                }
                else
                {
                    r = m;
                }

                m     = (r + l) >> 1;
                col_j = csr_col_ind[m];
            }

            // Check if a match has been found
            if(col_j == col_k)
            {
                // If a match has been found, do ILU computation
                csr_val[l] = rocsparse_fma(-local_val, csr_val[k], csr_val[l]);
            }
        }
    }

    // Make sure updated csr_val is written to global memory
    __threadfence();

    if(lid == 0)
    {
        // First lane writes "we are done" flag
        atomicOr(&done[row], 1);
    }
}
