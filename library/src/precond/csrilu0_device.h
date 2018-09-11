/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef CSRILU0_DEVICE_H
#define CSRILU0_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE, unsigned int HASH>
__global__ void csrilu0_hash_kernel(rocsparse_int m,
                                    const rocsparse_int* __restrict__ csr_row_ptr,
                                    const rocsparse_int* __restrict__ csr_col_ind,
                                    T* __restrict__ csr_val,
                                    const rocsparse_int* __restrict__ csr_diag_ind,
                                    rocsparse_int* __restrict__ done,
                                    rocsparse_int* __restrict__ map,
                                    rocsparse_int* __restrict__ zero_pivot,
                                    rocsparse_index_base idx_base)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int lid = tid & (WF_SIZE - 1);
    rocsparse_int idx = gid / WF_SIZE;

    __shared__ rocsparse_int stable[BLOCKSIZE * HASH];
    __shared__ rocsparse_int sdata[BLOCKSIZE * HASH];

    // Pointer to each wavefronts shared data
    rocsparse_int* table = &stable[(tid / WF_SIZE) * WF_SIZE * HASH];
    rocsparse_int* data  = &sdata[(tid / WF_SIZE) * WF_SIZE * HASH];

    // Initialize hash table with -1
    for(rocsparse_int j = lid; j < WF_SIZE * HASH; j += WF_SIZE)
    {
        table[j] = -1;
    }

    // Do not run out of bounds
    if(idx >= m)
    {
        return;
    }

    // Current row this wavefront is working on
    rocsparse_int row = map[idx];
    // Diagonal entry point of the current row
    rocsparse_int row_diag  = csr_diag_ind[row];
    rocsparse_int row_begin = csr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;

    // Fill hash table
    // Loop over columns of current row and fill hash table with row dependencies
    // Each lane processes one entry
    for(rocsparse_int j = row_begin + lid; j < row_end; j += WF_SIZE)
    {
        // Insert key into hash table
        rocsparse_int key = csr_col_ind[j];
        // Compute hash
        rocsparse_int hash = (key * 103) & (WF_SIZE * HASH - 1);

        // Hash operation
        while(true)
        {
            if(table[hash] == key)
            {
                // key is already inserted, done
                break;
            }
            else if(atomicCAS(&table[hash], -1, key) == -1)
            {
                // inserted key into the table, done
                data[hash] = j;
                break;
            }
            else
            {
                // collision, compute new hash
                hash = (hash + 1) & (WF_SIZE * HASH - 1);
            }
        }
    }

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
        rocsparse_int local_done = 0;
        while(!local_done)
        {
#if defined(__HIP_PLATFORM_HCC__)
            local_done = __atomic_load_n(&done[local_col], __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
            local_done      = atomicOr(&done[local_col], 0);
#endif
        }

// Load diagonal entry
#if defined(__HIP_PLATFORM_HCC__)
        T diag_val;
        __atomic_load(&csr_val[local_diag], &diag_val, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        T diag_val          = csr_val[local_diag];
#endif

        // Row has numerical zero diagonal
        if(diag_val == static_cast<T>(0))
        {
            if(lid == 0)
            {
                // We are looking for the first zero pivot
                atomicMin(zero_pivot, local_col);
            }

            // Skip this row if it has a zero pivot
            break;
        }

        csr_val[j] = local_val /= diag_val;

        // Loop over the row the current column index depends on
        // Each lane processes one entry
        for(rocsparse_int k = local_diag + 1 + lid; k < local_end; k += WF_SIZE)
        {
            // Get value from hash table
            rocsparse_int key = csr_col_ind[k];
            // Compute hash
            rocsparse_int hash = (key * 103) & (WF_SIZE * HASH - 1);

            // Hash operation
            while(true)
            {
                rocsparse_int val = table[hash];

                if(val == -1)
                {
                    // No entry for the key, done
                    break;
                }
                else if(val == key)
                {
// Entry found, do ILU computation
#if defined(__HIP_PLATFORM_HCC__)
                    T val_k;
                    __atomic_load(&csr_val[k], &val_k, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
                    T val_k = csr_val[k];
#endif
                    csr_val[data[hash]] -= local_val * val_k;
                    break;
                }

                // Collision, compute new hash
                hash = (hash + 1) & (WF_SIZE * HASH - 1);
            }
        }
    }

    if(lid == 0)
    {
// Lane 0 write "we are done" flag
#if defined(__HIP_PLATFORM_HCC__)
        __atomic_store_n(&done[row], 1, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        atomicOr(&done[row], 1);
#endif
    }
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__global__ void csrilu0_binsearch_kernel(rocsparse_int m,
                                         const rocsparse_int* __restrict__ csr_row_ptr,
                                         const rocsparse_int* __restrict__ csr_col_ind,
                                         T* __restrict__ csr_val,
                                         const rocsparse_int* __restrict__ csr_diag_ind,
                                         rocsparse_int* __restrict__ done,
                                         rocsparse_int* __restrict__ map,
                                         rocsparse_int* __restrict__ zero_pivot,
                                         rocsparse_index_base idx_base)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int lid = tid & (WF_SIZE - 1);
    rocsparse_int idx = gid / WF_SIZE;

    // Do not run out of bounds
    if(idx >= m)
    {
        return;
    }

    // Current row this wavefront is working on
    rocsparse_int row = map[idx];
    // Diagonal entry point of the current row
    rocsparse_int row_diag  = csr_diag_ind[row];
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
        rocsparse_int local_done = 0;
        while(!local_done)
        {
#if defined(__HIP_PLATFORM_HCC__)
            local_done = __atomic_load_n(&done[local_col], __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
            local_done = atomicOr(&done[local_col], 0);
#endif
        }

// Load diagonal entry
#if defined(__HIP_PLATFORM_HCC__)
        T diag_val;
        __atomic_load(&csr_val[local_diag], &diag_val, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        // TODO
        volatile T diag_val      = csr_val[local_diag];
#endif

        // Row has numerical zero diagonal
        if(diag_val == static_cast<T>(0))
        {
            if(lid == 0)
            {
                // We are looking for the first zero pivot
                atomicMin(zero_pivot, local_col);
            }

            // Skip this row if it has a zero pivot
            break;
        }

        csr_val[j] = local_val /= diag_val;

        // Loop over the row the current column index depends on
        // Each lane processes one entry
        rocsparse_int l = j + 1;
        for(rocsparse_int k = local_diag + 1 + lid; k < local_end; k += WF_SIZE)
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
#if defined(__HIP_PLATFORM_HCC__)
                T val_k;
                __atomic_load(&csr_val[k], &val_k, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
                volatile T val_k = csr_val[k];
#endif

                csr_val[l] -= local_val * val_k;
            }
        }
    }

    if(lid == 0)
    {
// Lane 0 write "we are done" flag
#if defined(__HIP_PLATFORM_HCC__)
        __atomic_store_n(&done[row], 1, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        atomicOr(&done[row], 1);
#endif
    }
}

#endif // CSRILU0_DEVICE_H
