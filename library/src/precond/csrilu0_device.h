/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
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

int wid = tid / WF_SIZE;

    __shared__ rocsparse_int stable[BLOCKSIZE / WF_SIZE][WF_SIZE * HASH];//[BLOCKSIZE * HASH];
    __shared__ rocsparse_int sdata[BLOCKSIZE / WF_SIZE][WF_SIZE * HASH];

    for(rocsparse_int j = 0; j < HASH; ++j)
    {
        stable[wid][lid + j * WF_SIZE] = -1;
    }

    if (idx >= m)
    {
         return;
    }

    rocsparse_int row = map[idx];
    rocsparse_int row_diag  = csr_diag_ind[row];

    // Row has structural zero diagonal, skip
    if(row_diag == -1)
    {
        if(lid == 0)
        {
            atomicMin(zero_pivot, row);
#if defined(__HIP_PLATFORM_HCC__)
            __atomic_store_n(&done[row], 1, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
            atomicOr(&done[row], 1);
#endif
        }

        return;
    }

    rocsparse_int row_begin = csr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;

    // Fill hash table
//    rocsparse_int* table = &stable[(tid / WF_SIZE) * WF_SIZE * HASH];
//    rocsparse_int* data  = &sdata[(tid / WF_SIZE) * WF_SIZE * HASH];
    rocsparse_int* table = stable[wid];
    rocsparse_int* data  = sdata[wid];

    for(rocsparse_int j = row_begin + lid; j < row_end; j += WF_SIZE)
    {
        // Insert key into hash table
        int key = csr_col_ind[j];
        int hash = (key * 103) & (WF_SIZE * HASH - 1);
    
        while(true)
        {
            if(table[hash] == key)
            {
                break;
            }
            else if(atomicCAS(&table[hash], -1, key) == -1)
            {
                data[hash] = j;
                break;
            }
            else
            {
                hash = (hash + 1) & (WF_SIZE * HASH - 1);
            }
        }
    }

    for(rocsparse_int j = row_begin; j < row_diag; ++j)
    {
        rocsparse_int local_col  = csr_col_ind[j] - idx_base;
        T local_val = csr_val[j];
        rocsparse_int local_end  = csr_row_ptr[local_col + 1] - idx_base;
        rocsparse_int local_diag = csr_diag_ind[local_col];

        // Row depends on structural zero diagonal
        if(local_diag == -1)
        {
            if(lid == 0)
            {
                atomicMin(zero_pivot, local_col);
            }

            break;
        }

        rocsparse_int local_done = 0;
        while(!local_done)
        {
#if defined(__HIP_PLATFORM_HCC__)
            local_done = __atomic_load_n(&done[local_col], __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
            local_done = atomicOr(&done[local_col], 0x0);
#endif
        }

#if defined(__HIP_PLATFORM_HCC__)
        T diag_val;
        __atomic_load(&csr_val[local_diag], &diag_val, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        T diag_val = csr_val[local_diag];
#endif

        // Row has numerical zero diagonal
        if(diag_val == 0.0)
        {
            if(lid == 0)
            {
                atomicMin(zero_pivot, local_col);
            }

            break;
        }

        csr_val[j] = local_val /= diag_val;

        for(rocsparse_int k = local_diag + 1 + lid; k < local_end; k += WF_SIZE)
        {
            // Get value from hash table
            int key = csr_col_ind[k];
            int hash = (key * 103) & (WF_SIZE * HASH - 1);

            while(true)
            {
                int val = table[hash];

                if(val == -1)
                {
                    break;
                }
                else if(val == key)
                {
#if defined(__HIP_PLATFORM_HCC__)
                    T val_k;
                    __atomic_load(&csr_val[k], &val_k, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
                    T val_k = csr_val[k];
#endif
                    csr_val[data[hash]] -= local_val * val_k;
                    break;
                }

                hash = (hash + 1) & (WF_SIZE * HASH - 1);
            }
        }
    }

    if(lid == 0)
    {
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

    if (idx >= m)
    {
         return;
    }

    rocsparse_int row = map[idx];
    rocsparse_int row_diag  = csr_diag_ind[row];

    // Row has structural zero diagonal, skip
    if(row_diag == -1)
    {
        if(lid == 0)
        {
            atomicMin(zero_pivot, row);
#if defined(__HIP_PLATFORM_HCC__)
            __atomic_store_n(&done[row], 1, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
            atomicOr(&done[row], 1);
#endif
        }

        return;
    }

    rocsparse_int row_begin = csr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;

    for(rocsparse_int j = row_begin; j < row_diag; ++j)
    {
        rocsparse_int local_col  = csr_col_ind[j] - idx_base;
        T local_val = csr_val[j];
        rocsparse_int local_end  = csr_row_ptr[local_col + 1] - idx_base;
        rocsparse_int local_diag = csr_diag_ind[local_col];

        // Row depends on structural zero diagonal
        if(local_diag == -1)
        {
            if(lid == 0)
            {
                atomicMin(zero_pivot, local_col);
            }

            break;
        }

        rocsparse_int local_done = 0;
        while(!local_done)
        {
#if defined(__HIP_PLATFORM_HCC__)
            local_done = __atomic_load_n(&done[local_col], __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
            local_done = atomicOr(&done[local_col], 0x0);
#endif
        }

#if defined(__HIP_PLATFORM_HCC__)
        T diag_val;
        __atomic_load(&csr_val[local_diag], &diag_val, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        // TODO
        volatile T diag_val = csr_val[local_diag];
#endif

        // Row has numerical zero diagonal
        if(diag_val == 0.0)
        {
            if(lid == 0)
            {
                atomicMin(zero_pivot, local_col);
            }

            break;
        }

        csr_val[j] = local_val /= diag_val;

        rocsparse_int l = j + 1;
        for(rocsparse_int k = local_diag + 1 + lid; k < local_end; k += WF_SIZE)
        {
            rocsparse_int r = row_end - 1;
            rocsparse_int m = (r + l) >> 1;
            rocsparse_int col_j = csr_col_ind[m];
            
            rocsparse_int col_k = csr_col_ind[k];
    
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
            
                m = (r + l) >> 1;
                col_j = csr_col_ind[m];
            }
    
            if(col_j == col_k)
            {
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
#if defined(__HIP_PLATFORM_HCC__)
        __atomic_store_n(&done[row], 1, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        atomicOr(&done[row], 1);
#endif
    }
}

#endif // CSRMV_DEVICE_H
