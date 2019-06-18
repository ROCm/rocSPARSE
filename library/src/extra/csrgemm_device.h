/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#ifndef CSRGEMM_DEVICE_H
#define CSRGEMM_DEVICE_H

#include "common.h"
#include "rocsparse.h"

#include <hip/hip_runtime.h>

// Compute number of intermediate products of each row
template <unsigned int WFSIZE>
__global__ void csrgemm_intermediate_products(rocsparse_int m,
                                              const rocsparse_int* __restrict__ csr_row_ptr_A,
                                              const rocsparse_int* __restrict__ csr_col_ind_A,
                                              const rocsparse_int* __restrict__ csr_row_ptr_B,
                                              const rocsparse_int* __restrict__ csr_row_ptr_D,
                                              rocsparse_int* __restrict__ int_prod,
                                              rocsparse_index_base idx_base_A,
                                              bool                 mul,
                                              bool                 add)
{
    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Each (sub)wavefront processes a row
    rocsparse_int row = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) / WFSIZE;

    // Bounds check
    if(row >= m)
    {
        return;
    }

    // Initialize intermediate product counter of current row
    rocsparse_int nprod = 0;

    // alpha * A * B part
    if(mul == true)
    {
        // Row begin and row end of A matrix
        rocsparse_int row_begin_A = csr_row_ptr_A[row] - idx_base_A;
        rocsparse_int row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

        // Loop over columns of A in current row
        for(rocsparse_int j = row_begin_A + lid; j < row_end_A; j += WFSIZE)
        {
            // Current column of A
            rocsparse_int col_A = csr_col_ind_A[j] - idx_base_A;

            // Accumulate non zero entries of B in row col_A
            nprod += (csr_row_ptr_B[col_A + 1] - csr_row_ptr_B[col_A]);
        }

        // Gather nprod
        rocsparse_wfreduce_sum<WFSIZE>(&nprod);
    }

    // Last lane writes result
    if(lid == WFSIZE - 1)
    {
        // beta * D part
        if(add == true)
        {
            nprod += (csr_row_ptr_D[row + 1] - csr_row_ptr_D[row]);
        }

        // Write number of intermediate products of the current row
        int_prod[row] = nprod;
    }
}

template <unsigned int BLOCKSIZE, unsigned int GROUPS>
static __device__ __forceinline__ void csrgemm_group_reduce(rocsparse_int tid,
                                                            rocsparse_int* __restrict__ data)
{
    // clang-format off
    if(BLOCKSIZE > 512 && tid < 512) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid + 512) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE > 256 && tid < 256) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid + 256) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE > 128 && tid < 128) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid + 128) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >  64 && tid <  64) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +  64) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >  32 && tid <  32) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +  32) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >  16 && tid <  16) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +  16) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >   8 && tid <   8) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +   8) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >   4 && tid <   4) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +   4) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >   2 && tid <   2) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +   2) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >   1 && tid <   1) for(unsigned int i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +   1) * GROUPS + i]; __syncthreads();
    // clang-format on
}

template <unsigned int BLOCKSIZE, unsigned int GROUPS>
__global__ void csrgemm_group_reduce_part1(rocsparse_int m,
                                           rocsparse_int* __restrict__ int_prod,
                                           rocsparse_int* __restrict__ group_size)
{
    rocsparse_int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // Shared memory for block reduction
    __shared__ rocsparse_int sdata[BLOCKSIZE * GROUPS];

    // Initialize shared memory
    for(unsigned int i = 0; i < GROUPS; ++i)
    {
        sdata[hipThreadIdx_x * GROUPS + i] = 0;
    }

    // Loop over rows
    for(; row < m; row += hipGridDim_x * hipBlockDim_x)
    {
        rocsparse_int nprod = int_prod[row];

        // clang-format off
             if(nprod <=    32) { ++sdata[hipThreadIdx_x * GROUPS + 0]; int_prod[row] = 0; }
        else if(nprod <=    64) { ++sdata[hipThreadIdx_x * GROUPS + 1]; int_prod[row] = 1; }
        else if(nprod <=   512) { ++sdata[hipThreadIdx_x * GROUPS + 2]; int_prod[row] = 2; }
        else if(nprod <=  1024) { ++sdata[hipThreadIdx_x * GROUPS + 3]; int_prod[row] = 3; }
        else if(nprod <=  2048) { ++sdata[hipThreadIdx_x * GROUPS + 4]; int_prod[row] = 4; }
        else if(nprod <=  4096) { ++sdata[hipThreadIdx_x * GROUPS + 5]; int_prod[row] = 5; }
        else if(nprod <=  8192) { ++sdata[hipThreadIdx_x * GROUPS + 6]; int_prod[row] = 6; }
        else                    { ++sdata[hipThreadIdx_x * GROUPS + 7]; int_prod[row] = 7; }
        // clang-format on
    }

    // Wait for all threads to finish
    __syncthreads();

    // Reduce block
    csrgemm_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

    // Write result
    if(hipThreadIdx_x < GROUPS)
    {
        group_size[hipBlockIdx_x * GROUPS + hipThreadIdx_x] = sdata[hipThreadIdx_x];
    }
}

template <unsigned int BLOCKSIZE, unsigned int GROUPS>
__global__ void csrgemm_group_reduce_part3(rocsparse_int* __restrict__ group_size)
{
    // Shared memory for block reduction
    __shared__ rocsparse_int sdata[BLOCKSIZE * GROUPS];

    // Copy global data to shared memory
    for(unsigned int i = hipThreadIdx_x; i < BLOCKSIZE * GROUPS; i += BLOCKSIZE)
    {
        sdata[i] = group_size[i];
    }

    // Wait for all threads to finish
    __syncthreads();

    // Reduce block
    csrgemm_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

    // Write result back to global memory
    if(hipThreadIdx_x < GROUPS)
    {
        group_size[hipThreadIdx_x] = sdata[hipThreadIdx_x];
    }
}

// Hash operation to insert key into hash table
template <unsigned int HASHVAL, unsigned int HASHSIZE>
static __device__ __forceinline__ rocsparse_int insert_key(rocsparse_int key,
                                                           rocsparse_int* __restrict__ table)
{
    rocsparse_int nins = 0;
    rocsparse_int hash = (key * HASHVAL) & (HASHSIZE - 1);

    while(true)
    {
        if(table[hash] == key)
        {
            // Element already present
            break;
        }
        else if(table[hash] == -1)
        {
            // If empty, add element with atomic
            if(atomicCAS(&table[hash], -1, key) == -1)
            {
                // Increment number of insertions
                ++nins;
                break;
            }
        }
        else
        {
            // Linear probing, when hash is collided, try next entry
            hash = (hash + 1) & (HASHSIZE - 1);
        }
    }

    return nins;
}

// Compute non-zero entries per row, where each row is processed by a single wavefront
template <unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int HASHSIZE, unsigned int HASHVAL>
__global__ void csrgemm_nnz_wf_per_row(rocsparse_int m,
                                       const rocsparse_int* __restrict__ offset,
                                       const rocsparse_int* __restrict__ perm,
                                       const rocsparse_int* __restrict__ csr_row_ptr_A,
                                       const rocsparse_int* __restrict__ csr_col_ind_A,
                                       const rocsparse_int* __restrict__ csr_row_ptr_B,
                                       const rocsparse_int* __restrict__ csr_col_ind_B,
                                       const rocsparse_int* __restrict__ csr_row_ptr_D,
                                       const rocsparse_int* __restrict__ csr_col_ind_D,
                                       rocsparse_int* __restrict__ row_nnz,
                                       rocsparse_index_base idx_base_A,
                                       rocsparse_index_base idx_base_B,
                                       rocsparse_index_base idx_base_D,
                                       bool                 mul,
                                       bool                 add)
{
    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);
    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each (sub)wavefront processes a row
    rocsparse_int row = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Hash table in shared memory
    __shared__ rocsparse_int stable[BLOCKSIZE / WFSIZE * HASHSIZE];

    // Local hash table
    rocsparse_int* table = &stable[wid * HASHSIZE];

    // Initialize hash table
    for(unsigned int i = lid; i < HASHSIZE; i += WFSIZE)
    {
        table[i] = -1;
    }

    // Bounds check
    if(row >= m)
    {
        return;
    }

    // Apply permutation, if available
    row = perm ? perm[row + *offset] : row;

    // Initialize row nnz
    rocsparse_int nnz = 0;

    // alpha * A * B part
    if(mul == true)
    {
        // Get row boundaries of the current row in A
        rocsparse_int row_begin_A = csr_row_ptr_A[row] - idx_base_A;
        rocsparse_int row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

        // Loop over columns of A in current row
        for(rocsparse_int j = row_begin_A + lid; j < row_end_A; j += WFSIZE)
        {
            // Column of A in current row
            rocsparse_int col_A = csr_col_ind_A[j] - idx_base_A;

            // Loop over columns of B in row col_A
            rocsparse_int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
            rocsparse_int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

            // Insert all columns of B into hash table
            for(rocsparse_int k = row_begin_B; k < row_end_B; ++k)
            {
                // Count the actual insertions to obtain row nnz of C
                nnz += insert_key<HASHVAL, HASHSIZE>(csr_col_ind_B[k] - idx_base_B, table);
            }
        }
    }

    // beta * D part
    if(add == true)
    {
        // Get row boundaries of the current row in D
        rocsparse_int row_begin_D = csr_row_ptr_D[row] - idx_base_D;
        rocsparse_int row_end_D   = csr_row_ptr_D[row + 1] - idx_base_D;

        // Loop over columns of D in current row and insert all columns of D into hash table
        for(rocsparse_int j = row_begin_D + lid; j < row_end_D; j += WFSIZE)
        {
            // Count the actual insertions to obtain row nnz of C
            nnz += insert_key<HASHVAL, HASHSIZE>(csr_col_ind_D[j] - idx_base_D, table);
        }
    }

    // Accumulate all row nnz within each (sub)wavefront to obtain the total row nnz
    // of the current row
    rocsparse_wfreduce_sum<WFSIZE>(&nnz);

    // Write result to global memory
    if(lid == WFSIZE - 1)
    {
        row_nnz[row] = nnz;
    }
}

// Compute non-zero entries per row, where each row is processed by a single block
template <unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int HASHSIZE, unsigned int HASHVAL>
__global__ void csrgemm_nnz_block_per_row(const rocsparse_int* __restrict__ offset,
                                          const rocsparse_int* __restrict__ perm,
                                          const rocsparse_int* __restrict__ csr_row_ptr_A,
                                          const rocsparse_int* __restrict__ csr_col_ind_A,
                                          const rocsparse_int* __restrict__ csr_row_ptr_B,
                                          const rocsparse_int* __restrict__ csr_col_ind_B,
                                          const rocsparse_int* __restrict__ csr_row_ptr_D,
                                          const rocsparse_int* __restrict__ csr_col_ind_D,
                                          rocsparse_int* __restrict__ row_nnz,
                                          rocsparse_index_base idx_base_A,
                                          rocsparse_index_base idx_base_B,
                                          rocsparse_index_base idx_base_D,
                                          bool                 mul,
                                          bool                 add)
{
    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);
    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each block processes a row (apply permutation)
    rocsparse_int row = perm[hipBlockIdx_x + *offset];

    // Hash table in shared memory
    __shared__ rocsparse_int table[HASHSIZE];

    // Initialize hash table
    for(unsigned int i = hipThreadIdx_x; i < HASHSIZE; i += BLOCKSIZE)
    {
        table[i] = -1;
    }

    // Wait for all threads to finish initialization
    __syncthreads();

    // Initialize row nnz
    rocsparse_int nnz = 0;

    // alpha * A * B part
    if(mul == true)
    {
        // Get row boundaries of the current row in A
        rocsparse_int row_begin_A = csr_row_ptr_A[row] - idx_base_A;
        rocsparse_int row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

        // Loop over columns of A in current row
        for(rocsparse_int j = row_begin_A + wid; j < row_end_A; j += BLOCKSIZE / WFSIZE)
        {
            // Column of A in current row
            rocsparse_int col_A = csr_col_ind_A[j] - idx_base_A;

            // Loop over columns of B in row col_A
            rocsparse_int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
            rocsparse_int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

            for(rocsparse_int k = row_begin_B + lid; k < row_end_B; k += WFSIZE)
            {
                // Count the actual insertions to obtain row nnz of C
                nnz += insert_key<HASHVAL, HASHSIZE>(csr_col_ind_B[k] - idx_base_B, table);
            }
        }
    }

    // beta * D part
    if(add == true)
    {
        // Get row boundaries of the current row in D
        rocsparse_int row_begin_D = csr_row_ptr_D[row] - idx_base_D;
        rocsparse_int row_end_D   = csr_row_ptr_D[row + 1] - idx_base_D;

        // Loop over columns of D in current row and insert all columns of D into hash table
        for(rocsparse_int j = row_begin_D + wid; j < row_end_D; j += BLOCKSIZE / WFSIZE)
        {
            // Count the actual insertions to obtain row nnz of C
            nnz += insert_key<HASHVAL, HASHSIZE>(csr_col_ind_D[j] - idx_base_D, table);
        }
    }

    // Wait for all threads to finish hash operation
    __syncthreads();

    // Accumulate all row nnz within each (sub)wavefront to obtain the total row nnz
    // of the current row
    rocsparse_wfreduce_sum<WFSIZE>(&nnz);

    // Write result to shared memory for final reduction by first wavefront
    if(lid == WFSIZE - 1)
    {
        table[wid] = nnz;
    }

    // Wait for all threads to finish reduction
    __syncthreads();

    // Gather row nnz for the whole block
    nnz = (hipThreadIdx_x < BLOCKSIZE / WFSIZE) ? table[hipThreadIdx_x] : 0;

    // First wavefront computes final sum
    rocsparse_wfreduce_sum<BLOCKSIZE / WFSIZE>(&nnz);

    // Write result to global memory
    if(hipThreadIdx_x == BLOCKSIZE / WFSIZE - 1)
    {
        row_nnz[row] = nnz;
    }
}

#endif // CSRGEMM_DEVICE_H
