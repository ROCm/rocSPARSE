/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef COOMV_DEVICE_H
#define COOMV_DEVICE_H

#include <hip/hip_runtime.h>

// Scale kernel for beta != 1.0
template <typename T>
__global__ void coomv_scale(rocsparse_int size, T scalar, T* __restrict__ data)
{
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= size)
    {
        return;
    }

    data[gid] *= scalar;
}

// Implementation motivated by papers 'Efficient Sparse Matrix-Vector Multiplication on CUDA',
// 'Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors' and
// 'Segmented operations for sparse matrix computation on vector multiprocessors'
template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WARPSIZE>
static __device__ void coomvn_general_warp_reduce(rocsparse_int nnz,
                                                  rocsparse_int loops,
                                                  T alpha,
                                                  const rocsparse_int* coo_row_ind,
                                                  const rocsparse_int* coo_col_ind,
                                                  const T* coo_val,
                                                  const T* x,
                                                  T* y,
                                                  rocsparse_int* row_block_red,
                                                  T* val_block_red,
                                                  rocsparse_index_base idx_base)
{
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocsparse_int tid = hipThreadIdx_x;

    // Lane index (0,...,WARPSIZE)
    rocsparse_int laneid = gid % WARPSIZE;
    // Warp index
    rocsparse_int warpid = gid / WARPSIZE;

    // Initialize block buffers
    if(laneid == 0)
    {
        row_block_red[warpid] = -1;
        val_block_red[warpid] = static_cast<T>(0);
    }

    // Global COO array index start for current warp
    rocsparse_int offset = warpid * loops * WARPSIZE;

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ rocsparse_int shared_row[BLOCKSIZE];
    __shared__ T shared_val[BLOCKSIZE];

    // Initialize shared memory
    shared_row[tid] = -1;
    shared_val[tid] = static_cast<T>(0);

    __syncthreads();

    // Quick return when thread is out of bounds
    if(offset + laneid >= nnz)
    {
        return;
    }

    rocsparse_int row;
    T val;

    // Current threads index into COO structure
    rocsparse_int idx = offset + laneid;

    // Each thread processes 'loop' COO entries
    while(idx < offset + loops * WARPSIZE)
    {
        // Get corresponding COO entry, if not out of bounds.
        // This can happen when processing more than 1 entry if
        // nnz % WARPSIZE != 0
        if(idx < nnz)
        {
            row = coo_row_ind[idx] - idx_base;
            val = alpha * coo_val[idx] * __ldg(x + coo_col_ind[idx] - idx_base);
        }
        else
        {
            row = -1;
            val = static_cast<T>(0);
        }

        // First thread in warp checks row index from previous loop
        // if it has been completed or if additional rows have to be
        // appended.
        if(idx > offset && laneid == 0)
        {
            rocsparse_int prevrow = shared_row[tid + WARPSIZE - 1];
            if(row == prevrow)
            {
                val += shared_val[tid + WARPSIZE - 1];
            }
            else if(prevrow >= 0)
            {
                y[prevrow] += shared_val[tid + WARPSIZE - 1];
            }
        }

        __syncthreads();

        // Update shared buffers
        shared_row[tid] = row;
        shared_val[tid] = val;

        __syncthreads();

#pragma unroll
        // Segmented warp reduction
        for(rocsparse_int j = 1; j < WARPSIZE; j <<= 1)
        {
            if(laneid >= j)
            {
                if(row == shared_row[tid - j])
                {
                    val += shared_val[tid - j];
                }
            }
            __syncthreads();

            shared_val[tid] = val;

            __syncthreads();
        }

        // All lanes but the last one write their result in y.
        // The last value might need to be appended by the next iteration.
        if(laneid < WARPSIZE - 1)
        {
            if(row != shared_row[tid + 1] && row >= 0)
            {
                y[row] += val;
            }
        }

        // Keep going for the next iteration
        idx += WARPSIZE;
    }

    // Write last entries into buffers for segmented block reduction
    if(laneid == WARPSIZE - 1)
    {
        row_block_red[warpid] = row;
        val_block_red[warpid] = val;
    }
}

// Segmented block reduction kernel
template <typename T, rocsparse_int BLOCKSIZE>
static __device__ void segmented_blockreduce(const rocsparse_int* rows, T* vals)
{
    rocsparse_int tid = hipThreadIdx_x;
#pragma unroll
    for(rocsparse_int j = 1; j < BLOCKSIZE; j <<= 1)
    {
        T val = static_cast<T>(0);
        if(tid >= j)
        {
            if(rows[tid] == rows[tid - j])
            {
                val = vals[tid - j];
            }
        }
        __syncthreads();

        vals[tid] += val;
        __syncthreads();
    }
}

// Do the final block reduction of the block reduction buffers back into global memory
template <typename T, rocsparse_int BLOCKSIZE>
__global__ void coomvn_general_block_reduce(rocsparse_int nnz,
                                            const rocsparse_int* __restrict__ row_block_red,
                                            const T* __restrict__ val_block_red,
                                            T* y)
{
    rocsparse_int tid = hipThreadIdx_x;

    // Quick return when thread is out of bounds
    if(tid >= nnz)
    {
        return;
    }

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ rocsparse_int shared_row[BLOCKSIZE];
    __shared__ T shared_val[BLOCKSIZE];

    // Loop over blocks that are subject for segmented reduction
    for(rocsparse_int i = tid; i < nnz; i += BLOCKSIZE)
    {
        // Copy data to reduction buffers
        shared_row[tid] = row_block_red[i];
        shared_val[tid] = val_block_red[i];

        __syncthreads();

        // Do segmented block reduction
        segmented_blockreduce<T, BLOCKSIZE>(shared_row, shared_val);

        // Add reduced sum to y if valid
        rocsparse_int row = shared_row[tid];
        if(row != shared_row[tid + 1] && row >= 0)
        {
            y[row] += shared_val[tid];
        }

        __syncthreads();
    }
}

#endif // COOMV_DEVICE_H
