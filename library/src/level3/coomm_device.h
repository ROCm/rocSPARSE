/*! \file */
/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef COOMM_DEVICE_H
#define COOMM_DEVICE_H

#include "common.h"

// Scale kernel for beta != 1.0
template <typename I, typename T>
__device__ void
    coomm_scale_device(I m, I n, T beta, T* __restrict__ data, I ld, rocsparse_order order)
{
    I gidx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    I gidy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(gidx >= m || gidy >= n)
    {
        return;
    }

    if(order == rocsparse_order_column)
    {
        data[gidx + ld * gidy] = data[gidx + ld * gidy] * beta;
    }
    else
    {
        data[gidy + ld * gidx] = data[gidy + ld * gidx] * beta;
    }
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, bool TRANSB, typename I, typename T>
static __device__ void coommnn_general_wf_segmented(I                    nnz,
                                                    I                    n,
                                                    I                    loops,
                                                    T                    alpha,
                                                    const I*             coo_row_ind,
                                                    const I*             coo_col_ind,
                                                    const T*             coo_val,
                                                    const T*             B,
                                                    I                    ldb,
                                                    T*                   C,
                                                    I                    ldc,
                                                    I*                   row_block_red,
                                                    T*                   val_block_red,
                                                    rocsparse_order      order,
                                                    rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;

    // Lane index (0,...,WF_SIZE)
    int lid = tid & (WF_SIZE - 1);
    // Wavefront index
    I wid = tid / WF_SIZE;

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ I shared_row[WF_SIZE];
    __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE + 1];

    I col    = BLOCKSIZE / WF_SIZE * hipBlockIdx_y + wid;
    I offset = hipBlockIdx_x * loops * WF_SIZE;

    // Current threads index into COO structure
    I idx = offset + lid;

    I row;
    T val;

    // Each thread processes 'loop' COO entries
    while(idx < offset + loops * WF_SIZE)
    {
        // Get corresponding COO entry
        I r = (idx < nnz) ? rocsparse_nontemporal_load(coo_row_ind + idx) - idx_base : -1;
        I c = (idx < nnz) ? rocsparse_nontemporal_load(coo_col_ind + idx) - idx_base : 0;
        T v = (idx < nnz) ? alpha * rocsparse_nontemporal_load(coo_val + idx) : static_cast<T>(0);

        row = r;

        if(!TRANSB)
        {
            val = (col < n) ? v * rocsparse_ldg(B + col * ldb + c) : static_cast<T>(0);
        }
        else
        {
            val = (col < n) ? v * rocsparse_ldg(B + c * ldb + col) : static_cast<T>(0);
        }

        // First thread in wavefront checks row index from previous loop
        // if it has been completed or if additional rows have to be
        // appended.
        if(idx > offset && lid == 0 && col < n)
        {
            I prevrow = shared_row[WF_SIZE - 1];
            if(row == prevrow)
            {
                val = val + shared_val[wid][WF_SIZE - 1];
            }
            else if(prevrow >= 0)
            {
                if(order == rocsparse_order_column)
                {
                    C[prevrow + col * ldc] = C[prevrow + col * ldc] + shared_val[wid][WF_SIZE - 1];
                }
                else
                {
                    C[col + prevrow * ldc] = C[col + prevrow * ldc] + shared_val[wid][WF_SIZE - 1];
                }
            }
        }

        __syncthreads();

        shared_val[wid][lid] = val;
        shared_row[lid]      = row;

        __syncthreads();

#pragma unroll
        // Segmented wavefront reduction
        for(int j = 1; j < WF_SIZE; j <<= 1)
        {
            if(lid >= j)
            {
                if(row == shared_row[lid - j])
                {
                    val = val + shared_val[wid][lid - j];
                }
            }
            __syncthreads();

            shared_val[wid][lid] = val;

            __syncthreads();
        }

        // All lanes but the last one write their result in C.
        // The last value might need to be appended by the next iteration.
        if(lid < WF_SIZE - 1 && col < n)
        {
            if(row != shared_row[lid + 1] && row >= 0)
            {
                if(order == rocsparse_order_column)
                {
                    C[row + col * ldc] = C[row + col * ldc] + val;
                }
                else
                {
                    C[col + row * ldc] = C[col + row * ldc] + val;
                }
            }
        }

        idx += WF_SIZE;
    }

    // Write last entries into buffers for segmented block reduction
    if(lid == WF_SIZE - 1 && col < n)
    {
        rocsparse_nontemporal_store(row, row_block_red + hipBlockIdx_x + hipGridDim_x * col);
        rocsparse_nontemporal_store(val, val_block_red + hipBlockIdx_x + hipGridDim_x * col);
    }
}

// Segmented block reduction kernel
template <unsigned int BLOCKSIZE, typename I, typename T>
static __device__ void segmented_blockreduce(const I* rows, T* vals)
{
    int tid = hipThreadIdx_x;

#pragma unroll
    for(int j = 1; j < BLOCKSIZE; j <<= 1)
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

        vals[tid] = vals[tid] + val;
        __syncthreads();
    }
}

// Do the final block reduction of the block reduction buffers back into global memory
template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void coommn_general_block_reduce(I nblocks,
                                     const I* __restrict__ row_block_red,
                                     const T* __restrict__ val_block_red,
                                     T*              C,
                                     I               ldc,
                                     rocsparse_order order)
{
    int tid = hipThreadIdx_x;

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ I shared_row[BLOCKSIZE];
    __shared__ T shared_val[BLOCKSIZE];

    shared_row[tid] = -1;
    shared_val[tid] = static_cast<T>(0);

    __syncthreads();

    I col = hipBlockIdx_x;

    for(I i = tid; i < nblocks; i += BLOCKSIZE)
    {
        // Copy data to reduction buffers
        shared_row[tid] = row_block_red[i + nblocks * col];
        shared_val[tid] = val_block_red[i + nblocks * col];

        __syncthreads();

        // Do segmented block reduction
        segmented_blockreduce<BLOCKSIZE>(shared_row, shared_val);

        // Add reduced sum to C if valid
        I row   = shared_row[tid];
        I rowp1 = (tid < BLOCKSIZE - 1) ? shared_row[tid + 1] : -1;

        if(row != rowp1 && row >= 0)
        {
            if(order == rocsparse_order_column)
            {
                C[row + ldc * col] = C[row + ldc * col] + shared_val[tid];
            }
            else
            {
                C[col + ldc * row] = C[col + ldc * row] + shared_val[tid];
            }
        }

        __syncthreads();
    }
}

template <unsigned int BLK_SIZE_X,
          unsigned int BLK_SIZE_Y,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T>
static __device__ void coommnn_general_wf_atomic(I                    nnz,
                                                 I                    n,
                                                 I                    nblocks,
                                                 T                    alpha,
                                                 const I*             coo_row_ind,
                                                 const I*             coo_col_ind,
                                                 const T*             coo_val,
                                                 const T*             B,
                                                 I                    ldb,
                                                 T*                   C,
                                                 I                    ldc,
                                                 rocsparse_order      order,
                                                 rocsparse_index_base idx_base)
{
    I idx = (BLK_SIZE_Y * hipBlockIdx_x + hipThreadIdx_y) * LOOPS;

    I col = hipBlockIdx_y * BLK_SIZE_X + hipThreadIdx_x;

    if(col >= n || idx >= nnz)
    {
        return;
    }

    T temp = static_cast<T>(0);

    I row = coo_row_ind[idx] - idx_base;

    I end = (idx + LOOPS > nnz) ? nnz - 1 : (idx + LOOPS) - 1;
    while(idx < end)
    {
        if(!TRANSB)
        {
            temp += coo_val[idx] * B[col * ldb + (coo_col_ind[idx] - idx_base)];
        }
        else
        {
            temp += coo_val[idx] * B[(coo_col_ind[idx] - idx_base) * ldb + col];
        }

        I nrow = coo_row_ind[idx + 1] - idx_base;
        if(row != nrow)
        {
            if(order == rocsparse_order_column)
            {
                atomicAdd(&C[col * ldc + row], alpha * temp);
            }
            else
            {
                atomicAdd(&C[row * ldc + col], alpha * temp);
            }

            row  = nrow;
            temp = static_cast<T>(0);
        }

        idx++;
    }

    if(!TRANSB)
    {
        temp += coo_val[idx] * B[col * ldb + (coo_col_ind[idx] - idx_base)];
    }
    else
    {
        temp += coo_val[idx] * B[(coo_col_ind[idx] - idx_base) * ldb + col];
    }

    if(order == rocsparse_order_column)
    {
        atomicAdd(&C[col * ldc + row], alpha * temp);
    }
    else
    {
        atomicAdd(&C[row * ldc + col], alpha * temp);
    }
}

#endif // COOMM_DEVICE_H
