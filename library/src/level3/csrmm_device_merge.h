/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

// See Yang C., Bulu? A., Owens J.D. (2018) Design Principles for Sparse Matrix Multiplication on the GPU.
// In: Aldinucci M., Padovani L., Torquati M. (eds) Euro-Par 2018: Parallel Processing. Euro-Par 2018.
// Lecture Notes in Computer Science, vol 11014. Springer, Cham. https://doi.org/10.1007/978-3-319-96983-1_48
template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          bool         TRANSB,
          typename I,
          typename J,
          typename T>
static ROCSPARSE_DEVICE_ILF void csrmmnt_merge_main_device(bool conj_A,
                                                           bool conj_B,
                                                           J    ncol,
                                                           J    M,
                                                           J    N,
                                                           J    K,
                                                           I    nnz,
                                                           T    alpha,
                                                           J* __restrict__ row_block_red,
                                                           T* __restrict__ val_block_red,
                                                           const J* __restrict__ row_limits,
                                                           const I* __restrict__ csr_row_ptr,
                                                           const J* __restrict__ csr_col_ind,
                                                           const T* __restrict__ csr_val,
                                                           const T* __restrict__ B,
                                                           J ldb,
                                                           T beta,
                                                           T* __restrict__ C,
                                                           J                    ldc,
                                                           rocsparse_order      order,
                                                           rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    int bid = hipBlockIdx_x;
    int lid = tid & (WF_SIZE - 1);
    int wid = tid / WF_SIZE;

    __shared__ J shared_row[BLOCKSIZE];
    __shared__ T shared_val[BLOCKSIZE * WF_SIZE];

    J left  = row_limits[bid];
    J right = row_limits[bid + 1];

    J row_ind = -1;
    J col_ind = 0;
    T val     = static_cast<T>(0);

    if((BLOCKSIZE * bid + tid) < nnz)
    {
        while(left < right)
        {
            J mid = (left + right) / 2;
            if((csr_row_ptr[mid + 1] - idx_base) <= (BLOCKSIZE * bid + tid))
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }
        row_ind = left;
        col_ind = rocsparse_nontemporal_load(&csr_col_ind[BLOCKSIZE * bid + tid]) - idx_base;
        val = alpha * conj_val(rocsparse_nontemporal_load(&csr_val[BLOCKSIZE * bid + tid]), conj_A);
    }

    shared_row[tid] = row_ind;
    __syncthreads();

    for(J colB = 0; colB < ncol; colB += WF_SIZE)
    {
        T valB[WF_SIZE];
        for(J i = 0; i < WF_SIZE; ++i)
        {
            T v = rocsparse_shfl(val, i, WF_SIZE);
            J c = __shfl(col_ind, i, WF_SIZE);

            if(!TRANSB)
            {
                valB[i] = v * conj_val(B[c + ldb * (colB + lid)], conj_B);
            }
            else
            {
                valB[i] = v * conj_val(B[ldb * c + colB + lid], conj_B);
            }
        }

        for(J i = 0; i < WF_SIZE; ++i)
        {
            shared_val[BLOCKSIZE * lid + WF_SIZE * wid + i] = valB[i];
        }
        __syncthreads();

        for(J i = 0; i < WF_SIZE; ++i)
        {
            valB[i] = shared_val[BLOCKSIZE * i + tid];
        }

        // segmented reduction
        for(J j = 1; j < BLOCKSIZE; j <<= 1)
        {
            if(tid >= j)
            {
                if(row_ind == shared_row[tid - j])
                {
                    for(J i = 0; i < WF_SIZE; ++i)
                    {
                        valB[i] = valB[i] + shared_val[BLOCKSIZE * i + tid - j];
                    }
                }
            }
            __syncthreads();
            for(J i = 0; i < WF_SIZE; ++i)
            {
                shared_val[BLOCKSIZE * i + tid] = valB[i];
            }
            __syncthreads();
        }

        // All lanes but the last one write their result in C.
        if(tid < (BLOCKSIZE - 1))
        {
            if(row_ind != shared_row[tid + 1] && row_ind >= 0)
            {
                if(order == rocsparse_order_column)
                {
                    for(J i = 0; i < WF_SIZE; ++i)
                    {
                        C[row_ind + ldc * (colB + i)] += valB[i];
                    }
                }
                else
                {
                    for(J i = 0; i < WF_SIZE; ++i)
                    {
                        C[colB + i + ldc * row_ind] += valB[i];
                    }
                }
            }
        }

        if(tid == (BLOCKSIZE - 1))
        {
            for(J i = 0; i < WF_SIZE; ++i)
            {
                val_block_red[hipGridDim_x * (colB + i) + bid] = valB[i];
            }
        }
    }

    if(tid == (BLOCKSIZE - 1))
    {
        row_block_red[bid] = row_ind;
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          bool         TRANSB,
          typename I,
          typename J,
          typename T>
static ROCSPARSE_DEVICE_ILF void csrmmnt_merge_remainder_device(bool conj_A,
                                                                bool conj_B,
                                                                J    offset,
                                                                J    M,
                                                                J    N,
                                                                J    K,
                                                                I    nnz,
                                                                T    alpha,
                                                                J* __restrict__ row_block_red,
                                                                T* __restrict__ val_block_red,
                                                                const J* __restrict__ row_limits,
                                                                const I* __restrict__ csr_row_ptr,
                                                                const J* __restrict__ csr_col_ind,
                                                                const T* __restrict__ csr_val,
                                                                const T* __restrict__ B,
                                                                J ldb,
                                                                T beta,
                                                                T* __restrict__ C,
                                                                J                    ldc,
                                                                rocsparse_order      order,
                                                                rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    int bid = hipBlockIdx_x;
    int lid = tid & (WF_SIZE - 1);
    int wid = tid / WF_SIZE;

    __shared__ J shared_row[BLOCKSIZE];
    __shared__ T shared_val[BLOCKSIZE * WF_SIZE];

    J left  = row_limits[bid];
    J right = row_limits[bid + 1];

    J row_ind = -1;
    J col_ind = 0;
    T val     = static_cast<T>(0);

    if(BLOCKSIZE * bid + tid < nnz)
    {
        while(left < right)
        {
            J mid = (left + right) / 2;
            if((csr_row_ptr[mid + 1] - idx_base) <= (BLOCKSIZE * bid + tid))
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }
        row_ind = left;
        col_ind = rocsparse_nontemporal_load(&csr_col_ind[BLOCKSIZE * bid + tid]) - idx_base;
        val = alpha * conj_val(rocsparse_nontemporal_load(&csr_val[BLOCKSIZE * bid + tid]), conj_A);
    }

    J colB = offset;

    T valB[WF_SIZE];
    for(J i = 0; i < WF_SIZE; ++i)
    {
        T v = rocsparse_shfl(val, i, WF_SIZE);
        J c = __shfl(col_ind, i, WF_SIZE);

        if(!TRANSB)
        {
            valB[i] = (colB + lid) < N ? v * conj_val(B[c + ldb * (colB + lid)], conj_B)
                                       : static_cast<T>(0);
        }
        else
        {
            valB[i] = (colB + lid) < N ? v * conj_val(B[ldb * c + (colB + lid)], conj_B)
                                       : static_cast<T>(0);
        }
    }

    __syncthreads();
    shared_row[tid] = row_ind;
    for(J i = 0; i < WF_SIZE; ++i)
    {
        shared_val[BLOCKSIZE * lid + WF_SIZE * wid + i] = valB[i];
    }
    __syncthreads();

    for(J i = 0; i < WF_SIZE; ++i)
    {
        valB[i] = shared_val[BLOCKSIZE * i + tid];
    }

    // segmented reduction
    for(J j = 1; j < BLOCKSIZE; j <<= 1)
    {
        if(tid >= j)
        {
            if(row_ind == shared_row[tid - j])
            {
                for(J i = 0; i < WF_SIZE; ++i)
                {
                    valB[i] = valB[i] + shared_val[BLOCKSIZE * i + tid - j];
                }
            }
        }
        __syncthreads();
        for(J i = 0; i < WF_SIZE; ++i)
        {
            shared_val[BLOCKSIZE * i + tid] = valB[i];
        }
        __syncthreads();
    }

    // All lanes but the last one write their result in C.
    if(tid < (BLOCKSIZE - 1))
    {
        if(row_ind != shared_row[tid + 1] && row_ind >= 0)
        {
            if(order == rocsparse_order_column)
            {
                for(J i = 0; i < WF_SIZE; ++i)
                {
                    if((colB + i) < N)
                    {
                        C[row_ind + ldc * (colB + i)] += valB[i];
                    }
                }
            }
            else
            {
                for(J i = 0; i < WF_SIZE; ++i)
                {
                    if((colB + i) < N)
                    {
                        C[colB + i + ldc * row_ind] += valB[i];
                    }
                }
            }
        }
    }

    if(tid == (BLOCKSIZE - 1))
    {
        row_block_red[bid] = row_ind;
        for(J i = 0; i < WF_SIZE; ++i)
        {
            if((colB + i) < N)
            {
                val_block_red[hipGridDim_x * (colB + i) + bid] = valB[i];
            }
        }
    }
}

// Segmented block reduction kernel
template <unsigned int BLOCKSIZE, typename I, typename T>
static ROCSPARSE_DEVICE_ILF void segmented_blockreduce(const I* __restrict__ rows,
                                                       T* __restrict__ vals)
{
    int tid = hipThreadIdx_x;

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
template <unsigned int BLOCKSIZE, typename I, typename J, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmmnt_general_block_reduce(I nblocks,
                                      const J* __restrict__ row_block_red,
                                      const T* __restrict__ val_block_red,
                                      T*              C,
                                      J               ldc,
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
        shared_row[tid] = row_block_red[i];
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

// Scale kernel for beta != 1.0
template <unsigned int BLOCKSIZE, typename I, typename T>
static ROCSPARSE_DEVICE_ILF void
    csrmmnn_merge_scale_device(I m, I n, T beta, T* __restrict__ data, I ld, rocsparse_order order)
{
    I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

    if(gid >= m * n)
    {
        return;
    }

    I wid = (order == rocsparse_order_column) ? gid / m : gid / n;
    I lid = (order == rocsparse_order_column) ? gid % m : gid % n;

    if(beta == static_cast<T>(0))
    {
        data[lid + ld * wid] = static_cast<T>(0);
    }
    else
    {
        data[lid + ld * wid] *= beta;
    }
}

template <unsigned int BLOCKSIZE, unsigned int NNZ_PER_BLOCK, typename I, typename J>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmmnn_merge_compute_row_limits(J m,
                                          I nblocks,
                                          I nnz,
                                          const I* __restrict__ csr_row_ptr,
                                          J* __restrict__ row_limits,
                                          rocsparse_index_base idx_base)
{
    I gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

    if(gid >= nblocks)
    {
        return;
    }

    I s0 = NNZ_PER_BLOCK * gid;

    J left  = 0;
    J right = m;
    J mid   = (left + right) / 2;
    while((csr_row_ptr[left] - idx_base) < s0 && left < mid && right > mid)
    {
        if((csr_row_ptr[mid] - idx_base) <= s0)
        {
            left = mid;
        }
        else
        {
            right = mid;
        }
        mid = (left + right) / 2;
    }

    row_limits[gid] = left;

    if(gid == nblocks - 1)
    {
        row_limits[gid + 1] = m;
    }
}
