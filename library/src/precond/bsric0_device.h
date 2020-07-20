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
#ifndef BSRIC0_DEVICE_H
#define BSRIC0_DEVICE_H

#include "common.h"

#include <hip/hip_runtime.h>

template <typename T,
          rocsparse_int BLOCKSIZE,
          rocsparse_int WFSIZE,
          rocsparse_int MAX_NNZB,
          rocsparse_int BSR_BLOCK_DIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsric0_small_maxnnzb_unrolled8x8_kernel(rocsparse_direction direction,
                                                 rocsparse_int       mb,
                                                 rocsparse_int       block_dim,
                                                 const rocsparse_int* __restrict__ bsr_row_ptr,
                                                 const rocsparse_int* __restrict__ bsr_col_ind,
                                                 T* __restrict__ bsr_val,
                                                 const rocsparse_int* __restrict__ bsr_diag_ind,
                                                 int* __restrict__ block_done,
                                                 const rocsparse_int* __restrict__ block_map,
                                                 rocsparse_int* __restrict__ zero_pivot,
                                                 rocsparse_index_base idx_base)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int tidz = hipThreadIdx_z;

    rocsparse_int gid = tidx + BSR_BLOCK_DIM * tidz;

    __shared__ rocsparse_int columns[MAX_NNZB];
    __shared__ rocsparse_int index[MAX_NNZB];
    __shared__ T             row_sum[BSR_BLOCK_DIM * BSR_BLOCK_DIM];
    __shared__ T             local_values[MAX_NNZB * BSR_BLOCK_DIM * BSR_BLOCK_DIM];
    __shared__ T             values[MAX_NNZB * BSR_BLOCK_DIM * BSR_BLOCK_DIM];

    // Current block row this wavefront is working on
    rocsparse_int block_row = block_map[hipBlockIdx_x];

    // Block diagonal entry point of the current block row
    rocsparse_int block_row_diag = bsr_diag_ind[block_row];

    // If one thread in the warp breaks here, then all threads in
    // the warp break so no divergence
    if(block_row_diag == -1)
    {
        __threadfence();

        if(gid == 0)
        {
            atomicMin(zero_pivot, block_row + idx_base);

            // Last lane in wavefront writes "we are done" flag for its block row
            atomicOr(&block_done[block_row], 1);
        }

        return;
    }

    // Block row entry point
    rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;

    // Load current row column indices into shared memory
    for(rocsparse_int j = block_row_begin + gid; j < block_row_diag + 1;
        j += BSR_BLOCK_DIM * BSR_BLOCK_DIM)
    {
        columns[j - block_row_begin] = bsr_col_ind[j] - idx_base;
    }

    // Load current row diagonal block into shared memory
    if(direction == rocsparse_direction_row)
    {
        values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
               + BSR_BLOCK_DIM * tidz + tidx]
            = bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * block_row_diag + BSR_BLOCK_DIM * tidz + tidx];
    }
    else
    {
        values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
               + BSR_BLOCK_DIM * tidz + tidx]
            = bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * block_row_diag + BSR_BLOCK_DIM * tidx + tidz];
    }

    // Block row sum accumulator
    row_sum[BSR_BLOCK_DIM * tidz + tidx] = static_cast<T>(0);

    __threadfence_block();

    // Loop over non-diagonal block columns of current block row
    for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
    {
        // Block column index currently being processes
        rocsparse_int block_col = columns[j - block_row_begin];

        // Beginning of the row that corresponds to block_col
        rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

        // Diagonal entry point of row block_col
        rocsparse_int local_block_diag = bsr_diag_ind[block_col];

        // Structural zero pivot, do not process this row
        if(local_block_diag == -1)
        {
            // If one thread in the warp breaks here, then all threads in
            // the warp break so no divergence
            break;
            //local_block_diag = block_row_diag - 1;
        }

        // Load current j block into shared memory
        if(direction == rocsparse_direction_row)
        {
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                   + tidx]
                = bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * j + BSR_BLOCK_DIM * tidz + tidx];
        }
        else
        {
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                   + tidx]
                = bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * j + BSR_BLOCK_DIM * tidx + tidz];
        }

        for(rocsparse_int p = local_block_begin + gid; p < local_block_diag + 1;
            p += BSR_BLOCK_DIM * BSR_BLOCK_DIM)
        {
            // Perform a binary search to find matching block columns
            rocsparse_int l = 0;
            rocsparse_int r = block_row_diag - block_row_begin;
            rocsparse_int m = (r + l) >> 1;

            rocsparse_int block_col_j = columns[m];
            rocsparse_int block_col_p = bsr_col_ind[p] - idx_base;

            // Binary search for block column
            while(l < r)
            {
                if(block_col_j < block_col_p)
                {
                    l = m + 1;
                }
                else
                {
                    r = m;
                }

                m           = (r + l) >> 1;
                block_col_j = columns[m];
            }

            // Check if a match has been found
            index[p - local_block_begin]
                = (block_col_j == block_col_p) ? BSR_BLOCK_DIM * BSR_BLOCK_DIM * m : -1;
        }

        __threadfence_block();

        // Spin loop until dependency has been resolved
        while(!atomicOr(&block_done[block_col], 0))
            ;

        __threadfence();

        // Load local block row into shared memory
        rocsparse_int count = 0;
        for(rocsparse_int l = local_block_begin; l < local_block_diag + 1; l++)
        {
            rocsparse_int idx = index[l - local_block_begin];

            if(idx != -1)
            {
                if(direction == rocsparse_direction_row)
                {
                    local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * count + BSR_BLOCK_DIM * tidz
                                 + tidx]
                        = bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * tidz + tidx];
                }
                else
                {
                    local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * count + BSR_BLOCK_DIM * tidz
                                 + tidx]
                        = bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * tidx + tidz];
                }

                index[count] = idx;

                count++;
            }
        }

        __threadfence_block();

        for(rocsparse_int k = 0; k < BSR_BLOCK_DIM; k++)
        {
            // Current value
            T val = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                           + BSR_BLOCK_DIM * tidz + k];

            // Load diagonal entry
            T diag_val
                = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (count - 1) + BSR_BLOCK_DIM * k + k];

            // Row has numerical zero pivot
            if(diag_val == static_cast<T>(0))
            {
                if(gid == 0)
                {
                    // We are looking for the first zero pivot
                    atomicMin(zero_pivot, block_col + idx_base);
                }

                diag_val = static_cast<T>(1);
            }

            // Compute reciprocal
            diag_val = static_cast<T>(1) / diag_val;

            // Local row sum
            T local_sum = static_cast<T>(0);

            for(rocsparse_int l = 0; l < count - 1; l++)
            {
                rocsparse_int idx = index[l];

                if(BSR_BLOCK_DIM >= 1)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * k + 0];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + 0];
                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
                if(BSR_BLOCK_DIM >= 2)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * k + 1];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + 1];
                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
                if(BSR_BLOCK_DIM >= 3)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * k + 2];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + 2];
                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
                if(BSR_BLOCK_DIM >= 4)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * k + 3];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + 3];
                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
                if(BSR_BLOCK_DIM >= 5)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * k + 4];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + 4];
                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
                if(BSR_BLOCK_DIM >= 6)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * k + 5];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + 5];
                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
                if(BSR_BLOCK_DIM >= 7)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * k + 6];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + 6];
                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
                if(BSR_BLOCK_DIM >= 8)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * l + BSR_BLOCK_DIM * k + 7];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + 7];
                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
            }

            rocsparse_int idx = index[count - 1];

            for(rocsparse_int p = 0; p < k; p++)
            {
                T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (count - 1) + BSR_BLOCK_DIM * k
                                    + p];
                T v2 = values[idx + BSR_BLOCK_DIM * tidz + p];
                local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
            }

            val = (val - local_sum) * diag_val;
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz + k]
                = val;

            __threadfence_block();

            row_sum[BSR_BLOCK_DIM * tidz + tidx] = rocsparse_fma(
                values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                       + k],
                rocsparse_conj(values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                                      + BSR_BLOCK_DIM * tidx + k]),
                row_sum[BSR_BLOCK_DIM * tidz + tidx]);

            __threadfence_block();
        }
    }

    // Handle diagonal block column of block row.
    for(rocsparse_int k = 0; k < BSR_BLOCK_DIM; k++)
    {
        rocsparse_int row_diag = BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                                 + BSR_BLOCK_DIM * k + k;

        if(k == tidz)
        {
            values[row_diag]
                = sqrt(rocsparse_abs(values[row_diag] - row_sum[BSR_BLOCK_DIM * k + k]));
        }

        __threadfence_block();

        // Load value
        T val = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                       + BSR_BLOCK_DIM * tidz + k];

        // Load diagonal entry
        T diag_val = values[row_diag];

        // Row has numerical zero pivot
        if(diag_val == static_cast<T>(0))
        {
            if(gid == 0)
            {
                // We are looking for the first zero pivot
                atomicMin(zero_pivot, block_row + idx_base);
            }

            // Normally would break here but to avoid divergence set diag_val to one and continue
            // The zero pivot has already been set so further computation does not matter
            diag_val = static_cast<T>(1);
        }

        // Compute reciprocal
        diag_val = static_cast<T>(1) / diag_val;

        // Local row sum
        T local_sum = row_sum[BSR_BLOCK_DIM * tidz + k];

        if(k < tidz)
        {
            val = (val - local_sum) * diag_val;
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                   + BSR_BLOCK_DIM * tidz + k]
                = val;

            __threadfence_block();

            row_sum[BSR_BLOCK_DIM * tidz + tidx] = rocsparse_fma(
                val,
                rocsparse_conj(
                    values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                           + BSR_BLOCK_DIM * tidx + k]),
                row_sum[BSR_BLOCK_DIM * tidz + tidx]);
        }

        __threadfence_block();
    }

    // Copy row block values from shared memory back to global memory
    for(rocsparse_int j = block_row_begin; j < block_row_diag + 1; j++)
    {
        if(direction == rocsparse_direction_row)
        {
            bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * j + BSR_BLOCK_DIM * tidz + tidx]
                = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                         + BSR_BLOCK_DIM * tidz + tidx];
        }
        else
        {
            bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * j + BSR_BLOCK_DIM * tidx + tidz]
                = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                         + BSR_BLOCK_DIM * tidz + tidx];
        }
    }

    __threadfence();

    // Last lane in wavefront writes "we are done" flag for its block row
    atomicOr(&block_done[block_row], 1);
}

template <typename T,
          rocsparse_int BLOCKSIZE,
          rocsparse_int WFSIZE,
          rocsparse_int MAX_NNZB,
          rocsparse_int BSR_BLOCK_DIM,
          rocsparse_int BLK_SIZE_Y>
__launch_bounds__(BLOCKSIZE) __global__ void bsric0_hash_small_maxnnzb_small_blockdim_kernel(
    rocsparse_direction direction,
    rocsparse_int       mb,
    rocsparse_int       block_dim,
    const rocsparse_int* __restrict__ bsr_row_ptr,
    const rocsparse_int* __restrict__ bsr_col_ind,
    T* __restrict__ bsr_val,
    const rocsparse_int* __restrict__ bsr_diag_ind,
    int* __restrict__ block_done,
    const rocsparse_int* __restrict__ block_map,
    rocsparse_int* __restrict__ zero_pivot,
    rocsparse_index_base idx_base)
{
    constexpr rocsparse_int BLK_SIZE_X = BSR_BLOCK_DIM;

    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int tidy = hipThreadIdx_y;
    rocsparse_int tidz = hipThreadIdx_z;

    rocsparse_int gid = tidx + BLK_SIZE_X * tidy + BLK_SIZE_X * BLK_SIZE_Y * tidz;
    rocsparse_int rid = tidx + BLK_SIZE_X * tidy;
    rocsparse_int lid = gid & (WFSIZE - 1);

    __shared__ rocsparse_int table[MAX_NNZB];
    __shared__ rocsparse_int data[MAX_NNZB];
    __shared__ rocsparse_int index[MAX_NNZB];
    __shared__ T             row_sum[BSR_BLOCK_DIM * BSR_BLOCK_DIM];
    __shared__ T             local_values[MAX_NNZB * BSR_BLOCK_DIM * BSR_BLOCK_DIM];
    __shared__ T             values[MAX_NNZB * BSR_BLOCK_DIM * BSR_BLOCK_DIM];

    // Current block row this wavefront is working on
    rocsparse_int block_row = block_map[hipBlockIdx_x];

    // Block diagonal entry point of the current block row
    rocsparse_int block_row_diag = bsr_diag_ind[block_row];

    // If one thread in the warp breaks here, then all threads in
    // the warp break so no divergence
    if(block_row_diag == -1)
    {
        __threadfence();

        if(lid == WFSIZE - 1)
        {
            atomicMin(zero_pivot, block_row + idx_base);

            // Last lane in wavefront writes "we are done" flag for its block row
            atomicOr(&block_done[block_row], 1);
        }

        return;
    }

    // Block row entry point
    rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;

    // Initialize hash table with -1
    for(rocsparse_int j = lid; j < MAX_NNZB; j += WFSIZE)
    {
        table[j] = -1;
    }

    __threadfence_block();

    // Fill hash table
    for(rocsparse_int j = block_row_begin + lid; j < block_row_diag + 1; j += WFSIZE)
    {
        // Insert block_key into hash table
        rocsparse_int block_key = bsr_col_ind[j] - idx_base;

        // Compute block hash
        rocsparse_int block_hash = (block_key * 103) % MAX_NNZB;

        // Hash operation
        while(true)
        {
            if(table[block_hash] == block_key)
            {
                // block_key is already inserted, block done
                break;
            }
            else if(atomicCAS(&table[block_hash], -1, block_key) == -1)
            {
                data[block_hash] = j - block_row_begin;

                break;
            }
            else
            {
                // collision, compute new block_hash
                block_hash = (block_hash + 1) % MAX_NNZB;
            }
        }
    }

    // Load current diagonal block into shared memory
    if(direction == rocsparse_direction_row)
    {
        values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
               + BSR_BLOCK_DIM * tidz + tidx]
            = (tidx < block_dim && tidz < block_dim)
                  ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidz + tidx]
                  : static_cast<T>(0);
    }
    else
    {
        values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
               + BSR_BLOCK_DIM * tidz + tidx]
            = (tidx < block_dim && tidz < block_dim)
                  ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidx + tidz]
                  : static_cast<T>(0);
    }

    // Block row sum accumulator
    row_sum[BSR_BLOCK_DIM * tidz + tidx] = static_cast<T>(0);

    __threadfence_block();

    // Loop over non-diagonal block columns of current block row
    for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
    {
        // Block column index currently being processes
        rocsparse_int block_col = bsr_col_ind[j] - idx_base;

        // Beginning of the row that corresponds to block_col
        rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

        // Diagonal entry point of row block_col
        rocsparse_int local_block_diag = bsr_diag_ind[block_col];

        // Structural zero pivot, do not process this row
        if(local_block_diag == -1)
        {
            // If one thread in the warp breaks here, then all threads in
            // the warp break so no divergence
            break;
            //local_block_diag = block_row_diag - 1;
        }

        // Load current j block into shared memory
        if(direction == rocsparse_direction_row)
        {
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                   + tidx]
                = (tidx < block_dim && tidz < block_dim)
                      ? bsr_val[block_dim * block_dim * j + block_dim * tidz + tidx]
                      : static_cast<T>(0);
        }
        else
        {
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                   + tidx]
                = (tidx < block_dim && tidz < block_dim)
                      ? bsr_val[block_dim * block_dim * j + block_dim * tidx + tidz]
                      : static_cast<T>(0);
        }

        for(rocsparse_int l = local_block_begin + tidy; l < local_block_diag + 1; l += BLK_SIZE_Y)
        {
            rocsparse_int block_key = bsr_col_ind[l] - idx_base;

            // Compute block hash
            rocsparse_int block_hash = (block_key * 103) % MAX_NNZB;

            rocsparse_int idx = -1;

            // Hash operation
            rocsparse_int count = 0;
            while(count < MAX_NNZB)
            {
                count++;

                if(table[block_hash] == -1)
                {
                    // no entry for the key, done
                    break;
                }
                else if(table[block_hash] == block_key)
                {
                    idx = BSR_BLOCK_DIM * BSR_BLOCK_DIM * data[block_hash];
                    break;
                }
                else
                {
                    // collision, compute new block_hash
                    block_hash = (block_hash + 1) % MAX_NNZB;
                }
            }

            index[l - local_block_begin] = idx;
        }

        __threadfence_block();

        // Spin loop until dependency has been resolved
        while(!atomicOr(&block_done[block_col], 0))
            ;

        __threadfence();

        // Load local block row into shared memory
        for(rocsparse_int l = local_block_begin + tidy; l < local_block_diag + 1; l += BLK_SIZE_Y)
        {
            rocsparse_int idx = index[l - local_block_begin];

            if(idx != -1)
            {
                if(direction == rocsparse_direction_row)
                {
                    local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (l - local_block_begin)
                                 + BSR_BLOCK_DIM * tidz + tidx]
                        = (tidx < block_dim && tidz < block_dim)
                              ? bsr_val[block_dim * block_dim * l + block_dim * tidz + tidx]
                              : static_cast<T>(0);
                }
                else
                {
                    local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (l - local_block_begin)
                                 + BSR_BLOCK_DIM * tidz + tidx]
                        = (tidx < block_dim && tidz < block_dim)
                              ? bsr_val[block_dim * block_dim * l + block_dim * tidx + tidz]
                              : static_cast<T>(0);
                }
            }
        }

        __threadfence_block();

        for(rocsparse_int k = 0; k < block_dim; k++)
        {
            // Current value
            T val = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                           + BSR_BLOCK_DIM * tidz + k];

            // Load diagonal entry
            T diag_val = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM
                                          * (local_block_diag - local_block_begin)
                                      + BSR_BLOCK_DIM * k + k];

            // Row has numerical zero pivot
            if(diag_val == static_cast<T>(0))
            {
                if(lid == WFSIZE - 1)
                {
                    // We are looking for the first zero pivot
                    atomicMin(zero_pivot, block_col + idx_base);
                }

                diag_val = static_cast<T>(1);
            }

            // Compute reciprocal
            diag_val = static_cast<T>(1) / diag_val;

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            for(rocsparse_int l = local_block_begin + tidy; l < local_block_diag; l += BLK_SIZE_Y)
            {
                rocsparse_int idx = index[l - local_block_begin];

                if(idx != -1)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (l - local_block_begin)
                                        + BSR_BLOCK_DIM * k + tidx];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + tidx];

                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
            }
            if(tidy == BLK_SIZE_Y - 1)
            {
                rocsparse_int idx = index[local_block_diag - local_block_begin];

                if(idx != -1 && tidx < k)
                {
                    T v1 = local_values[BSR_BLOCK_DIM * BSR_BLOCK_DIM
                                            * (local_block_diag - local_block_begin)
                                        + BSR_BLOCK_DIM * k + tidx];
                    T v2 = values[idx + BSR_BLOCK_DIM * tidz + tidx];

                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
            }

            // Accumulate row sum
            local_sum = rocsparse_wfreduce_sum<(BLK_SIZE_X * BLK_SIZE_Y)>(local_sum);

            // Last lane id computes the Cholesky factor and writes it to shared memory
            if(rid == (BLK_SIZE_X * BLK_SIZE_Y) - 1)
            {
                val = (val - local_sum) * diag_val;
                values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                       + k]
                    = val;
            }

            __threadfence_block();

            row_sum[BSR_BLOCK_DIM * tidz + tidx] = rocsparse_fma(
                values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                       + k],
                rocsparse_conj(values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                                      + BSR_BLOCK_DIM * tidx + k]),
                row_sum[BSR_BLOCK_DIM * tidz + tidx]);

            __threadfence_block();
        }
    }

    // Handle diagonal block column of block row.
    for(rocsparse_int k = 0; k < block_dim; k++)
    {
        rocsparse_int row_diag = BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                                 + BSR_BLOCK_DIM * k + k;

        if(k == tidz)
        {
            values[row_diag]
                = sqrt(rocsparse_abs(values[row_diag] - row_sum[BSR_BLOCK_DIM * k + k]));
        }

        __threadfence_block();

        // Load value
        T val = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                       + BSR_BLOCK_DIM * tidz + k];

        // Load diagonal entry
        T diag_val = values[row_diag];

        // Row has numerical zero pivot
        if(diag_val == static_cast<T>(0))
        {
            if(lid == WFSIZE - 1)
            {
                // We are looking for the first zero pivot
                atomicMin(zero_pivot, block_row + idx_base);
            }

            // Normally would break here but to avoid divergence set diag_val to one and continue
            // The zero pivot has already been set so further computation does not matter
            diag_val = static_cast<T>(1);
        }

        // Compute reciprocal
        diag_val = static_cast<T>(1) / diag_val;

        // Local row sum
        T local_sum = row_sum[BSR_BLOCK_DIM * tidz + k];

        if(k < tidz)
        {
            val = (val - local_sum) * diag_val;
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                   + BSR_BLOCK_DIM * tidz + k]
                = val;

            __threadfence_block();

            row_sum[BSR_BLOCK_DIM * tidz + tidx] = rocsparse_fma(
                val,
                rocsparse_conj(
                    values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                           + BSR_BLOCK_DIM * tidx + k]),
                row_sum[BSR_BLOCK_DIM * tidz + tidx]);
        }

        __threadfence_block();
    }

    if(tidx < block_dim && tidz < block_dim)
    {
        for(rocsparse_int j = block_row_begin + tidy; j < block_row_diag + 1; j += BLK_SIZE_Y)
        {
            if(direction == rocsparse_direction_row)
            {
                bsr_val[block_dim * block_dim * j + block_dim * tidz + tidx]
                    = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                             + BSR_BLOCK_DIM * tidz + tidx];
            }
            else
            {
                bsr_val[block_dim * block_dim * j + block_dim * tidx + tidz]
                    = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                             + BSR_BLOCK_DIM * tidz + tidx];
            }
        }
    }

    __threadfence();

    if(lid == WFSIZE - 1)
    {
        // Last lane in wavefront writes "we are done" flag for its block row
        atomicOr(&block_done[block_row], 1);
    }
}

template <typename T,
          rocsparse_int BLOCKSIZE,
          rocsparse_int WFSIZE,
          rocsparse_int MAX_NNZB,
          rocsparse_int BSR_BLOCK_DIM,
          rocsparse_int BLK_SIZE_Y>
__launch_bounds__(BLOCKSIZE) __global__ void bsric0_hash_large_maxnnzb_small_blockdim_kernel(
    rocsparse_direction direction,
    rocsparse_int       mb,
    rocsparse_int       block_dim,
    const rocsparse_int* __restrict__ bsr_row_ptr,
    const rocsparse_int* __restrict__ bsr_col_ind,
    T* __restrict__ bsr_val,
    const rocsparse_int* __restrict__ bsr_diag_ind,
    int* __restrict__ block_done,
    const rocsparse_int* __restrict__ block_map,
    rocsparse_int* __restrict__ zero_pivot,
    rocsparse_index_base idx_base)
{
    constexpr rocsparse_int BLK_SIZE_X = BSR_BLOCK_DIM;

    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int tidy = hipThreadIdx_y;
    rocsparse_int tidz = hipThreadIdx_z;

    rocsparse_int gid = tidx + BLK_SIZE_X * tidy + BLK_SIZE_X * BLK_SIZE_Y * tidz;
    rocsparse_int rid = tidx + BLK_SIZE_X * tidy;
    rocsparse_int lid = gid & (WFSIZE - 1);

    __shared__ rocsparse_int table[MAX_NNZB];
    __shared__ rocsparse_int data[MAX_NNZB];
    __shared__ rocsparse_int index[MAX_NNZB];
    __shared__ T             row_sum[BSR_BLOCK_DIM * BSR_BLOCK_DIM];
    __shared__ T             values[MAX_NNZB * BSR_BLOCK_DIM * BSR_BLOCK_DIM];

    // Current block row this wavefront is working on
    rocsparse_int block_row = block_map[hipBlockIdx_x];

    // Block diagonal entry point of the current block row
    rocsparse_int block_row_diag = bsr_diag_ind[block_row];

    // If one thread in the warp breaks here, then all threads in
    // the warp break so no divergence
    if(block_row_diag == -1)
    {
        __threadfence();

        if(lid == WFSIZE - 1)
        {
            atomicMin(zero_pivot, block_row + idx_base);

            // Last lane in wavefront writes "we are done" flag for its block row
            atomicOr(&block_done[block_row], 1);
        }

        return;
    }

    // Block row entry point
    rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;

    // Initialize hash table with -1
    for(rocsparse_int j = lid; j < MAX_NNZB; j += WFSIZE)
    {
        table[j] = -1;
    }

    __threadfence_block();

    // Fill hash table
    for(rocsparse_int j = block_row_begin + lid; j < block_row_diag + 1; j += WFSIZE)
    {
        // Insert block_key into hash table
        rocsparse_int block_key = bsr_col_ind[j] - idx_base;

        // Compute block hash
        rocsparse_int block_hash = (block_key * 103) % MAX_NNZB;

        // Hash operation
        while(true)
        {
            if(table[block_hash] == block_key)
            {
                // block_key is already inserted, block done
                break;
            }
            else if(atomicCAS(&table[block_hash], -1, block_key) == -1)
            {
                data[block_hash] = j - block_row_begin;

                break;
            }
            else
            {
                // collision, compute new block_hash
                block_hash = (block_hash + 1) % MAX_NNZB;
            }
        }
    }

    // Load current diagonal block into shared memory
    if(direction == rocsparse_direction_row)
    {
        values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
               + BSR_BLOCK_DIM * tidz + tidx]
            = (tidx < block_dim && tidz < block_dim)
                  ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidz + tidx]
                  : static_cast<T>(0);
    }
    else
    {
        values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
               + BSR_BLOCK_DIM * tidz + tidx]
            = (tidx < block_dim && tidz < block_dim)
                  ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidx + tidz]
                  : static_cast<T>(0);
    }

    // Block row sum accumulator
    row_sum[BSR_BLOCK_DIM * tidz + tidx] = static_cast<T>(0);

    __threadfence_block();

    // Loop over non-diagonal block columns of current block row
    for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
    {
        // Block column index currently being processes
        rocsparse_int block_col = bsr_col_ind[j] - idx_base;

        // Beginning of the row that corresponds to block_col
        rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

        // Diagonal entry point of row block_col
        rocsparse_int local_block_diag = bsr_diag_ind[block_col];

        // Structural zero pivot, do not process this row
        if(local_block_diag == -1)
        {
            // If one thread in the warp breaks here, then all threads in
            // the warp break so no divergence
            break;
            //local_block_diag = block_row_diag - 1;
        }

        // Load current j block into shared memory
        if(direction == rocsparse_direction_row)
        {
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                   + tidx]
                = (tidx < block_dim && tidz < block_dim)
                      ? bsr_val[block_dim * block_dim * j + block_dim * tidz + tidx]
                      : static_cast<T>(0);
        }
        else
        {
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                   + tidx]
                = (tidx < block_dim && tidz < block_dim)
                      ? bsr_val[block_dim * block_dim * j + block_dim * tidx + tidz]
                      : static_cast<T>(0);
        }

        for(rocsparse_int l = local_block_begin + tidy; l < local_block_diag + 1; l += BLK_SIZE_Y)
        {
            rocsparse_int block_key = bsr_col_ind[l] - idx_base;

            // Compute block hash
            rocsparse_int block_hash = (block_key * 103) % MAX_NNZB;

            rocsparse_int idx = -1;

            // Hash operation
            rocsparse_int count = 0;
            while(count < MAX_NNZB)
            {
                count++;

                if(table[block_hash] == -1)
                {
                    // no entry for the key, done
                    break;
                }
                else if(table[block_hash] == block_key)
                {
                    idx = BSR_BLOCK_DIM * BSR_BLOCK_DIM * data[block_hash];
                    break;
                }
                else
                {
                    // collision, compute new block_hash
                    block_hash = (block_hash + 1) % MAX_NNZB;
                }
            }

            index[l - local_block_begin] = idx;
        }

        __threadfence_block();

        // Spin loop until dependency has been resolved
        while(!atomicOr(&block_done[block_col], 0))
            ;

        __threadfence();

        for(rocsparse_int k = 0; k < block_dim; k++)
        {
            // Current value
            T val = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                           + BSR_BLOCK_DIM * tidz + k];

            // Load diagonal entry
            T diag_val = bsr_val[block_dim * block_dim * local_block_diag + block_dim * k + k];

            // Row has numerical zero pivot
            if(diag_val == static_cast<T>(0))
            {
                if(lid == WFSIZE - 1)
                {
                    // We are looking for the first zero pivot
                    atomicMin(zero_pivot, block_col + idx_base);
                }

                diag_val = static_cast<T>(1);
            }

            // Compute reciprocal
            diag_val = static_cast<T>(1) / diag_val;

            // Local row sum
            T local_sum = static_cast<T>(0);

            if(tidx < block_dim && tidz < block_dim)
            {
                // Loop over the row the current column index depends on
                // Each lane processes one entry
                for(rocsparse_int l = local_block_begin + tidy; l < local_block_diag;
                    l += BLK_SIZE_Y)
                {
                    rocsparse_int idx = index[l - local_block_begin];

                    if(idx != -1)
                    {
                        T v1 = static_cast<T>(0);
                        if(direction == rocsparse_direction_row)
                        {
                            v1 = bsr_val[block_dim * block_dim * l + block_dim * k + tidx];
                        }
                        else
                        {
                            v1 = bsr_val[block_dim * block_dim * l + block_dim * tidx + k];
                        }

                        T v2 = values[idx + BSR_BLOCK_DIM * tidz + tidx];

                        local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                    }
                }
                if(tidy == BLK_SIZE_Y - 1)
                {
                    rocsparse_int idx = index[local_block_diag - local_block_begin];

                    if(idx != -1 && tidx < k)
                    {
                        T v1 = static_cast<T>(0);
                        if(direction == rocsparse_direction_row)
                        {
                            v1 = bsr_val[block_dim * block_dim * local_block_diag + block_dim * k
                                         + tidx];
                        }
                        else
                        {
                            v1 = bsr_val[block_dim * block_dim * local_block_diag + block_dim * tidx
                                         + k];
                        }

                        T v2 = values[idx + BSR_BLOCK_DIM * tidz + tidx];

                        local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                    }
                }
            }

            // Accumulate row sum
            local_sum = rocsparse_wfreduce_sum<(BLK_SIZE_X * BLK_SIZE_Y)>(local_sum);

            // Last lane id computes the Cholesky factor and writes it to shared memory
            if(rid == (BLK_SIZE_X * BLK_SIZE_Y) - 1)
            {
                val = (val - local_sum) * diag_val;
                values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                       + k]
                    = val;
            }

            __threadfence_block();

            row_sum[BSR_BLOCK_DIM * tidz + tidx] = rocsparse_fma(
                values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin) + BSR_BLOCK_DIM * tidz
                       + k],
                rocsparse_conj(values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                                      + BSR_BLOCK_DIM * tidx + k]),
                row_sum[BSR_BLOCK_DIM * tidz + tidx]);

            __threadfence_block();
        }
    }

    // Handle diagonal block column of block row.
    for(rocsparse_int k = 0; k < block_dim; k++)
    {
        rocsparse_int row_diag = BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                                 + BSR_BLOCK_DIM * k + k;

        if(k == tidz)
        {
            values[row_diag]
                = sqrt(rocsparse_abs(values[row_diag] - row_sum[BSR_BLOCK_DIM * k + k]));
        }

        __threadfence_block();

        // Load value
        T val = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                       + BSR_BLOCK_DIM * tidz + k];

        // Load diagonal entry
        T diag_val = values[row_diag];

        // Row has numerical zero pivot
        if(diag_val == static_cast<T>(0))
        {
            if(lid == WFSIZE - 1)
            {
                // We are looking for the first zero pivot
                atomicMin(zero_pivot, block_row + idx_base);
            }

            // Normally would break here but to avoid divergence set diag_val to one and continue
            // The zero pivot has already been set so further computation does not matter
            diag_val = static_cast<T>(1);
        }

        // Compute reciprocal
        diag_val = static_cast<T>(1) / diag_val;

        // Local row sum
        T local_sum = row_sum[BSR_BLOCK_DIM * tidz + k];

        if(k < tidz)
        {
            val = (val - local_sum) * diag_val;
            values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                   + BSR_BLOCK_DIM * tidz + k]
                = val;

            __threadfence_block();

            row_sum[BSR_BLOCK_DIM * tidz + tidx] = rocsparse_fma(
                val,
                rocsparse_conj(
                    values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (block_row_diag - block_row_begin)
                           + BSR_BLOCK_DIM * tidx + k]),
                row_sum[BSR_BLOCK_DIM * tidz + tidx]);
        }

        __threadfence_block();
    }

    if(tidx < block_dim && tidz < block_dim)
    {
        for(rocsparse_int j = block_row_begin + tidy; j < block_row_diag + 1; j += BLK_SIZE_Y)
        {
            if(direction == rocsparse_direction_row)
            {
                bsr_val[block_dim * block_dim * j + block_dim * tidz + tidx]
                    = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                             + BSR_BLOCK_DIM * tidz + tidx];
            }
            else
            {
                bsr_val[block_dim * block_dim * j + block_dim * tidx + tidz]
                    = values[BSR_BLOCK_DIM * BSR_BLOCK_DIM * (j - block_row_begin)
                             + BSR_BLOCK_DIM * tidz + tidx];
            }
        }
    }

    __threadfence();

    if(lid == WFSIZE - 1)
    {
        // Last lane in wavefront writes "we are done" flag for its block row
        atomicOr(&block_done[block_row], 1);
    }
}

template <typename T,
          rocsparse_int BLOCKSIZE,
          rocsparse_int WFSIZE,
          rocsparse_int MAX_NNZB,
          rocsparse_int BSR_BLOCK_DIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsric0_hash_large_blockdim_kernel(rocsparse_direction direction,
                                           rocsparse_int       mb,
                                           rocsparse_int       block_dim,
                                           const rocsparse_int* __restrict__ bsr_row_ptr,
                                           const rocsparse_int* __restrict__ bsr_col_ind,
                                           T* __restrict__ bsr_val,
                                           const rocsparse_int* __restrict__ bsr_diag_ind,
                                           int* __restrict__ block_done,
                                           const rocsparse_int* __restrict__ block_map,
                                           rocsparse_int* __restrict__ zero_pivot,
                                           rocsparse_index_base idx_base)
{
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);
    rocsparse_int rid = lid & ((WFSIZE / BSR_BLOCK_DIM) - 1);

    __shared__ rocsparse_int table[MAX_NNZB];
    __shared__ rocsparse_int data[MAX_NNZB];
    __shared__ rocsparse_int index[MAX_NNZB];

    // Initialize hash table with -1
    for(rocsparse_int j = lid; j < MAX_NNZB; j += WFSIZE)
    {
        table[j] = -1;
    }

    __syncthreads();

    // Local row within BSR block
    rocsparse_int local_row = (lid / (WFSIZE / BSR_BLOCK_DIM));

    // Current block row this wavefront is working on
    rocsparse_int block_row = block_map[hipBlockIdx_x];

    // Block diagonal entry point of the current block row
    rocsparse_int block_row_diag = bsr_diag_ind[block_row];

    // If one thread in the warp breaks here, then all threads in
    // the warp break so no divergence
    if(block_row_diag == -1)
    {
        __threadfence();

        if(lid == WFSIZE - 1)
        {
            atomicMin(zero_pivot, block_row + idx_base);

            // Last lane in wavefront writes "we are done" flag for its block row
            atomicOr(&block_done[block_row], 1);
        }

        return;
    }

    // Block row entry point
    rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;

    // Row sum accumulator
    T row_sum = static_cast<T>(0);

    // Fill hash table
    for(rocsparse_int j = block_row_begin + lid; j < block_row_diag + 1; j += WFSIZE)
    {
        // Insert block_key into hash table
        rocsparse_int block_key = bsr_col_ind[j] - idx_base;

        // Compute block hash
        rocsparse_int block_hash = (block_key * 103) % MAX_NNZB;

        // Hash operation
        while(true)
        {
            if(table[block_hash] == block_key)
            {
                // block_key is already inserted, block done
                break;
            }
            else if(atomicCAS(&table[block_hash], -1, block_key) == -1)
            {
                data[block_hash] = j;

                break;
            }
            else
            {
                // collision, compute new block_hash
                block_hash = (block_hash + 1) % MAX_NNZB;
            }
        }
    }

    // Loop over non-diagonal block columns of current block row
    for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
    {
        // Block column index currently being processes
        rocsparse_int block_col = bsr_col_ind[j] - idx_base;

        // Beginning of the row that corresponds to block_col
        rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

        // Diagonal entry point of row block_col
        rocsparse_int local_block_diag = bsr_diag_ind[block_col];

        // Structural zero pivot, do not process this row
        if(local_block_diag == -1)
        {
            // If one thread in the warp breaks here, then all threads in
            // the warp break so no divergence
            break;
        }

        for(rocsparse_int l = local_block_begin + rid; l < local_block_diag + 1;
            l += (WFSIZE / BSR_BLOCK_DIM))
        {
            rocsparse_int block_key = bsr_col_ind[l] - idx_base;

            // Compute block hash
            // rocsparse_int block_hash = (block_key * 103) & (MAX_NNZB - 1);
            rocsparse_int block_hash = (block_key * 103) % MAX_NNZB;

            // Hash operation
            rocsparse_int idx   = -1;
            rocsparse_int count = 0;
            while(count < MAX_NNZB)
            {
                count++;

                if(table[block_hash] == -1)
                {
                    // no entry for the key, done
                    break;
                }
                else if(table[block_hash] == block_key)
                {
                    idx = block_dim * block_dim * data[block_hash];
                    break;
                }
                else
                {
                    // collision, compute new block_hash
                    block_hash = (block_hash + 1) % MAX_NNZB;
                }
            }

            index[l - local_block_begin] = idx;
        }

        // Spin loop until dependency has been resolved
        while(!atomicOr(&block_done[block_col], 0))
            ;

        __threadfence();

        for(rocsparse_int k = 0; k < block_dim; k++)
        {
            // Current value
            T val = static_cast<T>(0);

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Load diagonal entry
            T diag_val = static_cast<T>(1);

            if(local_row < block_dim)
            {
                if(direction == rocsparse_direction_row)
                {
                    val = bsr_val[block_dim * block_dim * j + block_dim * local_row + k];
                }
                else
                {
                    val = bsr_val[block_dim * block_dim * j + block_dim * k + local_row];
                }

                // Load diagonal entry
                diag_val = bsr_val[block_dim * block_dim * local_block_diag + block_dim * k + k];

                // Row has numerical zero pivot
                if(diag_val == static_cast<T>(0))
                {
                    if(rid == 0)
                    {
                        // We are looking for the first zero pivot
                        atomicMin(zero_pivot, block_col + idx_base);
                    }

                    diag_val = static_cast<T>(1);
                }

                // Compute reciprocal
                diag_val = static_cast<T>(1) / diag_val;

                // Loop over the row the current column index depends on
                // Each lane processes one entry
                T v1 = static_cast<T>(0);
                T v2 = static_cast<T>(0);
                for(rocsparse_int l = local_block_begin + rid; l < local_block_diag;
                    l += (WFSIZE / BSR_BLOCK_DIM))
                {
                    rocsparse_int idx = index[l - local_block_begin];

                    if(idx != -1)
                    {
                        for(rocsparse_int p = 0; p < block_dim; p++)
                        {
                            if(direction == rocsparse_direction_row)
                            {
                                v1 = bsr_val[block_dim * block_dim * l + block_dim * k + p];
                                v2 = bsr_val[idx + block_dim * local_row + p];
                            }
                            else
                            {
                                v1 = bsr_val[block_dim * block_dim * l + block_dim * p + k];
                                v2 = bsr_val[idx + block_dim * p + local_row];
                            }

                            local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                        }
                    }
                }

                if(rid == 0)
                {
                    rocsparse_int idx = index[local_block_diag - local_block_begin];

                    if(idx != -1)
                    {
                        for(rocsparse_int p = 0; p < k; p++)
                        {
                            T v1 = static_cast<T>(0);
                            T v2 = static_cast<T>(0);
                            if(direction == rocsparse_direction_row)
                            {
                                v1 = bsr_val[block_dim * block_dim * local_block_diag
                                             + block_dim * k + p];
                                v2 = bsr_val[idx + block_dim * local_row + p];
                            }
                            else
                            {
                                v1 = bsr_val[block_dim * block_dim * local_block_diag
                                             + block_dim * p + k];
                                v2 = bsr_val[idx + block_dim * p + local_row];
                            }

                            local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                        }
                    }
                }
            }

            // Accumulate row sum
            local_sum = rocsparse_wfreduce_sum<(WFSIZE / BSR_BLOCK_DIM)>(local_sum);

            if(local_row < block_dim)
            {
                // Last lane id computes the Cholesky factor and writes it to global memory
                if(rid == (WFSIZE / BSR_BLOCK_DIM) - 1)
                {
                    val     = (val - local_sum) * diag_val;
                    row_sum = rocsparse_fma(val, rocsparse_conj(val), row_sum);

                    if(direction == rocsparse_direction_row)
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * local_row + k] = val;
                    }
                    else
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * k + local_row] = val;
                    }
                }
            }
        }
    }

    __threadfence();

    // Handle diagonal block column of block row. All threads in the warp enter here or none do.
    for(rocsparse_int j = 0; j < block_dim; j++)
    {
        rocsparse_int row_diag = block_dim * block_dim * block_row_diag + block_dim * j + j;

        // Check if 'col' row is complete
        if(j == local_row)
        {
            if(rid == (WFSIZE / BSR_BLOCK_DIM) - 1)
            {
                T temp            = sqrt(rocsparse_abs(bsr_val[row_diag] - row_sum));
                bsr_val[row_diag] = temp;

                if(temp == static_cast<T>(0))
                {
                    // We are looking for the first zero pivot
                    atomicMin(zero_pivot, block_row + idx_base);
                }
            }
        }

        // Ensure previous writes to global memory are seen by all threads
        __threadfence();

        // Current value
        T val = static_cast<T>(0);

        // Local row sum
        T local_sum = static_cast<T>(0);

        // Diagonal entry
        T diag_val = static_cast<T>(0);

        if(j < local_row && local_row < block_dim)
        {
            // Load value
            if(direction == rocsparse_direction_row)
            {
                val = bsr_val[block_dim * block_dim * block_row_diag + block_dim * local_row + j];
            }
            else
            {
                val = bsr_val[block_dim * block_dim * block_row_diag + block_dim * j + local_row];
            }

            // Load diagonal entry
            diag_val = bsr_val[row_diag];

            // Row has numerical zero pivot
            if(diag_val == static_cast<T>(0))
            {
                if(rid == 0)
                {
                    // We are looking for the first zero pivot
                    atomicMin(zero_pivot, block_row + idx_base);
                }

                // Normally would break here but to avoid divergence set diag_val to one and continue
                // The zero pivot has already been set so further computation does not matter
                diag_val = static_cast<T>(1);
            }

            // Compute reciprocal
            diag_val = static_cast<T>(1) / diag_val;

            T v1 = static_cast<T>(0);
            T v2 = static_cast<T>(0);
            for(rocsparse_int k = block_row_begin + rid; k < block_row_diag;
                k += (WFSIZE / BSR_BLOCK_DIM))
            {
                for(rocsparse_int p = 0; p < block_dim; p++)
                {
                    if(direction == rocsparse_direction_row)
                    {
                        v1 = bsr_val[block_dim * block_dim * k + block_dim * j + p];
                        v2 = bsr_val[block_dim * block_dim * k + block_dim * local_row + p];
                    }
                    else
                    {
                        v1 = bsr_val[block_dim * block_dim * k + block_dim * p + j];
                        v2 = bsr_val[block_dim * block_dim * k + block_dim * p + local_row];
                    }

                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
            }

            if(rid == 0)
            {
                for(rocsparse_int p = 0; p < j; p++)
                {
                    if(direction == rocsparse_direction_row)
                    {
                        v1 = bsr_val[block_dim * block_dim * block_row_diag + block_dim * j + p];
                        v2 = bsr_val[block_dim * block_dim * block_row_diag + block_dim * local_row
                                     + p];
                    }
                    else
                    {
                        v1 = bsr_val[block_dim * block_dim * block_row_diag + block_dim * p + j];
                        v2 = bsr_val[block_dim * block_dim * block_row_diag + block_dim * p
                                     + local_row];
                    }

                    local_sum = rocsparse_fma(v1, rocsparse_conj(v2), local_sum);
                }
            }
        }

        // Accumulate row sum
        local_sum = rocsparse_wfreduce_sum<(WFSIZE / BSR_BLOCK_DIM)>(local_sum);

        if(j < local_row && local_row < block_dim)
        {
            // Last lane id computes the Cholesky factor and writes it to global memory
            if(rid == (WFSIZE / BSR_BLOCK_DIM) - 1)
            {
                val     = (val - local_sum) * diag_val;
                row_sum = rocsparse_fma(val, rocsparse_conj(val), row_sum);

                if(direction == rocsparse_direction_row)
                {
                    bsr_val[block_dim * block_dim * block_row_diag + block_dim * local_row + j]
                        = val;
                }
                else
                {
                    bsr_val[block_dim * block_dim * block_row_diag + block_dim * j + local_row]
                        = val;
                }
            }
        }
    }

    __threadfence();

    if(lid == WFSIZE - 1)
    {
        // Last lane in wavefront writes "we are done" flag for its block row
        atomicOr(&block_done[block_row], 1);
    }
}

template <typename T,
          unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int THREADS_PER_ROW,
          bool         SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsric0_binsearch_kernel(rocsparse_direction direction,
                                 rocsparse_int       mb,
                                 rocsparse_int       block_dim,
                                 const rocsparse_int* __restrict__ bsr_row_ptr,
                                 const rocsparse_int* __restrict__ bsr_col_ind,
                                 T* __restrict__ bsr_val,
                                 const rocsparse_int* __restrict__ bsr_diag_ind,
                                 int* __restrict__ block_done,
                                 const rocsparse_int* __restrict__ block_map,
                                 rocsparse_int* __restrict__ zero_pivot,
                                 rocsparse_index_base idx_base)
{
    int lid = hipThreadIdx_x & (WFSIZE - 1);
    int wid = hipThreadIdx_x / WFSIZE;
    int rid = lid & (THREADS_PER_ROW - 1);

    rocsparse_int idx = hipBlockIdx_x + wid;

    // Current block row this wavefront is working on
    rocsparse_int block_row = block_map[idx];

    // Block diagonal entry point of the current block row
    rocsparse_int block_row_diag = bsr_diag_ind[block_row];

    // If one thread in the warp breaks here, then all threads in
    // the warp break so no divergence
    if(block_row_diag == -1)
    {
        __threadfence();

        if(lid == WFSIZE - 1)
        {
            atomicMin(zero_pivot, block_row + idx_base);

            // Last lane in wavefront writes "we are done" flag for its block row
            atomicOr(&block_done[block_row], 1);
        }

        return;
    }

    // Block row entry point
    rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;
    rocsparse_int block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;

    for(rocsparse_int i = 0; i < block_dim; i += (WFSIZE / THREADS_PER_ROW))
    {
        rocsparse_int local_row = (hipThreadIdx_x / THREADS_PER_ROW) + i;

        __syncthreads();

        // Row sum accumulator
        T row_sum = static_cast<T>(0);

        // Loop over block columns of current block row
        for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
        {
            // Block column index currently being processes
            rocsparse_int block_col = bsr_col_ind[j] - idx_base;

            // Beginning of the block row that corresponds to block_col
            rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

            // Block diagonal entry point of block row 'block_col'
            rocsparse_int local_block_diag = bsr_diag_ind[block_col];

            // Structural zero pivot, do not process this block row
            if(local_block_diag == -1)
            {
                // If one thread in the warp breaks here, then all threads in
                // the warp break so no divergence
                break;
            }

            // Spin loop until dependency has been resolved
            int          local_done    = atomicOr(&block_done[block_col], 0);
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

                local_done = atomicOr(&block_done[block_col], 0);
            }

            __threadfence();

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                // Column index currently being processes
                rocsparse_int col = block_dim * block_col + k;

                // Current value
                T val = static_cast<T>(0);

                // Local row sum
                T local_sum = static_cast<T>(0);

                // Load diagonal entry
                T diag_val = static_cast<T>(1);

                if(local_row < block_dim)
                {
                    // Corresponding value
                    if(direction == rocsparse_direction_row)
                    {
                        val = bsr_val[block_dim * block_dim * j + block_dim * local_row + k];
                    }
                    else
                    {
                        val = bsr_val[block_dim * block_dim * j + block_dim * k + local_row];
                    }

                    diag_val
                        = bsr_val[block_dim * block_dim * local_block_diag + block_dim * k + k];

                    // Row has numerical zero pivot
                    if(diag_val == static_cast<T>(0))
                    {
                        if(rid == 0)
                        {
                            // We are looking for the first zero pivot
                            atomicMin(zero_pivot, block_col + idx_base);
                        }

                        // Normally would break here but to avoid divergence set diag_val to one and continue
                        // The zero pivot has already been set so further computation does not matter
                        diag_val = static_cast<T>(1);
                    }

                    // Compute reciprocal
                    diag_val = static_cast<T>(1) / diag_val;

                    // Loop over the row the current column index depends on
                    // Each lane processes one entry
                    for(rocsparse_int p = local_block_begin + rid; p < local_block_diag + 1;
                        p += THREADS_PER_ROW)
                    {
                        // Perform a binary search to find matching block columns
                        rocsparse_int l = block_row_begin;
                        rocsparse_int r = block_row_end - 1;
                        rocsparse_int m = (r + l) >> 1;

                        rocsparse_int block_col_j = bsr_col_ind[m] - idx_base;
                        rocsparse_int block_col_p = bsr_col_ind[p] - idx_base;

                        // Binary search for block column
                        while(l < r)
                        {
                            if(block_col_j < block_col_p)
                            {
                                l = m + 1;
                            }
                            else
                            {
                                r = m;
                            }

                            m           = (r + l) >> 1;
                            block_col_j = bsr_col_ind[m] - idx_base;
                        }

                        // Check if a match has been found
                        if(block_col_j == block_col_p)
                        {
                            for(rocsparse_int q = 0; q < block_dim; q++)
                            {
                                if(block_dim * block_col_p + q < col)
                                {
                                    T vp = static_cast<T>(0);
                                    T vj = static_cast<T>(0);
                                    if(direction == rocsparse_direction_row)
                                    {
                                        vp = bsr_val[block_dim * block_dim * p + block_dim * k + q];
                                        vj = bsr_val[block_dim * block_dim * m
                                                     + block_dim * local_row + q];
                                    }
                                    else
                                    {
                                        vp = bsr_val[block_dim * block_dim * p + block_dim * q + k];
                                        vj = bsr_val[block_dim * block_dim * m + block_dim * q
                                                     + local_row];
                                    }

                                    // If a match has been found, do linear combination
                                    local_sum = rocsparse_fma(vp, rocsparse_conj(vj), local_sum);
                                }
                            }
                        }
                    }
                }

                // Accumulate row sum
                local_sum = rocsparse_wfreduce_sum<THREADS_PER_ROW>(local_sum);

                if(local_row < block_dim)
                {
                    // Last lane id computes the Cholesky factor and writes it to global memory
                    if(rid == THREADS_PER_ROW - 1)
                    {
                        val     = (val - local_sum) * diag_val;
                        row_sum = rocsparse_fma(val, rocsparse_conj(val), row_sum);

                        if(direction == rocsparse_direction_row)
                        {
                            bsr_val[block_dim * block_dim * j + block_dim * local_row + k] = val;
                        }
                        else
                        {
                            bsr_val[block_dim * block_dim * j + block_dim * k + local_row] = val;
                        }
                    }
                }
            }
        }

        // Handle diagonal block column of block row
        for(rocsparse_int j = 0; j < block_dim; j++)
        {
            rocsparse_int row_diag = block_dim * block_dim * block_row_diag + block_dim * j + j;

            // Check if 'col' row is complete
            if(j == local_row)
            {
                if(rid == THREADS_PER_ROW - 1)
                {
                    T temp            = sqrt(rocsparse_abs(bsr_val[row_diag] - row_sum));
                    bsr_val[row_diag] = temp;

                    if(temp == static_cast<T>(0))
                    {
                        // We are looking for the first zero pivot
                        atomicMin(zero_pivot, block_row + idx_base);
                    }
                }
            }

            // Ensure previous writes to global memory are seen by all threads
            __threadfence();

            // Current value
            T val = static_cast<T>(0);

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Diagonal entry
            T diag_val = static_cast<T>(1);

            if(j < local_row && local_row < block_dim)
            {
                // Corresponding value
                if(direction == rocsparse_direction_row)
                {
                    val = bsr_val[block_dim * block_dim * block_row_diag + block_dim * local_row
                                  + j];
                }
                else
                {
                    val = bsr_val[block_dim * block_dim * block_row_diag + block_dim * j
                                  + local_row];
                }

                // Load diagonal entry
                diag_val = bsr_val[row_diag];

                // Row has numerical zero pivot
                if(diag_val == static_cast<T>(0))
                {
                    if(rid == 0)
                    {
                        // We are looking for the first zero pivot
                        atomicMin(zero_pivot, block_row + idx_base);
                    }

                    // Normally would break here but to avoid divergence set diag_val to one and continue
                    // The zero pivot has already been set so further computation does not matter
                    diag_val = static_cast<T>(1);
                }

                // Compute reciprocal
                diag_val = static_cast<T>(1) / diag_val;

                T vk = static_cast<T>(0);
                T vj = static_cast<T>(0);
                for(rocsparse_int k = block_row_begin + rid; k < block_row_diag;
                    k += THREADS_PER_ROW)
                {
                    for(rocsparse_int q = 0; q < block_dim; q++)
                    {
                        if(direction == rocsparse_direction_row)
                        {
                            vk = bsr_val[block_dim * block_dim * k + block_dim * j + q];
                            vj = bsr_val[block_dim * block_dim * k + block_dim * local_row + q];
                        }
                        else
                        {
                            vk = bsr_val[block_dim * block_dim * k + block_dim * q + j];
                            vj = bsr_val[block_dim * block_dim * k + block_dim * q + local_row];
                        }

                        // If a match has been found, do linear combination
                        local_sum = rocsparse_fma(vk, rocsparse_conj(vj), local_sum);
                    }
                }

                if(rid == 0)
                {
                    for(rocsparse_int q = 0; q < j; q++)
                    {
                        if(direction == rocsparse_direction_row)
                        {
                            vk = bsr_val[block_dim * block_dim * block_row_diag + block_dim * j
                                         + q];
                            vj = bsr_val[block_dim * block_dim * block_row_diag
                                         + block_dim * local_row + q];
                        }
                        else
                        {
                            vk = bsr_val[block_dim * block_dim * block_row_diag + block_dim * q
                                         + j];
                            vj = bsr_val[block_dim * block_dim * block_row_diag + block_dim * q
                                         + local_row];
                        }

                        // If a match has been found, do linear combination
                        local_sum = rocsparse_fma(vk, rocsparse_conj(vj), local_sum);
                    }
                }
            }

            // Accumulate row sum
            local_sum = rocsparse_wfreduce_sum<THREADS_PER_ROW>(local_sum);

            if(j < local_row && local_row < block_dim)
            {
                // Last lane id computes the Cholesky factor and writes it to global memory
                if(rid == THREADS_PER_ROW - 1)
                {
                    val     = (val - local_sum) * diag_val;
                    row_sum = rocsparse_fma(val, val, row_sum);

                    if(direction == rocsparse_direction_row)
                    {
                        bsr_val[block_dim * block_dim * block_row_diag + block_dim * local_row + j]
                            = val;
                    }
                    else
                    {
                        bsr_val[block_dim * block_dim * block_row_diag + block_dim * j + local_row]
                            = val;
                    }
                }
            }
        }
    }

    __threadfence();

    if(lid == WFSIZE - 1)
    {
        // Last lane writes "we are done" flag for current block row
        atomicOr(&block_done[block_row], 1);
    }
}

#endif // BSRIC0_DEVICE_H