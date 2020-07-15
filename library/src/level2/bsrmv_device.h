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
#ifndef BSRMV_DEVICE_H
#define BSRMV_DEVICE_H

#include "common.h"
#include "rocsparse.h"

#include <hip/hip_runtime.h>

// General BSRMV that works for any BSR block dimensions
template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__device__ void bsrmvn_general_device(rocsparse_direction dir,
                                      T                   alpha,
                                      const rocsparse_int* __restrict__ bsr_row_ptr,
                                      const rocsparse_int* __restrict__ bsr_col_ind,
                                      const T* __restrict__ bsr_val,
                                      rocsparse_int bsr_dim,
                                      const T* __restrict__ x,
                                      T beta,
                                      T* __restrict__ y,
                                      rocsparse_index_base idx_base)
{
    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each thread block processes a BSR row
    rocsparse_int row = hipBlockIdx_x;

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // Each wavefront processes a row of the BSR block.
    // If the number of BSR block rows exceed the number of wavefronts, each wavefront
    // processes multiple rows. 'bi' is the row index into the BSR block and 'bj' is
    // the column index.
    // BLOCKSIZE must be the square of WFSIZE.

    // Loop over the rows of the BSR block in chunks of WFSIZE, such that each
    // wavefront will process a row
    for(rocsparse_int bi = wid; bi < bsr_dim; bi += WFSIZE)
    {
        // BSR block row accumulator
        T sum = static_cast<T>(0);

        // Loop over all BSR blocks in the current row
        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            // BSR column index
            rocsparse_int col = bsr_col_ind[j] - idx_base;

            // Loop over the columns of the BSR block in chunks of WFSIZE, such that
            // each lane will process a single value of the BSR block
            for(rocsparse_int bj = lid; bj < bsr_dim; bj += WFSIZE)
            {
                // Each lane computes the sum of a specific entry over all BSR blocks in
                // the current row
                if(dir == rocsparse_direction_column)
                {
                    sum = rocsparse_fma(bsr_val[bsr_dim * bsr_dim * j + bsr_dim * bj + bi],
                                        x[bsr_dim * col + bj],
                                        sum);
                }
                else
                {
                    sum = rocsparse_fma(bsr_val[bsr_dim * bsr_dim * j + bsr_dim * bi + bj],
                                        x[bsr_dim * col + bj],
                                        sum);
                }
            }
        }

        // Each wavefront accumulates its BSR block row sum
        sum = rocsparse_wfreduce_sum<WFSIZE>(sum);

        // Last lane of each wavefront writes its result to global memory
        if(lid == WFSIZE - 1)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * bsr_dim + bi] = rocsparse_fma(beta, y[row * bsr_dim + bi], alpha * sum);
            }
            else
            {
                y[row * bsr_dim + bi] = alpha * sum;
            }
        }
    }
}

// BSRMV kernel for BSR block dimension of 2
template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__device__ void bsrmvn_2x2_device(rocsparse_int       mb,
                                  rocsparse_direction dir,
                                  T                   alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  const T* __restrict__ x,
                                  T beta,
                                  T* __restrict__ y,
                                  rocsparse_index_base idx_base)
{
    // BSR block dimension
    static constexpr int BSRDIM = 2;

    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows
    rocsparse_int row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

    // Do not run out of bounds
    if(row >= mb)
    {
        return;
    }

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // BSR block row accumulator
    T sum0 = static_cast<T>(0);
    T sum1 = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block
    for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
    {
        // Do not exceed the row
        if(j + lid < row_end)
        {
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[j + lid] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            if(dir == rocsparse_direction_column)
            {
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 0], x[col + 0], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 1], x[col + 0], sum1);

                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 2], x[col + 1], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 3], x[col + 1], sum1);
            }
            else
            {
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 0], x[col + 0], sum0);
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 1], x[col + 1], sum0);

                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 2], x[col + 0], sum1);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 3], x[col + 1], sum1);
            }
        }
    }

    // Each wavefront accumulates its BSR block row sum
    sum0 = rocsparse_wfreduce_sum<WFSIZE>(sum0);
    sum1 = rocsparse_wfreduce_sum<WFSIZE>(sum1);

    // Last lane of each wavefront writes the two row sums to global memory
    if(lid == WFSIZE - 1)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + 0] = rocsparse_fma(beta, y[row * BSRDIM + 0], alpha * sum0);
            y[row * BSRDIM + 1] = rocsparse_fma(beta, y[row * BSRDIM + 1], alpha * sum1);
        }
        else
        {
            y[row * BSRDIM + 0] = alpha * sum0;
            y[row * BSRDIM + 1] = alpha * sum1;
        }
    }
}

// BSRMV kernel for BSR block dimension of 3
template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__device__ void bsrmvn_3x3_device(rocsparse_int       mb,
                                  rocsparse_direction dir,
                                  T                   alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  const T* __restrict__ x,
                                  T beta,
                                  T* __restrict__ y,
                                  rocsparse_index_base idx_base)
{
    static constexpr int BSRDIM = 3;

    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each thread block processes a BSR row
    rocsparse_int row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

    if(row >= mb)
    {
        return;
    }

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // BSR block row accumulator
    T sum0 = static_cast<T>(0);
    T sum1 = static_cast<T>(0);
    T sum2 = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block
    for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
    {
        if(j + lid < row_end)
        {
            rocsparse_int col = (bsr_col_ind[j + lid] - idx_base) * BSRDIM;

            // Compute the sum of the three rows within the BSR blocks of the current
            // BSR row
            if(dir == rocsparse_direction_column)
            {
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 0], x[col + 0], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 1], x[col + 0], sum1);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 2], x[col + 0], sum2);

                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 3], x[col + 1], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 4], x[col + 1], sum1);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 5], x[col + 1], sum2);

                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 6], x[col + 2], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 7], x[col + 2], sum1);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 8], x[col + 2], sum2);
            }
            else
            {
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 0], x[col + 0], sum0);
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 1], x[col + 1], sum0);
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 2], x[col + 2], sum0);

                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 3], x[col + 0], sum1);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 4], x[col + 1], sum1);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 5], x[col + 2], sum1);

                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 6], x[col + 0], sum2);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 7], x[col + 1], sum2);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 8], x[col + 2], sum2);
            }
        }
    }

    // Each wavefront accumulates its BSR block row sum
    sum0 = rocsparse_wfreduce_sum<WFSIZE>(sum0);
    sum1 = rocsparse_wfreduce_sum<WFSIZE>(sum1);
    sum2 = rocsparse_wfreduce_sum<WFSIZE>(sum2);

    // Last lane of each wavefront writes its result to global memory
    if(lid == WFSIZE - 1)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + 0] = rocsparse_fma(beta, y[row * BSRDIM + 0], alpha * sum0);
            y[row * BSRDIM + 1] = rocsparse_fma(beta, y[row * BSRDIM + 1], alpha * sum1);
            y[row * BSRDIM + 2] = rocsparse_fma(beta, y[row * BSRDIM + 2], alpha * sum2);
        }
        else
        {
            y[row * BSRDIM + 0] = alpha * sum0;
            y[row * BSRDIM + 1] = alpha * sum1;
            y[row * BSRDIM + 2] = alpha * sum2;
        }
    }
}

// BSRMV kernel for BSR block dimension of 4
template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__device__ void bsrmvn_4x4_device(rocsparse_int       mb,
                                  rocsparse_direction dir,
                                  T                   alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  const T* __restrict__ x,
                                  T beta,
                                  T* __restrict__ y,
                                  rocsparse_index_base idx_base)
{
    // BSR block dimension
    static constexpr int BSRDIM = 4;

    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows
    rocsparse_int row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

    // Do not run out of bounds
    if(row >= mb)
    {
        return;
    }

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // BSR block row accumulator
    T sum0 = static_cast<T>(0);
    T sum1 = static_cast<T>(0);
    T sum2 = static_cast<T>(0);
    T sum3 = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block
    for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
    {
        // Do not exceed the row
        if(j + lid < row_end)
        {
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[j + lid] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            if(dir == rocsparse_direction_column)
            {
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 0], x[col + 0], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 1], x[col + 0], sum1);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 2], x[col + 0], sum2);
                sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 3], x[col + 0], sum3);

                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 4], x[col + 1], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 5], x[col + 1], sum1);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 6], x[col + 1], sum2);
                sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 7], x[col + 1], sum3);

                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 8], x[col + 2], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 9], x[col + 2], sum1);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 10], x[col + 2], sum2);
                sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 11], x[col + 2], sum3);

                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 12], x[col + 3], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 13], x[col + 3], sum1);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 14], x[col + 3], sum2);
                sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 15], x[col + 3], sum3);
            }
            else
            {
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 0], x[col + 0], sum0);
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 1], x[col + 1], sum0);
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 2], x[col + 2], sum0);
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 3], x[col + 3], sum0);

                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 4], x[col + 0], sum1);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 5], x[col + 1], sum1);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 6], x[col + 2], sum1);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 7], x[col + 3], sum1);

                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 8], x[col + 0], sum2);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 9], x[col + 1], sum2);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 10], x[col + 2], sum2);
                sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 11], x[col + 3], sum2);

                sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 12], x[col + 0], sum3);
                sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 13], x[col + 1], sum3);
                sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 14], x[col + 2], sum3);
                sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 15], x[col + 3], sum3);
            }
        }
    }

    // Each wavefront accumulates its BSR block row sum
    sum0 = rocsparse_wfreduce_sum<WFSIZE>(sum0);
    sum1 = rocsparse_wfreduce_sum<WFSIZE>(sum1);
    sum2 = rocsparse_wfreduce_sum<WFSIZE>(sum2);
    sum3 = rocsparse_wfreduce_sum<WFSIZE>(sum3);

    // Last lane of each wavefront writes the two row sums to global memory
    if(lid == WFSIZE - 1)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + 0] = rocsparse_fma(beta, y[row * BSRDIM + 0], alpha * sum0);
            y[row * BSRDIM + 1] = rocsparse_fma(beta, y[row * BSRDIM + 1], alpha * sum1);
            y[row * BSRDIM + 2] = rocsparse_fma(beta, y[row * BSRDIM + 2], alpha * sum2);
            y[row * BSRDIM + 3] = rocsparse_fma(beta, y[row * BSRDIM + 3], alpha * sum3);
        }
        else
        {
            y[row * BSRDIM + 0] = alpha * sum0;
            y[row * BSRDIM + 1] = alpha * sum1;
            y[row * BSRDIM + 2] = alpha * sum2;
            y[row * BSRDIM + 3] = alpha * sum3;
        }
    }
}

// BSRMV kernel for BSR block dimension of 5
template <typename T, unsigned int BLOCKSIZE>
__device__ void bsrmvn_5x5_device(rocsparse_int       mb,
                                  rocsparse_direction dir,
                                  T                   alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  const T* __restrict__ x,
                                  T beta,
                                  T* __restrict__ y,
                                  rocsparse_index_base idx_base)
{
    // BSR block dimension
    static constexpr int BSRDIM = 5;

    // BSR block lane id
    rocsparse_int lid = hipThreadIdx_x % BSRDIM;

    // Each thread block processes a single BSR row
    rocsparse_int row = hipBlockIdx_x;

    // Offset into x vector
    rocsparse_int idx
        = (dir == rocsparse_direction_column) ? ((hipThreadIdx_x / BSRDIM) % BSRDIM) : lid;

    // Number of BSR blocks processed at the same time
    const unsigned int NBLOCKS = BLOCKSIZE / (BSRDIM * BSRDIM);

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // BSR block row accumulator
    T sum = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block value
    for(rocsparse_int j = row_begin; j < row_end; j += NBLOCKS)
    {
        rocsparse_int k = j + hipThreadIdx_x / (BSRDIM * BSRDIM);

        // Do not exceed the row
        if(k < row_end)
        {
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[k] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            sum = rocsparse_fma(bsr_val[j * BSRDIM * BSRDIM + hipThreadIdx_x], x[col + idx], sum);
        }
    }

    // Accumulate each row sum of the BSR block
    __shared__ T sdata[BSRDIM * BSRDIM * NBLOCKS];

    sdata[hipThreadIdx_x] = sum;

    if(dir == rocsparse_direction_column)
    {
        if(hipThreadIdx_x < BLOCKSIZE - BSRDIM * 8)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 8];
        if(hipThreadIdx_x < BSRDIM * 4)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 4];
        if(hipThreadIdx_x < BSRDIM * 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 2];
        if(hipThreadIdx_x < BSRDIM * 1)
            sum = sdata[hipThreadIdx_x] + sdata[hipThreadIdx_x + BSRDIM * 1];
    }
    else
    {
        // Accumulate the row sum for different blocks
        if(hipThreadIdx_x < BSRDIM * BSRDIM)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * BSRDIM];

        // Reduce the intra block row sum
        if(lid < 1)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
        if(lid < 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];

        // Final reduction
        if(hipThreadIdx_x < BSRDIM)
            sum = sdata[hipThreadIdx_x * BSRDIM] + sdata[hipThreadIdx_x * BSRDIM + 1];
    }

    // First 5 threads write row sums to global memory
    if(hipThreadIdx_x < BSRDIM)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + hipThreadIdx_x]
                = rocsparse_fma(beta, y[row * BSRDIM + hipThreadIdx_x], alpha * sum);
        }
        else
        {
            y[row * BSRDIM + hipThreadIdx_x] = alpha * sum;
        }
    }
}

// BSRMV kernel for BSR block dimension of 8
template <typename T, unsigned int BLOCKSIZE>
__device__ void bsrmvn_8x8_device(rocsparse_int       mb,
                                  rocsparse_direction dir,
                                  T                   alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  const T* __restrict__ x,
                                  T beta,
                                  T* __restrict__ y,
                                  rocsparse_index_base idx_base)
{
    // BSR block dimension
    static constexpr int BSRDIM = 8;

    // BSR block lane id
    rocsparse_int lid = hipThreadIdx_x % BSRDIM;

    // Each thread block processes a single BSR row
    rocsparse_int row = hipBlockIdx_x;

    // Offset into x vector
    rocsparse_int idx
        = (dir == rocsparse_direction_column) ? ((hipThreadIdx_x / BSRDIM) % BSRDIM) : lid;

    // Number of BSR blocks processed at the same time
    const unsigned int NBLOCKS = BLOCKSIZE / (BSRDIM * BSRDIM);

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // BSR block row accumulator
    T sum = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block value
    for(rocsparse_int j = row_begin; j < row_end; j += NBLOCKS)
    {
        rocsparse_int k = j + hipThreadIdx_x / (BSRDIM * BSRDIM);

        // Do not exceed the row
        if(k < row_end)
        {
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[k] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            sum = rocsparse_fma(bsr_val[j * BSRDIM * BSRDIM + hipThreadIdx_x], x[col + idx], sum);
        }
    }

    // Accumulate each row sum of the BSR block
    __shared__ T sdata[BSRDIM * BSRDIM * NBLOCKS];

    sdata[hipThreadIdx_x] = sum;

    __syncthreads();

    if(dir == rocsparse_direction_column)
    {
        if(hipThreadIdx_x < BSRDIM * 8)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 8];
        if(hipThreadIdx_x < BSRDIM * 4)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 4];
        if(hipThreadIdx_x < BSRDIM * 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 2];
        if(hipThreadIdx_x < BSRDIM * 1)
            sum = sdata[hipThreadIdx_x] + sdata[hipThreadIdx_x + BSRDIM * 1];
    }
    else
    {
        // Accumulate the row sum for different blocks
        if(hipThreadIdx_x < BSRDIM * BSRDIM)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * BSRDIM];

        __syncthreads();

        // Reduce the intra block row sum
        if(lid < 4)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
        if(lid < 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];

        // Final reduction
        if(hipThreadIdx_x < BSRDIM)
            sum = sdata[hipThreadIdx_x * BSRDIM] + sdata[hipThreadIdx_x * BSRDIM + 1];
    }

    // First 8 threads write row sums to global memory
    if(hipThreadIdx_x < BSRDIM)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + hipThreadIdx_x]
                = rocsparse_fma(beta, y[row * BSRDIM + hipThreadIdx_x], alpha * sum);
        }
        else
        {
            y[row * BSRDIM + hipThreadIdx_x] = alpha * sum;
        }
    }
}

// BSRMV kernel for BSR block dimension of 16
template <typename T, unsigned int BLOCKSIZE>
__device__ void bsrmvn_16x16_device(rocsparse_int       mb,
                                    rocsparse_direction dir,
                                    T                   alpha,
                                    const rocsparse_int* __restrict__ bsr_row_ptr,
                                    const rocsparse_int* __restrict__ bsr_col_ind,
                                    const T* __restrict__ bsr_val,
                                    const T* __restrict__ x,
                                    T beta,
                                    T* __restrict__ y,
                                    rocsparse_index_base idx_base)
{
    // BSR block dimension
    static constexpr int BSRDIM = 16;

    // BSR block lane id
    rocsparse_int lid = hipThreadIdx_x % BSRDIM;

    // Each thread block processes a single BSR row
    rocsparse_int row = hipBlockIdx_x;

    // Offset into x vector
    rocsparse_int idx
        = (dir == rocsparse_direction_column) ? ((hipThreadIdx_x / BSRDIM) % BSRDIM) : lid;

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // BSR block row accumulator
    T sum = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block value
    for(rocsparse_int j = row_begin; j < row_end; ++j)
    {
        rocsparse_int k = j + hipThreadIdx_x / (BSRDIM * BSRDIM);

        // Do not exceed the row
        if(k < row_end)
        {
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[k] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            sum = rocsparse_fma(bsr_val[j * BSRDIM * BSRDIM + hipThreadIdx_x], x[col + idx], sum);
        }
    }

    // Accumulate each row sum of the BSR block
    __shared__ T sdata[BSRDIM * BSRDIM];

    sdata[hipThreadIdx_x] = sum;

    __syncthreads();

    if(dir == rocsparse_direction_column)
    {
        if(hipThreadIdx_x < BSRDIM * 8)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 8];
        __syncthreads();
        if(hipThreadIdx_x < BSRDIM * 4)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 4];
        if(hipThreadIdx_x < BSRDIM * 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 2];
        if(hipThreadIdx_x < BSRDIM * 1)
            sum = sdata[hipThreadIdx_x] + sdata[hipThreadIdx_x + BSRDIM * 1];
    }
    else
    {
        // Reduce the intra block row sum
        if(lid < 8)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 8];
        if(lid < 4)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
        if(lid < 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];

        // Final reduction
        if(hipThreadIdx_x < BSRDIM)
            sum = sdata[hipThreadIdx_x * BSRDIM] + sdata[hipThreadIdx_x * BSRDIM + 1];
    }

    // First 16 threads write row sums to global memory
    if(hipThreadIdx_x < BSRDIM)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + hipThreadIdx_x]
                = rocsparse_fma(beta, y[row * BSRDIM + hipThreadIdx_x], alpha * sum);
        }
        else
        {
            y[row * BSRDIM + hipThreadIdx_x] = alpha * sum;
        }
    }
}

// BSRMV kernel for BSR block dimension of 17 to 32
template <typename T, unsigned int BSRDIM>
__device__ void bsrmvn_17_32_device(rocsparse_int       mb,
                                    rocsparse_direction dir,
                                    T                   alpha,
                                    const rocsparse_int* __restrict__ bsr_row_ptr,
                                    const rocsparse_int* __restrict__ bsr_col_ind,
                                    const T* __restrict__ bsr_val,
                                    const T* __restrict__ x,
                                    T beta,
                                    T* __restrict__ y,
                                    rocsparse_index_base idx_base)
{
    // BSR block lane id
    rocsparse_int lid = hipThreadIdx_x % BSRDIM;

    // Each thread block processes a single BSR row
    rocsparse_int row = hipBlockIdx_x;

    // Offset into x vector
    rocsparse_int idx
        = (dir == rocsparse_direction_column) ? ((hipThreadIdx_x / BSRDIM) % BSRDIM) : lid;

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // BSR block row accumulator
    T sum = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block value
    for(rocsparse_int j = row_begin; j < row_end; ++j)
    {
        rocsparse_int k = j + hipThreadIdx_x / (BSRDIM * BSRDIM);

        // Do not exceed the row
        if(k < row_end)
        {
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[k] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            sum = rocsparse_fma(bsr_val[j * BSRDIM * BSRDIM + hipThreadIdx_x], x[col + idx], sum);
        }
    }

    // Accumulate each row sum of the BSR block
    __shared__ T sdata[BSRDIM * BSRDIM];

    sdata[hipThreadIdx_x] = sum;

    __syncthreads();

    if(dir == rocsparse_direction_column)
    {
        if(hipThreadIdx_x < BSRDIM * BSRDIM - BSRDIM * 16)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 16];
        __syncthreads();
        if(hipThreadIdx_x < BSRDIM * 8)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 8];
        __syncthreads();
        if(hipThreadIdx_x < BSRDIM * 4)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 4];
        __syncthreads();
        if(hipThreadIdx_x < BSRDIM * 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 2];
        if(hipThreadIdx_x < BSRDIM * 1)
            sum = sdata[hipThreadIdx_x] + sdata[hipThreadIdx_x + BSRDIM * 1];
    }
    else
    {
        // Reduce the intra block row sum
        if(lid < BSRDIM - 16)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 16];
        __syncthreads();
        if(lid < 8)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 8];
        __syncthreads();
        if(lid < 4)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
        __syncthreads();
        if(lid < 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];
        __syncthreads();

        // Final reduction
        if(hipThreadIdx_x < BSRDIM)
            sum = sdata[hipThreadIdx_x * BSRDIM] + sdata[hipThreadIdx_x * BSRDIM + 1];
    }

    // First bunch of threads write row sums to global memory
    if(hipThreadIdx_x < BSRDIM)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + hipThreadIdx_x]
                = rocsparse_fma(beta, y[row * BSRDIM + hipThreadIdx_x], alpha * sum);
        }
        else
        {
            y[row * BSRDIM + hipThreadIdx_x] = alpha * sum;
        }
    }
}

#endif // BSRMV_DEVICE_H
