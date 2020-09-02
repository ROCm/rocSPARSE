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
#ifndef BSRILU0_DEVICE_H
#define BSRILU0_DEVICE_H

#include "common.h"

#include <hip/hip_runtime.h>

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int BSRDIM>
__device__ void bsrilu0_2_8_device(rocsparse_direction  dir,
                                   rocsparse_int        mb,
                                   const rocsparse_int* bsr_row_ptr,
                                   const rocsparse_int* bsr_col_ind,
                                   T*                   bsr_val,
                                   const rocsparse_int* bsr_diag_ind,
                                   rocsparse_int        bsr_dim,
                                   int*                 done_array,
                                   const rocsparse_int* map,
                                   rocsparse_int*       zero_pivot,
                                   rocsparse_index_base idx_base,
                                   int                  boost,
                                   U                    boost_tol,
                                   T                    boost_val)
{
    // Current row this wavefront is working on
    rocsparse_int row = map[blockIdx.x];

    // Diagonal entry point of the current row
    rocsparse_int row_diag = bsr_diag_ind[row];

    // Row entry point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // Zero pivot tracker
    bool pivot = false;

    // Shared memory to cache BSR values
    __shared__ T sdata1[BSRDIM][BSRDIM + 1];
    __shared__ T sdata2[BSRDIM][BSRDIM + 1];

    // Check for structural pivot
    if(row_diag != -1)
    {
        // Process lower diagonal
        for(rocsparse_int j = row_begin; j < row_diag; ++j)
        {
            // Column index of current BSR block
            rocsparse_int bsr_col = bsr_col_ind[j] - idx_base;

            // Load row j into shared memory
            sdata2[threadIdx.y][threadIdx.x]
                = (threadIdx.x < bsr_dim && threadIdx.y < bsr_dim)
                      ? bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                      : static_cast<T>(0);

            // Process all lower matrix BSR blocks

            // Obtain corresponding row entry and exit point that corresponds with the
            // current BSR column. Actually, we skip all lower matrix column indices,
            // therefore starting with the diagonal entry.
            rocsparse_int diag_j    = bsr_diag_ind[bsr_col];
            rocsparse_int row_end_j = bsr_row_ptr[bsr_col + 1] - idx_base;

            // Check for structural pivot
            if(diag_j == -1)
            {
                pivot = true;
                break;
            }

            // Spin loop until dependency has been resolved
            while(!atomicOr(&done_array[bsr_col], 0))
                ;

            // Make sure dependencies are visible in global memory
            __threadfence();

            // Load updated BSR block into shared memory
            sdata1[threadIdx.y][threadIdx.x]
                = (threadIdx.x < bsr_dim && threadIdx.y < bsr_dim)
                      ? bsr_val[BSR_IND(diag_j, threadIdx.x, threadIdx.y, dir)]
                      : static_cast<T>(0);

            // Make sure all writes to shared memory are visible
            __threadfence_block();

            // Loop through all rows within the BSR block
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                // Load diagonal entry of the BSR block
                T diag = sdata1[bi][bi];
                T val  = sdata2[bi][threadIdx.x];

                // This has already been checked for zero by previous computations
                val /= diag;

                // Make sure val has been read before updating
                __threadfence_block();

                // Update
                if(threadIdx.y == 0)
                {
                    sdata2[bi][threadIdx.x] = val;
                }

                // Do linear combination
                rocsparse_int bj = bi + 1 + threadIdx.y;
                if(bj < bsr_dim)
                {
                    sdata2[bj][threadIdx.x]
                        = rocsparse_fma(-val, sdata1[bj][bi], sdata2[bj][threadIdx.x]);
                }

                __threadfence_block();
            }

            // Write row j back to global memory
            if(threadIdx.x < bsr_dim && threadIdx.y < bsr_dim)
            {
                bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                    = sdata2[threadIdx.y][threadIdx.x];
            }

            // Loop over upper offset pointer and do linear combination for nnz entry
            for(rocsparse_int k = diag_j + 1; k < row_end_j; ++k)
            {
                rocsparse_int bsr_col_k = bsr_col_ind[k] - idx_base;

                // Search for matching column index in current row
                rocsparse_int q         = row_begin + threadIdx.x + threadIdx.y * blockDim.x;
                rocsparse_int bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;

                // Check if match has been found by any thread in the wavefront
                while(bsr_col_j < bsr_col_k)
                {
                    q += WFSIZE;
                    bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;
                }

                // Check if match has been found by any thread in the wavefront
                int match = __ffsll(__ballot(bsr_col_j == bsr_col_k));

                // If match has been found, process it
                if(match)
                {
                    // Tell all other threads about the matching index
                    rocsparse_int m = __shfl(q, match - 1);

                    // Load BSR block from row k into shared memory
                    sdata1[threadIdx.y][threadIdx.x]
                        = bsr_val[BSR_IND(k, threadIdx.x, threadIdx.y, dir)];

                    // Make sure all writes to shared memory are visible
                    __threadfence_block();

                    T sum = static_cast<T>(0);

                    for(rocsparse_int bk = 0; bk < bsr_dim; ++bk)
                    {
                        sum = rocsparse_fma(sdata2[bk][threadIdx.x], sdata1[threadIdx.y][bk], sum);
                    }

                    // Write back to global row m
                    if(threadIdx.x < bsr_dim && threadIdx.y < bsr_dim)
                    {
                        // Do not pre-cache row m as we read/write only once
                        bsr_val[BSR_IND(m, threadIdx.x, threadIdx.y, dir)] -= sum;
                    }
                }

                __threadfence_block();
            }
        }

        // Process diagonal
        if(bsr_col_ind[row_diag] - idx_base == row)
        {
            // Load diagonal BSR block into shared memory
            sdata1[threadIdx.y][threadIdx.x]
                = (threadIdx.x < bsr_dim && threadIdx.y < bsr_dim)
                      ? bsr_val[BSR_IND(row_diag, threadIdx.x, threadIdx.y, dir)]
                      : static_cast<T>(0);

            __threadfence_block();

            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                // Load diagonal matrix entry
                T diag = sdata1[bi][bi];

                // Numeric boost
                if(boost)
                {
                    diag = (boost_tol >= rocsparse_abs(diag)) ? boost_val : diag;

                    __threadfence_block();

                    if(threadIdx.x == 0 && threadIdx.y == 0)
                    {
                        sdata1[bi][bi] = diag;
                    }
                }
                else
                {
                    // Check for numeric pivot
                    if(diag == static_cast<T>(0))
                    {
                        pivot = true;
                        continue;
                    }
                }

                rocsparse_int bk = bi + 1 + threadIdx.x;
                if(bk < bsr_dim)
                {
                    // Multiplication factor
                    T val = sdata1[bi][bk];
                    val /= diag;

                    // Make sure val has been read before updating
                    __threadfence_block();

                    // Update
                    if(threadIdx.y == 0)
                    {
                        sdata1[bi][bk] = val;
                    }

                    // Do linear combination
                    rocsparse_int bj = bi + 1 + threadIdx.y;
                    if(bj < bsr_dim)
                    {
                        sdata1[bj][bk] = rocsparse_fma(-val, sdata1[bj][bi], sdata1[bj][bk]);
                    }
                }
            }

            __threadfence_block();

            // Write diagonal BSR block back to global memory
            if(threadIdx.x < bsr_dim && threadIdx.y < bsr_dim)
            {
                bsr_val[BSR_IND(row_diag, threadIdx.x, threadIdx.y, dir)]
                    = sdata1[threadIdx.y][threadIdx.x];
            }
        }

        // Process upper diagonal BSR blocks
        for(rocsparse_int j = row_diag + 1; j < row_end; ++j)
        {
            __threadfence_block();

            // Load row j into shared memory
            sdata2[threadIdx.y][threadIdx.x]
                = (threadIdx.x < bsr_dim && threadIdx.y < bsr_dim)
                      ? bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                      : static_cast<T>(0);

            __threadfence_block();

            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                rocsparse_int bj = bi + 1 + threadIdx.y;
                if(bj < bsr_dim)
                {
                    sdata2[threadIdx.x][bj] = rocsparse_fma(
                        -sdata1[bi][bj], sdata2[threadIdx.x][bi], sdata2[threadIdx.x][bj]);
                }
            }

            __threadfence_block();

            // Write row j back to global memory
            if(threadIdx.x < bsr_dim && threadIdx.y < bsr_dim)
            {
                bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                    = sdata2[threadIdx.y][threadIdx.x];
            }
        }
    }
    else
    {
        // Structural pivot found
        pivot = true;
    }

    // Make sure updated csr_val is written to global memory
    __threadfence();

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        // First lane writes "we are done" flag
        atomicOr(&done_array[row], 1);

        if(pivot)
        {
            // Atomically set minimum zero pivot, if found
            atomicMin(zero_pivot, row + idx_base);
        }
    }
}

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__device__ void bsrilu0_general_device(rocsparse_direction  dir,
                                       rocsparse_int        mb,
                                       const rocsparse_int* bsr_row_ptr,
                                       const rocsparse_int* bsr_col_ind,
                                       T*                   bsr_val,
                                       const rocsparse_int* bsr_diag_ind,
                                       rocsparse_int        bsr_dim,
                                       int*                 done_array,
                                       const rocsparse_int* map,
                                       rocsparse_int*       zero_pivot,
                                       rocsparse_index_base idx_base,
                                       int                  boost,
                                       U                    boost_tol,
                                       T                    boost_val)
{
    int lid = hipThreadIdx_x & (WFSIZE - 1);
    int wid = hipThreadIdx_x / WFSIZE;

    // Index
    rocsparse_int idx = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= mb)
    {
        return;
    }

    // Current row this wavefront is working on
    rocsparse_int row = map[idx];

    // Diagonal entry point of the current row
    rocsparse_int row_diag = bsr_diag_ind[row];

    // Row entry point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // Zero pivot tracker
    bool pivot = false;

    // Check for structural pivot
    if(row_diag != -1)
    {
        // Process lower diagonal
        for(rocsparse_int j = row_begin; j < row_diag; ++j)
        {
            // Column index of current BSR block
            rocsparse_int bsr_col = bsr_col_ind[j] - idx_base;

            // Process all lower matrix BSR blocks

            // Obtain corresponding row entry and exit point that corresponds with the
            // current BSR column. Actually, we skip all lower matrix column indices,
            // therefore starting with the diagonal entry.
            rocsparse_int diag_j    = bsr_diag_ind[bsr_col];
            rocsparse_int row_end_j = bsr_row_ptr[bsr_col + 1] - idx_base;

            // Check for structural pivot
            if(diag_j == -1)
            {
                pivot = true;
                break;
            }

            // Spin loop until dependency has been resolved
            int          local_done    = atomicOr(&done_array[bsr_col], 0);
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

                local_done = atomicOr(&done_array[bsr_col], 0);
            }

            // Make sure dependencies are visible in global memory
            __threadfence();

            // Loop through all rows within the BSR block
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                // Load diagonal entry of the BSR block
                T diag = bsr_val[BSR_IND(diag_j, bi, bi, dir)];

                // Loop through all rows
                for(rocsparse_int bk = lid; bk < bsr_dim; bk += WFSIZE)
                {
                    T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                    // This has already been checked for zero by previous computations
                    val /= diag;

                    // Update
                    bsr_val[BSR_IND(j, bk, bi, dir)] = val;

                    // Do linear combination

                    // Loop through all columns above the diagonal of the BSR block
                    for(rocsparse_int bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = rocsparse_fma(-val,
                                            bsr_val[BSR_IND(diag_j, bi, bj, dir)],
                                            bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }

            // Loop over upper offset pointer and do linear combination for nnz entry
            for(rocsparse_int k = diag_j + 1; k < row_end_j; ++k)
            {
                rocsparse_int bsr_col_k = bsr_col_ind[k] - idx_base;

                // Search for matching column index in current row
                rocsparse_int q         = row_begin + lid;
                rocsparse_int bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;

                // Check if match has been found by any thread in the wavefront
                while(bsr_col_j < bsr_col_k)
                {
                    q += WFSIZE;
                    bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;
                }

                // Check if match has been found by any thread in the wavefront
                int match = __ffsll(__ballot(bsr_col_j == bsr_col_k));

                // If match has been found, process it
                if(match)
                {
                    // Tell all other threads about the matching index
                    rocsparse_int m = __shfl(q, match - 1);

                    for(rocsparse_int bi = lid; bi < bsr_dim; bi += WFSIZE)
                    {
                        for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
                        {
                            T sum = static_cast<T>(0);

                            for(rocsparse_int bk = 0; bk < bsr_dim; ++bk)
                            {
                                sum = rocsparse_fma(bsr_val[BSR_IND(j, bi, bk, dir)],
                                                    bsr_val[BSR_IND(k, bk, bj, dir)],
                                                    sum);
                            }

                            bsr_val[BSR_IND(m, bi, bj, dir)] -= sum;
                        }
                    }
                }
            }
        }

        // Process diagonal
        if(bsr_col_ind[row_diag] - idx_base == row)
        {
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                // Load diagonal matrix entry
                T diag = bsr_val[BSR_IND(row_diag, bi, bi, dir)];

                // Numeric boost
                if(boost)
                {
                    diag = (boost_tol >= rocsparse_abs(diag)) ? boost_val : diag;

                    if(lid == 0)
                    {
                        bsr_val[BSR_IND(row_diag, bi, bi, dir)] = diag;
                    }
                }
                else
                {
                    // Check for numeric pivot
                    if(diag == static_cast<T>(0))
                    {
                        pivot = true;
                        continue;
                    }
                }

                for(rocsparse_int bk = bi + 1 + lid; bk < bsr_dim; bk += WFSIZE)
                {
                    // Multiplication factor
                    T val = bsr_val[BSR_IND(row_diag, bk, bi, dir)];
                    val /= diag;

                    // Update
                    bsr_val[BSR_IND(row_diag, bk, bi, dir)] = val;

                    // Do linear combination
                    for(rocsparse_int bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(row_diag, bk, bj, dir)]
                            = rocsparse_fma(-val,
                                            bsr_val[BSR_IND(row_diag, bi, bj, dir)],
                                            bsr_val[BSR_IND(row_diag, bk, bj, dir)]);
                    }
                }
            }
        }

        // Process upper diagonal BSR blocks
        for(rocsparse_int j = row_diag + 1; j < row_end; ++j)
        {
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                for(rocsparse_int bk = lid; bk < bsr_dim; bk += WFSIZE)
                {
                    for(rocsparse_int bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bj, bk, dir)]
                            = rocsparse_fma(-bsr_val[BSR_IND(row_diag, bj, bi, dir)],
                                            bsr_val[BSR_IND(j, bi, bk, dir)],
                                            bsr_val[BSR_IND(j, bj, bk, dir)]);
                    }
                }
            }
        }
    }
    else
    {
        // Structural pivot found
        pivot = true;
    }

    // Make sure updated csr_val is written to global memory
    __threadfence();

    if(lid == 0)
    {
        // First lane writes "we are done" flag
        atomicOr(&done_array[row], 1);

        if(pivot)
        {
            // Atomically set minimum zero pivot, if found
            atomicMin(zero_pivot, row + idx_base);
        }
    }
}

#endif // BSRILU0_DEVICE_H
