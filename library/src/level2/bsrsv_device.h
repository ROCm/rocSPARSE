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
#ifndef BSRSV_DEVICE_H
#define BSRSV_DEVICE_H

#include "common.h"

#include <hip/hip_runtime.h>

template <typename T, unsigned int WFSIZE, unsigned int DIMY, unsigned int BSRDIM>
__launch_bounds__(WFSIZE* DIMY) __global__ void bsrsv_gather(rocsparse_int nnzb,
                                                             const rocsparse_int* __restrict__ perm,
                                                             const T* __restrict__ bsr_val_A,
                                                             T* __restrict__ bsr_val_T,
                                                             rocsparse_int bsr_dim)
{
    rocsparse_int lid = hipThreadIdx_x & (BSRDIM - 1);
    rocsparse_int wid = hipThreadIdx_x / BSRDIM;

    // Non-permuted nnz index
    rocsparse_int i = hipBlockIdx_x * DIMY + hipThreadIdx_y;

    // Do not exceed the number of elements
    if(i >= nnzb)
    {
        return;
    }

    // Load the permuted nnz index
    rocsparse_int j = perm[i];

    // Gather values from A and store them to T with respect to the
    // given row / column permutation
    for(rocsparse_int bi = lid; bi < bsr_dim; bi += BSRDIM)
    {
        for(rocsparse_int bj = wid; bj < bsr_dim; bj += BSRDIM)
        {
            bsr_val_T[bsr_dim * bsr_dim * i + bi + bj * bsr_dim]
                = bsr_val_A[bsr_dim * bsr_dim * j + bj + bi * bsr_dim];
        }
    }
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__device__ void bsrsv_lower_general_device(rocsparse_int mb,
                                           T             alpha,
                                           const rocsparse_int* __restrict__ bsr_row_ptr,
                                           const rocsparse_int* __restrict__ bsr_col_ind,
                                           const T* __restrict__ bsr_val,
                                           rocsparse_int bsr_dim,
                                           const T* __restrict__ x,
                                           T* __restrict__ y,
                                           int* __restrict__ done_array,
                                           rocsparse_int* __restrict__ map,
                                           rocsparse_int* __restrict__ zero_pivot,
                                           rocsparse_index_base idx_base,
                                           rocsparse_diag_type  diag_type,
                                           rocsparse_direction  dir)
{
    int lid = hipThreadIdx_x & (WFSIZE - 1);
    int wid = hipThreadIdx_x / WFSIZE;

    // Index into the row map
    rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= mb)
    {
        return;
    }

    // Get the BSR row this wavefront will operate on
    rocsparse_int row = map[idx];

    // Current row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // Initialize local_col with mb
    rocsparse_int local_col = mb;

    // Initialize y with alpha and x
    for(rocsparse_int bi = lid; bi < bsr_dim; bi += WFSIZE)
    {
        y[row * bsr_dim + bi] = alpha * x[row * bsr_dim + bi];
    }

    // Loop over the current row
    rocsparse_int j;
    for(j = row_begin; j < row_end; ++j)
    {
        // Current column index
        local_col = bsr_col_ind[j] - idx_base;

        // Processing lower triangular

        // Ignore all diagonal entries and above
        if(local_col >= row)
        {
            break;
        }

        // Spin loop until dependency has been resolved
        int          local_done    = atomicOr(&done_array[local_col], 0);
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

            local_done = atomicOr(&done_array[local_col], 0);
        }

        // Wait for y to be visible globally
        __threadfence();

        // Local sum computation
        for(rocsparse_int bi = lid; bi < bsr_dim; bi += WFSIZE)
        {
            // Local sum accumulator
            T local_sum = static_cast<T>(0);

            for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
            {
                local_sum = rocsparse_fma(
                    bsr_val[BSR_IND(j, bi, bj, dir)], y[local_col * bsr_dim + bj], local_sum);
            }

            // Write local sum to y
            y[row * bsr_dim + bi] -= local_sum;
        }
    }

    bool pivot = false;

    // Process diagonal
    if(local_col == row)
    {
        for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
        {
            // Load diagonal matrix entry
            T diag = (diag_type == rocsparse_diag_type_non_unit)
                         ? bsr_val[bsr_dim * bsr_dim * j + bi + bi * bsr_dim]
                         : static_cast<T>(1);

            // Load result of bi-th BSR row
            T val = y[row * bsr_dim + bi];

            // Check for numerical pivot
            if(diag == static_cast<T>(0))
            {
                pivot = true;
            }
            else
            {
                // Divide result of bi-th BSR row by diagonal entry
                y[row * bsr_dim + bi] = val /= diag;
            }

            // Update remaining non-diagonal entries
            for(rocsparse_int bj = bi + lid + 1; bj < bsr_dim; bj += WFSIZE)
            {
                y[row * bsr_dim + bj] -= val * bsr_val[BSR_IND(j, bj, bi, dir)];
            }
        }
    }

    // Make sure y is written to global memory before setting "row is done" flag
    __threadfence();

    // Write "row is done" flag
    if(lid == 0)
    {
        atomicOr(&done_array[row], 1);

        if(pivot == true)
        {
            atomicMin(zero_pivot, row + idx_base);
        }
    }
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__device__ void bsrsv_upper_general_device(rocsparse_int mb,
                                           T             alpha,
                                           const rocsparse_int* __restrict__ bsr_row_ptr,
                                           const rocsparse_int* __restrict__ bsr_col_ind,
                                           const T* __restrict__ bsr_val,
                                           rocsparse_int bsr_dim,
                                           const T* __restrict__ x,
                                           T* __restrict__ y,
                                           int* __restrict__ done_array,
                                           rocsparse_int* __restrict__ map,
                                           rocsparse_int* __restrict__ zero_pivot,
                                           rocsparse_index_base idx_base,
                                           rocsparse_diag_type  diag_type,
                                           rocsparse_direction  dir)
{
    int lid = hipThreadIdx_x & (WFSIZE - 1);
    int wid = hipThreadIdx_x / WFSIZE;

    // Index into the row map
    rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= mb)
    {
        return;
    }

    // Get the BSR row this wavefront will operate on
    rocsparse_int row = map[idx];

    // Current row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // Initialize local_col with mb
    rocsparse_int local_col = mb;

    // Initialize y with alpha and x
    for(rocsparse_int bi = lid; bi < bsr_dim; bi += WFSIZE)
    {
        y[row * bsr_dim + bi] = alpha * x[row * bsr_dim + bi];
    }

    // Loop over the current row
    rocsparse_int j;
    for(j = row_end - 1; j >= row_begin; --j)
    {
        // Current column index
        local_col = bsr_col_ind[j] - idx_base;

        // Processing upper triangular

        // Ignore all diagonal entries and below
        if(local_col <= row)
        {
            break;
        }

        // Spin loop until dependency has been resolved
        int          local_done    = atomicOr(&done_array[local_col], 0);
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

            local_done = atomicOr(&done_array[local_col], 0);
        }

        // Wait for y to be visible globally
        __threadfence();

        // Local sum computation
        for(rocsparse_int bi = lid; bi < bsr_dim; bi += WFSIZE)
        {
            // Local sum accumulator
            T local_sum = static_cast<T>(0);

            for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
            {
                local_sum = rocsparse_fma(
                    bsr_val[BSR_IND(j, bi, bj, dir)], y[local_col * bsr_dim + bj], local_sum);
            }

            // Write local sum to y
            y[row * bsr_dim + bi] -= local_sum;
        }
    }

    bool pivot = false;

    // Process diagonal
    if(local_col == row)
    {
        for(rocsparse_int bi = bsr_dim - 1; bi >= 0; --bi)
        {
            // Load diagonal matrix entry
            T diag = (diag_type == rocsparse_diag_type_non_unit)
                         ? bsr_val[bsr_dim * bsr_dim * j + bi + bi * bsr_dim]
                         : static_cast<T>(1);

            // Load result of bi-th BSR row
            T val = y[row * bsr_dim + bi];

            // Check for numerical pivot
            if(diag == static_cast<T>(0))
            {
                pivot = true;
            }
            else
            {
                // Divide result of bi-th BSR row by diagonal entry
                y[row * bsr_dim + bi] = val /= diag;
            }

            // Update remaining non-diagonal entries
            for(rocsparse_int bj = lid; bj < bi; bj += WFSIZE)
            {
                y[row * bsr_dim + bj] -= val * bsr_val[BSR_IND(j, bj, bi, dir)];
            }
        }
    }

    // Make sure y is written to global memory before setting "row is done" flag
    __threadfence();

    // Write "row is done" flag
    if(lid == 0)
    {
        atomicOr(&done_array[row], 1);

        if(pivot == true)
        {
            atomicMin(zero_pivot, row + idx_base);
        }
    }
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, rocsparse_int BSRDIM, bool SLEEP>
__device__ void bsrsv_lower_shared_device(rocsparse_int mb,
                                          T             alpha,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          const T* __restrict__ bsr_val,
                                          rocsparse_int bsr_dim,
                                          const T* __restrict__ x,
                                          T* __restrict__ y,
                                          int* __restrict__ done_array,
                                          rocsparse_int* __restrict__ map,
                                          rocsparse_int* __restrict__ zero_pivot,
                                          rocsparse_index_base idx_base,
                                          rocsparse_diag_type  diag_type,
                                          rocsparse_direction  dir)
{
    int lid = hipThreadIdx_x & (WFSIZE - 1);
    int wid = hipThreadIdx_x / WFSIZE;

    // Index into the row map
    rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= mb)
    {
        return;
    }

    // Get the BSR row this wavefront will operate on
    rocsparse_int row = map[idx];

    // Current row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // Initialize local_col with mb
    rocsparse_int local_col = mb;

    // Initialize local summation variable with alpha and x
    T local_sum = alpha * ((lid < bsr_dim) ? x[row * bsr_dim + lid] : static_cast<T>(0));

    // Shared memory to hold BSR blocks and updated sums
    __shared__ T sdata1[BLOCKSIZE / WFSIZE * BSRDIM * BSRDIM];
    __shared__ T sdata2[BLOCKSIZE / WFSIZE * BSRDIM];

    T* bsr_values  = &sdata1[wid * BSRDIM * BSRDIM];
    T* bsr_updates = &sdata2[wid * BSRDIM];

    // Loop over the current row
    for(rocsparse_int j = row_begin; j < row_end; ++j)
    {
        // Current column index
        local_col = bsr_col_ind[j] - idx_base;

        // Load BSR block values
        // Each wavefront loads a full BSR block into shared memory
        // Pad remaining entries with zero
        int bi = lid & (BSRDIM - 1);
        int bj = lid / BSRDIM;

        for(rocsparse_int k = bj; k < BSRDIM; k += WFSIZE / BSRDIM)
        {
            bsr_values[bi + k * BSRDIM] = (bi < bsr_dim && k < bsr_dim)
                                              ? bsr_val[BSR_IND(j, bi, k, dir)]
                                              : static_cast<T>(0);
        }

        // Processing lower triangular

        // Ignore all diagonal entries and above
        if(local_col >= row)
        {
            break;
        }

        // Spin loop until dependency has been resolved
        int          local_done    = atomicOr(&done_array[local_col], 0);
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

            local_done = atomicOr(&done_array[local_col], 0);
        }

        // Wait for y to be visible globally
        __threadfence();

        // Load all updated dependencies into shared memory
        if(lid < BSRDIM)
        {
            bsr_updates[lid] = (lid < bsr_dim) ? y[local_col * bsr_dim + lid] : static_cast<T>(0);
        }

        __threadfence_block();

        // Local sum computation
        if(lid < bsr_dim)
        {
            for(rocsparse_int bj = 0; bj < BSRDIM; ++bj)
            {
                local_sum
                    = rocsparse_fma(-bsr_values[lid + bj * BSRDIM], bsr_updates[bj], local_sum);
            }
        }
    }

    // Initialize zero pivot
    bool pivot = false;

    // Process diagonal
    if(local_col == row)
    {
        for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
        {
            // Load diagonal matrix entry
            T diag = (diag_type == rocsparse_diag_type_non_unit) ? bsr_values[bi + bi * BSRDIM]
                                                                 : static_cast<T>(1);

            // Load result of bi-th BSR row
            T val = rocsparse_shfl(local_sum, bi);

            // Check for numerical pivot
            if(diag == static_cast<T>(0))
            {
                pivot = true;
            }
            else
            {
                // Divide result of bi-th row by diagonal entry
                val /= diag;
            }

            // Update remaining non-diagonal entries
            if(lid < bsr_dim)
            {
                if(bi < lid)
                {
                    local_sum = rocsparse_fma(-val, bsr_values[lid + bi * BSRDIM], local_sum);
                }
                else if(lid == bi)
                {
                    local_sum = val;
                }
            }
        }
    }

    if(lid < bsr_dim)
    {
        // Store the rows results in y
        y[row * bsr_dim + lid] = local_sum;
    }

    // Make sure y is written to global memory before setting "row is done" flag
    __threadfence();

    if(lid == 0)
    {
        // Write "row is done" flag
        atomicOr(&done_array[row], 1);

        // Find the minimum pivot, if applicable
        if(pivot == true)
        {
            atomicMin(zero_pivot, row + idx_base);
        }
    }
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, rocsparse_int BSRDIM, bool SLEEP>
__device__ void bsrsv_upper_shared_device(rocsparse_int mb,
                                          T             alpha,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          const T* __restrict__ bsr_val,
                                          rocsparse_int bsr_dim,
                                          const T* __restrict__ x,
                                          T* __restrict__ y,
                                          int* __restrict__ done_array,
                                          rocsparse_int* __restrict__ map,
                                          rocsparse_int* __restrict__ zero_pivot,
                                          rocsparse_index_base idx_base,
                                          rocsparse_diag_type  diag_type,
                                          rocsparse_direction  dir)
{
    int lid = hipThreadIdx_x & (WFSIZE - 1);
    int wid = hipThreadIdx_x / WFSIZE;

    // Index into the row map
    rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= mb)
    {
        return;
    }

    // Get the BSR row this wavefront will operate on
    rocsparse_int row = map[idx];

    // Current row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // Initialize local_col with mb
    rocsparse_int local_col = mb;

    // Initialize local summation variable with alpha and x
    T local_sum = alpha * ((lid < bsr_dim) ? x[row * bsr_dim + lid] : static_cast<T>(0));

    // Shared memory to hold BSR blocks and updated sums
    __shared__ T sdata1[BLOCKSIZE / WFSIZE * BSRDIM * BSRDIM];
    __shared__ T sdata2[BLOCKSIZE / WFSIZE * BSRDIM];

    T* bsr_values  = &sdata1[wid * BSRDIM * BSRDIM];
    T* bsr_updates = &sdata2[wid * BSRDIM];

    // Loop over the current row
    for(rocsparse_int j = row_end - 1; j >= row_begin; --j)
    {
        // Current column index
        local_col = bsr_col_ind[j] - idx_base;

        // Load BSR block values
        // Each wavefront loads a full BSR block into shared memory
        // Pad remaining entries with zero
        int bi = lid & (BSRDIM - 1);
        int bj = lid / BSRDIM;

        for(rocsparse_int k = bj; k < BSRDIM; k += WFSIZE / BSRDIM)
        {
            bsr_values[bi + k * BSRDIM] = (bi < bsr_dim && k < bsr_dim)
                                              ? bsr_val[BSR_IND(j, bi, k, dir)]
                                              : static_cast<T>(0);
        }

        // Processing upper triangular

        // Ignore all diagonal entries and below
        if(local_col <= row)
        {
            break;
        }

        // Spin loop until dependency has been resolved
        int          local_done    = atomicOr(&done_array[local_col], 0);
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

            local_done = atomicOr(&done_array[local_col], 0);
        }

        // Wait for y to be visible globally
        __threadfence();

        // Load all updated dependencies into shared memory
        if(lid < BSRDIM)
        {
            bsr_updates[lid] = (lid < bsr_dim) ? y[local_col * bsr_dim + lid] : static_cast<T>(0);
        }

        __threadfence_block();

        // Local sum computation
        if(lid < bsr_dim)
        {
            for(rocsparse_int bj = 0; bj < BSRDIM; ++bj)
            {
                local_sum
                    = rocsparse_fma(-bsr_values[lid + bj * BSRDIM], bsr_updates[bj], local_sum);
            }
        }
    }

    // Initialize zero pivot
    bool pivot = false;

    // Process diagonal
    if(local_col == row)
    {
        for(rocsparse_int bi = bsr_dim - 1; bi >= 0; --bi)
        {
            // Load diagonal matrix entry
            T diag = (diag_type == rocsparse_diag_type_non_unit) ? bsr_values[bi + bi * BSRDIM]
                                                                 : static_cast<T>(1);

            // Load result of bi-th BSR row
            T val = rocsparse_shfl(local_sum, bi);

            // Check for numerical pivot
            if(diag == static_cast<T>(0))
            {
                pivot = true;
            }
            else
            {
                // Divide result of bi-th row by diagonal entry
                val /= diag;
            }

            // Update remaining non-diagonal entries
            if(lid < bsr_dim)
            {
                if(bi > lid)
                {
                    local_sum = rocsparse_fma(-val, bsr_values[lid + bi * BSRDIM], local_sum);
                }
                else if(lid == bi)
                {
                    local_sum = val;
                }
            }
        }
    }

    if(lid < bsr_dim)
    {
        // Store the rows results in y
        y[row * bsr_dim + lid] = local_sum;
    }

    // Make sure y is written to global memory before setting "row is done" flag
    __threadfence();

    if(lid == 0)
    {
        // Write "row is done" flag
        atomicOr(&done_array[row], 1);

        // Find the minimum pivot, if applicable
        if(pivot == true)
        {
            atomicMin(zero_pivot, row + idx_base);
        }
    }
}

#endif // BSRSV_DEVICE_H
