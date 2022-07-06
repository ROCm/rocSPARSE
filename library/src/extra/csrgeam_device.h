/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef CSRGEAM_DEVICE_H
#define CSRGEAM_DEVICE_H

#include "common.h"

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void csrgeam_index_base(rocsparse_int* nnz)
{
    --(*nnz);
}

// Compute non-zero entries per row, where each row is processed by a wavefront.
// Splitting row into several chunks such that we can use shared memory to store whether
// a column index is populated or not.
template <unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrgeam_nnz_multipass_device(rocsparse_int m,
                                      rocsparse_int n,
                                      const rocsparse_int* __restrict__ csr_row_ptr_A,
                                      const rocsparse_int* __restrict__ csr_col_ind_A,
                                      const rocsparse_int* __restrict__ csr_row_ptr_B,
                                      const rocsparse_int* __restrict__ csr_col_ind_B,
                                      rocsparse_int* __restrict__ row_nnz,
                                      rocsparse_index_base idx_base_A,
                                      rocsparse_index_base idx_base_B)
{
    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each wavefront processes a row
    rocsparse_int row = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(row >= m)
    {
        return;
    }

    // Row nnz marker
    __shared__ bool stable[BLOCKSIZE];
    bool*           table = &stable[wid * WFSIZE];

    // Get row entry and exit point of A
    rocsparse_int row_begin_A = csr_row_ptr_A[row] - idx_base_A;
    rocsparse_int row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

    // Get row entry and exit point of B
    rocsparse_int row_begin_B = csr_row_ptr_B[row] - idx_base_B;
    rocsparse_int row_end_B   = csr_row_ptr_B[row + 1] - idx_base_B;

    // Load the first column of the current row from A and B to set the starting
    // point for the first chunk
    rocsparse_int col_A = (row_begin_A < row_end_A) ? csr_col_ind_A[row_begin_A] - idx_base_A : n;
    rocsparse_int col_B = (row_begin_B < row_end_B) ? csr_col_ind_B[row_begin_B] - idx_base_B : n;

    // Begin of the current row chunk
    rocsparse_int chunk_begin = min(col_A, col_B);

    // Initialize the row nnz for the full (wavefront-wide) row
    rocsparse_int nnz = 0;

    // Initialize the index for column access into A and B
    row_begin_A += lid;
    row_begin_B += lid;

    // Loop over the chunks until the end of both rows (A and B) has been reached (which
    // is the number of total columns n)
    while(true)
    {
        // Initialize row nnz table
        table[lid] = false;

        __threadfence_block();

        // Initialize the beginning of the next chunk
        rocsparse_int min_col = n;

        // Loop over all columns of A, starting with the first entry that did not fit
        // into the previous chunk
        for(; row_begin_A < row_end_A; row_begin_A += WFSIZE)
        {
            // Get the column of A
            rocsparse_int col_A = csr_col_ind_A[row_begin_A] - idx_base_A;

            // Get the column of A shifted by the chunk_begin
            rocsparse_int shf_A = col_A - chunk_begin;

            // Check if this column of A is within the chunk
            if(shf_A < WFSIZE)
            {
                // Mark this column in shared memory
                table[shf_A] = true;
            }
            else
            {
                // Store the first column index of A that exceeds the current chunk
                min_col = min(min_col, col_A);
                break;
            }
        }

        // Loop over all columns of B, starting with the first entry that did not fit
        // into the previous chunk
        for(; row_begin_B < row_end_B; row_begin_B += WFSIZE)
        {
            // Get the column of B
            rocsparse_int col_B = csr_col_ind_B[row_begin_B] - idx_base_B;

            // Get the column of B shifted by the chunk_begin
            rocsparse_int shf_B = col_B - chunk_begin;

            // Check if this column of B is within the chunk
            if(shf_B < WFSIZE)
            {
                // Mark this column in shared memory
                table[shf_B] = true;
            }
            else
            {
                // Store the first column index of B that exceeds the current chunk
                min_col = min(min_col, col_B);
                break;
            }
        }

        __threadfence_block();

        // Compute the chunk's number of non-zeros of the row and add it to the global
        // row nnz counter
        nnz += __popcll(__ballot(table[lid]));

        // Gather wavefront-wide minimum for the next chunks starting column index
        // Using shfl_xor here so that each thread in the wavefront obtains the final
        // result
        for(unsigned int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            min_col = min(min_col, __shfl_xor(min_col, i));
        }

        // Each thread sets the new chunk beginning
        chunk_begin = min_col;

        // Once the chunk beginning has reached the total number of columns n,
        // we are done
        if(chunk_begin >= n)
        {
            break;
        }
    }

    // Last thread in each wavefront writes the accumulated total row nnz to global
    // memory
    if(lid == WFSIZE - 1)
    {
        row_nnz[row] = nnz;
    }
}

// Compute matrix addition, where each row is processed by a wavefront.
// Splitting row into several chunks such that we can use shared memory to store whether
// a column index is populated or not.
template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T>
__device__ void csrgeam_fill_multipass_device(rocsparse_int m,
                                              rocsparse_int n,
                                              T             alpha,
                                              const rocsparse_int* __restrict__ csr_row_ptr_A,
                                              const rocsparse_int* __restrict__ csr_col_ind_A,
                                              const T* __restrict__ csr_val_A,
                                              T beta,
                                              const rocsparse_int* __restrict__ csr_row_ptr_B,
                                              const rocsparse_int* __restrict__ csr_col_ind_B,
                                              const T* __restrict__ csr_val_B,
                                              const rocsparse_int* __restrict__ csr_row_ptr_C,
                                              rocsparse_int* __restrict__ csr_col_ind_C,
                                              T* __restrict__ csr_val_C,
                                              rocsparse_index_base idx_base_A,
                                              rocsparse_index_base idx_base_B,
                                              rocsparse_index_base idx_base_C)
{
    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each wavefront processes a row
    rocsparse_int row = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(row >= m)
    {
        return;
    }

    // Row entry marker and value accumulator
    __shared__ bool stable[BLOCKSIZE];
    __shared__ T    sdata[BLOCKSIZE];

    bool* table = &stable[wid * WFSIZE];
    T*    data  = &sdata[wid * WFSIZE];

    // Get row entry and exit point of A
    rocsparse_int row_begin_A = csr_row_ptr_A[row] - idx_base_A;
    rocsparse_int row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

    // Get row entry and exit point of B
    rocsparse_int row_begin_B = csr_row_ptr_B[row] - idx_base_B;
    rocsparse_int row_end_B   = csr_row_ptr_B[row + 1] - idx_base_B;

    // Get row entry point of C
    rocsparse_int row_begin_C = csr_row_ptr_C[row] - idx_base_C;

    // Load the first column of the current row from A and B to set the starting
    // point for the first chunk
    rocsparse_int col_A = (row_begin_A < row_end_A) ? csr_col_ind_A[row_begin_A] - idx_base_A : n;
    rocsparse_int col_B = (row_begin_B < row_end_B) ? csr_col_ind_B[row_begin_B] - idx_base_B : n;

    // Begin of the current row chunk
    rocsparse_int chunk_begin = min(col_A, col_B);

    // Initialize the index for column access into A and B
    row_begin_A += lid;
    row_begin_B += lid;

    // Loop over the chunks until the end of both rows (A and B) has been reached (which
    // is the number of total columns n)
    while(true)
    {
        // Initialize row nnz table and value accumulator
        table[lid] = false;
        data[lid]  = static_cast<T>(0);

        __threadfence_block();

        // Initialize the beginning of the next chunk
        rocsparse_int min_col = n;

        // Loop over all columns of A, starting with the first entry that did not fit
        // into the previous chunk
        for(; row_begin_A < row_end_A; row_begin_A += WFSIZE)
        {
            // Get the column of A
            rocsparse_int col = csr_col_ind_A[row_begin_A] - idx_base_A;

            // Get the column of A shifted by the chunk_begin
            rocsparse_int shf_A = col - chunk_begin;

            // Check if this column of A is within the chunk
            if(shf_A < WFSIZE)
            {
                // Mark nnz
                table[shf_A] = true;

                // Initialize with value of A
                data[shf_A] = alpha * csr_val_A[row_begin_A];
            }
            else
            {
                // Store the first column index of A that exceeds the current chunk
                min_col = min(min_col, col);
                break;
            }
        }

        // Loop over all columns of B, starting with the first entry that did not fit
        // into the previous chunk
        for(; row_begin_B < row_end_B; row_begin_B += WFSIZE)
        {
            // Get the column of B
            rocsparse_int col = csr_col_ind_B[row_begin_B] - idx_base_B;

            // Get the column of B shifted by the chunk_begin
            rocsparse_int shf_B = col - chunk_begin;

            // Check if this column of B is within the chunk
            if(shf_B < WFSIZE)
            {
                // Mark nnz
                table[shf_B] = true;

                // Add values of B
                data[shf_B] = rocsparse_fma(beta, csr_val_B[row_begin_B], data[shf_B]);
            }
            else
            {
                // Store the first column index of B that exceeds the current chunk
                min_col = min(min_col, col);
                break;
            }
        }

        __threadfence_block();

        // Each lane checks whether there is an non-zero entry to fill or not
        bool has_nnz = table[lid];

        // Obtain the bitmask that marks the position of each non-zero entry
        unsigned long long mask = __ballot(has_nnz);

        // If the lane has an nnz assign, it must be filled into C
        if(has_nnz)
        {
            rocsparse_int offset;

            // Compute the lane's fill position in C
            if(WFSIZE == 32)
            {
                offset = __popc(mask & (0xffffffff >> (WFSIZE - 1 - lid)));
            }
            else
            {
                offset = __popcll(mask & (0xffffffffffffffff >> (WFSIZE - 1 - lid)));
            }

            // Fill C
            csr_col_ind_C[row_begin_C + offset - 1] = lid + chunk_begin + idx_base_C;
            csr_val_C[row_begin_C + offset - 1]     = data[lid];
        }

        // Shift the row entry to C by the number of total nnz of the current row
        row_begin_C += __popcll(mask);

        // Gather wavefront-wide minimum for the next chunks starting column index
        // Using shfl_xor here so that each thread in the wavefront obtains the final
        // result
        for(unsigned int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            min_col = min(min_col, __shfl_xor(min_col, i));
        }

        // Each thread sets the new chunk beginning
        chunk_begin = min_col;

        // Once the chunk beginning has reached the total number of columns n,
        // we are done
        if(chunk_begin >= n)
        {
            break;
        }
    }
}

#endif // CSRGEAM_DEVICE_H
