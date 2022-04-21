/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef CSRMV_SYMM_DEVICE_H
#define CSRMV_SYMM_DEVICE_H

#include "common.h"

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename J, typename T>
static ROCSPARSE_DEVICE_ILF void csrmvn_symm_general_device(bool                 conj,
                                                            J                    m,
                                                            T                    alpha,
                                                            const I*             csr_row_ptr,
                                                            const J*             csr_col_ind,
                                                            const T*             csr_val,
                                                            const T*             x,
                                                            T                    beta,
                                                            T*                   y,
                                                            rocsparse_index_base idx_base)
{
    int lid = hipThreadIdx_x & (WF_SIZE - 1);

    J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    J nwf = hipGridDim_x * BLOCKSIZE / WF_SIZE;

    // Loop over rows
    for(J row = gid / WF_SIZE; row < m; row += nwf)
    {
        // Each wavefront processes one row
        I row_start = csr_row_ptr[row] - idx_base;
        I row_end   = csr_row_ptr[row + 1] - idx_base;

        T sum = static_cast<T>(0);

        // Loop over non-zero elements
        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            T val = conj_val(csr_val[j], conj);
            sum   = rocsparse_fma(alpha * val, rocsparse_ldg(x + csr_col_ind[j] - idx_base), sum);
        }

        // Obtain row sum using parallel reduction
        sum = rocsparse_wfreduce_sum<WF_SIZE>(sum);

        // First thread of each wavefront writes result into global memory
        if(lid == WF_SIZE - 1)
        {
            if(beta == static_cast<T>(0))
            {
                y[row] = sum;
            }
            else
            {
                y[row] = rocsparse_fma(beta, y[row], sum);
            }
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename J, typename T>
static ROCSPARSE_DEVICE_ILF void csrmvt_symm_general_device(bool                 conj,
                                                            J                    m,
                                                            T                    alpha,
                                                            const I*             csr_row_ptr,
                                                            const J*             csr_col_ind,
                                                            const T*             csr_val,
                                                            const T*             x,
                                                            T*                   y,
                                                            rocsparse_index_base idx_base)
{
    int lid = hipThreadIdx_x & (WF_SIZE - 1);

    J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    J inc = hipGridDim_x * BLOCKSIZE / WF_SIZE;

    for(J row = gid / WF_SIZE; row < m; row += inc)
    {
        I row_begin = csr_row_ptr[row] - idx_base;
        I row_end   = csr_row_ptr[row + 1] - idx_base;
        T row_val   = alpha * x[row];

        for(I j = row_begin + lid; j < row_end; j += WF_SIZE)
        {
            J col = csr_col_ind[j] - idx_base;

            if(col != row)
            {
                T val = conj_val(csr_val[j], conj);
                atomicAdd(&y[col], val * row_val);
            }
        }
    }
}

template <typename I>
static inline ROCSPARSE_DEVICE_ILF I
    binSearch(const I* csr_row_ptr, I row, I stop_row, I n, rocsparse_index_base base)
{
    I l = row, r = stop_row - 1, box = 0, mid;

    while(l <= r)
    {
        mid = (l + r) / 2;
        if(mid > r)
            break;

        if(mid == row && n < (csr_row_ptr[mid + 1] - base))
        {
            box = mid + 1;
            break;
        }
        else if(n >= (csr_row_ptr[mid - 1] - base) && n < (csr_row_ptr[mid] - base))
        {
            box = mid;
            break;
        }
        else if(mid == (stop_row - 1) && n >= (csr_row_ptr[mid] - base))
        {
            box = mid + 1;
            break;
        }
        else if((csr_row_ptr[mid] - base) > n)
            r = mid - 1;
        else
            l = mid + 1;
    }

    return (box == 0) ? 0 : box - 1;
}

static inline ROCSPARSE_DEVICE_ILF int lowerPowerOf2(int num)
{
    num--;

    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;

    num++;

    num >>= 1;

    return num;
}

template <rocsparse_int BLOCKSIZE,
          rocsparse_int MAX_ROWS,
          rocsparse_int WG_SIZE,
          typename I,
          typename J,
          typename T>
ROCSPARSE_DEVICE_ILF void csrmvn_symm_adaptive_device(bool                 conj,
                                                      I                    nnz,
                                                      I                    max_rows,
                                                      const I*             row_blocks,
                                                      T                    alpha,
                                                      const I*             csr_row_ptr,
                                                      const J*             csr_col_ind,
                                                      const T*             csr_val,
                                                      const T*             x,
                                                      T                    beta,
                                                      T*                   y,
                                                      rocsparse_index_base idx_base)
{
    __shared__ T partial_sums[BLOCKSIZE];
    __shared__ T cols_in_rows[MAX_ROWS];

    int gid = hipBlockIdx_x;
    int lid = hipThreadIdx_x;

    for(J i = 0; i < BLOCKSIZE; i += WG_SIZE)
    {
        partial_sums[lid + i] = static_cast<T>(0);
    }

    __syncthreads();

    // The row blocks buffer holds a packed set of information used to inform each
    // workgroup about how to do its work:
    //
    // |6666 5555 5555 5544 4444 4444 3333 3333|3322 2222|2222 1111 1111 1100 0000 0000|
    // |3210 9876 5432 1098 7654 3210 9876 5432|1098 7654|3210 9876 5432 1098 7654 3210|
    // |------------Row Information------------|----flag^|---WG ID within a long row---|
    //
    // The upper 32 bits of each rowBlock entry tell the workgroup the ID of the first
    // row it will be working on. When one workgroup calculates multiple rows, this
    // rowBlock entry and the next one tell it the range of rows to work on.
    // The lower 24 bits are used whenever multiple workgroups calculate a single long
    // row. This tells each workgroup its ID within that row, so it knows which
    // part of the row to operate on.
    // Bit 24 is a flag bit used so that the multiple WGs calculating a long row can
    // know when the first workgroup for that row has finished initializing the output
    // value. While this bit is the same as the first workgroup's flag bit, this
    // workgroup will spin-loop.
    I row      = row_blocks[gid];
    I stop_row = row_blocks[gid + 1];

    if((stop_row - row > 2))
    {
        // CSR-Stream case. See Sections III.A and III.B in the SC'14 paper:
        // "Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format"
        // for a detailed description of CSR-Stream.
        // In a nutshell, the idea is to use all of the threads to stream the matrix
        // values into the local memory in a fast, coalesced manner. After that, the
        // per-row reductions are done out of the local memory, which is designed
        // to handle non-coalsced accesses.

        // The best method for reducing the local memory values depends on the number
        // of rows. The SC'14 paper discusses a "CSR-Scalar" style reduction where
        // each thread reduces its own row. This yields good performance if there
        // are many (relatively short) rows. However, if they are few (relatively
        // long) rows, it's actually better to perform a tree-style reduction where
        // multiple threads team up to reduce the same row.

        // The calculations below tell you how many threads this workgroup can allocate
        // to each row, assuming that every row gets the same number of threads.
        // We want the closest lower-power-of-2 to this number -- that is how many
        // threads can work in each row's reduction using our algorithm.
        int possibleThreadsRed = hipBlockDim_x / (stop_row - row);
        int numThreadsForRed   = lowerPowerOf2(possibleThreadsRed);

        I local_row = row + lid;

        // Stream all of this row block's matrix values into local memory.
        // Perform the matvec in parallel with this work.
        I col = csr_row_ptr[row] + lid - idx_base;

        // Stream all of this row blocks' matrix values into local memory
        // Only do the unrolled loop if it won't overflow the buffer
        if(col + BLOCKSIZE - WG_SIZE < nnz)
        {
            // This allows loop unrolling, since BLOCKSIZE is a compile-time constant
            for(J i = 0; i < BLOCKSIZE; i += WG_SIZE)
            {
                partial_sums[lid + i] = alpha * conj_val(csr_val[col + i], conj);
            }
        }
        else
        {
            // This is required so that we stay in bounds for csr_val[].
            // Otherwise, if the matrix's endpoints don't line up with BLOCKSIZE,
            // we will buffer overflow. On today's dGPUs, this doesn't cause problems.
            // The values are within a dGPU's page, which is zeroed out on allocation.
            // However, this may change in the future (e.g. with shared virtual memory.)
            // This causes a minor performance loss because this is the last workgroup
            // to be launched, and this loop can't be unrolled.
            I max_to_load = csr_row_ptr[stop_row] - csr_row_ptr[row];
            for(I i = 0; (lid + i) < max_to_load; i += WG_SIZE)
            {
                partial_sums[lid + i] = alpha * conj_val(csr_val[col + i], conj);
            }
        }

        // Upper triangular
        // Initialize the cols_in_rows
        for(I l = lid; l < max_rows; l += WG_SIZE)
        {
            cols_in_rows[l] = static_cast<T>(0);
        }

        __syncthreads();

        // max_rows is the maximum number of rows any block will handle
        // It is used to size the local memory for cols_in_rows, so once your
        // stop row for this block is more than this value, we need to make
        // sure we can offset it. Otherwise, we would blow past the end of
        // the local memory.
        I stop_cols_idx = (stop_row < max_rows) ? 0 : (stop_row - max_rows);
        if(col + BLOCKSIZE - WG_SIZE < nnz)
        {
            for(J i = 0; i < BLOCKSIZE; i += WG_SIZE)
            {
                // Need to prep some data for the upper triangular calculation
                I myRow = binSearch(csr_row_ptr, row, stop_row, (col + i), idx_base);
                J myCol = csr_col_ind[col + i] - idx_base;

                // Coming in, partial_sums contains the matrix data, so this allows
                // us to reach into the output and calculate this piece of the upper
                // triangular.
                if((myCol != myRow) && (col + i) < (csr_row_ptr[stop_row] - idx_base))
                {
                    if(myCol >= (stop_cols_idx) && myCol < stop_row)
                        atomicAdd(&cols_in_rows[myCol - (stop_cols_idx)],
                                  (partial_sums[lid + i] * x[myRow]));
                    else
                        atomicAdd(&y[myCol], (partial_sums[lid + i] * x[myRow]));
                }

                // For the lower triangular, the matrix value is already in partial_sums.
                // Calculate the mat*x for thie lower triangular and place it into
                // partial_sums.
                partial_sums[lid + i] *= x[myCol];
            }
        }
        else
        {
            I max_to_load = csr_row_ptr[stop_row] - csr_row_ptr[row];
            for(I i = 0; (lid + i) < max_to_load; i += WG_SIZE)
            {
                // Need to prep some data for the upper triangular calculation
                I myRow = binSearch(csr_row_ptr, row, stop_row, (col + i), idx_base);
                J myCol = csr_col_ind[col + i] - idx_base;

                // Coming in, partial_sums contains the matrix data, so this allows
                // us to reach into the output and calculate this piece of the upper
                // triangular.
                if((myCol != myRow) && (col + i) < (csr_row_ptr[stop_row] - idx_base))
                {
                    if(myCol >= (stop_cols_idx) && myCol < stop_row)
                        atomicAdd(&cols_in_rows[myCol - (stop_cols_idx)],
                                  (partial_sums[lid + i] * x[myRow]));
                    else
                        atomicAdd(&y[myCol], (partial_sums[lid + i] * x[myRow]));
                }

                // For the lower triangular, the matrix value is already in partial_sums.
                // Calculate the mat*x for thie lower triangular and place it into
                // partial_sums.
                partial_sums[lid + i] *= x[myCol];
            }
        }

        __syncthreads();

        I end_cols_idx = (stop_row < max_rows) ? stop_row : max_rows;

        for(I l = lid; l < (end_cols_idx - (stop_row - row)); l += WG_SIZE)
        {
            atomicAdd(&y[stop_cols_idx + l], cols_in_rows[l]);
        }

        __syncthreads();

        // Lower Triangular
        if(numThreadsForRed > 1)
        {
            // In this case, we want to have the workgroup perform a tree-style reduction
            // of each row. {numThreadsForRed} adjacent threads team up to linearly reduce
            // a row into {numThreadsForRed} locations in local memory.
            // After that, a single thread from each of those teams linearly walks through
            // the local memory values for that row and reduces to the final output value.
            T temp = static_cast<T>(0);

            // {numThreadsForRed} adjacent threads all work on the same row, so their
            // start and end values are the same.
            int st                = lid / numThreadsForRed;
            I   local_first_val   = (csr_row_ptr[row + st] - csr_row_ptr[row]);
            I   local_last_val    = csr_row_ptr[row + st + 1] - csr_row_ptr[row];
            I   workForEachThread = (local_last_val - local_first_val) / numThreadsForRed;
            int threadInBlock     = lid & (numThreadsForRed - 1);

            // Not all row blocks are full -- they may have an odd number of rows. As such,
            // we need to ensure that adjacent-groups only work on real data for this rowBlock.
            if(st < (stop_row - row))
            {
                // only works when numThreadsForRed is a power of 2
                for(I i = 0; i < workForEachThread; i++)
                {
                    temp += partial_sums[local_first_val + i * numThreadsForRed + threadInBlock];
                }

                I local_cur_val = local_first_val + numThreadsForRed * workForEachThread;
                if(threadInBlock < local_last_val - local_cur_val)
                {
                    temp += partial_sums[local_cur_val + threadInBlock];
                }
            }
            __syncthreads();
            partial_sums[lid] = temp;
            __syncthreads();

            // Step one of this two-stage reduction is done. Now each row has {numThreadsForRed}
            // values sitting in the local memory. This next step takes the first thread from
            // each of the adjacent-groups and uses it to walk through those values and reduce
            // them into a final output value for the row.
            temp = static_cast<T>(0);
            if(lid < (stop_row - row))
            {
                for(int i = 0; i < numThreadsForRed; i++)
                {
                    temp += partial_sums[lid * numThreadsForRed + i];
                }
                temp += cols_in_rows[lid
                                     + (end_cols_idx
                                        - (stop_row - row))]; // sum from upper triangular matrix
                atomicAdd(&y[row + lid], temp);
            }
        }
        else
        {
            // In this case, we want to have each thread perform the reduction for a single row.
            // Essentially, this looks like performing CSR-Scalar, except it is computed out of local memory.
            // However, this reduction is also much faster than CSR-Scalar, because local memory
            // is designed for scatter-gather operations.
            // We need a while loop because there may be more rows than threads in the WG.
            while(local_row < stop_row)
            {
                I local_first_val = (csr_row_ptr[local_row] - csr_row_ptr[row]);
                I local_last_val  = csr_row_ptr[local_row + 1] - csr_row_ptr[row];
                T temp            = static_cast<T>(0);
                for(I local_cur_val = local_first_val; local_cur_val < local_last_val;
                    local_cur_val++)
                {
                    temp += partial_sums[local_cur_val];
                }

                // After you've done the reduction into the temp register,
                // put that into the output for each row.
                temp += cols_in_rows[end_cols_idx - stop_row
                                     + local_row]; // sum from upper triangular matrix
                atomicAdd(&y[local_row], temp);
                local_row += hipBlockDim_x;
            }
        }
    }
    else
    {
        // Thread ID in block
        int t     = hipThreadIdx_x;
        I   myRow = row;

        // Lower triangular
        while(myRow < stop_row)
        {
            I vecStart = csr_row_ptr[myRow] - idx_base;
            I vecEnd   = csr_row_ptr[myRow + 1] - idx_base;
            T mySum    = static_cast<T>(0);
            for(I j = vecStart + t; j < vecEnd; j += WG_SIZE)
            {
                J col = csr_col_ind[j] - idx_base;
                mySum += conj_val(csr_val[j], conj) * x[col];
            }

            partial_sums[t] = mySum;
            __syncthreads();

            // Assumes BLOCKSIZE = 4 * WG_SIZE and both BLOCKSIZE, WG_SIZE are powers of 2
            if(BLOCKSIZE > 256)
            {
                if(t < 256)
                {
                    partial_sums[t]
                        += partial_sums[t + 256] + partial_sums[t + 512] + partial_sums[t + 768];
                }
                __syncthreads();
            }
            if(BLOCKSIZE > 64)
            {
                if(t < 64)
                {
                    partial_sums[t]
                        += partial_sums[t + 64] + partial_sums[t + 128] + partial_sums[t + 192];
                }
                __syncthreads();
            }
            if(BLOCKSIZE > 16)
            {
                if(t < 16)
                {
                    partial_sums[t]
                        += partial_sums[t + 16] + partial_sums[t + 32] + partial_sums[t + 48];
                }
                __syncthreads();
            }
            if(BLOCKSIZE > 4)
            {
                if(t < 4)
                {
                    partial_sums[t]
                        += partial_sums[t + 4] + partial_sums[t + 8] + partial_sums[t + 12];
                }
                __syncthreads();
            }
            if(BLOCKSIZE > 1)
            {
                if(t < 1)
                {
                    partial_sums[t]
                        += partial_sums[t + 1] + partial_sums[t + 2] + partial_sums[t + 3];
                }
                __syncthreads();
            }

            // Write result
            if(t == 0)
            {
                atomicAdd(&y[myRow], alpha * partial_sums[0]);
            }
            myRow++;
        }

        // Upper Triangular
        I vecStart = csr_row_ptr[row] - idx_base;
        I VecEnd   = csr_row_ptr[stop_row] - idx_base;
        for(I j = vecStart + t; j < VecEnd; j += WG_SIZE)
        {
            I myRow2 = binSearch(csr_row_ptr, row, stop_row, j, idx_base);
            J myCol  = csr_col_ind[j] - idx_base;
            if(myCol != myRow2)
            {
                atomicAdd(&y[myCol], (alpha * conj_val(csr_val[j], conj) * x[myRow2]));
            }
        }
    }
}

#endif // CSRMV_SYMM_DEVICE_H
