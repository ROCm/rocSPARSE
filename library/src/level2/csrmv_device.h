/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
static ROCSPARSE_DEVICE_ILF void csrmvn_general_device(bool                 conj,
                                                       J                    m,
                                                       T                    alpha,
                                                       const I*             row_offset_begin,
                                                       const I*             row_offset_end,
                                                       const J*             csr_col_ind,
                                                       const A*             csr_val,
                                                       const X*             x,
                                                       T                    beta,
                                                       Y*                   y,
                                                       rocsparse_index_base idx_base)
{
    int lid = hipThreadIdx_x & (WF_SIZE - 1);

    J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    J nwf = hipGridDim_x * (BLOCKSIZE / WF_SIZE);

    // Loop over rows
    for(J row = gid / WF_SIZE; row < m; row += nwf)
    {
        // Each wavefront processes one row
        I row_start = row_offset_begin[row] - idx_base;
        I row_end   = row_offset_end[row] - idx_base;

        T sum = static_cast<T>(0);

        // Loop over non-zero elements
        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            sum = rocsparse_fma<T>(alpha * conj_val(csr_val[j], conj),
                                   rocsparse_ldg(x + csr_col_ind[j] - idx_base),
                                   sum);
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
                y[row] = rocsparse_fma<T>(beta, y[row], sum);
            }
        }
    }
}

template <typename J, typename Y, typename T>
static ROCSPARSE_DEVICE_ILF void csrmvt_scale_device(J size, T scalar, Y* data)
{
    J idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= size)
    {
        return;
    }

    data[idx] *= scalar;
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
static ROCSPARSE_DEVICE_ILF void csrmvt_general_device(bool                 conj,
                                                       J                    m,
                                                       T                    alpha,
                                                       const I*             csr_row_ptr_begin,
                                                       const I*             csr_row_ptr_end,
                                                       const J*             csr_col_ind,
                                                       const A*             csr_val,
                                                       const X*             x,
                                                       Y*                   y,
                                                       rocsparse_index_base idx_base)
{
    int lid = hipThreadIdx_x & (WF_SIZE - 1);

    J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    J inc = hipGridDim_x * (BLOCKSIZE / WF_SIZE);

    for(J row = gid / WF_SIZE; row < m; row += inc)
    {
        I row_begin = csr_row_ptr_begin[row] - idx_base;
        I row_end   = csr_row_ptr_end[row] - idx_base;
        T row_val   = alpha * x[row];

        for(I j = row_begin + lid; j < row_end; j += WF_SIZE)
        {
            J col = csr_col_ind[j] - idx_base;
            A val = conj_val(csr_val[j], conj);

            rocsparse_atomic_add(&y[col], row_val * val);
        }
    }
}

template <typename I, typename T>
static ROCSPARSE_DEVICE_ILF T
    sum2_reduce(T cur_sum, T* partial, int lid, I max_size, int reduc_size)
{
    if(max_size > reduc_size)
    {
        cur_sum = cur_sum + partial[lid + reduc_size];
        __syncthreads();
        partial[lid] = cur_sum;
    }
    return cur_sum;
}

template <rocsparse_int BLOCKSIZE,
          rocsparse_int BLOCK_MULTIPLIER,
          rocsparse_int ROWS_FOR_VECTOR,
          rocsparse_int WG_SIZE,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
static ROCSPARSE_DEVICE_ILF void csrmvn_adaptive_device(bool                 conj,
                                                        I                    nnz,
                                                        const I*             row_blocks,
                                                        unsigned int*        wg_flags,
                                                        const J*             wg_ids,
                                                        T                    alpha,
                                                        const I*             csr_row_ptr,
                                                        const J*             csr_col_ind,
                                                        const A*             csr_val,
                                                        const X*             x,
                                                        T                    beta,
                                                        Y*                   y,
                                                        rocsparse_index_base idx_base)
{
    __shared__ T partialSums[BLOCKSIZE];

    int lid = hipThreadIdx_x;
    int gid = hipBlockIdx_x;

    // The row blocks buffer holds information used to inform each
    // workgroup about how to do its work:
    //
    // The rowBlock entry tell the workgroup the ID of the first
    // row it will be working on. When one workgroup calculates multiple rows, this
    // rowBlock entry and the next one tell it the range of rows to work on.
    // The wg_ids are used whenever multiple workgroups calculate a single long
    // row. This tells each workgroup its ID within that row, so it knows which
    // part of the row to operate on.
    // Alternately, on short row blocks, wg_ids are used to communicate
    // the number of threads that should be used for the reduction. Pre-calculating
    // this on the CPU-side results in a noticeable performance uplift on many matrices.
    // wg_flags contains the flag bit used so that the multiple WGs calculating a long row can
    // know when the first workgroup for that row has finished initializing the output
    // value. While this bit is the same as the first workgroup's flag bit, this
    // workgroup will spin-loop.
    I row      = row_blocks[gid];
    I stop_row = row_blocks[gid + 1];
    J num_rows = stop_row - row;

    // Get the workgroup within this long row ID
    J wg = wg_ids[gid];

    // Any workgroup only calculates, at most, BLOCK_MULTIPLIER*BLOCKSIZE items in a row.
    // If there are more items in this row, we assign more workgroups.
    I vecStart = (I)wg * (I)BLOCK_MULTIPLIER * BLOCKSIZE + csr_row_ptr[row] - idx_base;
    I vecEnd   = min(csr_row_ptr[row + 1] - idx_base, vecStart + BLOCK_MULTIPLIER * BLOCKSIZE);

    T temp_sum = static_cast<T>(0);

    // If the next row block starts more than 2 rows away, then we choose CSR-Stream.
    // If this is zero (long rows) or one (final workgroup in a long row, or a single
    // row in a row block), we want to use the CSR-Vector algorithm(s).
    // We have found, through experimentation, that CSR-Vector is generally faster
    // when working on 2 rows, due to its simplicity and better reduction method.
    if(num_rows > ROWS_FOR_VECTOR)
    {
        // CSR-Stream case. See Sections III.A and III.B in the SC'14 paper:
        // Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format
        // for a detailed description of CSR-Stream.
        // In a nutshell, the idea is to use all of the threads to stream the matrix
        // values into the local memory in a fast, coalesced manner. After that, the
        // per-row reductions are done out of the local memory, which is designed
        // to handle non-coalsced accesses.

        // The best method for reducing the local memory values depends on the number
        // of rows. The SC'14 paper discusses a CSR-Scalar style reduction where
        // each thread reduces its own row. This yields good performance if there
        // are many (relatively short) rows. However, if they are few (relatively
        // long) rows, it's actually better to perform a tree-style reduction where
        // multiple threads team up to reduce the same row.

        // The calculation below tells you how many threads this workgroup can allocate
        // to each row, assuming that every row gets the same number of threads.
        // We want the closest lower (or equal) power-of-2 to this number --
        // that is how many threads can work in each row's reduction using our algorithm.
        // For instance, with workgroup size 256, 2 rows = 128 threads, 3 rows = 64
        // threads, 4 rows = 64 threads, 5 rows = 32 threads, etc.
        // int numThreadsForRed = get_local_size(0) >> ((CHAR_BIT*sizeof(unsigned
        // int))-clz(num_rows-1));
        J numThreadsForRed = wg; // Same calculation as above, done on host.

        // Stream all of this row block's matrix values into local memory.
        // Perform the matvec in parallel with this work.
        I col = csr_row_ptr[row] + lid - idx_base;
        if(col + BLOCKSIZE - WG_SIZE < nnz)
        {
            for(J i = 0; i < BLOCKSIZE; i += WG_SIZE)
            {
                partialSums[lid + i]
                    = alpha * conj_val(csr_val[col + i], conj) * x[csr_col_ind[col + i] - idx_base];
            }
        }
        else
        {
            // This is required so that we stay in bounds for csr_val[] and csr_col_ind[].
            // Otherwise, if the matrix's endpoints don't line up with BLOCKSIZE,
            // we will buffer overflow. On today's dGPUs, this doesn't cause problems.
            // The values are within a dGPU's page, which is zeroed out on allocation.
            // However, this may change in the future (e.g. with shared virtual memory.)
            // This causes a minor performance loss because this is the last workgroup
            // to be launched, and this loop can't be unrolled.
            for(I i = 0; col + i < csr_row_ptr[stop_row] - idx_base; i += WG_SIZE)
            {
                partialSums[lid + i]
                    = alpha * conj_val(csr_val[col + i], conj) * x[csr_col_ind[col + i] - idx_base];
            }
        }
        __syncthreads();

        if(numThreadsForRed > 1)
        {
            // In this case, we want to have the workgroup perform a tree-style reduction
            // of each row. {numThreadsForRed} adjacent threads team up to linearly reduce
            // a row into {numThreadsForRed} locations in local memory.
            // After that, the entire workgroup does a parallel reduction, and each
            // row ends up with an individual answer.

            // {numThreadsForRed} adjacent threads all work on the same row, so their
            // start and end values are the same.
            // numThreadsForRed guaranteed to be a power of two, so the clz code below
            // avoids an integer divide.
            // size_t st = lid/numThreadsForRed;
            I local_row       = row + (lid >> (31 - __clz(numThreadsForRed)));
            J local_first_val = csr_row_ptr[local_row] - csr_row_ptr[row];
            J local_last_val  = csr_row_ptr[local_row + 1] - csr_row_ptr[row];
            J threadInBlock   = lid & (numThreadsForRed - 1);

            // Not all row blocks are full -- they may have an odd number of rows. As such,
            // we need to ensure that adjacent-groups only work on real data for this rowBlock.
            if(local_row < stop_row)
            {
                // This is dangerous -- will infinite loop if your last value is within
                // numThreadsForRed of MAX_UINT. Noticable performance gain to avoid a
                // long induction variable here, though.
                for(J local_cur_val = local_first_val + threadInBlock;
                    local_cur_val < local_last_val;
                    local_cur_val += numThreadsForRed)
                {
                    temp_sum = temp_sum + partialSums[local_cur_val];
                }
            }
            __syncthreads();

            partialSums[lid] = temp_sum;

            // Step one of this two-stage reduction is done. Now each row has {numThreadsForRed}
            // values sitting in the local memory. This means that, roughly, the beginning of
            // LDS is full up to {workgroup size} entries.
            // Now we perform a parallel reduction that sums together the answers for each
            // row in parallel, leaving us an answer in 'temp_sum' for each row.
            for(int i = (WG_SIZE >> 1); i > 0; i >>= 1)
            {
                __syncthreads();
                temp_sum = sum2_reduce(temp_sum, partialSums, lid, numThreadsForRed, i);
            }

            if(threadInBlock == 0 && local_row < stop_row)
            {
                // All of our write-outs check to see if the output vector should first be zeroed.
                // If so, just do a write rather than a read-write. Measured to be a slight (~5%)
                // performance improvement.
                if(beta != static_cast<T>(0))
                {
                    temp_sum = rocsparse_fma<T>(beta, y[local_row], temp_sum);
                }
                y[local_row] = temp_sum;
            }
        }
        else
        {
            // In this case, we want to have each thread perform the reduction for a single row.
            // Essentially, this looks like performing CSR-Scalar, except it is computed out of
            // local memory.
            // However, this reduction is also much faster than CSR-Scalar, because local memory
            // is designed for scatter-gather operations.
            // We need a while loop because there may be more rows than threads in the WG.
            I local_row = row + lid;
            while(local_row < stop_row)
            {
                J local_first_val = (csr_row_ptr[local_row] - csr_row_ptr[row]);
                J local_last_val  = csr_row_ptr[local_row + 1] - csr_row_ptr[row];
                temp_sum          = static_cast<T>(0);
                for(J local_cur_val = local_first_val; local_cur_val < local_last_val;
                    ++local_cur_val)
                {
                    temp_sum = temp_sum + partialSums[local_cur_val];
                }

                // After you've done the reduction into the temp_sum register,
                // put that into the output for each row.
                if(beta != static_cast<T>(0))
                {
                    temp_sum = rocsparse_fma<T>(beta, y[local_row], temp_sum);
                }

                y[local_row] = temp_sum;
                local_row += WG_SIZE;
            }
        }
    }
    else if(num_rows >= 1 && !wg) // CSR-Vector case.
    {
        // ^^ The above check says that if this workgroup is supposed to work on <= ROWS_VECTOR
        // number of rows then we should do the CSR-Vector algorithm. If we want this row to be
        // done with CSR-LongRows, then all of its workgroups (except the last one) will have the
        // same stop_row and row. The final workgroup in a LongRow will have stop_row and row
        // different, but the internal wg number will be non-zero.

        // If this workgroup is operating on multiple rows (because CSR-Stream is poor for small
        // numbers of rows), then it needs to iterate until it reaches the stop_row.
        // We don't check <= stop_row because of the potential for unsigned overflow.
        while(row < stop_row)
        {
            // Any workgroup only calculates, at most, BLOCKSIZE items in this row.
            // If there are more items in this row, we use CSR-LongRows.
            temp_sum = static_cast<T>(0);
            vecStart = csr_row_ptr[row] - idx_base;
            vecEnd   = csr_row_ptr[row + 1] - idx_base;

            // Load in a bunch of partial results into your register space, rather than LDS (no
            // contention)
            // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
            // Using a long induction variable to make sure unsigned int overflow doesn't break
            // things.
            for(I j = vecStart + lid; j < vecEnd; j += WG_SIZE)
            {
                temp_sum = rocsparse_fma<T>(
                    alpha * conj_val(csr_val[j], conj), x[csr_col_ind[j] - idx_base], temp_sum);
            }

            partialSums[lid] = temp_sum;

            __syncthreads();

            // Reduce partial sums
            rocsparse_blockreduce_sum<WG_SIZE>(lid, partialSums);

            if(lid == 0)
            {
                temp_sum = partialSums[0];

                if(beta != static_cast<T>(0))
                {
                    temp_sum = rocsparse_fma<T>(beta, y[row], temp_sum);
                }

                y[row] = temp_sum;
            }
            ++row;
        }
    }
    else
    {
        // In CSR-LongRows, we have more than one workgroup calculating this row.
        // The output values for those types of rows are stored using atomic_add, because
        // more than one parallel workgroup's value makes up the final answer.
        // Unfortunately, this makes it difficult to do y=Ax, rather than y=Ax+y, because
        // the values still left in y will be added in using the atomic_add.
        //
        // Our solution is to have the first workgroup in one of these long-rows cases
        // properly initaizlie the output vector. All the other workgroups working on this
        // row will spin-loop until that workgroup finishes its work.

        // First, figure out which workgroup you are in the row.
        // You can use that to find the global ID for the first workgroup calculating
        // this long row.
        J             first_wg_in_row = gid - wg_ids[gid];
        unsigned long compare_value   = wg_flags[gid];

        // wg_flags[first_wg_in_row] in the first workgroup is the flag that everyone waits on.
        if(gid == first_wg_in_row && lid == 0)
        {
            // The first workgroup handles the output initialization.
            T out_val = y[row];
            temp_sum  = (beta - static_cast<T>(1)) * out_val;
            atomicXor(&wg_flags[first_wg_in_row], 1U); // Release other workgroups.
        }
        // For every other workgroup, wg_flags[first_wg_in_row] holds the value they wait on.
        // If your flag == first_wg's flag, you spin loop.
        // The first workgroup will eventually flip this flag, and you can move forward.
        __syncthreads();
        while(gid != first_wg_in_row && lid == 0
              && ((rocsparse_atomic_max(&wg_flags[first_wg_in_row], 0U)) == compare_value))
            ;
        __syncthreads();

        // After you've passed the barrier, update your local flag to make sure that
        // the next time through, you know what to wait on.
        if(gid != first_wg_in_row && lid == 0)
            wg_flags[gid] ^= 1U;

        // All but the final workgroup in a long-row collaboration have the same start_row
        // and stop_row. They only run for one iteration.
        // Load in a bunch of partial results into your register space, rather than LDS (no
        // contention)
        // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
        for(I j = vecStart + lid; j < vecEnd; j += WG_SIZE)
        {
            temp_sum = rocsparse_fma<T>(
                alpha * conj_val(csr_val[j], conj), x[csr_col_ind[j] - idx_base], temp_sum);
        }

        partialSums[lid] = temp_sum;

        __syncthreads();

        // Reduce partial sums
        rocsparse_blockreduce_sum<WG_SIZE>(lid, partialSums);

        if(lid == 0)
        {
            rocsparse_atomic_add(y + row, partialSums[0]);
        }
    }
}
