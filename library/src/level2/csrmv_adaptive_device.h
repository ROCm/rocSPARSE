#pragma once
#ifndef CSRMV_ADAPTIVE_DEVICE_H
#define CSRMV_ADAPTIVE_DEVICE_H

#include <hip/hip_runtime.h>

static inline __device__ float atomic_add_float_extended(float* ptr, float temp, float* old_sum)
{
    return atomicAdd(ptr, temp);
}

static inline __device__ double atomic_add_float_extended(double* ptr, double temp, double* old_sum)
{
    unsigned long long newVal;
    unsigned long long prevVal;
    do
    {
        prevVal = __double_as_longlong(*ptr);
        newVal  = __double_as_longlong(temp + *ptr);
    } while(atomicCAS((unsigned long long*)ptr, prevVal, newVal) != prevVal);
    if(old_sum != 0)
        *old_sum = (double)prevVal;
    return (double)newVal;
}

template <typename T>
static inline __device__ T
sum2_reduce(T cur_sum, T* partial, int lid, int max_size, int reduc_size)
{
    if(max_size > reduc_size)
    {
        cur_sum += partial[lid + reduc_size];
        __syncthreads();
        partial[lid] = cur_sum;
    }
    return cur_sum;
}

template <typename T,
          rocsparse_int BLOCKSIZE,
          rocsparse_int BLOCK_MULTIPLIER,
          rocsparse_int ROWS_FOR_VECTOR,
          rocsparse_int WG_BITS,
          rocsparse_int ROW_BITS,
          rocsparse_int WG_SIZE>
__device__ void csrmvn_adaptive_device(unsigned long long* row_blocks,
                                       T alpha,
                                       const rocsparse_int* csr_row_ptr,
                                       const rocsparse_int* csr_col_ind,
                                       const T* csr_val,
                                       const T* x,
                                       T beta,
                                       T* y,
                                       rocsparse_index_base idx_base)
{
    __shared__ T partialSums[BLOCKSIZE];
    unsigned int gid = hipBlockIdx_x;
    unsigned int lid = hipThreadIdx_x;

    // The row blocks buffer holds a packed set of information used to inform each
    // workgroup about how to do its work:
    //
    // |6666 5555 5555 5544 4444 4444 3333 3333|3322 2222|2222 1111 1111 1100 0000 0000|
    // |3210 9876 5432 1098 7654 3210 9876 5432|1098 7654|3210 9876 5432 1098 7654 3210|
    // |------------Row Information------------|--------^|---WG ID within a long row---|
    // |                                       |    flag/|or # reduce threads for short|
    //
    // The upper 32 bits of each rowBlock entry tell the workgroup the ID of the first
    // row it will be working on. When one workgroup calculates multiple rows, this
    // rowBlock entry and the next one tell it the range of rows to work on.
    // The lower 24 bits are used whenever multiple workgroups calculate a single long
    // row. This tells each workgroup its ID within that row, so it knows which
    // part of the row to operate on.
    // Alternately, on short row blocks, the lower bits are used to communicate
    // the number of threads that should be used for the reduction. Pre-calculating
    // this on the CPU-side results in a noticable performance uplift on many matrices.
    // Bit 24 is a flag bit used so that the multiple WGs calculating a long row can
    // know when the first workgroup for that row has finished initializing the output
    // value. While this bit is the same as the first workgroup's flag bit, this
    // workgroup will spin-loop.
    unsigned int row = ((row_blocks[gid] >> (64 - ROW_BITS)) & ((1ULL << ROW_BITS) - 1ULL));
    unsigned int stop_row =
        ((row_blocks[gid + 1] >> (64 - ROW_BITS)) & ((1ULL << ROW_BITS) - 1ULL));
    unsigned int num_rows = stop_row - row;

    // Get the workgroup within this long row ID out of the bottom bits of the row block.
    unsigned int wg = row_blocks[gid] & ((1 << WG_BITS) - 1);

    // Any workgroup only calculates, at most, BLOCK_MULTIPLIER*BLOCKSIZE items in a row.
    // If there are more items in this row, we assign more workgroups.
//    unsigned int vecStart = hc::__mad24(wg, (unsigned int)(BLOCK_MULTIPLIER * BLOCKSIZE), (unsigned int)csr_row_ptr[row]);
    unsigned int vecStart = ((wg >> 8) << 8) * (((unsigned int)(BLOCK_MULTIPLIER * BLOCKSIZE) >> 8) << 8) + csr_row_ptr[row];
    unsigned int vecEnd = (csr_row_ptr[row + 1] > vecStart + BLOCK_MULTIPLIER * BLOCKSIZE)
                              ? vecStart + BLOCK_MULTIPLIER * BLOCKSIZE
                              : csr_row_ptr[row + 1];

    T temp_sum  = 0.;

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
        unsigned int numThreadsForRed = wg; // Same calculation as above, done on host.

        // Stream all of this row block's matrix values into local memory.
        // Perform the matvec in parallel with this work.
        unsigned int col = csr_row_ptr[row] + lid;
        if(gid != (gridDim.x - 1))
        {
            for(int i                = 0; i < BLOCKSIZE; i += WG_SIZE)
                partialSums[lid + i] = alpha * csr_val[col + i] * x[csr_col_ind[col + i]];
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
            for(int i                = 0; col + i < csr_row_ptr[stop_row]; i += WG_SIZE)
                partialSums[lid + i] = alpha * csr_val[col + i] * x[csr_col_ind[col + i]];
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
            // avoids an integer divide. ~2% perf gain in EXTRA_PRECISION.
            // size_t st = lid/numThreadsForRed;
            unsigned int local_row       = row + (lid >> (31 - __clz(numThreadsForRed)));
            unsigned int local_first_val = csr_row_ptr[local_row] - csr_row_ptr[row];
            unsigned int local_last_val  = csr_row_ptr[local_row + 1] - csr_row_ptr[row];
            unsigned int threadInBlock   = lid & (numThreadsForRed - 1);

            // Not all row blocks are full -- they may have an odd number of rows. As such,
            // we need to ensure that adjacent-groups only work on real data for this rowBlock.
            if(local_row < stop_row)
            {
                // This is dangerous -- will infinite loop if your last value is within
                // numThreadsForRed of MAX_UINT. Noticable performance gain to avoid a
                // long induction variable here, though.
                for(unsigned int local_cur_val = local_first_val + threadInBlock;
                    local_cur_val < local_last_val;
                    local_cur_val += numThreadsForRed)
                    temp_sum += partialSums[local_cur_val];
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
                if(beta != 0.)
                    temp_sum += beta * y[local_row];
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
            unsigned int local_row = row + lid;
            while(local_row < stop_row)
            {
                int local_first_val = (csr_row_ptr[local_row] - csr_row_ptr[row]);
                int local_last_val  = csr_row_ptr[local_row + 1] - csr_row_ptr[row];
                temp_sum            = 0.;
                for(int local_cur_val = local_first_val; local_cur_val < local_last_val;
                    local_cur_val++)
                    temp_sum += partialSums[local_cur_val];

                // After you've done the reduction into the temp_sum register,
                // put that into the output for each row.
                if(beta != 0.)
                    temp_sum += beta * y[local_row];
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
            temp_sum  = 0.;
            vecStart  = csr_row_ptr[row];
            vecEnd    = csr_row_ptr[row + 1];

            // Load in a bunch of partial results into your register space, rather than LDS (no
            // contention)
            // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
            // Using a long induction variable to make sure unsigned int overflow doesn't break
            // things.
            for(unsigned long long j = vecStart + lid; j < vecEnd; j += WG_SIZE)
            {
                unsigned int col = csr_col_ind[(unsigned int)j];
                temp_sum += alpha * csr_val[(unsigned int)j] * x[col];
            }

            partialSums[lid] = temp_sum;

            // Reduce partial sums
            for(int i = (WG_SIZE >> 1); i > 0; i >>= 1)
            {
                __syncthreads();
                temp_sum = sum2_reduce(temp_sum, partialSums, lid, WG_SIZE, i);
            }

            if(lid == 0U)
            {
                if(beta != 0.)
                    temp_sum += beta * y[row];
                y[row] = temp_sum;
            }
            row++;
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

        // First, figure out which workgroup you are in the row. Bottom 24 bits.
        // You can use that to find the global ID for the first workgroup calculating
        // this long row.
        unsigned int first_wg_in_row = gid - (row_blocks[gid] & ((1ULL << WG_BITS) - 1ULL));
        unsigned int compare_value   = row_blocks[gid] & (1ULL << WG_BITS);

        // Bit 24 in the first workgroup is the flag that everyone waits on.
        if(gid == first_wg_in_row && lid == 0ULL)
        {
            // The first workgroup handles the output initialization.
            T out_val = y[row];
            temp_sum           = (beta - 1.) * out_val;
            atomicXor(&row_blocks[first_wg_in_row], (1ULL << WG_BITS)); // Release other workgroups.
        }
        // For every other workgroup, bit 24 holds the value they wait on.
        // If your bit 24 == first_wg's bit 24, you spin loop.
        // The first workgroup will eventually flip this bit, and you can move forward.
        __syncthreads();
        while(
            gid != first_wg_in_row && lid == 0U &&
            ((atomicMax(&row_blocks[first_wg_in_row], 0ULL) & (1ULL << WG_BITS)) == compare_value))
            ;
        __syncthreads();

        // After you've passed the barrier, update your local flag to make sure that
        // the next time through, you know what to wait on.
        if(gid != first_wg_in_row && lid == 0ULL)
            row_blocks[gid] ^= (1ULL << WG_BITS);

        // All but the final workgroup in a long-row collaboration have the same start_row
        // and stop_row. They only run for one iteration.
        // Load in a bunch of partial results into your register space, rather than LDS (no
        // contention)
        // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
        unsigned int col = vecStart + lid;
        if(row == stop_row) // inner thread, we can hardcode/unroll this loop
        {
            // Don't put BLOCK_MULTIPLIER*BLOCKSIZE as the stop point, because
            // some GPU compilers will *aggressively* unroll this loop.
            // That increases register pressure and reduces occupancy.
            for(int j = 0; j < (int)(vecEnd - col); j += WG_SIZE)
            {
                temp_sum += alpha * csr_val[col + j] * x[csr_col_ind[col + j]];
#if 2 * WG_SIZE <= BLOCK_MULTIPLIER * BLOCKSIZE
                // If you can, unroll this loop once. It somewhat helps performance.
                j += WG_SIZE;
                temp_sum += alpha * csr_val[col + j] * x[csr_col_ind[col + j]];
#endif
            }
        }
        else
        {
            for(int j = 0; j < (int)(vecEnd - col); j += WG_SIZE)
                temp_sum += alpha * csr_val[col + j] * x[csr_col_ind[col + j]];
        }

        partialSums[lid] = temp_sum;

        // Reduce partial sums
        for(int i = (WG_SIZE >> 1); i > 0; i >>= 1)
        {
            __syncthreads();
            temp_sum = sum2_reduce(temp_sum, partialSums, lid, WG_SIZE, i);
        }

        if(lid == 0U)
        {
            atomic_add_float_extended(&y[row], temp_sum, 0);
        }
    }
}

#endif // CSRMV_ADAPTIVE_DEVICE_H
