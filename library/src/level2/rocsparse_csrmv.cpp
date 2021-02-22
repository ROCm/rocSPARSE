/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_csrmv.hpp"
#include "definitions.h"
#include "utility.h"

#include "csrmv_device.h"

#define BLOCK_SIZE 1024
#define BLOCK_MULTIPLIER 3
#define ROWS_FOR_VECTOR 1
#define WG_BITS 24
#define ROW_BITS 32
#define WG_SIZE 256

__attribute__((unused)) static unsigned int flp2(unsigned int x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x - (x >> 1);
}

// Short rows in CSR-Adaptive are batched together into a single row block.
// If there are a relatively small number of these, then we choose to do
// a horizontal reduction (groups of threads all reduce the same row).
// If there are many threads (e.g. more threads than the maximum size
// of our workgroup) then we choose to have each thread serially reduce
// the row.
// This function calculates the number of threads that could team up
// to reduce these groups of rows. For instance, if you have a
// workgroup size of 256 and 4 rows, you could have 64 threads
// working on each row. If you have 5 rows, only 32 threads could
// reliably work on each row because our reduction assumes power-of-2.
static unsigned long long numThreadsForReduction(unsigned long long num_rows)
{
#if defined(__INTEL_COMPILER)
    return WG_SIZE >> (_bit_scan_reverse(num_rows - 1) + 1);
#elif(defined(__clang__) && __has_builtin(__builtin_clz)) \
    || !defined(__clang) && defined(__GNUG__)             \
           && ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 30202)
    return (WG_SIZE >> (8 * sizeof(int) - __builtin_clz(num_rows - 1)));
#elif defined(_MSC_VER) && (_MSC_VER >= 1400)
    unsigned long long bit_returned;
    _BitScanReverse(&bit_returned, (num_rows - 1));
    return WG_SIZE >> (bit_returned + 1);
#else
    return flp2(WG_SIZE / num_rows);
#endif
}

template <typename I>
static inline void ComputeRowBlocks(unsigned long long* rowBlocks,
                                    size_t&             rowBlockSize,
                                    const I*            rowDelimiters,
                                    I                   nRows,
                                    bool                allocate_row_blocks = true)
{
    unsigned long long* rowBlocksBase;

    // Start at one because of rowBlock[0]
    I total_row_blocks = 1;

    if(allocate_row_blocks)
    {
        rowBlocksBase = rowBlocks;
        *rowBlocks    = 0;
        ++rowBlocks;
    }

    unsigned long long sum = 0;
    unsigned long long i;
    unsigned long long last_i = 0;

    // Check to ensure nRows can fit in 32 bits
    // NOTE: There is a flaw here.
    // LCOV_EXCL_START
    if(static_cast<unsigned long long>(nRows)
       > static_cast<unsigned long long>(std::pow(2, ROW_BITS)))
    {
        fprintf(stderr, "nrow does not fit in 32 bits\n");
        exit(1);
    }
    // LCOV_EXCL_STOP

    I consecutive_long_rows = 0;
    for(i = 1; i <= static_cast<unsigned long long>(nRows); ++i)
    {
        I row_length = (rowDelimiters[i] - rowDelimiters[i - 1]);
        sum += row_length;

        // The following section of code calculates whether you're moving between
        // a series of "short" rows and a series of "long" rows.
        // This is because the reduction in CSR-Adaptive likes things to be
        // roughly the same length. Long rows can be reduced horizontally.
        // Short rows can be reduced one-thread-per-row. Try not to mix them.
        if(row_length > 128)
        {
            ++consecutive_long_rows;
        }
        else if(consecutive_long_rows > 0)
        {
            // If it turns out we WERE in a long-row region, cut if off now.
            if(row_length < 32) // Now we're in a short-row region
            {
                consecutive_long_rows = -1;
            }
            else
            {
                consecutive_long_rows++;
            }
        }

        // If you just entered into a "long" row from a series of short rows,
        // then we need to make sure we cut off those short rows. Put them in
        // their own workgroup.
        if(consecutive_long_rows == 1)
        {
            // Assuming there *was* a previous workgroup. If not, nothing to do here.
            if(i - last_i > 1)
            {
                if(allocate_row_blocks)
                {
                    *rowBlocks = ((i - 1) << (64 - ROW_BITS));

                    // If this row fits into CSR-Stream, calculate how many rows
                    // can be used to do a parallel reduction.
                    // Fill in the low-order bits with the numThreadsForRed
                    if(((i - 1) - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
                    {
                        *(rowBlocks - 1) |= numThreadsForReduction((i - 1) - last_i);
                    }

                    ++rowBlocks;
                }

                ++total_row_blocks;
                last_i = i - 1;
                sum    = row_length;
            }
        }
        else if(consecutive_long_rows == -1)
        {
            // We see the first short row after some long ones that
            // didn't previously fill up a row block.
            if(allocate_row_blocks)
            {
                *rowBlocks = ((i - 1) << (64 - ROW_BITS));
                if(((i - 1) - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
                {
                    *(rowBlocks - 1) |= numThreadsForReduction((i - 1) - last_i);
                }

                ++rowBlocks;
            }

            ++total_row_blocks;
            last_i                = i - 1;
            sum                   = row_length;
            consecutive_long_rows = 0;
        }

        // Now, what's up with this row? What did it do?

        // exactly one row results in non-zero elements to be greater than blockSize
        // This is csr-vector case; bottom WGBITS == workgroup ID
        if((i - last_i == 1) && sum > static_cast<unsigned long long>(BLOCK_SIZE))
        {
            I numWGReq = static_cast<I>(
                std::ceil(static_cast<double>(row_length) / (BLOCK_MULTIPLIER * BLOCK_SIZE)));

            // Check to ensure #workgroups can fit in WGBITS bits, if not
            // then the last workgroup will do all the remaining work
            numWGReq = (numWGReq < static_cast<I>(std::pow(2, WG_BITS)))
                           ? numWGReq
                           : static_cast<I>(std::pow(2, WG_BITS));

            if(allocate_row_blocks)
            {
                for(I w = 1; w < numWGReq; ++w)
                {
                    *rowBlocks = ((i - 1) << (64 - ROW_BITS));
                    *rowBlocks |= static_cast<unsigned long long>(w);
                    ++rowBlocks;
                }

                *rowBlocks = (i << (64 - ROW_BITS));
                ++rowBlocks;
            }

            total_row_blocks += numWGReq;
            last_i                = i;
            sum                   = 0;
            consecutive_long_rows = 0;
        }
        // more than one row results in non-zero elements to be greater than blockSize
        // This is csr-stream case; bottom WGBITS = number of parallel reduction threads
        else if((i - last_i > 1) && sum > static_cast<unsigned long long>(BLOCK_SIZE))
        {
            // This row won't fit, so back off one.
            --i;

            if(allocate_row_blocks)
            {
                *rowBlocks = (i << (64 - ROW_BITS));
                if((i - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
                {
                    *(rowBlocks - 1) |= numThreadsForReduction(i - last_i);
                }

                ++rowBlocks;
            }

            ++total_row_blocks;
            last_i                = i;
            sum                   = 0;
            consecutive_long_rows = 0;
        }
        // This is csr-stream case; bottom WGBITS = number of parallel reduction threads
        else if(sum == static_cast<unsigned long long>(BLOCK_SIZE))
        {
            if(allocate_row_blocks)
            {
                *rowBlocks = (i << (64 - ROW_BITS));
                if((i - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
                {
                    *(rowBlocks - 1) |= numThreadsForReduction(i - last_i);
                }

                ++rowBlocks;
            }

            ++total_row_blocks;
            last_i                = i;
            sum                   = 0;
            consecutive_long_rows = 0;
        }
    }

    // If we didn't fill a row block with the last row, make sure we don't lose it.
    if(allocate_row_blocks
       && (*(rowBlocks - 1) >> (64 - ROW_BITS)) != static_cast<unsigned long long>(nRows))
    {
        *rowBlocks = (static_cast<unsigned long long>(nRows) << (64 - ROW_BITS));
        if((nRows - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
        {
            *(rowBlocks - 1) |= numThreadsForReduction(i - last_i);
        }

        ++rowBlocks;
    }

    ++total_row_blocks;

    if(allocate_row_blocks)
    {
        size_t dist = std::distance(rowBlocksBase, rowBlocks);
        assert((2 * dist) <= rowBlockSize);
        // Update the size of rowBlocks to reflect the actual amount of memory used
        // We're multiplying the size by two because the extended precision form of
        // CSR-Adaptive requires more space for the final global reduction.
        rowBlockSize = 2 * dist;
    }
    else
    {
        rowBlockSize = 2 * total_row_blocks;
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrmv_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   J                         m,
                                                   J                         n,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   rocsparse_mat_info        info)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csrmv_analysis",
              trans,
              m,
              n,
              nnz,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info);

    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    // Check index base
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || csr_col_ind == nullptr || csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Clear csrmv info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrmv_info(info->csrmv_info));

    // Create csrmv info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrmv_info(&info->csrmv_info));

    // Stream
    hipStream_t stream = handle->stream;

    // row blocks size
    info->csrmv_info->size = 0;

    // Temporary arrays to hold device data
    std::vector<I> hptr(m + 1);
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        hptr.data(), csr_row_ptr, sizeof(I) * (m + 1), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Determine row blocks array size
    ComputeRowBlocks<I>((unsigned long long*)NULL, info->csrmv_info->size, hptr.data(), m, false);

    // Create row blocks structure
    std::vector<unsigned long long> row_blocks(info->csrmv_info->size, 0);

    ComputeRowBlocks<I>(row_blocks.data(), info->csrmv_info->size, hptr.data(), m, true);

    // Allocate memory on device to hold csrmv info, if required
    if(info->csrmv_info->size > 0)
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->csrmv_info->row_blocks,
                                      sizeof(unsigned long long) * info->csrmv_info->size));

        // Copy row blocks information to device
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(info->csrmv_info->row_blocks,
                                           row_blocks.data(),
                                           sizeof(unsigned long long) * info->csrmv_info->size,
                                           hipMemcpyHostToDevice,
                                           stream));

        // Wait for device transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }

    // Store some pointers to verify correct execution
    info->csrmv_info->trans       = trans;
    info->csrmv_info->m           = m;
    info->csrmv_info->n           = n;
    info->csrmv_info->nnz         = nnz;
    info->csrmv_info->descr       = descr;
    info->csrmv_info->csr_row_ptr = csr_row_ptr;
    info->csrmv_info->csr_col_ind = csr_col_ind;

    return rocsparse_status_success;
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) __global__
    void csrmvn_general_kernel(J m,
                               U alpha_device_host,
                               const I* __restrict__ csr_row_ptr,
                               const J* __restrict__ csr_col_ind,
                               const T* __restrict__ csr_val,
                               const T* __restrict__ x,
                               U beta_device_host,
                               T* __restrict__ y,
                               rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
    {
        csrmvn_general_device<BLOCKSIZE, WF_SIZE>(
            m, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, idx_base);
    }
}

template <typename I, typename J, typename T, typename U>
__launch_bounds__(WG_SIZE) __global__
    void csrmvn_adaptive_kernel(unsigned long long* __restrict__ row_blocks,
                                U alpha_device_host,
                                const I* __restrict__ csr_row_ptr,
                                const J* __restrict__ csr_col_ind,
                                const T* __restrict__ csr_val,
                                const T* __restrict__ x,
                                U beta_device_host,
                                T* __restrict__ y,
                                rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
    {
        csrmvn_adaptive_device<BLOCK_SIZE,
                               BLOCK_MULTIPLIER,
                               ROWS_FOR_VECTOR,
                               WG_BITS,
                               ROW_BITS,
                               WG_SIZE>(
            row_blocks, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, idx_base);
    }
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmv_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   J                         m,
                                                   J                         n,
                                                   I                         nnz,
                                                   U                         alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   const T*                  x,
                                                   U                         beta_device_host,
                                                   T*                        y)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Run different csrmv kernels
    if(trans == rocsparse_operation_none)
    {
#define CSRMVN_DIM 512
        J nnz_per_row = nnz / m;

        dim3 csrmvn_blocks((m - 1) / CSRMVN_DIM + 1);
        dim3 csrmvn_threads(CSRMVN_DIM);

        if(handle->wavefront_size == 32)
        {
            // LCOV_EXCL_START
            if(nnz_per_row < 4)
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 2>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            else if(nnz_per_row < 8)
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 4>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            else if(nnz_per_row < 16)
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 8>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            else if(nnz_per_row < 32)
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 16>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            else
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 32>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            // LCOV_EXCL_STOP
        }
        else
        {
            assert(handle->wavefront_size == 64);
            if(nnz_per_row < 4)
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 2>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            else if(nnz_per_row < 8)
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 4>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            else if(nnz_per_row < 16)
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 8>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            else if(nnz_per_row < 32)
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 16>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            else if(nnz_per_row < 64)
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 32>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
            else
            {
                hipLaunchKernelGGL((csrmvn_general_kernel<CSRMVN_DIM, 64>),
                                   csrmvn_blocks,
                                   csrmvn_threads,
                                   0,
                                   stream,
                                   m,
                                   alpha_device_host,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
            }
        }

#undef CSRMVN_DIM
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmv_adaptive_template_dispatch(rocsparse_handle    handle,
                                                            rocsparse_operation trans,
                                                            J                   m,
                                                            J                   n,
                                                            I                   nnz,
                                                            U                   alpha_device_host,
                                                            const rocsparse_mat_descr descr,
                                                            const T*                  csr_val,
                                                            const I*                  csr_row_ptr,
                                                            const J*                  csr_col_ind,
                                                            rocsparse_csrmv_info      info,
                                                            const T*                  x,
                                                            U  beta_device_host,
                                                            T* y)
{
    // Check if info matches current matrix and options
    if(info->trans != trans)
    {
        return rocsparse_status_invalid_value;
    }

    if(info->m != m || info->n != n || info->nnz != nnz)
    {
        return rocsparse_status_invalid_size;
    }

    if(info->descr != descr)
    {
        return rocsparse_status_invalid_value;
    }

    if(info->csr_row_ptr != csr_row_ptr || info->csr_col_ind != csr_col_ind)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different csrmv kernels
    if(trans == rocsparse_operation_none)
    {
        dim3 csrmvn_blocks((info->size / 2) - 1);
        dim3 csrmvn_threads(WG_SIZE);
        hipLaunchKernelGGL((csrmvn_adaptive_kernel),
                           csrmvn_blocks,
                           csrmvn_threads,
                           0,
                           stream,
                           info->row_blocks,
                           alpha_device_host,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           x,
                           beta_device_host,
                           y,
                           descr->base);
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrmv_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          J                         m,
                                          J                         n,
                                          I                         nnz,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          rocsparse_mat_info        info,
                                          const T*                  x,
                                          const T*                  beta_device_host,
                                          T*                        y)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmv"),
                  trans,
                  m,
                  n,
                  nnz,
                  *alpha_device_host,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)x,
                  *beta_device_host,
                  (const void*&)y,
                  (const void*&)info);

        log_bench(handle,
                  "./rocsparse-bench -f csrmv -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx> "
                  "--alpha",
                  *alpha_device_host,
                  "--beta",
                  *beta_device_host);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmv"),
                  trans,
                  m,
                  n,
                  nnz,
                  (const void*&)alpha_device_host,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)x,
                  (const void*&)beta_device_host,
                  (const void*&)y);
    }

    // Check index base
    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(alpha_device_host == nullptr || beta_device_host == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Another quick return.
    //
    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    //
    // Check the rest of pointer arguments
    //
    if(csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || x == nullptr
       || y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(info == nullptr || info->csrmv_info == nullptr)
    {
        // If csrmv info is not available, call csrmv general
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {

            return rocsparse_csrmv_template_dispatch(handle,
                                                     trans,
                                                     m,
                                                     n,
                                                     nnz,
                                                     alpha_device_host,
                                                     descr,
                                                     csr_val,
                                                     csr_row_ptr,
                                                     csr_col_ind,
                                                     x,
                                                     beta_device_host,
                                                     y);
        }
        else
        {
            return rocsparse_csrmv_template_dispatch(handle,
                                                     trans,
                                                     m,
                                                     n,
                                                     nnz,
                                                     *alpha_device_host,
                                                     descr,
                                                     csr_val,
                                                     csr_row_ptr,
                                                     csr_col_ind,
                                                     x,
                                                     *beta_device_host,
                                                     y);
        }
    }
    else
    {
        // If csrmv info is available, call csrmv adaptive
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {

            return rocsparse_csrmv_adaptive_template_dispatch(handle,
                                                              trans,
                                                              m,
                                                              n,
                                                              nnz,
                                                              alpha_device_host,
                                                              descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              info->csrmv_info,
                                                              x,
                                                              beta_device_host,
                                                              y);
        }
        else
        {

            return rocsparse_csrmv_adaptive_template_dispatch(handle,
                                                              trans,
                                                              m,
                                                              n,
                                                              nnz,
                                                              *alpha_device_host,
                                                              descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              info->csrmv_info,
                                                              x,
                                                              *beta_device_host,
                                                              y);
        }
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                              \
    template rocsparse_status rocsparse_csrmv_analysis_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans,                                              \
        JTYPE                     m,                                                  \
        JTYPE                     n,                                                  \
        ITYPE                     nnz,                                                \
        const rocsparse_mat_descr descr,                                              \
        const TTYPE*              csr_val,                                            \
        const ITYPE*              csr_row_ptr,                                        \
        const JTYPE*              csr_col_ind,                                        \
        rocsparse_mat_info        info);                                                     \
    template rocsparse_status rocsparse_csrmv_template<ITYPE, JTYPE, TTYPE>(          \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans,                                              \
        JTYPE                     m,                                                  \
        JTYPE                     n,                                                  \
        ITYPE                     nnz,                                                \
        const TTYPE*              alpha_device_host,                                  \
        const rocsparse_mat_descr descr,                                              \
        const TTYPE*              csr_val,                                            \
        const ITYPE*              csr_row_ptr,                                        \
        const JTYPE*              csr_col_ind,                                        \
        rocsparse_mat_info        info,                                               \
        const TTYPE*              x,                                                  \
        const TTYPE*              beta_device_host,                                   \
        TTYPE*                    y);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

//
// rocsparse_xcsrmv_analysis
//
#define C_IMPL(NAME, TYPE)                                                             \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                 \
                                     rocsparse_operation       trans,                  \
                                     rocsparse_int             m,                      \
                                     rocsparse_int             n,                      \
                                     rocsparse_int             nnz,                    \
                                     const rocsparse_mat_descr descr,                  \
                                     const TYPE*               csr_val,                \
                                     const rocsparse_int*      csr_row_ptr,            \
                                     const rocsparse_int*      csr_col_ind,            \
                                     rocsparse_mat_info        info)                   \
    {                                                                                  \
        return rocsparse_csrmv_analysis_template(                                      \
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info); \
    }

C_IMPL(rocsparse_scsrmv_analysis, float);
C_IMPL(rocsparse_dcsrmv_analysis, double);
C_IMPL(rocsparse_ccsrmv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrmv_analysis, rocsparse_double_complex);

#undef C_IMPL

//
// rocsparse_xcsrmv
//
#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             m,           \
                                     rocsparse_int             n,           \
                                     rocsparse_int             nnz,         \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     rocsparse_mat_info        info,        \
                                     const TYPE*               x,           \
                                     const TYPE*               beta,        \
                                     TYPE*                     y)           \
    {                                                                       \
        return rocsparse_csrmv_template(handle,                             \
                                        trans,                              \
                                        m,                                  \
                                        n,                                  \
                                        nnz,                                \
                                        alpha,                              \
                                        descr,                              \
                                        csr_val,                            \
                                        csr_row_ptr,                        \
                                        csr_col_ind,                        \
                                        info,                               \
                                        x,                                  \
                                        beta,                               \
                                        y);                                 \
    }

C_IMPL(rocsparse_scsrmv, float);
C_IMPL(rocsparse_dcsrmv, double);
C_IMPL(rocsparse_ccsrmv, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrmv, rocsparse_double_complex);
#undef C_IMPL

extern "C" rocsparse_status rocsparse_csrmv_clear(rocsparse_handle handle, rocsparse_mat_info info)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle, "rocsparse_csrmv_clear", (const void*&)info);

    // Destroy csrmv info struct
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrmv_info(info->csrmv_info));
    info->csrmv_info = nullptr;

    return rocsparse_status_success;
}
