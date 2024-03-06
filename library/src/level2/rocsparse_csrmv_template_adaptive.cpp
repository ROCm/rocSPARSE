/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"
#include "control.h"
#include "rocsparse_csrmv.hpp"
#include "utility.h"

#include "csrmv_device.h"
#include "csrmv_symm_device.h"
#include <vector>

#define BLOCK_SIZE 1024
#define BLOCK_MULTIPLIER 3
#define ROWS_FOR_VECTOR 1
#define WG_SIZE 256

namespace rocsparse
{
    __attribute__((unused)) static uint32_t flp2(uint32_t x)
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
    static uint64_t numThreadsForReduction(uint64_t num_rows)
    {
#if defined(__INTEL_COMPILER)
        return WG_SIZE >> (_bit_scan_reverse(num_rows - 1) + 1);
#elif(defined(__clang__) && __has_builtin(__builtin_clz)) \
    || !defined(__clang) && defined(__GNUG__)             \
           && ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 30202)
        return (WG_SIZE >> (8 * sizeof(int) - __builtin_clz(num_rows - 1)));
#elif defined(_MSC_VER) && (_MSC_VER >= 1400)
        uint64_t bit_returned;
        _BitScanReverse(&bit_returned, (num_rows - 1));
        return WG_SIZE >> (bit_returned + 1);
#else
        return flp2(WG_SIZE / num_rows);
#endif
    }

    template <typename I>
    static inline I maxRowsInABlock(const I* rowBlocks, size_t rowBlockSize)
    {
        I max = 0;
        for(size_t i = 1; i < rowBlockSize; i++)
        {
            I current_row = rowBlocks[i];
            I prev_row    = rowBlocks[i - 1];

            if(max < current_row - prev_row)
                max = current_row - prev_row;
        }
        return max;
    }

    template <typename I, typename J>
    static inline void ComputeRowBlocks(I*       rowBlocks,
                                        J*       wgIds,
                                        size_t&  rowBlockSize,
                                        const I* rowDelimiters,
                                        I        nRows,
                                        bool     allocate_row_blocks = true)
    {
        I* rowBlocksBase;

        // Start at one because of rowBlock[0]
        I total_row_blocks = 1;

        if(allocate_row_blocks)
        {
            rowBlocksBase = rowBlocks;
            *rowBlocks    = 0;
            *wgIds        = 0;
            ++rowBlocks;
            ++wgIds;
        }

        I sum = 0;
        I i;
        I last_i = 0;

        I consecutive_long_rows = 0;
        for(i = 1; i <= nRows; ++i)
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
                        *rowBlocks = i - 1;

                        // If this row fits into CSR-Stream, calculate how many rows
                        // can be used to do a parallel reduction.
                        // Fill in the low-order bits with the numThreadsForRed
                        if(((i - 1) - last_i) > static_cast<I>(ROWS_FOR_VECTOR))
                        {
                            *(wgIds - 1) |= numThreadsForReduction((i - 1) - last_i);
                        }

                        ++rowBlocks;
                        ++wgIds;
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
                    *rowBlocks = i - 1;
                    if(((i - 1) - last_i) > static_cast<I>(ROWS_FOR_VECTOR))
                    {
                        *(wgIds - 1) |= numThreadsForReduction((i - 1) - last_i);
                    }

                    ++rowBlocks;
                    ++wgIds;
                }

                ++total_row_blocks;
                last_i                = i - 1;
                sum                   = row_length;
                consecutive_long_rows = 0;
            }

            // Now, what's up with this row? What did it do?

            // exactly one row results in non-zero elements to be greater than blockSize
            // This is csr-vector case;
            if((i - last_i == 1) && sum > static_cast<I>(BLOCK_SIZE))
            {
                I numWGReq = static_cast<I>(rocsparse::ceil(static_cast<double>(row_length)
                                                            / (BLOCK_MULTIPLIER * BLOCK_SIZE)));

                // Check to ensure #workgroups can fit in 32 bits, if not
                // then the last workgroup will do all the remaining work
                // Note: Maximum number of workgroups is 2^31-1 = 2147483647
                static constexpr I maxNumberOfWorkgroups = static_cast<I>(INT_MAX);
                numWGReq = (numWGReq < maxNumberOfWorkgroups) ? numWGReq : maxNumberOfWorkgroups;

                if(allocate_row_blocks)
                {
                    for(I w = 1; w < numWGReq; ++w)
                    {
                        *rowBlocks = (i - 1);
                        *wgIds |= static_cast<J>(w);

                        ++rowBlocks;
                        ++wgIds;
                    }

                    *rowBlocks = i;
                    ++rowBlocks;
                    ++wgIds;
                }

                total_row_blocks += numWGReq;
                last_i                = i;
                sum                   = 0;
                consecutive_long_rows = 0;
            }
            // more than one row results in non-zero elements to be greater than blockSize
            // This is csr-stream case; wgIds holds number of parallel reduction threads
            else if((i - last_i > 1) && sum > static_cast<I>(BLOCK_SIZE))
            {
                // This row won't fit, so back off one.
                --i;

                if(allocate_row_blocks)
                {
                    *rowBlocks = i;
                    if((i - last_i) > static_cast<I>(ROWS_FOR_VECTOR))
                    {
                        *(wgIds - 1) |= numThreadsForReduction(i - last_i);
                    }

                    ++rowBlocks;
                    ++wgIds;
                }

                ++total_row_blocks;
                last_i                = i;
                sum                   = 0;
                consecutive_long_rows = 0;
            }
            // This is csr-stream case; wgIds holds number of parallel reduction threads
            else if(sum == static_cast<I>(BLOCK_SIZE))
            {
                if(allocate_row_blocks)
                {
                    *rowBlocks = i;
                    if((i - last_i) > static_cast<I>(ROWS_FOR_VECTOR))
                    {
                        *(wgIds - 1) |= numThreadsForReduction(i - last_i);
                    }

                    ++rowBlocks;
                    ++wgIds;
                }

                ++total_row_blocks;
                last_i                = i;
                sum                   = 0;
                consecutive_long_rows = 0;
            }
        }

        // If we didn't fill a row block with the last row, make sure we don't lose it.
        if(allocate_row_blocks && *(rowBlocks - 1) != nRows)
        {
            *rowBlocks = nRows;
            if((nRows - last_i) > static_cast<I>(ROWS_FOR_VECTOR))
            {
                *(wgIds - 1) |= numThreadsForReduction(i - last_i);
            }

            ++rowBlocks;
        }

        ++total_row_blocks;

        if(allocate_row_blocks)
        {
            size_t dist = std::distance(rowBlocksBase, rowBlocks);

            assert((dist) <= rowBlockSize);
            // Update the size of rowBlocks to reflect the actual amount of memory used
            rowBlockSize = dist;
        }
        else
        {
            rowBlockSize = total_row_blocks;
        }
    }
}

template <typename I, typename J, typename A>
rocsparse_status
    rocsparse::csrmv_analysis_adaptive_template_dispatch(rocsparse_handle          handle,
                                                         rocsparse_operation       trans,
                                                         J                         m,
                                                         J                         n,
                                                         I                         nnz,
                                                         const rocsparse_mat_descr descr,
                                                         const A*                  csr_val,
                                                         const I*                  csr_row_ptr,
                                                         const J*                  csr_col_ind,
                                                         rocsparse_mat_info        info)
{
    // Clear csrmv info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_csrmv_info(info->csrmv_info));

    // Create csrmv info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_csrmv_info(&info->csrmv_info));

    // Stream
    hipStream_t stream = handle->stream;

    // row blocks size
    info->csrmv_info->adaptive.size = 0;

    // Temporary arrays to hold device data
    std::vector<I> hptr(m + 1);
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        hptr.data(), csr_row_ptr, sizeof(I) * (m + 1), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Determine row blocks array size
    ComputeRowBlocks<I, J>(
        (I*)NULL, (J*)NULL, info->csrmv_info->adaptive.size, hptr.data(), m, false);

    // Create row blocks, workgroup flag, and workgroup data structures
    std::vector<I>        row_blocks(info->csrmv_info->adaptive.size, 0);
    std::vector<uint32_t> wg_flags(info->csrmv_info->adaptive.size, 0);
    std::vector<J>        wg_ids(info->csrmv_info->adaptive.size, 0);

    ComputeRowBlocks<I, J>(
        row_blocks.data(), wg_ids.data(), info->csrmv_info->adaptive.size, hptr.data(), m, true);

    if(descr->type == rocsparse_matrix_type_symmetric)
    {
        info->csrmv_info->max_rows
            = maxRowsInABlock(row_blocks.data(), info->csrmv_info->adaptive.size);
    }

    // Allocate memory on device to hold csrmv info, if required
    if(info->csrmv_info->adaptive.size > 0)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync((void**)&info->csrmv_info->adaptive.row_blocks,
                                                     sizeof(I) * info->csrmv_info->adaptive.size,
                                                     handle->stream));
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync((void**)&info->csrmv_info->adaptive.wg_flags,
                                     sizeof(uint32_t) * info->csrmv_info->adaptive.size,
                                     handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync((void**)&info->csrmv_info->adaptive.wg_ids,
                                                     sizeof(J) * info->csrmv_info->adaptive.size,
                                                     handle->stream));

        // Copy row blocks information to device
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(info->csrmv_info->adaptive.row_blocks,
                                           row_blocks.data(),
                                           sizeof(I) * info->csrmv_info->adaptive.size,
                                           hipMemcpyHostToDevice,
                                           stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(info->csrmv_info->adaptive.wg_flags,
                                           wg_flags.data(),
                                           sizeof(uint32_t) * info->csrmv_info->adaptive.size,
                                           hipMemcpyHostToDevice,
                                           stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(info->csrmv_info->adaptive.wg_ids,
                                           wg_ids.data(),
                                           sizeof(J) * info->csrmv_info->adaptive.size,
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

    info->csrmv_info->index_type_I = rocsparse::get_indextype<I>();
    info->csrmv_info->index_type_J = rocsparse::get_indextype<J>();

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename I, typename J, typename A, typename X, typename Y, typename U>
    ROCSPARSE_KERNEL(WG_SIZE)
    void csrmvn_adaptive_kernel(bool conj,
                                I    nnz,
                                const I* __restrict__ row_blocks,
                                uint32_t* __restrict__ wg_flags,
                                const J* __restrict__ wg_ids,
                                U alpha_device_host,
                                const I* __restrict__ csr_row_ptr,
                                const J* __restrict__ csr_col_ind,
                                const A* __restrict__ csr_val,
                                const X* __restrict__ x,
                                U beta_device_host,
                                Y* __restrict__ y,
                                rocsparse_index_base idx_base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        auto beta  = rocsparse::load_scalar_device_host(beta_device_host);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::
                csrmvn_adaptive_device<BLOCK_SIZE, BLOCK_MULTIPLIER, ROWS_FOR_VECTOR, WG_SIZE>(
                    conj,
                    nnz,
                    row_blocks,
                    wg_flags,
                    wg_ids,
                    alpha,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    x,
                    beta,
                    y,
                    idx_base);
        }
    }

    template <rocsparse_int MAX_ROWS,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename U>
    ROCSPARSE_KERNEL(WG_SIZE)
    void csrmvn_symm_adaptive_kernel(bool conj,
                                     I    nnz,
                                     I    max_rows,
                                     const I* __restrict__ row_blocks,
                                     U alpha_device_host,
                                     const I* __restrict__ csr_row_ptr,
                                     const J* __restrict__ csr_col_ind,
                                     const A* __restrict__ csr_val,
                                     const X* __restrict__ x,
                                     U beta_device_host,
                                     Y* __restrict__ y,
                                     rocsparse_index_base idx_base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        auto beta  = rocsparse::load_scalar_device_host(beta_device_host);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmvn_symm_adaptive_device<BLOCK_SIZE, MAX_ROWS, WG_SIZE>(conj,
                                                                                  nnz,
                                                                                  max_rows,
                                                                                  row_blocks,
                                                                                  alpha,
                                                                                  csr_row_ptr,
                                                                                  csr_col_ind,
                                                                                  csr_val,
                                                                                  x,
                                                                                  beta,
                                                                                  y,
                                                                                  idx_base);
        }
    }
}

template <typename I, typename J, typename A, typename X, typename Y, typename U>
ROCSPARSE_KERNEL(WG_SIZE)
void csrmvn_symm_large_adaptive_kernel(bool conj,
                                       I    nnz,
                                       const I* __restrict__ row_blocks,
                                       U alpha_device_host,
                                       const I* __restrict__ csr_row_ptr,
                                       const J* __restrict__ csr_col_ind,
                                       const A* __restrict__ csr_val,
                                       const X* __restrict__ x,
                                       U beta_device_host,
                                       Y* __restrict__ y,
                                       rocsparse_index_base idx_base)
{
    auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
    auto beta  = rocsparse::load_scalar_device_host(beta_device_host);
    if(alpha != 0 || beta != 1)
    {
        rocsparse::csrmvn_symm_large_adaptive_device<BLOCK_SIZE, WG_SIZE>(
            conj, nnz, row_blocks, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, idx_base);
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
rocsparse_status rocsparse::csrmv_adaptive_template_dispatch(rocsparse_handle    handle,
                                                             rocsparse_operation trans,
                                                             J                   m,
                                                             J                   n,
                                                             I                   nnz,
                                                             U                   alpha_device_host,
                                                             const rocsparse_mat_descr descr,
                                                             const A*                  csr_val,
                                                             const I*                  csr_row_ptr,
                                                             const J*                  csr_col_ind,
                                                             rocsparse_csrmv_info      info,
                                                             const X*                  x,
                                                             U    beta_device_host,
                                                             Y*   y,
                                                             bool force_conj)
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG_POINTER(10, info);

    bool conj = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    // Check if info matches current matrix and options
    ROCSPARSE_CHECKARG_ENUM(1, trans);

    ROCSPARSE_CHECKARG(10, info, (info->trans != trans), rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG(
        1, trans, (trans != rocsparse_operation_none), rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG(10,
                       info,
                       (info->m != m || info->n != n || info->nnz != nnz),
                       rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(10, info, (info->descr != descr), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(10,
                       info,
                       (info->csr_row_ptr != csr_row_ptr || info->csr_col_ind != csr_col_ind),
                       rocsparse_status_invalid_pointer);

    // Stream
    hipStream_t stream = handle->stream;

    if(descr->type == rocsparse_matrix_type_general
       || descr->type == rocsparse_matrix_type_triangular)
    {
        // Run different csrmv kernels
        dim3 csrmvn_blocks((info->adaptive.size) - 1);
        dim3 csrmvn_threads(WG_SIZE);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmvn_adaptive_kernel),
                                           csrmvn_blocks,
                                           csrmvn_threads,
                                           0,
                                           stream,
                                           conj,
                                           nnz,
                                           static_cast<I*>(info->adaptive.row_blocks),
                                           info->adaptive.wg_flags,
                                           static_cast<J*>(info->adaptive.wg_ids),
                                           alpha_device_host,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           x,
                                           beta_device_host,
                                           y,
                                           descr->base);
    }
    else if(descr->type == rocsparse_matrix_type_symmetric)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::scale_array<256>),
                                           dim3((m - 1) / 256 + 1),
                                           dim3(256),
                                           0,
                                           stream,
                                           m,
                                           y,
                                           beta_device_host);

        dim3 csrmvn_blocks(info->adaptive.size - 1);
        dim3 csrmvn_threads(WG_SIZE);

        I max_rows = static_cast<I>(info->max_rows);
        if(max_rows <= 64)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmvn_symm_adaptive_kernel<64>),
                                               csrmvn_blocks,
                                               csrmvn_threads,
                                               0,
                                               stream,
                                               conj,
                                               nnz,
                                               max_rows,
                                               static_cast<I*>(info->adaptive.row_blocks),
                                               alpha_device_host,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               x,
                                               beta_device_host,
                                               y,
                                               descr->base);
        }
        else if(max_rows <= 128)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmvn_symm_adaptive_kernel<128>),
                                               csrmvn_blocks,
                                               csrmvn_threads,
                                               0,
                                               stream,
                                               conj,
                                               nnz,
                                               max_rows,
                                               static_cast<I*>(info->adaptive.row_blocks),
                                               alpha_device_host,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               x,
                                               beta_device_host,
                                               y,
                                               descr->base);
        }
        else if(max_rows <= 256)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmvn_symm_adaptive_kernel<256>),
                                               csrmvn_blocks,
                                               csrmvn_threads,
                                               0,
                                               stream,
                                               conj,
                                               nnz,
                                               max_rows,
                                               static_cast<I*>(info->adaptive.row_blocks),
                                               alpha_device_host,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               x,
                                               beta_device_host,
                                               y,
                                               descr->base);
        }
        else if(max_rows <= 512)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmvn_symm_adaptive_kernel<512>),
                                               csrmvn_blocks,
                                               csrmvn_threads,
                                               0,
                                               stream,
                                               conj,
                                               nnz,
                                               max_rows,
                                               static_cast<I*>(info->adaptive.row_blocks),
                                               alpha_device_host,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               x,
                                               beta_device_host,
                                               y,
                                               descr->base);
        }
        else if(max_rows <= 1024)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmvn_symm_adaptive_kernel<1024>),
                                               csrmvn_blocks,
                                               csrmvn_threads,
                                               0,
                                               stream,
                                               conj,
                                               nnz,
                                               max_rows,
                                               static_cast<I*>(info->adaptive.row_blocks),
                                               alpha_device_host,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               x,
                                               beta_device_host,
                                               y,
                                               descr->base);
        }
        else if(max_rows <= 2048)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmvn_symm_adaptive_kernel<2048>),
                                               csrmvn_blocks,
                                               csrmvn_threads,
                                               0,
                                               stream,
                                               conj,
                                               nnz,
                                               max_rows,
                                               static_cast<I*>(info->adaptive.row_blocks),
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
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrmvn_symm_large_adaptive_kernel),
                                               csrmvn_blocks,
                                               csrmvn_threads,
                                               0,
                                               stream,
                                               conj,
                                               nnz,
                                               static_cast<I*>(info->adaptive.row_blocks),
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
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, ATYPE)                                            \
    template rocsparse_status rocsparse::csrmv_analysis_adaptive_template_dispatch( \
        rocsparse_handle          handle,                                           \
        rocsparse_operation       trans,                                            \
        JTYPE                     m,                                                \
        JTYPE                     n,                                                \
        ITYPE                     nnz,                                              \
        const rocsparse_mat_descr descr,                                            \
        const ATYPE*              csr_val,                                          \
        const ITYPE*              csr_row_ptr,                                      \
        const JTYPE*              csr_col_ind,                                      \
        rocsparse_mat_info        info);

// Uniform precision
INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE(int32_t, int32_t, int8_t);
INSTANTIATE(int64_t, int32_t, int8_t);
INSTANTIATE(int64_t, int64_t, int8_t);

#undef INSTANTIATE

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, UTYPE)              \
    template rocsparse_status rocsparse::csrmv_adaptive_template_dispatch<TTYPE>( \
        rocsparse_handle          handle,                                         \
        rocsparse_operation       trans,                                          \
        JTYPE                     m,                                              \
        JTYPE                     n,                                              \
        ITYPE                     nnz,                                            \
        UTYPE                     alpha_device_host,                              \
        const rocsparse_mat_descr descr,                                          \
        const ATYPE*              csr_val,                                        \
        const ITYPE*              csr_row_ptr,                                    \
        const JTYPE*              csr_col_ind,                                    \
        rocsparse_csrmv_info      info,                                           \
        const XTYPE*              x,                                              \
        UTYPE                     beta_device_host,                               \
        YTYPE*                    y,                                              \
        bool                      force_conj);

// Uniform precision
INSTANTIATE(float, int32_t, int32_t, float, float, float, float);
INSTANTIATE(float, int64_t, int32_t, float, float, float, float);
INSTANTIATE(float, int64_t, int64_t, float, float, float, float);
INSTANTIATE(double, int32_t, int32_t, double, double, double, double);
INSTANTIATE(double, int64_t, int32_t, double, double, double, double);
INSTANTIATE(double, int64_t, int64_t, double, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

INSTANTIATE(float, int32_t, int32_t, float, float, float, const float*);
INSTANTIATE(float, int64_t, int32_t, float, float, float, const float*);
INSTANTIATE(float, int64_t, int64_t, float, float, float, const float*);
INSTANTIATE(double, int32_t, int32_t, double, double, double, const double*);
INSTANTIATE(double, int64_t, int32_t, double, double, double, const double*);
INSTANTIATE(double, int64_t, int64_t, double, double, double, const double*);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);

// Mixed percision
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float, float);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(double, int32_t, int32_t, float, double, double, double);
INSTANTIATE(double, int64_t, int32_t, float, double, double, double);
INSTANTIATE(double, int64_t, int64_t, float, double, double, double);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float, const float*);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float, const float*);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float, const float*);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(double, int32_t, int32_t, float, double, double, const double*);
INSTANTIATE(double, int64_t, int32_t, float, double, double, const double*);
INSTANTIATE(double, int64_t, int64_t, float, double, double, const double*);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);

#undef INSTANTIATE
