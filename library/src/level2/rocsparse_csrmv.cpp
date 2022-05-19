/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
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
#include "common.h"
#include "definitions.h"
#include "utility.h"

#include "csrmv_device.h"
#include "csrmv_symm_device.h"

#define BLOCK_SIZE 1024
#define BLOCK_MULTIPLIER 3
#define ROWS_FOR_VECTOR 1
#define WG_SIZE 256

#define LAUNCH_CSRMVN_GENERAL(wfsize)                                     \
    csrmvn_general_kernel<CSRMVN_DIM, wfsize>                             \
        <<<csrmvn_blocks, csrmvn_threads, 0, stream>>>(conj,              \
                                                       m,                 \
                                                       alpha_device_host, \
                                                       csr_row_ptr,       \
                                                       csr_col_ind,       \
                                                       csr_val,           \
                                                       x,                 \
                                                       beta_device_host,  \
                                                       y,                 \
                                                       descr->base)

#define LAUNCH_CSRMVT(wfsize)                                                                \
    csrmvt_general_kernel<CSRMVT_DIM, wfsize><<<csrmvt_blocks, csrmvt_threads, 0, stream>>>( \
        conj, m, alpha_device_host, csr_row_ptr, csr_col_ind, csr_val, x, y, descr->base)

#define LAUNCH_CSRMVN_SYMM_GENERAL(wfsize)                                \
    csrmvn_symm_general_kernel<CSRMV_SYMM_DIM, wfsize>                    \
        <<<csrmvn_blocks, csrmvn_threads, 0, stream>>>(conj,              \
                                                       m,                 \
                                                       alpha_device_host, \
                                                       csr_row_ptr,       \
                                                       csr_col_ind,       \
                                                       csr_val,           \
                                                       x,                 \
                                                       beta_device_host,  \
                                                       y,                 \
                                                       descr->base)

#define LAUNCH_CSRMVT_SYMM(wfsize)                      \
    csrmvt_symm_general_kernel<CSRMV_SYMM_DIM, wfsize>  \
        <<<csrmvt_blocks, csrmvt_threads, 0, stream>>>( \
            conj, m, alpha_device_host, csr_row_ptr, csr_col_ind, csr_val, x, y, descr->base)

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
            I numWGReq = static_cast<I>(
                std::ceil(static_cast<double>(row_length) / (BLOCK_MULTIPLIER * BLOCK_SIZE)));

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

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general
       && descr->type != rocsparse_matrix_type_triangular
       && descr->type != rocsparse_matrix_type_symmetric)
    {
        return rocsparse_status_not_implemented;
    }

    if(descr->type == rocsparse_matrix_type_symmetric
       || descr->type == rocsparse_matrix_type_triangular)
    {
        if(m != n)
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    if(m == 0 || n == 0)
    {
        if(nnz != 0)
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Quick return if possible
    if(m == 0 && n == 0 && nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Another quick return.
    if(m == 0 || n == 0 || nnz == 0)
    {
        // No matrix analysis required as matrix never accessed
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (csr_col_ind == nullptr && csr_val == nullptr))
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
    ComputeRowBlocks<I, J>((I*)NULL, (J*)NULL, info->csrmv_info->size, hptr.data(), m, false);

    // Create row blocks, workgroup flag, and workgroup data structures
    std::vector<I>            row_blocks(info->csrmv_info->size, 0);
    std::vector<unsigned int> wg_flags(info->csrmv_info->size, 0);
    std::vector<J>            wg_ids(info->csrmv_info->size, 0);

    ComputeRowBlocks<I, J>(
        row_blocks.data(), wg_ids.data(), info->csrmv_info->size, hptr.data(), m, true);

    if(descr->type == rocsparse_matrix_type_symmetric)
    {
        info->csrmv_info->max_rows = maxRowsInABlock(row_blocks.data(), info->csrmv_info->size);
    }

    // Allocate memory on device to hold csrmv info, if required
    if(info->csrmv_info->size > 0)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMalloc((void**)&info->csrmv_info->row_blocks,
                                                sizeof(I) * info->csrmv_info->size));
        RETURN_IF_HIP_ERROR(rocsparse_hipMalloc((void**)&info->csrmv_info->wg_flags,
                                                sizeof(unsigned int) * info->csrmv_info->size));
        RETURN_IF_HIP_ERROR(rocsparse_hipMalloc((void**)&info->csrmv_info->wg_ids,
                                                sizeof(J) * info->csrmv_info->size));

        // Copy row blocks information to device
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(info->csrmv_info->row_blocks,
                                           row_blocks.data(),
                                           sizeof(I) * info->csrmv_info->size,
                                           hipMemcpyHostToDevice,
                                           stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(info->csrmv_info->wg_flags,
                                           wg_flags.data(),
                                           sizeof(unsigned int) * info->csrmv_info->size,
                                           hipMemcpyHostToDevice,
                                           stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(info->csrmv_info->wg_ids,
                                           wg_ids.data(),
                                           sizeof(J) * info->csrmv_info->size,
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

    info->csrmv_info->index_type_I
        = (sizeof(I) == sizeof(uint16_t))
              ? rocsparse_indextype_u16
              : ((sizeof(I) == sizeof(int32_t)) ? rocsparse_indextype_i32
                                                : rocsparse_indextype_i64);
    info->csrmv_info->index_type_J
        = (sizeof(J) == sizeof(uint16_t))
              ? rocsparse_indextype_u16
              : ((sizeof(J) == sizeof(int32_t)) ? rocsparse_indextype_i32
                                                : rocsparse_indextype_i64);

    return rocsparse_status_success;
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmvn_general_kernel(bool conj,
                               J    m,
                               U    alpha_device_host,
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
            conj, m, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, idx_base);
    }
}

template <unsigned int BLOCKSIZE, typename J, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmvt_scale_kernel(J size, U scalar_device_host, T* __restrict__ data)
{
    auto scalar = load_scalar_device_host(scalar_device_host);
    csrmvt_scale_device(size, scalar, data);
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmvt_general_kernel(bool conj,
                               J    m,
                               U    alpha_device_host,
                               const I* __restrict__ csr_row_ptr,
                               const J* __restrict__ csr_col_ind,
                               const T* __restrict__ csr_val,
                               const T* __restrict__ x,
                               T* __restrict__ y,
                               rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != static_cast<T>(0))
    {
        csrmvt_general_device<BLOCKSIZE, WF_SIZE>(
            conj, m, alpha, csr_row_ptr, csr_col_ind, csr_val, x, y, idx_base);
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmvn_symm_general_kernel(bool conj,
                                    J    m,
                                    U    alpha_device_host,
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
        csrmvn_symm_general_device<BLOCKSIZE, WF_SIZE>(
            conj, m, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, idx_base);
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmvt_symm_general_kernel(bool conj,
                                    J    m,
                                    U    alpha_device_host,
                                    const I* __restrict__ csr_row_ptr,
                                    const J* __restrict__ csr_col_ind,
                                    const T* __restrict__ csr_val,
                                    const T* __restrict__ x,
                                    T* __restrict__ y,
                                    rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != static_cast<T>(0))
    {
        csrmvt_symm_general_device<BLOCKSIZE, WF_SIZE>(
            conj, m, alpha, csr_row_ptr, csr_col_ind, csr_val, x, y, idx_base);
    }
}

template <typename I, typename J, typename T, typename U>
__launch_bounds__(WG_SIZE) ROCSPARSE_KERNEL
    void csrmvn_adaptive_kernel(bool conj,
                                I    nnz,
                                const I* __restrict__ row_blocks,
                                unsigned int* __restrict__ wg_flags,
                                const J* __restrict__ wg_ids,
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
        csrmvn_adaptive_device<BLOCK_SIZE, BLOCK_MULTIPLIER, ROWS_FOR_VECTOR, WG_SIZE>(conj,
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

template <rocsparse_int MAX_ROWS, typename I, typename J, typename T, typename U>
__launch_bounds__(WG_SIZE) ROCSPARSE_KERNEL
    void csrmvn_symm_adaptive_kernel(bool conj,
                                     I    nnz,
                                     I    max_rows,
                                     const I* __restrict__ row_blocks,
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
        csrmvn_symm_adaptive_device<BLOCK_SIZE, MAX_ROWS, WG_SIZE>(conj,
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
                                                   T*                        y,
                                                   bool                      force_conj)
{
    bool conj = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    // Stream
    hipStream_t stream = handle->stream;

    // Average nnz per row
    J nnz_per_row = nnz / m;

    if(descr->type == rocsparse_matrix_type_general
       || descr->type == rocsparse_matrix_type_triangular)
    {
        // Run different csrmv kernels
        if(trans == rocsparse_operation_none)
        {
#define CSRMVN_DIM 512
            dim3 csrmvn_blocks((m - 1) / CSRMVN_DIM + 1);
            dim3 csrmvn_threads(CSRMVN_DIM);

            if(nnz_per_row < 4)
            {
                LAUNCH_CSRMVN_GENERAL(2);
            }
            else if(nnz_per_row < 8)
            {
                LAUNCH_CSRMVN_GENERAL(4);
            }
            else if(nnz_per_row < 16)
            {
                LAUNCH_CSRMVN_GENERAL(8);
            }
            else if(nnz_per_row < 32)
            {
                LAUNCH_CSRMVN_GENERAL(16);
            }
            else if(nnz_per_row < 64 || handle->wavefront_size == 32)
            {
                LAUNCH_CSRMVN_GENERAL(32);
            }
            else
            {
                LAUNCH_CSRMVN_GENERAL(64);
            }

#undef CSRMVN_DIM
        }
        else
        {
#define CSRMVT_DIM 256
            // Scale y with beta
            csrmvt_scale_kernel<CSRMVT_DIM>
                <<<(n - 1) / CSRMVT_DIM + 1, CSRMVT_DIM, 0, stream>>>(n, beta_device_host, y);

            rocsparse_int max_blocks = 1024;
            rocsparse_int min_blocks = (m - 1) / CSRMVT_DIM + 1;

            dim3 csrmvt_blocks(std::min(min_blocks, max_blocks));
            dim3 csrmvt_threads(CSRMVT_DIM);

            if(nnz_per_row < 4)
            {
                LAUNCH_CSRMVT(4);
            }
            else if(nnz_per_row < 8)
            {
                LAUNCH_CSRMVT(8);
            }
            else if(nnz_per_row < 16)
            {
                LAUNCH_CSRMVT(16);
            }
            else if(nnz_per_row < 32 || handle->wavefront_size == 32)
            {
                LAUNCH_CSRMVT(32);
            }
            else
            {
                LAUNCH_CSRMVT(64);
            }
#undef CSRMVT_DIM
        }
    }
    else if(descr->type == rocsparse_matrix_type_symmetric)
    {
#define CSRMV_SYMM_DIM 256
        dim3 csrmvn_blocks((m - 1) / CSRMV_SYMM_DIM + 1);
        dim3 csrmvn_threads(CSRMV_SYMM_DIM);

        if(nnz_per_row < 4)
        {
            LAUNCH_CSRMVN_SYMM_GENERAL(2);
        }
        else if(nnz_per_row < 8)
        {
            LAUNCH_CSRMVN_SYMM_GENERAL(4);
        }
        else if(nnz_per_row < 16)
        {
            LAUNCH_CSRMVN_SYMM_GENERAL(8);
        }
        else if(nnz_per_row < 32)
        {
            LAUNCH_CSRMVN_SYMM_GENERAL(16);
        }
        else if(nnz_per_row < 64 || handle->wavefront_size == 32)
        {
            LAUNCH_CSRMVN_SYMM_GENERAL(32);
        }
        else
        {
            LAUNCH_CSRMVN_SYMM_GENERAL(64);
        }

        rocsparse_int max_blocks = 1024;
        rocsparse_int min_blocks = (m - 1) / CSRMV_SYMM_DIM + 1;

        dim3 csrmvt_blocks(std::min(min_blocks, max_blocks));
        dim3 csrmvt_threads(CSRMV_SYMM_DIM);

        if(nnz_per_row < 4)
        {
            LAUNCH_CSRMVT_SYMM(4);
        }
        else if(nnz_per_row < 8)
        {
            LAUNCH_CSRMVT_SYMM(8);
        }
        else if(nnz_per_row < 16)
        {
            LAUNCH_CSRMVT_SYMM(16);
        }
        else if(nnz_per_row < 32 || handle->wavefront_size == 32)
        {
            LAUNCH_CSRMVT_SYMM(32);
        }
        else
        {
            LAUNCH_CSRMVT_SYMM(64);
        }
#undef CSRMV_SYMM_DIM
    }
    else
    {
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
                                                            U    beta_device_host,
                                                            T*   y,
                                                            bool force_conj)
{
    bool conj = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    // Check if info matches current matrix and options
    if(info->trans != trans)
    {
        return rocsparse_status_invalid_value;
    }

    if(trans != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
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

    if(descr->type == rocsparse_matrix_type_general
       || descr->type == rocsparse_matrix_type_triangular)
    {
        // Run different csrmv kernels
        dim3 csrmvn_blocks((info->size) - 1);
        dim3 csrmvn_threads(WG_SIZE);
        hipLaunchKernelGGL((csrmvn_adaptive_kernel),
                           csrmvn_blocks,
                           csrmvn_threads,
                           0,
                           stream,
                           conj,
                           nnz,
                           static_cast<I*>(info->row_blocks),
                           info->wg_flags,
                           static_cast<J*>(info->wg_ids),
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
        hipLaunchKernelGGL((scale_array<256>),
                           dim3((m - 1) / 256 + 1),
                           dim3(256),
                           0,
                           stream,
                           m,
                           y,
                           beta_device_host);

        dim3 csrmvn_blocks(info->size - 1);
        dim3 csrmvn_threads(WG_SIZE);

        I max_rows = static_cast<I>(info->max_rows);
        if(max_rows <= 64)
        {
            hipLaunchKernelGGL((csrmvn_symm_adaptive_kernel<64>),
                               csrmvn_blocks,
                               csrmvn_threads,
                               0,
                               stream,
                               conj,
                               nnz,
                               max_rows,
                               static_cast<I*>(info->row_blocks),
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
            hipLaunchKernelGGL((csrmvn_symm_adaptive_kernel<128>),
                               csrmvn_blocks,
                               csrmvn_threads,
                               0,
                               stream,
                               conj,
                               nnz,
                               max_rows,
                               static_cast<I*>(info->row_blocks),
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
            hipLaunchKernelGGL((csrmvn_symm_adaptive_kernel<256>),
                               csrmvn_blocks,
                               csrmvn_threads,
                               0,
                               stream,
                               conj,
                               nnz,
                               max_rows,
                               static_cast<I*>(info->row_blocks),
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
            hipLaunchKernelGGL((csrmvn_symm_adaptive_kernel<512>),
                               csrmvn_blocks,
                               csrmvn_threads,
                               0,
                               stream,
                               conj,
                               nnz,
                               max_rows,
                               static_cast<I*>(info->row_blocks),
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
            hipLaunchKernelGGL((csrmvn_symm_adaptive_kernel<1024>),
                               csrmvn_blocks,
                               csrmvn_threads,
                               0,
                               stream,
                               conj,
                               nnz,
                               max_rows,
                               static_cast<I*>(info->row_blocks),
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
            hipLaunchKernelGGL((csrmvn_symm_adaptive_kernel<2048>),
                               csrmvn_blocks,
                               csrmvn_threads,
                               0,
                               stream,
                               conj,
                               nnz,
                               max_rows,
                               static_cast<I*>(info->row_blocks),
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
            return rocsparse_status_not_implemented;
        }
    }
    else
    {
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
                                          T*                        y,
                                          bool                      force_conj)
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
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrmv"),
              trans,
              m,
              n,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)x,
              LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
              (const void*&)y);

    log_bench(handle,
              "./rocsparse-bench -f csrmv -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> ",
              "--alpha",
              LOG_BENCH_SCALAR_VALUE(handle, alpha_device_host),
              "--beta",
              LOG_BENCH_SCALAR_VALUE(handle, beta_device_host));

    // Check transpose
    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    if(descr->type != rocsparse_matrix_type_general
       && descr->type != rocsparse_matrix_type_triangular
       && descr->type != rocsparse_matrix_type_symmetric)
    {
        return rocsparse_status_not_implemented;
    }

    if(descr->type == rocsparse_matrix_type_symmetric
       || descr->type == rocsparse_matrix_type_triangular)
    {
        if(m != n)
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    if(m == 0 || n == 0)
    {
        if(nnz != 0)
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Quick return if possible
    if(m == 0 && n == 0 && nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Another quick return.
    if(m == 0 || n == 0 || nnz == 0)
    {
        // matrix never accessed however still need to update y vector
        rocsparse_int ysize = (trans == rocsparse_operation_none) ? m : n;
        if(ysize > 0)
        {
            if(y == nullptr && beta_device_host == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                hipLaunchKernelGGL((scale_array<256>),
                                   dim3((ysize - 1) / 256 + 1),
                                   dim3(256),
                                   0,
                                   handle->stream,
                                   ysize,
                                   y,
                                   beta_device_host);
            }
            else
            {
                hipLaunchKernelGGL((scale_array<256>),
                                   dim3((ysize - 1) / 256 + 1),
                                   dim3(256),
                                   0,
                                   handle->stream,
                                   ysize,
                                   y,
                                   *beta_device_host);
            }
        }

        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(alpha_device_host == nullptr || beta_device_host == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Another quick return.
    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    if(csr_row_ptr == nullptr || x == nullptr || y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (csr_col_ind == nullptr && csr_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(info == nullptr || info->csrmv_info == nullptr || trans != rocsparse_operation_none)
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
                                                     y,
                                                     force_conj);
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
                                                     y,
                                                     force_conj);
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
                                                              y,
                                                              force_conj);
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
                                                              y,
                                                              force_conj);
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
        TTYPE*                    y,                                                  \
        bool                      force_conj);

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
#undef INSTANTIATE

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
                                        y,                                  \
                                        false);                             \
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
