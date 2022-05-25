/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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

#include "rocsparse_csr2gebsr.hpp"
#include "definitions.h"
#include "utility.h"

#include "csr2gebsr_device.h"
#include <rocprim/rocprim.hpp>

#define launch_csr2gebsr_fast_kernel(direction, block_size, segment_size, wf_size)            \
    hipLaunchKernelGGL((csr2gebsr_fast_kernel<direction, block_size, segment_size, wf_size>), \
                       grid_size,                                                             \
                       block_size,                                                            \
                       0,                                                                     \
                       stream,                                                                \
                       m,                                                                     \
                       n,                                                                     \
                       mb,                                                                    \
                       nb,                                                                    \
                       row_block_dim,                                                         \
                       col_block_dim,                                                         \
                       csr_descr->base,                                                       \
                       csr_val,                                                               \
                       csr_row_ptr,                                                           \
                       csr_col_ind,                                                           \
                       bsr_descr->base,                                                       \
                       bsr_val,                                                               \
                       bsr_row_ptr,                                                           \
                       bsr_col_ind);

template <typename T>
rocsparse_status rocsparse_csr2gebsr_buffer_size_template(rocsparse_handle          handle,
                                                          rocsparse_direction       direction,
                                                          rocsparse_int             m,
                                                          rocsparse_int             n,
                                                          const rocsparse_mat_descr csr_descr,
                                                          const T*                  csr_val,
                                                          const rocsparse_int*      csr_row_ptr,
                                                          const rocsparse_int*      csr_col_ind,
                                                          rocsparse_int             row_block_dim,
                                                          rocsparse_int             col_block_dim,
                                                          size_t*                   p_buffer_size)
{
    //
    // Check for valid handle
    //
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    //
    // Check matrix descriptors
    //
    if(csr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Logging
    //
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2gebsr_buffer_size"),
              direction,
              m,
              n,
              csr_descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              row_block_dim,
              col_block_dim,
              (const void*&)p_buffer_size);

    log_bench(handle,
              "./rocsparse-bench -f csr2gebsr_buffer_size -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    //
    // Check direction
    //
    if(rocsparse_enum_utils::is_invalid(direction))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix sorting mode
    if(csr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    //
    // Check sizes
    //
    if(m < 0 || n < 0 || row_block_dim < 0 || col_block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    //
    // Check buffer size argument
    //
    if(p_buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    rocsparse_pointer_mode mode;
    rocsparse_get_pointer_mode(handle, &mode);

    //
    // Quick return if possible
    //
    if(m == 0 || n == 0 || row_block_dim == 0 || col_block_dim == 0)
    {
        //
        // Do not return 0 as buffer size
        //
        static constexpr size_t host_buffer_size = 4 * sizeof(rocsparse_int);
        if(mode == rocsparse_pointer_mode_host)
        {
            p_buffer_size[0] = host_buffer_size;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                hipMemcpy(p_buffer_size, &host_buffer_size, sizeof(size_t), hipMemcpyHostToDevice));
        }
        return rocsparse_status_success;
    }

    //
    // Check pointer arguments
    //
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

    {
        size_t host_buffer_size = 512 * sizeof(rocsparse_int);
        if(mode == rocsparse_pointer_mode_host)
        {
            p_buffer_size[0] = host_buffer_size;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                hipMemcpy(p_buffer_size, &host_buffer_size, sizeof(size_t), hipMemcpyHostToDevice));
        }
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csr2gebsr_template(rocsparse_handle          handle,
                                              rocsparse_direction       direction,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              const rocsparse_mat_descr csr_descr,
                                              const T*                  csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              const rocsparse_mat_descr bsr_descr,
                                              T*                        bsr_val,
                                              rocsparse_int*            bsr_row_ptr,
                                              rocsparse_int*            bsr_col_ind,
                                              rocsparse_int             row_block_dim,
                                              rocsparse_int             col_block_dim,
                                              void*                     p_buffer)
{
    //
    // Check for valid handle
    //
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    //
    // Check matrix descriptors
    //
    if(csr_descr == nullptr || bsr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Logging
    //
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2gebsr"),
              direction,
              m,
              n,
              csr_descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              bsr_descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              row_block_dim,
              col_block_dim,
              (const void*&)p_buffer);

    log_bench(handle, "./rocsparse-bench -f csr2gebsr -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    //
    // Check direction
    //
    if(rocsparse_enum_utils::is_invalid(direction))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix sorting mode
    if(csr_descr->storage_mode != rocsparse_storage_mode_sorted
       || bsr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    //
    // Check sizes
    //
    if(m < 0 || n < 0 || row_block_dim < 0 || col_block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    //
    // Quick return if possible
    //
    if(m == 0 || n == 0 || row_block_dim == 0 || col_block_dim == 0)
    {
        return rocsparse_status_success;
    }

    //
    // Check pointer arguments
    //
    if(csr_row_ptr == nullptr || bsr_row_ptr == nullptr || p_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((bsr_val == nullptr && bsr_col_ind != nullptr)
       || (bsr_val != nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    hipStream_t   stream = handle->stream;
    rocsparse_int mb     = (m + row_block_dim - 1) / row_block_dim;
    rocsparse_int nb     = (n + col_block_dim - 1) / col_block_dim;

    //
    // Stream
    //
    rocsparse_int start = 0;
    rocsparse_int end   = 0;
    RETURN_IF_HIP_ERROR(
        hipMemcpy(&end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(
        hipMemcpy(&start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));

    rocsparse_int nnzb = (end - start);

    if(nnzb != 0 && (bsr_val == nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Set bsr val to zero.
    //
    RETURN_IF_HIP_ERROR(hipMemset(bsr_val, 0, nnzb * row_block_dim * col_block_dim * sizeof(T)));
    if(row_block_dim == 1 && col_block_dim == 1)
    {
        constexpr rocsparse_int block_size = 256;
        rocsparse_int           grid_size  = mb / block_size;
        if(mb % block_size != 0)
        {
            grid_size++;
        }

        dim3 blocks(grid_size);
        dim3 threads(block_size);

        hipLaunchKernelGGL((csr2gebsr_kernel_bm1_bn1<block_size>),
                           blocks,
                           threads,
                           0,
                           stream,
                           m,
                           n,
                           mb,
                           nb,
                           csr_descr->base,
                           csr_val,
                           csr_row_ptr,
                           csr_col_ind,
                           bsr_descr->base,
                           bsr_val,
                           bsr_row_ptr,
                           bsr_col_ind);

        return rocsparse_status_success;
    }
    else if(row_block_dim == 1)
    {

        constexpr rocsparse_int block_size = 256;
        rocsparse_int           grid_size  = ((m + 1) + block_size - 1) / block_size;
        dim3                    blocks(grid_size);
        dim3                    threads(block_size);
        hipLaunchKernelGGL((csr2gebsr_kernel_bm1<block_size>),
                           blocks,
                           threads,
                           0,
                           stream,
                           m,
                           n,
                           mb,
                           nb,
                           csr_descr->base,
                           csr_val,
                           csr_row_ptr,
                           csr_col_ind,

                           direction,
                           bsr_descr->base,
                           bsr_val,
                           bsr_row_ptr,
                           bsr_col_ind,
                           row_block_dim,
                           col_block_dim);

        return rocsparse_status_success;
    }

    // Common case where BSR block dimension is small
    if(row_block_dim <= 32)
    {
        // A 64 thread wavefront is decomposed as:
        //      |    bank 0        bank 1       bank 2         bank 3
        // row 0|  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15 |
        // row 1| 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31 |
        // row 2| 32 33 34 35 | 36 37 38 39 | 40 41 42 43 | 44 45 46 47 |
        // row 3| 48 49 50 51 | 52 53 54 55 | 56 57 58 59 | 60 61 62 63 |
        //
        // Segments can be of size 4 (quarter row), 8 (half row), 16 (full row),
        // 32 (half wavefront), or 64 (full wavefront). We assign one segment per
        // BSR block row where the segment size matches the block dimension as
        // closely as possible while still being greater than or equal to the
        // block dimension.

        rocsparse_int block_size   = row_block_dim > 16 ? 32 : 16;
        rocsparse_int segment_size = row_block_dim;

        // round segment_size up to next power of 2
        segment_size--;
        segment_size |= segment_size >> 1;
        segment_size |= segment_size >> 2;
        segment_size |= segment_size >> 4;
        segment_size |= segment_size >> 8;
        segment_size |= segment_size >> 16;
        segment_size++;

        if(handle->wavefront_size == 32)
        {
            constexpr rocsparse_int wf_size = 32;

            rocsparse_int segments_per_wf              = wf_size / segment_size;
            rocsparse_int number_of_wf_segments_needed = (m + row_block_dim - 1) / row_block_dim;
            rocsparse_int number_of_wfs_needed
                = (number_of_wf_segments_needed + segments_per_wf - 1) / segments_per_wf;
            rocsparse_int grid_size
                = (wf_size * number_of_wfs_needed + block_size - 1) / block_size;

            if(direction == rocsparse_direction_row)
            {
                if(row_block_dim <= 2)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 16, 2, wf_size);
                }
                else if(row_block_dim <= 4)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 16, 4, wf_size);
                }
                else if(row_block_dim <= 8)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 16, 8, wf_size);
                }
                else if(row_block_dim <= 16)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 16, 16, wf_size);
                }
                else
                {
                    // (row_block_dim <= 32)
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 32, 32, wf_size);
                }
            }
            else
            {
                if(row_block_dim <= 2)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 16, 2, wf_size);
                }
                else if(row_block_dim <= 4)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 16, 4, wf_size);
                }
                else if(row_block_dim <= 8)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 16, 8, wf_size);
                }
                else if(row_block_dim <= 16)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 16, 16, wf_size);
                }
                else
                {
                    //(row_block_dim <= 32)
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 32, 32, wf_size);
                }
            }
        }
        else if(handle->wavefront_size == 64)
        {
            constexpr rocsparse_int wf_size = 64;

            rocsparse_int segments_per_wf              = wf_size / segment_size;
            rocsparse_int number_of_wf_segments_needed = (m + row_block_dim - 1) / row_block_dim;
            rocsparse_int number_of_wfs_needed
                = (number_of_wf_segments_needed + segments_per_wf - 1) / segments_per_wf;
            rocsparse_int grid_size
                = (wf_size * number_of_wfs_needed + block_size - 1) / block_size;

            if(direction == rocsparse_direction_row)
            {
                if(row_block_dim <= 2)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 16, 2, wf_size);
                }
                else if(row_block_dim <= 4)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 16, 4, wf_size);
                }
                else if(row_block_dim <= 8)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 16, 8, wf_size);
                }
                else if(row_block_dim <= 16)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 16, 16, wf_size);
                }
                else
                {
                    // (row_block_dim <= 32)
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_row, 32, 32, wf_size);
                }
            }
            else
            {
                if(row_block_dim <= 2)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 16, 2, wf_size);
                }
                else if(row_block_dim <= 4)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 16, 4, wf_size);
                }
                else if(row_block_dim <= 8)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 16, 8, wf_size);
                }
                else if(row_block_dim <= 16)
                {
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 16, 16, wf_size);
                }
                else
                {
                    // (row_block_dim <= 32)
                    launch_csr2gebsr_fast_kernel(rocsparse_direction_column, 32, 32, wf_size);
                }
            }
        }
        else
        {
            return rocsparse_status_arch_mismatch;
        }
    }
    // Uncommon (exceptional) case where BSR block dimension is large
    else
    {
        // Each segment handles a block row where each thread in the segment
        // can work on multiple rows in the block row (row_block_dim > segment_size)
        constexpr rocsparse_int block_size   = 32;
        constexpr rocsparse_int segment_size = 32;
        constexpr rocsparse_int wf_size      = 32;

        rocsparse_int rows_per_segment = (row_block_dim + segment_size - 1) / segment_size;
        rocsparse_int segments_per_wf  = wf_size / segment_size;
        rocsparse_int number_of_wf_segments_needed = (m + row_block_dim - 1) / row_block_dim;
        rocsparse_int number_of_wfs_needed
            = (number_of_wf_segments_needed + segments_per_wf - 1) / segments_per_wf;
        rocsparse_int grid_size = (wf_size * number_of_wfs_needed + block_size - 1) / block_size;

        size_t buffer_size
            = grid_size * block_size
              * (3 * rows_per_segment * sizeof(rocsparse_int) + rows_per_segment * sizeof(T));

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= buffer_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&temp_storage_ptr, buffer_size));
            temp_alloc = true;
        }

        char* temp = reinterpret_cast<char*>(temp_storage_ptr);

        rocsparse_int* temp1 = reinterpret_cast<rocsparse_int*>(temp);
        T*             temp2 = reinterpret_cast<T*>(
            temp + grid_size * block_size * 3 * rows_per_segment * sizeof(rocsparse_int));

        hipLaunchKernelGGL((csr2gebsr_general_kernel<block_size, segment_size, wf_size>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           handle->stream,
                           direction,
                           m,
                           n,
                           mb,
                           nb,
                           row_block_dim,
                           col_block_dim,
                           rows_per_segment,
                           csr_descr->base,
                           csr_val,
                           csr_row_ptr,
                           csr_col_ind,
                           bsr_descr->base,
                           bsr_val,
                           bsr_row_ptr,
                           bsr_col_ind,
                           temp1,
                           temp2);

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFree(temp_storage_ptr));
        }
    }

    return rocsparse_status_success;
}

#define launch_csr2gebsr_nnz_fast_kernel(block_size, segment_size, wf_size)            \
    hipLaunchKernelGGL((csr2gebsr_nnz_fast_kernel<block_size, segment_size, wf_size>), \
                       dim3(grid_size),                                                \
                       dim3(block_size),                                               \
                       0,                                                              \
                       handle->stream,                                                 \
                       m,                                                              \
                       n,                                                              \
                       mb,                                                             \
                       nb,                                                             \
                       row_block_dim,                                                  \
                       col_block_dim,                                                  \
                       csr_descr->base,                                                \
                       csr_row_ptr,                                                    \
                       csr_col_ind,                                                    \
                       bsr_descr->base,                                                \
                       bsr_row_ptr);

extern "C" rocsparse_status rocsparse_csr2gebsr_nnz(rocsparse_handle          handle,
                                                    rocsparse_direction       direction,
                                                    rocsparse_int             m,
                                                    rocsparse_int             n,
                                                    const rocsparse_mat_descr csr_descr,
                                                    const rocsparse_int*      csr_row_ptr,
                                                    const rocsparse_int*      csr_col_ind,
                                                    const rocsparse_mat_descr bsr_descr,
                                                    rocsparse_int*            bsr_row_ptr,
                                                    rocsparse_int             row_block_dim,
                                                    rocsparse_int             col_block_dim,
                                                    rocsparse_int*            bsr_nnz_devhost,
                                                    void*                     p_buffer)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check matrix descriptors
    if(csr_descr == nullptr || bsr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csr2gebsr_nnz",
              direction,
              m,
              n,
              csr_descr,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              bsr_descr,
              (const void*&)bsr_row_ptr,
              row_block_dim,
              col_block_dim,
              (const void*&)bsr_nnz_devhost,
              (const void*&)p_buffer);

    log_bench(handle, "./rocsparse-bench -f csr2gebsr_nnz", "--mtx <matrix.mtx>");

    //
    // Check direction
    //
    if(rocsparse_enum_utils::is_invalid(direction))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix sorting mode
    if(csr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }
    if(bsr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    //
    // Check sizes
    //
    if(m < 0 || n < 0 || row_block_dim < 0 || col_block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    //
    // Quick return if possible, before checking pointer arguments.
    //
    if(m == 0 || n == 0 || row_block_dim == 0 || col_block_dim == 0)
    {
        if(bsr_nnz_devhost != nullptr)
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(bsr_nnz_devhost, 0, sizeof(rocsparse_int), handle->stream));
            }
            else
            {
                *bsr_nnz_devhost = 0;
            }
        }

        return rocsparse_status_success;
    }

    //
    // Check pointer arguments
    //
    if(csr_row_ptr == nullptr || bsr_row_ptr == nullptr || bsr_nnz_devhost == nullptr
       || p_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(
            hipMemcpy(&end, &csr_row_ptr[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&start, &csr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        rocsparse_int nnz = (end - start);

        if(nnz != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    rocsparse_int mb = (m + row_block_dim - 1) / row_block_dim;
    rocsparse_int nb = (n + col_block_dim - 1) / col_block_dim;

    // If block dimension is one then BSR is equal to CSR
    if(row_block_dim == 1 && col_block_dim == 1)
    {
        constexpr rocsparse_int block_size = 256;
        rocsparse_int           grid_size  = ((m + 1) + block_size - 1) / block_size;
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL(csr2gebsr_nnz_kernel_bm1_bn1<block_size>,
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               handle->stream,
                               m,
                               csr_descr->base,
                               csr_row_ptr,
                               bsr_descr->base,
                               bsr_row_ptr,
                               bsr_nnz_devhost);
        }
        else
        {
            hipLaunchKernelGGL(csr2gebsr_nnz_kernel_bm1_bn1<block_size>,
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               handle->stream,
                               m,
                               csr_descr->base,
                               csr_row_ptr,
                               bsr_descr->base,
                               bsr_row_ptr);

            rocsparse_int hstart = 0;
            rocsparse_int hend   = 0;
            RETURN_IF_HIP_ERROR(
                hipMemcpy(&hend, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(
                hipMemcpy(&hstart, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
            *bsr_nnz_devhost = hend - hstart;
        }

        return rocsparse_status_success;
    }
    else if(row_block_dim == 1)
    {

        constexpr rocsparse_int block_size = 256;
        rocsparse_int           grid_size  = ((m + 1) + block_size - 1) / block_size;
        hipLaunchKernelGGL(csr2gebsr_nnz_kernel_bm1<block_size>,
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           handle->stream,
                           m,
                           csr_descr->base,
                           csr_row_ptr,
                           csr_col_ind,
                           bsr_descr->base,
                           bsr_row_ptr,
                           col_block_dim);

        // Perform inclusive scan on bsr row pointer array
        auto   op = rocprim::plus<rocsparse_int>();
        size_t temp_storage_size_bytes;
        RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                    temp_storage_size_bytes,
                                                    bsr_row_ptr,
                                                    bsr_row_ptr,
                                                    mb + 1,
                                                    op,
                                                    handle->stream));

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= temp_storage_size_bytes)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&temp_storage_ptr, temp_storage_size_bytes));
            temp_alloc = true;
        }

        RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(temp_storage_ptr,
                                                    temp_storage_size_bytes,
                                                    bsr_row_ptr,
                                                    bsr_row_ptr,
                                                    mb + 1,
                                                    op,
                                                    handle->stream));

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFree(temp_storage_ptr));
        }
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL(csr2gebsr_nnz_compute_nnz_total_kernel<1>,
                               dim3(1),
                               dim3(1),
                               0,
                               handle->stream,
                               mb,
                               bsr_row_ptr,
                               bsr_nnz_devhost);
        }
        else
        {
            rocsparse_int hstart = 0;
            rocsparse_int hend   = 0;
            RETURN_IF_HIP_ERROR(
                hipMemcpy(&hend, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(
                hipMemcpy(&hstart, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
            *bsr_nnz_devhost = hend - hstart;
        }

        return rocsparse_status_success;
    }

    // Common case where BSR block dimension is small
    if(row_block_dim <= 32)
    {
        // A 64 thread wavefront is decomposed as:
        //      |    bank 0        bank 1       bank 2         bank 3
        // row 0|  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15 |
        // row 1| 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31 |
        // row 2| 32 33 34 35 | 36 37 38 39 | 40 41 42 43 | 44 45 46 47 |
        // row 3| 48 49 50 51 | 52 53 54 55 | 56 57 58 59 | 60 61 62 63 |
        //
        // Segments can be of size 4 (quarter row), 8 (half row), 16 (full row),
        // or 32 (half wavefront). We assign one segment per BSR block row where
        // the segment size matches the block dimension as closely as possible
        // while still being greater than or equal to the block dimension.

        rocsparse_int block_size   = row_block_dim > 16 ? 32 : 16;
        rocsparse_int segment_size = row_block_dim;

        // round segment_size up to next power of 2
        segment_size--;
        segment_size |= segment_size >> 1;
        segment_size |= segment_size >> 2;
        segment_size |= segment_size >> 4;
        segment_size |= segment_size >> 8;
        segment_size |= segment_size >> 16;
        segment_size++;

        if(handle->wavefront_size == 32)
        {
            constexpr rocsparse_int wf_size = 32;

            rocsparse_int segments_per_wf              = wf_size / segment_size;
            rocsparse_int number_of_wf_segments_needed = (m + row_block_dim - 1) / row_block_dim;
            rocsparse_int number_of_wfs_needed
                = (number_of_wf_segments_needed + segments_per_wf - 1) / segments_per_wf;
            rocsparse_int grid_size
                = (wf_size * number_of_wfs_needed + block_size - 1) / block_size;

            if(row_block_dim <= 2)
            {
                launch_csr2gebsr_nnz_fast_kernel(16, 2, wf_size);
            }
            else if(row_block_dim <= 4)
            {
                launch_csr2gebsr_nnz_fast_kernel(16, 4, wf_size);
            }
            else if(row_block_dim <= 8)
            {
                launch_csr2gebsr_nnz_fast_kernel(16, 8, wf_size);
            }
            else if(row_block_dim <= 16)
            {
                launch_csr2gebsr_nnz_fast_kernel(16, 16, wf_size);
            }
            else
            {
                // (row_block_dim <= 32)
                launch_csr2gebsr_nnz_fast_kernel(32, 32, wf_size);
            }
        }
        else if(handle->wavefront_size == 64)
        {
            constexpr rocsparse_int wf_size = 64;

            rocsparse_int segments_per_wf              = wf_size / segment_size;
            rocsparse_int number_of_wf_segments_needed = (m + row_block_dim - 1) / row_block_dim;
            rocsparse_int number_of_wfs_needed
                = (number_of_wf_segments_needed + segments_per_wf - 1) / segments_per_wf;
            rocsparse_int grid_size
                = (wf_size * number_of_wfs_needed + block_size - 1) / block_size;

            if(row_block_dim <= 2)
            {
                launch_csr2gebsr_nnz_fast_kernel(16, 2, wf_size);
            }
            else if(row_block_dim <= 4)
            {
                launch_csr2gebsr_nnz_fast_kernel(16, 4, wf_size);
            }
            else if(row_block_dim <= 8)
            {
                launch_csr2gebsr_nnz_fast_kernel(16, 8, wf_size);
            }
            else if(row_block_dim <= 16)
            {
                launch_csr2gebsr_nnz_fast_kernel(16, 16, wf_size);
            }
            else
            {
                // (row_block_dim <= 32)
                launch_csr2gebsr_nnz_fast_kernel(32, 32, wf_size);
            }
        }
        else
        {
            return rocsparse_status_arch_mismatch;
        }
    }
    // Uncommon (exceptional) case where BSR block dimension is large
    else
    {
        // Each segment handles a block row where each thread in the segment
        // can work on multiple rows in the block row (row_block_dim > segment_size)
        constexpr rocsparse_int block_size   = 32;
        constexpr rocsparse_int segment_size = 32;
        constexpr rocsparse_int wf_size      = 32;

        rocsparse_int rows_per_segment = (row_block_dim + segment_size - 1) / segment_size;
        rocsparse_int segments_per_wf  = wf_size / segment_size;
        rocsparse_int number_of_wf_segments_needed = (m + row_block_dim - 1) / row_block_dim;
        rocsparse_int number_of_wfs_needed
            = (number_of_wf_segments_needed + segments_per_wf - 1) / segments_per_wf;
        rocsparse_int grid_size = (wf_size * number_of_wfs_needed + block_size - 1) / block_size;

        size_t buffer_size = grid_size * block_size * 2 * rows_per_segment * sizeof(rocsparse_int);

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= buffer_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&temp_storage_ptr, buffer_size));
            temp_alloc = true;
        }

        rocsparse_int* temp1 = (rocsparse_int*)temp_storage_ptr;

        hipLaunchKernelGGL((csr2gebsr_nnz_general_kernel<block_size, segment_size, wf_size>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           handle->stream,
                           m,
                           n,
                           mb,
                           nb,
                           row_block_dim,
                           col_block_dim,
                           rows_per_segment,
                           csr_descr->base,
                           csr_row_ptr,
                           csr_col_ind,
                           bsr_descr->base,
                           bsr_row_ptr,
                           temp1);

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFree(temp_storage_ptr));
        }
    }

    // Perform inclusive scan on bsr row pointer array
    auto   op = rocprim::plus<rocsparse_int>();
    size_t temp_storage_size_bytes;
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(
        nullptr, temp_storage_size_bytes, bsr_row_ptr, bsr_row_ptr, mb + 1, op, handle->stream));

    bool  temp_alloc       = false;
    void* temp_storage_ptr = nullptr;
    if(handle->buffer_size >= temp_storage_size_bytes)
    {
        temp_storage_ptr = handle->buffer;
        temp_alloc       = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&temp_storage_ptr, temp_storage_size_bytes));
        temp_alloc = true;
    }

    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(temp_storage_ptr,
                                                temp_storage_size_bytes,
                                                bsr_row_ptr,
                                                bsr_row_ptr,
                                                mb + 1,
                                                op,
                                                handle->stream));

    if(temp_alloc)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(temp_storage_ptr));
    }

    // Compute bsr_nnz
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL(csr2gebsr_nnz_compute_nnz_total_kernel<1>,
                           dim3(1),
                           dim3(1),
                           0,
                           handle->stream,
                           mb,
                           bsr_row_ptr,
                           bsr_nnz_devhost);
    }
    else
    {
        rocsparse_int hstart = 0;
        rocsparse_int hend   = 0;
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&hend, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&hstart, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        *bsr_nnz_devhost = hend - hstart;
    }

    return rocsparse_status_success;
}

//
// C INTERFACE
//
#define C_IMPL(NAME, TYPE)                                                       \
    rocsparse_status NAME##_buffer_size(rocsparse_handle          handle,        \
                                        rocsparse_direction       direction,     \
                                        rocsparse_int             m,             \
                                        rocsparse_int             n,             \
                                        const rocsparse_mat_descr csr_descr,     \
                                        const TYPE*               csr_val,       \
                                        const rocsparse_int*      csr_row_ptr,   \
                                        const rocsparse_int*      csr_col_ind,   \
                                        rocsparse_int             row_block_dim, \
                                        rocsparse_int             col_block_dim, \
                                        size_t*                   p_buffer_size) \
    {                                                                            \
        return rocsparse_csr2gebsr_buffer_size_template(handle,                  \
                                                        direction,               \
                                                        m,                       \
                                                        n,                       \
                                                        csr_descr,               \
                                                        csr_val,                 \
                                                        csr_row_ptr,             \
                                                        csr_col_ind,             \
                                                        row_block_dim,           \
                                                        col_block_dim,           \
                                                        p_buffer_size);          \
    }                                                                            \
                                                                                 \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,           \
                                     rocsparse_direction       direction,        \
                                     rocsparse_int             m,                \
                                     rocsparse_int             n,                \
                                     const rocsparse_mat_descr csr_descr,        \
                                     const TYPE*               csr_val,          \
                                     const rocsparse_int*      csr_row_ptr,      \
                                     const rocsparse_int*      csr_col_ind,      \
                                     const rocsparse_mat_descr bsr_descr,        \
                                     TYPE*                     bsr_val,          \
                                     rocsparse_int*            bsr_row_ptr,      \
                                     rocsparse_int*            bsr_col_ind,      \
                                     rocsparse_int             row_block_dim,    \
                                     rocsparse_int             col_block_dim,    \
                                     void*                     p_buffer)         \
                                                                                 \
    {                                                                            \
        return rocsparse_csr2gebsr_template(handle,                              \
                                            direction,                           \
                                            m,                                   \
                                            n,                                   \
                                            csr_descr,                           \
                                            csr_val,                             \
                                            csr_row_ptr,                         \
                                            csr_col_ind,                         \
                                            bsr_descr,                           \
                                            bsr_val,                             \
                                            bsr_row_ptr,                         \
                                            bsr_col_ind,                         \
                                            row_block_dim,                       \
                                            col_block_dim,                       \
                                            p_buffer);                           \
    }

C_IMPL(rocsparse_scsr2gebsr, float);
C_IMPL(rocsparse_dcsr2gebsr, double);
C_IMPL(rocsparse_ccsr2gebsr, rocsparse_float_complex);
C_IMPL(rocsparse_zcsr2gebsr, rocsparse_double_complex);
#undef C_IMPL
