/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_csr2bsr.hpp"
#include "definitions.h"
#include "utility.h"

#include "csr2bsr_device.h"

#include <rocprim/rocprim.hpp>

#define launch_csr2bsr_2_32_kernel(T, direction, block_size, segment_size)         \
    hipLaunchKernelGGL((csr2bsr_2_32_kernel<direction, block_size, segment_size>), \
                       grid_size,                                                  \
                       block_size,                                                 \
                       0,                                                          \
                       stream,                                                     \
                       m,                                                          \
                       n,                                                          \
                       mb,                                                         \
                       nb,                                                         \
                       block_dim,                                                  \
                       csr_descr->base,                                            \
                       csr_val,                                                    \
                       csr_row_ptr,                                                \
                       csr_col_ind,                                                \
                       bsr_descr->base,                                            \
                       bsr_val,                                                    \
                       bsr_row_ptr,                                                \
                       bsr_col_ind);

#define launch_csr2bsr_33_64_kernel(T, direction, block_size, rows_per_segment)         \
    hipLaunchKernelGGL((csr2bsr_33_64_kernel<direction, block_size, rows_per_segment>), \
                       grid_size,                                                       \
                       block_size,                                                      \
                       0,                                                               \
                       stream,                                                          \
                       m,                                                               \
                       n,                                                               \
                       mb,                                                              \
                       nb,                                                              \
                       block_dim,                                                       \
                       csr_descr->base,                                                 \
                       csr_val,                                                         \
                       csr_row_ptr,                                                     \
                       csr_col_ind,                                                     \
                       bsr_descr->base,                                                 \
                       bsr_val,                                                         \
                       bsr_row_ptr,                                                     \
                       bsr_col_ind);

template <typename T>
rocsparse_status rocsparse_csr2bsr_template(rocsparse_handle          handle,
                                            rocsparse_direction       direction,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const rocsparse_mat_descr csr_descr,
                                            const T*                  csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            rocsparse_int             block_dim,
                                            const rocsparse_mat_descr bsr_descr,
                                            T*                        bsr_val,
                                            rocsparse_int*            bsr_row_ptr,
                                            rocsparse_int*            bsr_col_ind)
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
              replaceX<T>("rocsparse_Xcsr2bsr"),
              direction,
              m,
              n,
              csr_descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              block_dim,
              bsr_descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind);

    log_bench(handle, "./rocsparse-bench -f csr2bsr -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check direction
    if(rocsparse_enum_utils::is_invalid(direction))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || n < 0 || block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || block_dim == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || bsr_row_ptr == nullptr)
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

    if(csr_val == nullptr && csr_col_ind == nullptr)
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

    rocsparse_int mb = (m + block_dim - 1) / block_dim;
    rocsparse_int nb = (n + block_dim - 1) / block_dim;

    rocsparse_int start = 0;
    rocsparse_int end   = 0;

    RETURN_IF_HIP_ERROR(
        hipMemcpy(&end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(
        hipMemcpy(&start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));

    rocsparse_int nnzb = (end - start);

    if(bsr_val == nullptr && bsr_col_ind == nullptr)
    {
        if(nnzb != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    if(bsr_val != nullptr)
    {
        hipMemset(bsr_val, 0, nnzb * block_dim * block_dim * sizeof(T));
    }

    // Stream
    hipStream_t stream = handle->stream;

    if(block_dim == 1)
    {
        constexpr rocsparse_int block_size = 256;
        rocsparse_int           grid_size  = mb / block_size;
        if(mb % block_size != 0)
        {
            grid_size++;
        }

        dim3 blocks(grid_size);
        dim3 threads(block_size);

        hipLaunchKernelGGL((csr2bsr_block_dim_equals_one_kernel<block_size>),
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

    // Common case where BSR block dimension is small
    if(block_dim <= 64)
    {
        // A 32 thread wavefront is decomposed as:
        //      |    bank 0        bank 1       bank 2         bank 3
        // row 0|  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15 |
        // row 1| 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31 |
        //
        // Segments can be of size 4 (quarter row), 8 (half row), 16 (full row),
        // or 32 (wavefront). We assign one segment per BSR block row where the
        // segment size matches the block dimension as closely as possible while
        // still being greater than or equal to the block dimension.
        rocsparse_int block_size   = block_dim > 16 ? 32 : 16;
        rocsparse_int segment_size = block_dim == 1 ? 2 : block_dim;

        // round segment_size up to next power of 2
        segment_size--;
        segment_size |= segment_size >> 1;
        segment_size |= segment_size >> 2;
        segment_size |= segment_size >> 4;
        segment_size |= segment_size >> 8;
        segment_size |= segment_size >> 16;
        segment_size++;

        if(block_dim > 32)
        {
            segment_size = block_size;
        }

        rocsparse_int segments_per_block = block_size / segment_size;
        rocsparse_int grid_size          = (mb + segments_per_block - 1) / segments_per_block;

        if(direction == rocsparse_direction_row)
        {
            if(block_dim <= 2)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_row, 16, 2);
            }
            else if(block_dim <= 4)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_row, 16, 4);
            }
            else if(block_dim <= 8)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_row, 16, 8);
            }
            else if(block_dim <= 16)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_row, 16, 16);
            }
            else if(block_dim <= 32)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_row, 32, 32);
            }
            else
            {
                // (block_dim <= 64)
                launch_csr2bsr_33_64_kernel(T, rocsparse_direction_row, 32, 2);
            }
        }
        else
        {
            if(block_dim <= 2)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_column, 16, 2);
            }
            else if(block_dim <= 4)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_column, 16, 4);
            }
            else if(block_dim <= 8)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_column, 16, 8);
            }
            else if(block_dim <= 16)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_column, 16, 16);
            }
            else if(block_dim <= 32)
            {
                launch_csr2bsr_2_32_kernel(T, rocsparse_direction_column, 32, 32);
            }
            else
            {
                // (block_dim <= 64)
                launch_csr2bsr_33_64_kernel(T, rocsparse_direction_column, 32, 2);
            }
        }
    }
    else
    {
        // Use a blocksize of 32 to handle each block row
        constexpr rocsparse_int block_size       = 32;
        rocsparse_int           rows_per_segment = (block_dim + block_size - 1) / block_size;

        rocsparse_int grid_size = mb;

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
            RETURN_IF_HIP_ERROR(hipMalloc(&temp_storage_ptr, buffer_size));
            temp_alloc = true;
        }

        char* temp = reinterpret_cast<char*>(temp_storage_ptr);

        rocsparse_int* temp1 = reinterpret_cast<rocsparse_int*>(temp);
        T*             temp2 = reinterpret_cast<T*>(
            temp + grid_size * block_size * 3 * rows_per_segment * sizeof(rocsparse_int));

        hipLaunchKernelGGL((csr2bsr_65_inf_kernel<block_size>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           handle->stream,
                           direction,
                           m,
                           n,
                           mb,
                           nb,
                           block_dim,
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
            RETURN_IF_HIP_ERROR(hipFree(temp_storage_ptr));
        }
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define launch_csr2bsr_nnz_2_32_kernel(block_size, segment_size)            \
    hipLaunchKernelGGL((csr2bsr_nnz_2_32_kernel<block_size, segment_size>), \
                       dim3(grid_size),                                     \
                       dim3(block_size),                                    \
                       0,                                                   \
                       handle->stream,                                      \
                       m,                                                   \
                       n,                                                   \
                       mb,                                                  \
                       nb,                                                  \
                       block_dim,                                           \
                       csr_descr->base,                                     \
                       csr_row_ptr,                                         \
                       csr_col_ind,                                         \
                       bsr_descr->base,                                     \
                       bsr_row_ptr);

#define launch_csr2bsr_nnz_33_64_kernel(block_size, rows_per_segment)            \
    hipLaunchKernelGGL((csr2bsr_nnz_33_64_kernel<block_size, rows_per_segment>), \
                       dim3(grid_size),                                          \
                       dim3(block_size),                                         \
                       0,                                                        \
                       handle->stream,                                           \
                       m,                                                        \
                       n,                                                        \
                       mb,                                                       \
                       nb,                                                       \
                       block_dim,                                                \
                       csr_descr->base,                                          \
                       csr_row_ptr,                                              \
                       csr_col_ind,                                              \
                       bsr_descr->base,                                          \
                       bsr_row_ptr);

extern "C" rocsparse_status rocsparse_csr2bsr_nnz(rocsparse_handle          handle,
                                                  rocsparse_direction       direction,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr csr_descr,
                                                  const rocsparse_int*      csr_row_ptr,
                                                  const rocsparse_int*      csr_col_ind,
                                                  rocsparse_int             block_dim,
                                                  const rocsparse_mat_descr bsr_descr,
                                                  rocsparse_int*            bsr_row_ptr,
                                                  rocsparse_int*            bsr_nnz)
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
              "rocsparse_csr2bsr_nnz",
              direction,
              m,
              n,
              csr_descr,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              block_dim,
              bsr_descr,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_nnz);

    log_bench(handle, "./rocsparse-bench -f csr2bsr_nnz", "--mtx <matrix.mtx>");

    // Check direction
    if(rocsparse_enum_utils::is_invalid(direction))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || n < 0 || block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || block_dim == 0)
    {
        if(bsr_nnz != nullptr)
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(bsr_nnz, 0, sizeof(rocsparse_int), handle->stream));
            }
            else
            {
                *bsr_nnz = 0;
            }
        }

        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || bsr_row_ptr == nullptr || bsr_nnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    rocsparse_int mb = (m + block_dim - 1) / block_dim;
    rocsparse_int nb = (n + block_dim - 1) / block_dim;

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

    // If block dimension is one then BSR is equal to CSR
    if(block_dim == 1)
    {
        constexpr rocsparse_int block_size = 256;
        rocsparse_int           grid_size  = ((m + 1) + block_size - 1) / block_size;
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL(csr2bsr_nnz_block_dim_equals_one_kernel<block_size>,
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               handle->stream,
                               m,
                               csr_descr->base,
                               csr_row_ptr,
                               bsr_descr->base,
                               bsr_row_ptr,
                               bsr_nnz);
        }
        else
        {
            hipLaunchKernelGGL(csr2bsr_nnz_block_dim_equals_one_kernel<block_size>,
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               handle->stream,
                               m,
                               csr_descr->base,
                               csr_row_ptr,
                               bsr_descr->base,
                               bsr_row_ptr);

            rocsparse_int start = 0;
            rocsparse_int end   = 0;
            RETURN_IF_HIP_ERROR(
                hipMemcpy(&end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(
                hipMemcpy(&start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
            *bsr_nnz = end - start;
        }

        return rocsparse_status_success;
    }

    // Common case where BSR block dimension is small
    if(block_dim <= 64)
    {
        // A 32 thread wavefront is decomposed as:
        //      |    bank 0        bank 1       bank 2         bank 3
        // row 0|  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15 |
        // row 1| 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31 |
        //
        // Segments can be of size 4 (quarter row), 8 (half row), 16 (full row),
        // or 32 (wavefront). We assign one segment per BSR block row where the
        // segment size matches the block dimension as closely as possible while
        // still being greater than or equal to the block dimension.
        rocsparse_int block_size   = block_dim > 16 ? 32 : 16;
        rocsparse_int segment_size = block_dim == 1 ? 2 : block_dim;

        // round segment_size up to next power of 2
        segment_size--;
        segment_size |= segment_size >> 1;
        segment_size |= segment_size >> 2;
        segment_size |= segment_size >> 4;
        segment_size |= segment_size >> 8;
        segment_size |= segment_size >> 16;
        segment_size++;

        if(block_dim > 32)
        {
            segment_size = block_size;
        }

        rocsparse_int segments_per_block = block_size / segment_size;
        rocsparse_int grid_size          = (mb + segments_per_block - 1) / segments_per_block;

        if(block_dim <= 2)
        {
            launch_csr2bsr_nnz_2_32_kernel(16, 2);
        }
        else if(block_dim <= 4)
        {
            launch_csr2bsr_nnz_2_32_kernel(16, 4);
        }
        else if(block_dim <= 8)
        {
            launch_csr2bsr_nnz_2_32_kernel(16, 8);
        }
        else if(block_dim <= 16)
        {
            launch_csr2bsr_nnz_2_32_kernel(16, 16);
        }
        else if(block_dim <= 32)
        {
            launch_csr2bsr_nnz_2_32_kernel(32, 32);
        }
        else
        {
            // (block_dim <= 64)
            launch_csr2bsr_nnz_33_64_kernel(32, 2);
        }
    }
    else
    {
        // Use a blocksize of 32 to handle each block row
        constexpr rocsparse_int block_size       = 32;
        rocsparse_int           rows_per_segment = (block_dim + block_size - 1) / block_size;

        rocsparse_int grid_size = mb;

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
            RETURN_IF_HIP_ERROR(hipMalloc(&temp_storage_ptr, buffer_size));
            temp_alloc = true;
        }

        rocsparse_int* temp1 = (rocsparse_int*)temp_storage_ptr;

        hipLaunchKernelGGL((csr2bsr_nnz_65_inf_kernel<block_size>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           handle->stream,
                           m,
                           n,
                           mb,
                           nb,
                           block_dim,
                           rows_per_segment,
                           csr_descr->base,
                           csr_row_ptr,
                           csr_col_ind,
                           bsr_descr->base,
                           bsr_row_ptr,
                           temp1);

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(hipFree(temp_storage_ptr));
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
        RETURN_IF_HIP_ERROR(hipMalloc(&temp_storage_ptr, temp_storage_size_bytes));
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
        RETURN_IF_HIP_ERROR(hipFree(temp_storage_ptr));
    }

    // Compute bsr_nnz
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL(csr2bsr_nnz_compute_nnz_total_kernel<1>,
                           dim3(1),
                           dim3(1),
                           0,
                           handle->stream,
                           mb,
                           bsr_row_ptr,
                           bsr_nnz);
    }
    else
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        *bsr_nnz = end - start;
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsr2bsr(rocsparse_handle          handle,
                                               rocsparse_direction       direction,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const rocsparse_mat_descr csr_descr,
                                               const float*              csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_int             block_dim,
                                               const rocsparse_mat_descr bsr_descr,
                                               float*                    bsr_val,
                                               rocsparse_int*            bsr_row_ptr,
                                               rocsparse_int*            bsr_col_ind)
{
    return rocsparse_csr2bsr_template(handle,
                                      direction,
                                      m,
                                      n,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      block_dim,
                                      bsr_descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind);
}

extern "C" rocsparse_status rocsparse_dcsr2bsr(rocsparse_handle          handle,
                                               rocsparse_direction       direction,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const rocsparse_mat_descr csr_descr,
                                               const double*             csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_int             block_dim,
                                               const rocsparse_mat_descr bsr_descr,
                                               double*                   bsr_val,
                                               rocsparse_int*            bsr_row_ptr,
                                               rocsparse_int*            bsr_col_ind)
{
    return rocsparse_csr2bsr_template(handle,
                                      direction,
                                      m,
                                      n,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      block_dim,
                                      bsr_descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind);
}

extern "C" rocsparse_status rocsparse_ccsr2bsr(rocsparse_handle               handle,
                                               rocsparse_direction            direction,
                                               rocsparse_int                  m,
                                               rocsparse_int                  n,
                                               const rocsparse_mat_descr      csr_descr,
                                               const rocsparse_float_complex* csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               const rocsparse_int*           csr_col_ind,
                                               rocsparse_int                  block_dim,
                                               const rocsparse_mat_descr      bsr_descr,
                                               rocsparse_float_complex*       bsr_val,
                                               rocsparse_int*                 bsr_row_ptr,
                                               rocsparse_int*                 bsr_col_ind)
{
    return rocsparse_csr2bsr_template(handle,
                                      direction,
                                      m,
                                      n,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      block_dim,
                                      bsr_descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind);
}

extern "C" rocsparse_status rocsparse_zcsr2bsr(rocsparse_handle                handle,
                                               rocsparse_direction             direction,
                                               rocsparse_int                   m,
                                               rocsparse_int                   n,
                                               const rocsparse_mat_descr       csr_descr,
                                               const rocsparse_double_complex* csr_val,
                                               const rocsparse_int*            csr_row_ptr,
                                               const rocsparse_int*            csr_col_ind,
                                               rocsparse_int                   block_dim,
                                               const rocsparse_mat_descr       bsr_descr,
                                               rocsparse_double_complex*       bsr_val,
                                               rocsparse_int*                  bsr_row_ptr,
                                               rocsparse_int*                  bsr_col_ind)
{
    return rocsparse_csr2bsr_template(handle,
                                      direction,
                                      m,
                                      n,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      block_dim,
                                      bsr_descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind);
}
