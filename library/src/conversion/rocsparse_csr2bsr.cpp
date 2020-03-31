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
#include "rocsparse.h"

#include "rocsparse_csr2bsr.hpp"

#include <hip/hip_runtime_api.h>
#include <rocprim/rocprim.hpp>

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define launch_csr2bsr_nnz_fast_kernel(block_size, bsr_block_dim, segment_size)          \
    hipLaunchKernelGGL(csr2bsr_nnz_fast_kernel<block_size, bsr_block_dim, segment_size>, \
                       dim3(grid_size),                                                  \
                       dim3(block_size),                                                 \
                       0,                                                                \
                       handle->stream,                                                   \
                       m,                                                                \
                       n,                                                                \
                       mb,                                                               \
                       nb,                                                               \
                       csr_descr->base,                                                  \
                       csr_row_ptr,                                                      \
                       csr_col_ind,                                                      \
                       bsr_descr->base,                                                  \
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
    if(direction != rocsparse_direction_row && direction != rocsparse_direction_column)
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
    if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_nnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    rocsparse_int mb = (m + block_dim - 1) / block_dim;
    rocsparse_int nb = (n + block_dim - 1) / block_dim;

    // If block dimension is one then BSR is equal to CSR
    if(block_dim == 1)
    {
        constexpr rocsparse_int block_size = 256;
        rocsparse_int           grid_size  = ((m + 1) + block_size - 1) / block_size;
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL(csr2bsr_nnz_block_dim_equals_one_kernel,
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
            hipLaunchKernelGGL(csr2bsr_nnz_block_dim_equals_one_kernel,
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
            *bsr_nnz = hend - hstart;
        }

        return rocsparse_status_success;
    }

    // Common case where BSR block dimension is small
    if(block_dim <= 16)
    {
        // A 64 thread wavefront is decomposed as:
        //      |    bank 0        bank 1       bank 2         bank 3
        // row 0|  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15 |
        // row 1| 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31 |
        // row 2| 32 33 34 35 | 36 37 38 39 | 40 41 42 43 | 44 45 46 47 |
        // row 3| 48 49 50 51 | 52 53 54 55 | 56 57 58 59 | 60 61 62 63 |
        //
        // Segments can be of size 4 (quarter row), 8 (half row), or 16 (full row).
        // We assign one segment per BSR block row where the segment size matches the
        // block dimension as closely as possible while still being greater than or
        // equal to the block dimension.

        constexpr rocsparse_int wf_size      = 64;
        constexpr rocsparse_int block_size   = 1024;
        rocsparse_int           segment_size = block_dim;

        // round segment_size up to next power of 2
        segment_size--;
        segment_size |= segment_size >> 1;
        segment_size |= segment_size >> 2;
        segment_size |= segment_size >> 4;
        segment_size |= segment_size >> 8;
        segment_size |= segment_size >> 16;
        segment_size++;

        // smallest segment size allowed is 4 as wavefront banks are 4 threads
        if(segment_size == 2)
        {
            segment_size = 4;
        }

        rocsparse_int segments_per_wf              = wf_size / segment_size;
        rocsparse_int number_of_wf_segments_needed = (m + block_dim - 1) / block_dim;
        rocsparse_int number_of_wfs_needed
            = (number_of_wf_segments_needed + segments_per_wf - 1) / segments_per_wf;
        rocsparse_int grid_size = (wf_size * number_of_wfs_needed + block_size - 1) / block_size;

        if(block_dim == 2)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 2, 4);
        }
        else if(block_dim == 3)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 3, 4);
        }
        else if(block_dim == 4)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 4, 4);
        }
        else if(block_dim == 5)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 5, 8);
        }
        else if(block_dim == 6)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 6, 8);
        }
        else if(block_dim == 7)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 7, 8);
        }
        else if(block_dim == 8)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 8, 8);
        }
        else if(block_dim == 9)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 9, 16);
        }
        else if(block_dim == 10)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 10, 16);
        }
        else if(block_dim == 11)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 11, 16);
        }
        else if(block_dim == 12)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 12, 16);
        }
        else if(block_dim == 13)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 13, 16);
        }
        else if(block_dim == 14)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 14, 16);
        }
        else if(block_dim == 15)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 15, 16);
        }
        else if(block_dim == 16)
        {
            launch_csr2bsr_nnz_fast_kernel(block_size, 16, 16);
        }
    }
    // Uncommon (exceptional) case where BSR block dimension is large
    else
    {
        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= mb * block_dim * sizeof(rocsparse_int))
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                hipMalloc(&temp_storage_ptr, mb * block_dim * sizeof(rocsparse_int)));
            temp_alloc = true;
        }

        rocsparse_int block_size = block_dim;
        rocsparse_int grid_size  = (m + block_size - 1) / block_size;
        hipLaunchKernelGGL(csr2bsr_nnz_slow_kernel,
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           handle->stream,
                           m,
                           n,
                           nb,
                           csr_descr->base,
                           csr_row_ptr,
                           csr_col_ind,
                           block_dim,
                           bsr_descr->base,
                           bsr_row_ptr,
                           (rocsparse_int*)temp_storage_ptr);

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
        hipLaunchKernelGGL(csr2bsr_nnz_compute_nnz_total_kernel,
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
        rocsparse_int hstart = 0;
        rocsparse_int hend   = 0;
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&hend, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&hstart, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        *bsr_nnz = hend - hstart;
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
