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

#pragma once
#ifndef ROCSPARSE_CSR2CSR_COMPRESS_HPP
#define ROCSPARSE_CSR2CSR_COMPRESS_HPP

#include "csr2csr_compress_device.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_csr2csr_compress_template(rocsparse_handle          handle,
                                                     rocsparse_int             m,
                                                     rocsparse_int             n,
                                                     const rocsparse_mat_descr descr_A,
                                                     const T*                  csr_val_A,
                                                     const rocsparse_int*      csr_col_ind_A,
                                                     const rocsparse_int*      csr_row_ptr_A,
                                                     rocsparse_int             nnz_A,
                                                     const rocsparse_int*      nnz_per_row,
                                                     T*                        csr_val_C,
                                                     rocsparse_int*            csr_col_ind_C,
                                                     rocsparse_int*            csr_row_ptr_C,
                                                     T                         tol)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2csr_compress"),
              m,
              n,
              descr_A,
              (const void*&)csr_val_A,
              (const void*&)csr_col_ind_A,
              (const void*&)csr_row_ptr_A,
              nnz_A,
              (const void*&)nnz_per_row,
              (const void*&)csr_val_C,
              (const void*&)csr_col_ind_C,
              (const void*&)csr_row_ptr_C,
              tol);

    log_bench(
        handle, "./rocsparse-bench -f csr2csr_compress -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz_A < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check tolerance
    if(std::real(tol) < std::real(static_cast<T>(0)))
    {
        return rocsparse_status_invalid_value;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(nnz_per_row == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_val_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    constexpr rocsparse_int block_size = 1024;
    rocsparse_int           grid_size  = (m + block_size - 1) / block_size;

    // Copy nnz_per_row to csr_row_ptr_C array
    hipLaunchKernelGGL((csr2csr_fill_row_ptr_kernel<block_size>),
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       stream,
                       m,
                       descr_A->base,
                       nnz_per_row,
                       csr_row_ptr_C);

    // Perform inclusive scan on csr row pointer array
    auto   op = rocprim::plus<rocsparse_int>();
    size_t temp_storage_size_bytes;
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(
        nullptr, temp_storage_size_bytes, csr_row_ptr_C, csr_row_ptr_C, m + 1, op, stream));

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
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                m + 1,
                                                op,
                                                stream));

    // Mean number of elements per row in the input CSR matrix
    rocsparse_int mean_nnz_per_row = nnz_A / m;

    // A wavefront is divided into segments of size 2, 4, 8, 16, or 32 threads (or 64 in the case of 64
    // thread wavefronts) depending on the mean number of elements per CSR matrix row. Each row in the
    // matrix is then handled by a single segment.
    if(handle->wavefront_size == 32)
    {
        if(mean_nnz_per_row < 4)
        {
            constexpr rocsparse_int segments_per_block = block_size / 2;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 2, 32>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
        else if(mean_nnz_per_row < 8)
        {
            constexpr rocsparse_int segments_per_block = block_size / 4;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 4, 32>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
        else if(mean_nnz_per_row < 16)
        {
            constexpr rocsparse_int segments_per_block = block_size / 8;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 8, 32>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
        else if(mean_nnz_per_row < 32)
        {
            constexpr rocsparse_int segments_per_block = block_size / 16;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 16, 32>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
        else
        {
            constexpr rocsparse_int segments_per_block = block_size / 32;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 32, 32>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
    }
    else if(handle->wavefront_size == 64)
    {
        if(mean_nnz_per_row < 4)
        {
            constexpr rocsparse_int segments_per_block = block_size / 2;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 2, 64>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
        else if(mean_nnz_per_row < 8)
        {
            constexpr rocsparse_int segments_per_block = block_size / 4;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 4, 64>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
        else if(mean_nnz_per_row < 16)
        {
            constexpr rocsparse_int segments_per_block = block_size / 8;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 8, 64>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
        else if(mean_nnz_per_row < 32)
        {
            constexpr rocsparse_int segments_per_block = block_size / 16;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 16, 64>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
        else if(mean_nnz_per_row < 64)
        {
            constexpr rocsparse_int segments_per_block = block_size / 32;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 32, 64>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
        else
        {
            constexpr rocsparse_int segments_per_block = block_size / 64;
            grid_size = (m + segments_per_block - 1) / segments_per_block;

            hipLaunchKernelGGL((csr2csr_compress_kernel<T, block_size, segments_per_block, 64, 64>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               n,
                               descr_A->base,
                               csr_val_A,
                               csr_col_ind_A,
                               csr_row_ptr_A,
                               nnz_A,
                               nnz_per_row,
                               csr_val_C,
                               csr_col_ind_C,
                               csr_row_ptr_C,
                               tol);
        }
    }
    else
    {
        return rocsparse_status_arch_mismatch;
    }

    if(temp_alloc)
    {
        RETURN_IF_HIP_ERROR(hipFree(temp_storage_ptr));
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSR2CSR_COMPRESS_HPP
