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
#ifndef ROCSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_HPP
#define ROCSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_HPP

#include "csr2csr_compress_device.h"
#include "definitions.h"
#include "handle.h"
#include "prune_dense2csr_by_percentage_device.h"
#include "prune_dense2csr_device.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

template <rocsparse_int DIM_X, rocsparse_int DIM_Y, typename T>
__launch_bounds__(DIM_X* DIM_Y) __global__
    void prune_dense2csr_nnz_kernel_host_pointer(rocsparse_int m,
                                                 rocsparse_int n,
                                                 const T* __restrict__ A,
                                                 rocsparse_int lda,
                                                 T             threshold,
                                                 rocsparse_int* __restrict__ nnz_per_rows)
{
    prune_dense2csr_nnz_kernel<DIM_X, DIM_Y, T>(m, n, A, lda, threshold, nnz_per_rows);
}

template <rocsparse_int DIM_X, rocsparse_int DIM_Y, typename T>
__launch_bounds__(DIM_X* DIM_Y) __global__
    void prune_dense2csr_nnz_kernel_device_pointer(rocsparse_int m,
                                                   rocsparse_int n,
                                                   const T* __restrict__ A,
                                                   rocsparse_int lda,
                                                   const T*      threshold,
                                                   rocsparse_int* __restrict__ nnz_per_rows)
{
    prune_dense2csr_nnz_kernel<DIM_X, DIM_Y, T>(m, n, A, lda, *threshold, nnz_per_rows);
}

template <rocsparse_int NUMROWS_PER_BLOCK, rocsparse_int WF_SIZE, typename T>
__launch_bounds__(WF_SIZE* NUMROWS_PER_BLOCK) __global__
    void prune_dense2csr_kernel_device_pointer(rocsparse_index_base base,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               const T* __restrict__ A,
                                               rocsparse_int lda,
                                               const T*      threshold,
                                               T* __restrict__ csr_val,
                                               const rocsparse_int* __restrict__ csr_row_ptr,
                                               rocsparse_int* __restrict__ csr_col_ind)
{
    prune_dense2csr_kernel<NUMROWS_PER_BLOCK, WF_SIZE, T>(
        base, m, n, A, lda, *threshold, csr_val, csr_row_ptr, csr_col_ind);
}

template <typename T>
rocsparse_status
    rocsparse_prune_dense2csr_by_percentage_buffer_size_template(rocsparse_handle handle,
                                                                 rocsparse_int    m,
                                                                 rocsparse_int    n,
                                                                 const T*         A,
                                                                 rocsparse_int    lda,
                                                                 T                percentage,
                                                                 const rocsparse_mat_descr descr,
                                                                 const T*                  csr_val,
                                                                 const rocsparse_int* csr_row_ptr,
                                                                 const rocsparse_int* csr_col_ind,
                                                                 rocsparse_mat_info   info,
                                                                 size_t*              buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xprune_dense2csr_by_percentage_buffer_size"),
              m,
              n,
              (const void*&)A,
              lda,
              percentage,
              descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              info,
              (void*&)buffer_size);

    log_bench(handle,
              "./rocsparse-bench -f prune_dense2csr_by_percentage_buffer_size -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    *buffer_size = 2 * m * n * sizeof(T);

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status
    rocsparse_prune_dense2csr_nnz_by_percentage_template(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const T*                  A,
                                                         rocsparse_int             lda,
                                                         T                         percentage,
                                                         const rocsparse_mat_descr descr,
                                                         rocsparse_int*            csr_row_ptr,
                                                         rocsparse_int*     nnz_total_dev_host_ptr,
                                                         rocsparse_mat_info info,
                                                         void*              temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xprune_dense2csr_nnz_by_percentage"),
              m,
              n,
              (const void*&)A,
              lda,
              percentage,
              descr,
              (void*&)csr_row_ptr,
              (void*&)nnz_total_dev_host_ptr,
              info,
              (void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f prune_dense2csr_nnz_by_percentage -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check sizes
    if(m < 0 || n < 0 || lda < m || percentage < static_cast<T>(0.0)
       || percentage > static_cast<T>(100.0))
    {
        return rocsparse_status_invalid_size;
    }

    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        if(nullptr != nnz_total_dev_host_ptr)
        {
            rocsparse_pointer_mode mode;
            rocsparse_status       status = rocsparse_get_pointer_mode(handle, &mode);
            if(rocsparse_status_success != status)
            {
                return status;
            }

            constexpr rocsparse_int block_size = 1024;
            rocsparse_int           grid_size  = (m + block_size - 1) / block_size;
            hipLaunchKernelGGL((fill_row_ptr_kernel<block_size>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               descr->base,
                               csr_row_ptr);

            if(rocsparse_pointer_mode_device == mode)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(
                    nnz_total_dev_host_ptr, 0, sizeof(rocsparse_int), handle->stream));
            }
            else
            {
                *nnz_total_dev_host_ptr = 0;
            }
        }

        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(A == nullptr || csr_row_ptr == nullptr || nnz_total_dev_host_ptr == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    rocsparse_int nnz_A = m * n;
    rocsparse_int pos   = std::ceil(nnz_A * (percentage / 100)) - 1;
    pos                 = std::min(pos, nnz_A - 1);
    pos                 = std::max(pos, 0);

    T* output = reinterpret_cast<T*>(temp_buffer);

    // Compute absolute value of A and store in first half of output array
    {
        dim3 grid((nnz_A - 1) / 256 + 1);
        dim3 threads(256);

        hipLaunchKernelGGL((abs_kernel<256, T>), grid, threads, 0, stream, m, n, A, lda, output);
    }

    // Determine amount of temporary storage needed for rocprim sort and inclusive scan and allocate if necessary
    size_t temp_storage_size_bytes_sort = 0;
    size_t temp_storage_size_bytes_scan = 0;

    RETURN_IF_HIP_ERROR(rocprim::radix_sort_keys(
        nullptr, temp_storage_size_bytes_sort, output, (output + nnz_A), nnz_A));

    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                temp_storage_size_bytes_scan,
                                                csr_row_ptr,
                                                csr_row_ptr,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                handle->stream));

    size_t temp_storage_size_bytes
        = std::max(temp_storage_size_bytes_sort, temp_storage_size_bytes_scan);

    // Device buffer should be sufficient for rocprim in most cases
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

    // perform sort on first half of output array and store result in second half of output array
    rocprim::radix_sort_keys(
        temp_storage_ptr, temp_storage_size_bytes, output, (output + nnz_A), nnz_A);

    T* d_threshold = &output[nnz_A + pos];

    static constexpr int NNZ_DIM_X = 64;
    static constexpr int NNZ_DIM_Y = 16;

    dim3 grid((m - 1) / (NNZ_DIM_X * 4) + 1);
    dim3 threads(NNZ_DIM_X, NNZ_DIM_Y);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((prune_dense2csr_nnz_kernel_device_pointer<NNZ_DIM_X, NNZ_DIM_Y, T>),
                           grid,
                           threads,
                           0,
                           stream,
                           m,
                           n,
                           A,
                           lda,
                           d_threshold,
                           &csr_row_ptr[1]);
    }
    else
    {
        T h_threshold = 0;
        RETURN_IF_HIP_ERROR(hipMemcpy(&h_threshold, d_threshold, sizeof(T), hipMemcpyDeviceToHost));
        hipLaunchKernelGGL((prune_dense2csr_nnz_kernel_host_pointer<NNZ_DIM_X, NNZ_DIM_Y, T>),
                           grid,
                           threads,
                           0,
                           stream,
                           m,
                           n,
                           A,
                           lda,
                           h_threshold,
                           &csr_row_ptr[1]);
    }

    // Store threshold at first entry in output array
    RETURN_IF_HIP_ERROR(hipMemcpy(output, d_threshold, sizeof(T), hipMemcpyDeviceToDevice));

    // Compute csr_row_ptr with the right index base.
    rocsparse_int first_value = descr->base;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        csr_row_ptr, &first_value, sizeof(rocsparse_int), hipMemcpyHostToDevice, handle->stream));

    // Perform actual inclusive sum
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(temp_storage_ptr,
                                                temp_storage_size_bytes,
                                                csr_row_ptr,
                                                csr_row_ptr,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                handle->stream));
    // Free rocprim buffer, if allocated
    if(temp_alloc == true)
    {
        RETURN_IF_HIP_ERROR(hipFree(temp_storage_ptr));
    }

    // Extract nnz_total_dev_host_ptr
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        dim3 grid(1);
        dim3 threads(1);
        hipLaunchKernelGGL(nnz_total_device_kernel,
                           grid,
                           threads,
                           0,
                           stream,
                           m,
                           csr_row_ptr,
                           nnz_total_dev_host_ptr);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz_total_dev_host_ptr, &csr_row_ptr[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        *nnz_total_dev_host_ptr -= descr->base;
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_prune_dense2csr_by_percentage_template(rocsparse_handle handle,
                                                                  rocsparse_int    m,
                                                                  rocsparse_int    n,
                                                                  const T*         A,
                                                                  rocsparse_int    lda,
                                                                  T                percentage,
                                                                  const rocsparse_mat_descr descr,
                                                                  T*                        csr_val,
                                                                  const rocsparse_int* csr_row_ptr,
                                                                  rocsparse_int*       csr_col_ind,
                                                                  rocsparse_mat_info   info,
                                                                  void*                temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xprune_dense2csr_by_percentage"),
              m,
              n,
              (const void*&)A,
              lda,
              percentage,
              descr,
              (void*&)csr_val,
              (const void*&)csr_row_ptr,
              (void*&)csr_col_ind,
              info,
              (void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f prune_dense2csr_by_percentage -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check sizes
    if(m < 0 || n < 0 || lda < m || percentage < static_cast<T>(0.0)
       || percentage > static_cast<T>(100.0))
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(A == nullptr || csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    T* d_threshold = &(reinterpret_cast<T*>(temp_buffer))[0];

    static constexpr rocsparse_int data_ratio = sizeof(T) / sizeof(float);

    if(handle->wavefront_size == 32)
    {
        static constexpr rocsparse_int WF_SIZE         = 32;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        dim3                           blocks((m - 1) / NROWS_PER_BLOCK + 1);
        dim3                           threads(WF_SIZE * NROWS_PER_BLOCK);

        hipLaunchKernelGGL((prune_dense2csr_kernel_device_pointer<NROWS_PER_BLOCK, WF_SIZE, T>),
                           blocks,
                           threads,
                           0,
                           stream,
                           descr->base,
                           m,
                           n,
                           A,
                           lda,
                           d_threshold,
                           csr_val,
                           csr_row_ptr,
                           csr_col_ind);
    }
    else
    {
        static constexpr rocsparse_int WF_SIZE         = 64;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        dim3                           blocks((m - 1) / NROWS_PER_BLOCK + 1);
        dim3                           threads(WF_SIZE * NROWS_PER_BLOCK);

        hipLaunchKernelGGL((prune_dense2csr_kernel_device_pointer<NROWS_PER_BLOCK, WF_SIZE, T>),
                           blocks,
                           threads,
                           0,
                           stream,
                           descr->base,
                           m,
                           n,
                           A,
                           lda,
                           d_threshold,
                           csr_val,
                           csr_row_ptr,
                           csr_col_ind);
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_HPP
