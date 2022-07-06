/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_prune_csr2csr.hpp"
#include "definitions.h"
#include "utility.h"

#include "csr2csr_compress_device.h"
#include "nnz_compress_device.h"
#include <rocprim/rocprim.hpp>

template <rocsparse_int BLOCK_SIZE,
          rocsparse_int SEGMENTS_PER_BLOCK,
          rocsparse_int SEGMENT_SIZE,
          rocsparse_int WF_SIZE,
          typename T,
          typename U>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void nnz_compress_kernel(rocsparse_int        m,
                             rocsparse_index_base idx_base_A,
                             const T* __restrict__ csr_val_A,
                             const rocsparse_int* __restrict__ csr_row_ptr_A,
                             rocsparse_int* __restrict__ nnz_per_row,
                             U threshold_device_host)
{
    auto threshold = load_scalar_device_host(threshold_device_host);
    nnz_compress_device<BLOCK_SIZE, SEGMENTS_PER_BLOCK, SEGMENT_SIZE, WF_SIZE>(
        m, idx_base_A, csr_val_A, csr_row_ptr_A, nnz_per_row, threshold);
}

template <rocsparse_int BLOCK_SIZE,
          rocsparse_int SEGMENTS_PER_BLOCK,
          rocsparse_int SEGMENT_SIZE,
          rocsparse_int WF_SIZE,
          typename T,
          typename U>
__launch_bounds__(BLOCK_SIZE) ROCSPARSE_KERNEL
    void csr2csr_compress_kernel(rocsparse_int        m,
                                 rocsparse_int        n,
                                 rocsparse_index_base idx_base_A,
                                 const T* __restrict__ csr_val_A,
                                 const rocsparse_int* __restrict__ csr_row_ptr_A,
                                 const rocsparse_int* __restrict__ csr_col_ind_A,
                                 rocsparse_int        nnz_A,
                                 rocsparse_index_base idx_base_C,
                                 T* __restrict__ csr_val_C,
                                 const rocsparse_int* __restrict__ csr_row_ptr_C,
                                 rocsparse_int* __restrict__ csr_col_ind_C,
                                 U threshold_device_host)
{
    auto threshold = load_scalar_device_host(threshold_device_host);
    csr2csr_compress_device<BLOCK_SIZE, SEGMENTS_PER_BLOCK, SEGMENT_SIZE, WF_SIZE>(m,
                                                                                   n,
                                                                                   idx_base_A,
                                                                                   csr_val_A,
                                                                                   csr_row_ptr_A,
                                                                                   csr_col_ind_A,
                                                                                   nnz_A,
                                                                                   idx_base_C,
                                                                                   csr_val_C,
                                                                                   csr_row_ptr_C,
                                                                                   csr_col_ind_C,
                                                                                   threshold);
}

template <rocsparse_int BLOCK_SIZE, rocsparse_int SEGMENT_SIZE, rocsparse_int WF_SIZE, typename T>
void nnz_compress(rocsparse_handle     handle,
                  rocsparse_int        m,
                  rocsparse_index_base idx_base_A,
                  const T* __restrict__ csr_val_A,
                  const rocsparse_int* __restrict__ csr_row_ptr_A,
                  rocsparse_int* __restrict__ nnz_per_row,
                  const T* threshold)
{
    constexpr rocsparse_int SEGMENTS_PER_BLOCK = BLOCK_SIZE / SEGMENT_SIZE;
    rocsparse_int           grid_size          = (m + SEGMENTS_PER_BLOCK - 1) / SEGMENTS_PER_BLOCK;

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL(
            (nnz_compress_kernel<BLOCK_SIZE, SEGMENTS_PER_BLOCK, SEGMENT_SIZE, WF_SIZE>),
            dim3(grid_size),
            dim3(BLOCK_SIZE),
            0,
            handle->stream,
            m,
            idx_base_A,
            csr_val_A,
            csr_row_ptr_A,
            nnz_per_row,
            threshold);
    }
    else
    {
        hipLaunchKernelGGL(
            (nnz_compress_kernel<BLOCK_SIZE, SEGMENTS_PER_BLOCK, SEGMENT_SIZE, WF_SIZE>),
            dim3(grid_size),
            dim3(BLOCK_SIZE),
            0,
            handle->stream,
            m,
            idx_base_A,
            csr_val_A,
            csr_row_ptr_A,
            nnz_per_row,
            *threshold);
    }
}

template <rocsparse_int BLOCK_SIZE, rocsparse_int SEGMENT_SIZE, rocsparse_int WF_SIZE, typename T>
void csr2csr_compress(rocsparse_handle     handle,
                      rocsparse_int        m,
                      rocsparse_int        n,
                      rocsparse_index_base idx_base_A,
                      const T* __restrict__ csr_val_A,
                      const rocsparse_int* __restrict__ csr_row_ptr_A,
                      const rocsparse_int* __restrict__ csr_col_ind_A,
                      rocsparse_int        nnz_A,
                      rocsparse_index_base idx_base_C,
                      T* __restrict__ csr_val_C,
                      const rocsparse_int* __restrict__ csr_row_ptr_C,
                      rocsparse_int* __restrict__ csr_col_ind_C,
                      const T* threshold)
{
    constexpr rocsparse_int SEGMENTS_PER_BLOCK = BLOCK_SIZE / SEGMENT_SIZE;
    rocsparse_int           grid_size          = (m + SEGMENTS_PER_BLOCK - 1) / SEGMENTS_PER_BLOCK;

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL(
            (csr2csr_compress_kernel<BLOCK_SIZE, SEGMENTS_PER_BLOCK, SEGMENT_SIZE, WF_SIZE>),
            dim3(grid_size),
            dim3(BLOCK_SIZE),
            0,
            handle->stream,
            m,
            n,
            idx_base_A,
            csr_val_A,
            csr_row_ptr_A,
            csr_col_ind_A,
            nnz_A,
            idx_base_C,
            csr_val_C,
            csr_row_ptr_C,
            csr_col_ind_C,
            threshold);
    }
    else
    {
        hipLaunchKernelGGL(
            (csr2csr_compress_kernel<BLOCK_SIZE, SEGMENTS_PER_BLOCK, SEGMENT_SIZE, WF_SIZE>),
            dim3(grid_size),
            dim3(BLOCK_SIZE),
            0,
            handle->stream,
            m,
            n,
            idx_base_A,
            csr_val_A,
            csr_row_ptr_A,
            csr_col_ind_A,
            nnz_A,
            idx_base_C,
            csr_val_C,
            csr_row_ptr_C,
            csr_col_ind_C,
            *threshold);
    }
}

template <typename T>
rocsparse_status rocsparse_prune_csr2csr_buffer_size_template(rocsparse_handle          handle,
                                                              rocsparse_int             m,
                                                              rocsparse_int             n,
                                                              rocsparse_int             nnz_A,
                                                              const rocsparse_mat_descr csr_descr_A,
                                                              const T*                  csr_val_A,
                                                              const rocsparse_int* csr_row_ptr_A,
                                                              const rocsparse_int* csr_col_ind_A,
                                                              const T*             threshold,
                                                              const rocsparse_mat_descr csr_descr_C,
                                                              const T*                  csr_val_C,
                                                              const rocsparse_int* csr_row_ptr_C,
                                                              const rocsparse_int* csr_col_ind_C,
                                                              size_t*              buffer_size)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xprune_csr2csr_buffer_size"),
              m,
              n,
              nnz_A,
              csr_descr_A,
              (const void*&)csr_val_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              (const void*&)threshold,
              csr_descr_C,
              (const void*&)csr_val_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)csr_col_ind_C,
              (const void*&)buffer_size);

    log_bench(handle,
              "./rocsparse-bench -f prune_csr2csr_buffer_size -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(csr_descr_A == nullptr || csr_descr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix sorting mode
    if(csr_descr_A->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }
    if(csr_descr_C->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz_A < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(csr_row_ptr_A == nullptr || csr_row_ptr_C == nullptr || buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_A == nullptr && csr_col_ind_A != nullptr)
       || (csr_val_A != nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_C == nullptr && csr_col_ind_C != nullptr)
       || (csr_val_C != nullptr && csr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && (csr_val_A == nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *buffer_size = 4;

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_prune_csr2csr_nnz_template(rocsparse_handle          handle,
                                                      rocsparse_int             m,
                                                      rocsparse_int             n,
                                                      rocsparse_int             nnz_A,
                                                      const rocsparse_mat_descr csr_descr_A,
                                                      const T*                  csr_val_A,
                                                      const rocsparse_int*      csr_row_ptr_A,
                                                      const rocsparse_int*      csr_col_ind_A,
                                                      const T*                  threshold,
                                                      const rocsparse_mat_descr csr_descr_C,
                                                      rocsparse_int*            csr_row_ptr_C,
                                                      rocsparse_int* nnz_total_dev_host_ptr,
                                                      void*          temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xprune_csr2csr_nnz"),
              m,
              n,
              nnz_A,
              csr_descr_A,
              (const void*&)csr_val_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              (const void*&)threshold,
              csr_descr_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)nnz_total_dev_host_ptr,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f prune_csr2csr_nnz -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(csr_descr_A == nullptr || csr_descr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix sorting mode
    if(csr_descr_A->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }
    if(csr_descr_C->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz_A < 0)
    {
        return rocsparse_status_invalid_size;
    }

    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0 || nnz_A == 0)
    {
        if(nnz_total_dev_host_ptr != nullptr && csr_row_ptr_C != nullptr)
        {
            rocsparse_pointer_mode mode;
            rocsparse_status       status = rocsparse_get_pointer_mode(handle, &mode);
            if(rocsparse_status_success != status)
            {
                return status;
            }

            constexpr rocsparse_int block_size = 1024;
            rocsparse_int           grid_size  = (m + block_size - 1) / block_size;
            hipLaunchKernelGGL((fill_row_ptr_device<block_size>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               m,
                               csr_descr_C->base,
                               csr_row_ptr_C);

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
    if(csr_row_ptr_A == nullptr || csr_row_ptr_C == nullptr || nnz_total_dev_host_ptr == nullptr
       || threshold == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_A == nullptr && csr_col_ind_A != nullptr)
       || (csr_val_A != nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && (csr_val_A == nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    constexpr rocsparse_int block_size = 1024;

    // Mean number of elements per row in the input CSR matrix
    rocsparse_int mean_nnz_per_row = nnz_A / m;

    // A wavefront is divided into segments of size 2, 4, 8, 16, or 32 threads (or 64 in the case of 64
    // thread wavefronts) depending on the mean number of elements per CSR matrix row. Each row in the
    // matrix is then handled by a single segment.
    if(handle->wavefront_size == 32)
    {
        if(mean_nnz_per_row < 4)
        {
            nnz_compress<block_size, 2, 32>(handle,
                                            m,
                                            csr_descr_A->base,
                                            csr_val_A,
                                            csr_row_ptr_A,
                                            &csr_row_ptr_C[1],
                                            threshold);
        }
        else if(mean_nnz_per_row < 8)
        {
            nnz_compress<block_size, 4, 32>(handle,
                                            m,
                                            csr_descr_A->base,
                                            csr_val_A,
                                            csr_row_ptr_A,
                                            &csr_row_ptr_C[1],
                                            threshold);
        }
        else if(mean_nnz_per_row < 16)
        {
            nnz_compress<block_size, 8, 32>(handle,
                                            m,
                                            csr_descr_A->base,
                                            csr_val_A,
                                            csr_row_ptr_A,
                                            &csr_row_ptr_C[1],
                                            threshold);
        }
        else if(mean_nnz_per_row < 32)
        {
            nnz_compress<block_size, 16, 32>(handle,
                                             m,
                                             csr_descr_A->base,
                                             csr_val_A,
                                             csr_row_ptr_A,
                                             &csr_row_ptr_C[1],
                                             threshold);
        }
        else
        {
            nnz_compress<block_size, 32, 32>(handle,
                                             m,
                                             csr_descr_A->base,
                                             csr_val_A,
                                             csr_row_ptr_A,
                                             &csr_row_ptr_C[1],
                                             threshold);
        }
    }
    else if(handle->wavefront_size == 64)
    {
        if(mean_nnz_per_row < 4)
        {
            nnz_compress<block_size, 2, 64>(handle,
                                            m,
                                            csr_descr_A->base,
                                            csr_val_A,
                                            csr_row_ptr_A,
                                            &csr_row_ptr_C[1],
                                            threshold);
        }
        else if(mean_nnz_per_row < 8)
        {
            nnz_compress<block_size, 4, 64>(handle,
                                            m,
                                            csr_descr_A->base,
                                            csr_val_A,
                                            csr_row_ptr_A,
                                            &csr_row_ptr_C[1],
                                            threshold);
        }
        else if(mean_nnz_per_row < 16)
        {
            nnz_compress<block_size, 8, 64>(handle,
                                            m,
                                            csr_descr_A->base,
                                            csr_val_A,
                                            csr_row_ptr_A,
                                            &csr_row_ptr_C[1],
                                            threshold);
        }
        else if(mean_nnz_per_row < 32)
        {
            nnz_compress<block_size, 16, 64>(handle,
                                             m,
                                             csr_descr_A->base,
                                             csr_val_A,
                                             csr_row_ptr_A,
                                             &csr_row_ptr_C[1],
                                             threshold);
        }
        else if(mean_nnz_per_row < 64)
        {
            nnz_compress<block_size, 32, 64>(handle,
                                             m,
                                             csr_descr_A->base,
                                             csr_val_A,
                                             csr_row_ptr_A,
                                             &csr_row_ptr_C[1],
                                             threshold);
        }
        else
        {
            nnz_compress<block_size, 64, 64>(handle,
                                             m,
                                             csr_descr_A->base,
                                             csr_val_A,
                                             csr_row_ptr_A,
                                             &csr_row_ptr_C[1],
                                             threshold);
        }
    }
    else
    {
        return rocsparse_status_arch_mismatch;
    }

    // Compute csr_row_ptr_C with the right index base.
    rocsparse_int first_value = csr_descr_C->base;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        csr_row_ptr_C, &first_value, sizeof(rocsparse_int), hipMemcpyHostToDevice, handle->stream));

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
        RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&temp_storage_ptr, temp_storage_size_bytes));
        temp_alloc = true;
    }

    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(temp_storage_ptr,
                                                temp_storage_size_bytes,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                m + 1,
                                                op,
                                                stream));

    // compute nnz_total_dev_host_ptr
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((compute_nnz_from_row_ptr_array_kernel<1>),
                           dim3(1),
                           dim3(1),
                           0,
                           stream,
                           m,
                           csr_row_ptr_C,
                           nnz_total_dev_host_ptr);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(nnz_total_dev_host_ptr,
                                           &csr_row_ptr_C[m],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        *nnz_total_dev_host_ptr -= csr_descr_C->base;
    }

    if(temp_alloc)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(temp_storage_ptr));
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_prune_csr2csr_template(rocsparse_handle          handle,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  rocsparse_int             nnz_A,
                                                  const rocsparse_mat_descr csr_descr_A,
                                                  const T*                  csr_val_A,
                                                  const rocsparse_int*      csr_row_ptr_A,
                                                  const rocsparse_int*      csr_col_ind_A,
                                                  const T*                  threshold,
                                                  const rocsparse_mat_descr csr_descr_C,
                                                  T*                        csr_val_C,
                                                  const rocsparse_int*      csr_row_ptr_C,
                                                  rocsparse_int*            csr_col_ind_C,
                                                  void*                     temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xprune_csr2csr"),
              m,
              n,
              nnz_A,
              csr_descr_A,
              (const void*&)csr_val_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              (const void*&)threshold,
              csr_descr_C,
              (const void*&)csr_val_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)csr_col_ind_C,
              (const void*&)temp_buffer);

    log_bench(
        handle, "./rocsparse-bench -f prune_csr2csr -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(csr_descr_A == nullptr || csr_descr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix sorting mode
    if(csr_descr_A->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }
    if(csr_descr_C->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz_A < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr_A == nullptr || csr_row_ptr_C == nullptr || threshold == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_A == nullptr && csr_col_ind_A != nullptr)
       || (csr_val_A != nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_C == nullptr && csr_col_ind_C != nullptr)
       || (csr_val_C != nullptr && csr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && (csr_val_A == nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_val_C == nullptr && csr_col_ind_C == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr_C[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                           &csr_row_ptr_C[0],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        rocsparse_int nnz = (end - start);

        if(nnz != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    constexpr rocsparse_int block_size = 1024;

    // Mean number of elements per row in the input CSR matrix
    rocsparse_int mean_nnz_per_row = nnz_A / m;

    // A wavefront is divided into segments of size 2, 4, 8, 16, or 32 threads (or 64 in the case of 64
    // thread wavefronts) depending on the mean number of elements per CSR matrix row. Each row in the
    // matrix is then handled by a single segment.
    if(handle->wavefront_size == 32)
    {
        if(mean_nnz_per_row < 4)
        {
            csr2csr_compress<block_size, 2, 32>(handle,
                                                m,
                                                n,
                                                csr_descr_A->base,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                nnz_A,
                                                csr_descr_C->base,
                                                csr_val_C,
                                                csr_row_ptr_C,
                                                csr_col_ind_C,
                                                threshold);
        }
        else if(mean_nnz_per_row < 8)
        {
            csr2csr_compress<block_size, 4, 32>(handle,
                                                m,
                                                n,
                                                csr_descr_A->base,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                nnz_A,
                                                csr_descr_C->base,
                                                csr_val_C,
                                                csr_row_ptr_C,
                                                csr_col_ind_C,
                                                threshold);
        }
        else if(mean_nnz_per_row < 16)
        {
            csr2csr_compress<block_size, 8, 32>(handle,
                                                m,
                                                n,
                                                csr_descr_A->base,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                nnz_A,
                                                csr_descr_C->base,
                                                csr_val_C,
                                                csr_row_ptr_C,
                                                csr_col_ind_C,
                                                threshold);
        }
        else if(mean_nnz_per_row < 32)
        {
            csr2csr_compress<block_size, 16, 32>(handle,
                                                 m,
                                                 n,
                                                 csr_descr_A->base,
                                                 csr_val_A,
                                                 csr_row_ptr_A,
                                                 csr_col_ind_A,
                                                 nnz_A,
                                                 csr_descr_C->base,
                                                 csr_val_C,
                                                 csr_row_ptr_C,
                                                 csr_col_ind_C,
                                                 threshold);
        }
        else
        {
            csr2csr_compress<block_size, 32, 32>(handle,
                                                 m,
                                                 n,
                                                 csr_descr_A->base,
                                                 csr_val_A,
                                                 csr_row_ptr_A,
                                                 csr_col_ind_A,
                                                 nnz_A,
                                                 csr_descr_C->base,
                                                 csr_val_C,
                                                 csr_row_ptr_C,
                                                 csr_col_ind_C,
                                                 threshold);
        }
    }
    else if(handle->wavefront_size == 64)
    {
        if(mean_nnz_per_row < 4)
        {
            csr2csr_compress<block_size, 2, 64>(handle,
                                                m,
                                                n,
                                                csr_descr_A->base,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                nnz_A,
                                                csr_descr_C->base,
                                                csr_val_C,
                                                csr_row_ptr_C,
                                                csr_col_ind_C,
                                                threshold);
        }
        else if(mean_nnz_per_row < 8)
        {
            csr2csr_compress<block_size, 4, 64>(handle,
                                                m,
                                                n,
                                                csr_descr_A->base,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                nnz_A,
                                                csr_descr_C->base,
                                                csr_val_C,
                                                csr_row_ptr_C,
                                                csr_col_ind_C,
                                                threshold);
        }
        else if(mean_nnz_per_row < 16)
        {
            csr2csr_compress<block_size, 8, 64>(handle,
                                                m,
                                                n,
                                                csr_descr_A->base,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                nnz_A,
                                                csr_descr_C->base,
                                                csr_val_C,
                                                csr_row_ptr_C,
                                                csr_col_ind_C,
                                                threshold);
        }
        else if(mean_nnz_per_row < 32)
        {
            csr2csr_compress<block_size, 16, 64>(handle,
                                                 m,
                                                 n,
                                                 csr_descr_A->base,
                                                 csr_val_A,
                                                 csr_row_ptr_A,
                                                 csr_col_ind_A,
                                                 nnz_A,
                                                 csr_descr_C->base,
                                                 csr_val_C,
                                                 csr_row_ptr_C,
                                                 csr_col_ind_C,
                                                 threshold);
        }
        else if(mean_nnz_per_row < 64)
        {
            csr2csr_compress<block_size, 32, 64>(handle,
                                                 m,
                                                 n,
                                                 csr_descr_A->base,
                                                 csr_val_A,
                                                 csr_row_ptr_A,
                                                 csr_col_ind_A,
                                                 nnz_A,
                                                 csr_descr_C->base,
                                                 csr_val_C,
                                                 csr_row_ptr_C,
                                                 csr_col_ind_C,
                                                 threshold);
        }
        else
        {
            csr2csr_compress<block_size, 64, 64>(handle,
                                                 m,
                                                 n,
                                                 csr_descr_A->base,
                                                 csr_val_A,
                                                 csr_row_ptr_A,
                                                 csr_col_ind_A,
                                                 nnz_A,
                                                 csr_descr_C->base,
                                                 csr_val_C,
                                                 csr_row_ptr_C,
                                                 csr_col_ind_C,
                                                 threshold);
        }
    }
    else
    {
        return rocsparse_status_arch_mismatch;
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status
    rocsparse_sprune_csr2csr_buffer_size(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         rocsparse_int             nnz_A,
                                         const rocsparse_mat_descr csr_descr_A,
                                         const float*              csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         const rocsparse_int*      csr_col_ind_A,
                                         const float*              threshold,
                                         const rocsparse_mat_descr csr_descr_C,
                                         const float*              csr_val_C,
                                         const rocsparse_int*      csr_row_ptr_C,
                                         const rocsparse_int*      csr_col_ind_C,
                                         size_t*                   buffer_size)
{
    return rocsparse_prune_csr2csr_buffer_size_template(handle,
                                                        m,
                                                        n,
                                                        nnz_A,
                                                        csr_descr_A,
                                                        csr_val_A,
                                                        csr_row_ptr_A,
                                                        csr_col_ind_A,
                                                        threshold,
                                                        csr_descr_C,
                                                        csr_val_C,
                                                        csr_row_ptr_C,
                                                        csr_col_ind_C,
                                                        buffer_size);
}

extern "C" rocsparse_status
    rocsparse_dprune_csr2csr_buffer_size(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         rocsparse_int             nnz_A,
                                         const rocsparse_mat_descr csr_descr_A,
                                         const double*             csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         const rocsparse_int*      csr_col_ind_A,
                                         const double*             threshold,
                                         const rocsparse_mat_descr csr_descr_C,
                                         const double*             csr_val_C,
                                         const rocsparse_int*      csr_row_ptr_C,
                                         const rocsparse_int*      csr_col_ind_C,
                                         size_t*                   buffer_size)
{
    return rocsparse_prune_csr2csr_buffer_size_template(handle,
                                                        m,
                                                        n,
                                                        nnz_A,
                                                        csr_descr_A,
                                                        csr_val_A,
                                                        csr_row_ptr_A,
                                                        csr_col_ind_A,
                                                        threshold,
                                                        csr_descr_C,
                                                        csr_val_C,
                                                        csr_row_ptr_C,
                                                        csr_col_ind_C,
                                                        buffer_size);
}

extern "C" rocsparse_status rocsparse_sprune_csr2csr_nnz(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         rocsparse_int             nnz_A,
                                                         const rocsparse_mat_descr csr_descr_A,
                                                         const float*              csr_val_A,
                                                         const rocsparse_int*      csr_row_ptr_A,
                                                         const rocsparse_int*      csr_col_ind_A,
                                                         const float*              threshold,
                                                         const rocsparse_mat_descr csr_descr_C,
                                                         rocsparse_int*            csr_row_ptr_C,
                                                         rocsparse_int* nnz_total_dev_host_ptr,
                                                         void*          temp_buffer)
{
    return rocsparse_prune_csr2csr_nnz_template(handle,
                                                m,
                                                n,
                                                nnz_A,
                                                csr_descr_A,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                threshold,
                                                csr_descr_C,
                                                csr_row_ptr_C,
                                                nnz_total_dev_host_ptr,
                                                temp_buffer);
}

extern "C" rocsparse_status rocsparse_dprune_csr2csr_nnz(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         rocsparse_int             nnz_A,
                                                         const rocsparse_mat_descr csr_descr_A,
                                                         const double*             csr_val_A,
                                                         const rocsparse_int*      csr_row_ptr_A,
                                                         const rocsparse_int*      csr_col_ind_A,
                                                         const double*             threshold,
                                                         const rocsparse_mat_descr csr_descr_C,
                                                         rocsparse_int*            csr_row_ptr_C,
                                                         rocsparse_int* nnz_total_dev_host_ptr,
                                                         void*          temp_buffer)
{
    return rocsparse_prune_csr2csr_nnz_template(handle,
                                                m,
                                                n,
                                                nnz_A,
                                                csr_descr_A,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                threshold,
                                                csr_descr_C,
                                                csr_row_ptr_C,
                                                nnz_total_dev_host_ptr,
                                                temp_buffer);
}

extern "C" rocsparse_status rocsparse_sprune_csr2csr(rocsparse_handle          handle,
                                                     rocsparse_int             m,
                                                     rocsparse_int             n,
                                                     rocsparse_int             nnz_A,
                                                     const rocsparse_mat_descr csr_descr_A,
                                                     const float*              csr_val_A,
                                                     const rocsparse_int*      csr_row_ptr_A,
                                                     const rocsparse_int*      csr_col_ind_A,
                                                     const float*              threshold,
                                                     const rocsparse_mat_descr csr_descr_C,
                                                     float*                    csr_val_C,
                                                     const rocsparse_int*      csr_row_ptr_C,
                                                     rocsparse_int*            csr_col_ind_C,
                                                     void*                     temp_buffer)
{
    return rocsparse_prune_csr2csr_template(handle,
                                            m,
                                            n,
                                            nnz_A,
                                            csr_descr_A,
                                            csr_val_A,
                                            csr_row_ptr_A,
                                            csr_col_ind_A,
                                            threshold,
                                            csr_descr_C,
                                            csr_val_C,
                                            csr_row_ptr_C,
                                            csr_col_ind_C,
                                            temp_buffer);
}

extern "C" rocsparse_status rocsparse_dprune_csr2csr(rocsparse_handle          handle,
                                                     rocsparse_int             m,
                                                     rocsparse_int             n,
                                                     rocsparse_int             nnz_A,
                                                     const rocsparse_mat_descr csr_descr_A,
                                                     const double*             csr_val_A,
                                                     const rocsparse_int*      csr_row_ptr_A,
                                                     const rocsparse_int*      csr_col_ind_A,
                                                     const double*             threshold,
                                                     const rocsparse_mat_descr csr_descr_C,
                                                     double*                   csr_val_C,
                                                     const rocsparse_int*      csr_row_ptr_C,
                                                     rocsparse_int*            csr_col_ind_C,
                                                     void*                     temp_buffer)
{
    return rocsparse_prune_csr2csr_template(handle,
                                            m,
                                            n,
                                            nnz_A,
                                            csr_descr_A,
                                            csr_val_A,
                                            csr_row_ptr_A,
                                            csr_col_ind_A,
                                            threshold,
                                            csr_descr_C,
                                            csr_val_C,
                                            csr_row_ptr_C,
                                            csr_col_ind_C,
                                            temp_buffer);
}
