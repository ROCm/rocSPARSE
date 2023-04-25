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

#include "rocsparse_prune_dense2csr.hpp"
#include "definitions.h"
#include "utility.h"

#include "csr2csr_compress_device.h"
#include "prune_dense2csr_device.h"
#include <rocprim/rocprim.hpp>

template <rocsparse_int DIM_X, rocsparse_int DIM_Y, typename T, typename U>
__launch_bounds__(DIM_X* DIM_Y) ROCSPARSE_KERNEL
    void prune_dense2csr_nnz_kernel(rocsparse_int m,
                                    rocsparse_int n,
                                    const T* __restrict__ A,
                                    rocsparse_int lda,
                                    U             threshold_device_host,
                                    rocsparse_int* __restrict__ nnz_per_rows)
{
    auto threshold = load_scalar_device_host(threshold_device_host);
    prune_dense2csr_nnz_device<DIM_X, DIM_Y>(m, n, A, lda, threshold, nnz_per_rows);
}

template <rocsparse_int NUMROWS_PER_BLOCK, rocsparse_int WF_SIZE, typename T, typename U>
__launch_bounds__(WF_SIZE* NUMROWS_PER_BLOCK) ROCSPARSE_KERNEL
    void prune_dense2csr_kernel(rocsparse_index_base base,
                                rocsparse_int        m,
                                rocsparse_int        n,
                                const T* __restrict__ A,
                                rocsparse_int lda,
                                U             threshold_device_host,
                                T* __restrict__ csr_val,
                                const rocsparse_int* __restrict__ csr_row_ptr,
                                rocsparse_int* __restrict__ csr_col_ind)
{
    auto threshold = load_scalar_device_host(threshold_device_host);
    prune_dense2csr_device<NUMROWS_PER_BLOCK, WF_SIZE>(
        base, m, n, A, lda, threshold, csr_val, csr_row_ptr, csr_col_ind);
}

template <typename T>
rocsparse_status rocsparse_prune_dense2csr_buffer_size_template(rocsparse_handle          handle,
                                                                rocsparse_int             m,
                                                                rocsparse_int             n,
                                                                const T*                  A,
                                                                rocsparse_int             lda,
                                                                const T*                  threshold,
                                                                const rocsparse_mat_descr descr,
                                                                const T*                  csr_val,
                                                                const rocsparse_int* csr_row_ptr,
                                                                const rocsparse_int* csr_col_ind,
                                                                size_t*              buffer_size)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check matrix descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xprune_dense2csr_buffer_size"),
              m,
              n,
              (const void*&)A,
              lda,
              (const void*&)threshold,
              descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)buffer_size);

    log_bench(handle,
              "./rocsparse-bench -f prune_dense2csr_buffer_size -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || lda < m)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(A == nullptr || threshold == nullptr || csr_row_ptr == nullptr || buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *buffer_size = 4;

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_prune_dense2csr_nnz_template(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const T*                  A,
                                                        rocsparse_int             lda,
                                                        const T*                  threshold,
                                                        const rocsparse_mat_descr descr,
                                                        rocsparse_int*            csr_row_ptr,
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
              replaceX<T>("rocsparse_Xprune_dense2csr_nnz"),
              m,
              n,
              (const void*&)A,
              lda,
              (const void*&)threshold,
              descr,
              (const void*&)csr_row_ptr,
              (const void*&)nnz_total_dev_host_ptr,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f prune_dense2csr_nnz -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || lda < m)
    {
        return rocsparse_status_invalid_size;
    }

    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        if(nnz_total_dev_host_ptr != nullptr && csr_row_ptr != nullptr)
        {
            hipLaunchKernelGGL((set_array_to_value<256>),
                               dim3(m / 256 + 1),
                               dim3(256),
                               0,
                               stream,
                               (m + 1),
                               csr_row_ptr,
                               static_cast<rocsparse_int>(descr->base));

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
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
    if(A == nullptr || threshold == nullptr || csr_row_ptr == nullptr
       || nnz_total_dev_host_ptr == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    static constexpr int NNZ_DIM_X = 64;
    static constexpr int NNZ_DIM_Y = 16;
    rocsparse_int        blocks    = (m - 1) / (NNZ_DIM_X * 4) + 1;

    dim3 grid(blocks);
    dim3 threads(NNZ_DIM_X, NNZ_DIM_Y);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((prune_dense2csr_nnz_kernel<NNZ_DIM_X, NNZ_DIM_Y, T>),
                           grid,
                           threads,
                           0,
                           stream,
                           m,
                           n,
                           A,
                           lda,
                           threshold,
                           &csr_row_ptr[1]);
    }
    else
    {
        hipLaunchKernelGGL((prune_dense2csr_nnz_kernel<NNZ_DIM_X, NNZ_DIM_Y, T>),
                           grid,
                           threads,
                           0,
                           stream,
                           m,
                           n,
                           A,
                           lda,
                           *threshold,
                           &csr_row_ptr[1]);
    }

    // Compute csr_row_ptr with the right index base.
    rocsparse_int first_value = descr->base;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        csr_row_ptr, &first_value, sizeof(rocsparse_int), hipMemcpyHostToDevice, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    // Obtain rocprim buffer size
    size_t temp_storage_bytes = 0;
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                temp_storage_bytes,
                                                csr_row_ptr,
                                                csr_row_ptr,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                handle->stream));

    // Get rocprim buffer
    bool  d_temp_alloc;
    void* d_temp_storage;

    // Device buffer should be sufficient for rocprim in most cases
    if(handle->buffer_size >= temp_storage_bytes)
    {
        d_temp_storage = handle->buffer;
        d_temp_alloc   = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&d_temp_storage, temp_storage_bytes, handle->stream));
        d_temp_alloc = true;
    }

    // Perform actual inclusive sum
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(d_temp_storage,
                                                temp_storage_bytes,
                                                csr_row_ptr,
                                                csr_row_ptr,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                handle->stream));

    // Free rocprim buffer, if allocated
    if(d_temp_alloc == true)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(d_temp_storage, handle->stream));
    }

    // Extract nnz_total_dev_host_ptr
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL(nnz_total_device_kernel,
                           dim3(1),
                           dim3(1),
                           0,
                           stream,
                           m,
                           csr_row_ptr,
                           nnz_total_dev_host_ptr);
    }
    else
    {
        rocsparse_int start, end;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &csr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        *nnz_total_dev_host_ptr = end - start;
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_prune_dense2csr_template(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    rocsparse_int             n,
                                                    const T*                  A,
                                                    rocsparse_int             lda,
                                                    const T*                  threshold,
                                                    const rocsparse_mat_descr descr,
                                                    T*                        csr_val,
                                                    const rocsparse_int*      csr_row_ptr,
                                                    rocsparse_int*            csr_col_ind,
                                                    void*                     temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xprune_dense2csr"),
              m,
              n,
              (const void*&)A,
              lda,
              (const void*&)threshold,
              descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)temp_buffer);

    log_bench(
        handle, "./rocsparse-bench -f prune_dense2csr -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || lda < m)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(A == nullptr || threshold == nullptr || csr_row_ptr == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_val == nullptr && csr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &csr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        rocsparse_int nnz = (end - start);

        if(nnz != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Stream
    hipStream_t stream = handle->stream;

    static constexpr rocsparse_int data_ratio = sizeof(T) / sizeof(float);

    if(handle->wavefront_size == 32)
    {
        static constexpr rocsparse_int WF_SIZE         = 32;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        dim3                           blocks((m - 1) / NROWS_PER_BLOCK + 1);
        dim3                           threads(WF_SIZE * NROWS_PER_BLOCK);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL((prune_dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE, T>),
                               blocks,
                               threads,
                               0,
                               stream,
                               descr->base,
                               m,
                               n,
                               A,
                               lda,
                               threshold,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
        else
        {
            hipLaunchKernelGGL((prune_dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE, T>),
                               blocks,
                               threads,
                               0,
                               stream,
                               descr->base,
                               m,
                               n,
                               A,
                               lda,
                               *threshold,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
    }
    else
    {
        static constexpr rocsparse_int WF_SIZE         = 64;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        dim3                           blocks((m - 1) / NROWS_PER_BLOCK + 1);
        dim3                           threads(WF_SIZE * NROWS_PER_BLOCK);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL((prune_dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE, T>),
                               blocks,
                               threads,
                               0,
                               stream,
                               descr->base,
                               m,
                               n,
                               A,
                               lda,
                               threshold,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
        else
        {
            hipLaunchKernelGGL((prune_dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE, T>),
                               blocks,
                               threads,
                               0,
                               stream,
                               descr->base,
                               m,
                               n,
                               A,
                               lda,
                               *threshold,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
    }

    return rocsparse_status_success;
}

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status rocsparse_sprune_dense2csr_buffer_size(rocsparse_handle handle,
                                                                   rocsparse_int    m,
                                                                   rocsparse_int    n,
                                                                   const float*     A,
                                                                   rocsparse_int    lda,
                                                                   const float*     threshold,
                                                                   const rocsparse_mat_descr descr,
                                                                   const float*         csr_val,
                                                                   const rocsparse_int* csr_row_ptr,
                                                                   const rocsparse_int* csr_col_ind,
                                                                   size_t*              buffer_size)
{
    return rocsparse_prune_dense2csr_buffer_size_template(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size);
}

extern "C" rocsparse_status rocsparse_dprune_dense2csr_buffer_size(rocsparse_handle handle,
                                                                   rocsparse_int    m,
                                                                   rocsparse_int    n,
                                                                   const double*    A,
                                                                   rocsparse_int    lda,
                                                                   const double*    threshold,
                                                                   const rocsparse_mat_descr descr,
                                                                   const double*        csr_val,
                                                                   const rocsparse_int* csr_row_ptr,
                                                                   const rocsparse_int* csr_col_ind,
                                                                   size_t*              buffer_size)
{
    return rocsparse_prune_dense2csr_buffer_size_template(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size);
}

extern "C" rocsparse_status rocsparse_sprune_dense2csr_nnz(rocsparse_handle          handle,
                                                           rocsparse_int             m,
                                                           rocsparse_int             n,
                                                           const float*              A,
                                                           rocsparse_int             lda,
                                                           const float*              threshold,
                                                           const rocsparse_mat_descr descr,
                                                           rocsparse_int*            csr_row_ptr,
                                                           rocsparse_int* nnz_total_dev_host_ptr,
                                                           void*          temp_buffer)
{
    return rocsparse_prune_dense2csr_nnz_template(
        handle, m, n, A, lda, threshold, descr, csr_row_ptr, nnz_total_dev_host_ptr, temp_buffer);
}

extern "C" rocsparse_status rocsparse_dprune_dense2csr_nnz(rocsparse_handle          handle,
                                                           rocsparse_int             m,
                                                           rocsparse_int             n,
                                                           const double*             A,
                                                           rocsparse_int             lda,
                                                           const double*             threshold,
                                                           const rocsparse_mat_descr descr,
                                                           rocsparse_int*            csr_row_ptr,
                                                           rocsparse_int* nnz_total_dev_host_ptr,
                                                           void*          temp_buffer)
{
    return rocsparse_prune_dense2csr_nnz_template(
        handle, m, n, A, lda, threshold, descr, csr_row_ptr, nnz_total_dev_host_ptr, temp_buffer);
}

extern "C" rocsparse_status rocsparse_sprune_dense2csr(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       const float*              A,
                                                       rocsparse_int             lda,
                                                       const float*              threshold,
                                                       const rocsparse_mat_descr descr,
                                                       float*                    csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       rocsparse_int*            csr_col_ind,
                                                       void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_template(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

extern "C" rocsparse_status rocsparse_dprune_dense2csr(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       const double*             A,
                                                       rocsparse_int             lda,
                                                       const double*             threshold,
                                                       const rocsparse_mat_descr descr,
                                                       double*                   csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       rocsparse_int*            csr_col_ind,
                                                       void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_template(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}
