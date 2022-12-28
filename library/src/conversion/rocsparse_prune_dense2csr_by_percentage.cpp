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

#include "rocsparse_prune_dense2csr_by_percentage.hpp"
#include "definitions.h"
#include "utility.h"

#include "csr2csr_compress_device.h"
#include "prune_dense2csr_by_percentage_device.h"
#include "prune_dense2csr_device.h"
#include <rocprim/rocprim.hpp>

template <rocsparse_int DIM_X, rocsparse_int DIM_Y, typename T, typename U>
ROCSPARSE_KERNEL(DIM_X* DIM_Y)
void prune_dense2csr_nnz_kernel2(rocsparse_int m,
                                 rocsparse_int n,
                                 const T* __restrict__ A,
                                 rocsparse_int lda,
                                 U             threshold_device_host,
                                 rocsparse_int* __restrict__ nnz_per_rows)
{
    auto threshold = load_scalar_device_host(threshold_device_host);
    prune_dense2csr_nnz_device<DIM_X, DIM_Y>(m, n, A, lda, threshold, nnz_per_rows);
}

template <rocsparse_int NUMROWS_PER_BLOCK, rocsparse_int WF_SIZE, typename T>
ROCSPARSE_KERNEL(WF_SIZE* NUMROWS_PER_BLOCK)
void prune_dense2csr_kernel2_device_pointer(rocsparse_index_base base,
                                            rocsparse_int        m,
                                            rocsparse_int        n,
                                            const T* __restrict__ A,
                                            rocsparse_int lda,
                                            const T*      threshold,
                                            T* __restrict__ csr_val,
                                            const rocsparse_int* __restrict__ csr_row_ptr,
                                            rocsparse_int* __restrict__ csr_col_ind)
{
    prune_dense2csr_device<NUMROWS_PER_BLOCK, WF_SIZE>(
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

    if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
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
              (const void*&)buffer_size);

    log_bench(handle,
              "./rocsparse-bench -f prune_dense2csr_by_percentage_buffer_size -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || lda < m || percentage < static_cast<T>(0.0)
       || percentage > static_cast<T>(100.0))
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(A == nullptr || csr_row_ptr == nullptr || buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *buffer_size = sizeof(T) * 2 * m * n;
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
              (const void*&)csr_row_ptr,
              (const void*&)nnz_total_dev_host_ptr,
              info,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f prune_dense2csr_nnz_by_percentage -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
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
        if(nnz_total_dev_host_ptr != nullptr && csr_row_ptr != nullptr)
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
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&temp_storage_ptr, temp_storage_size_bytes, handle->stream));
        temp_alloc = true;
    }

    // perform sort on first half of output array and store result in second half of output array
    rocprim::radix_sort_keys(
        temp_storage_ptr, temp_storage_size_bytes, output, (output + nnz_A), nnz_A);

    const T* d_threshold = &output[nnz_A + pos];

    static constexpr int NNZ_DIM_X = 64;
    static constexpr int NNZ_DIM_Y = 16;

    {
        dim3 grid((m - 1) / (NNZ_DIM_X * 4) + 1);
        dim3 threads(NNZ_DIM_X, NNZ_DIM_Y);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL((prune_dense2csr_nnz_kernel2<NNZ_DIM_X, NNZ_DIM_Y>),
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
            T h_threshold = static_cast<T>(0);
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &h_threshold, d_threshold, sizeof(T), hipMemcpyDeviceToHost, handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            hipLaunchKernelGGL((prune_dense2csr_nnz_kernel2<NNZ_DIM_X, NNZ_DIM_Y>),
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
    }
    // Store threshold at first entry in output array
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(output, d_threshold, sizeof(T), hipMemcpyDeviceToDevice, handle->stream));

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
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
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
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(nnz_total_dev_host_ptr,
                                           &csr_row_ptr[m],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

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
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              info,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f prune_dense2csr_by_percentage -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
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
    if(A == nullptr || csr_row_ptr == nullptr || temp_buffer == nullptr)
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

    T* d_threshold = &(reinterpret_cast<T*>(temp_buffer))[0];

    static constexpr rocsparse_int data_ratio = sizeof(T) / sizeof(float);

    if(handle->wavefront_size == 32)
    {
        static constexpr rocsparse_int WF_SIZE         = 32;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        dim3                           blocks((m - 1) / NROWS_PER_BLOCK + 1);
        dim3                           threads(WF_SIZE * NROWS_PER_BLOCK);

        hipLaunchKernelGGL((prune_dense2csr_kernel2_device_pointer<NROWS_PER_BLOCK, WF_SIZE, T>),
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

        hipLaunchKernelGGL((prune_dense2csr_kernel2_device_pointer<NROWS_PER_BLOCK, WF_SIZE, T>),
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

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status
    rocsparse_sprune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const float*              A,
                                                         rocsparse_int             lda,
                                                         float                     percentage,
                                                         const rocsparse_mat_descr descr,
                                                         const float*              csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size)
{
    return rocsparse_prune_dense2csr_by_percentage_buffer_size_template(handle,
                                                                        m,
                                                                        n,
                                                                        A,
                                                                        lda,
                                                                        percentage,
                                                                        descr,
                                                                        csr_val,
                                                                        csr_row_ptr,
                                                                        csr_col_ind,
                                                                        info,
                                                                        buffer_size);
}

extern "C" rocsparse_status
    rocsparse_dprune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const double*             A,
                                                         rocsparse_int             lda,
                                                         double                    percentage,
                                                         const rocsparse_mat_descr descr,
                                                         const double*             csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size)
{
    return rocsparse_prune_dense2csr_by_percentage_buffer_size_template(handle,
                                                                        m,
                                                                        n,
                                                                        A,
                                                                        lda,
                                                                        percentage,
                                                                        descr,
                                                                        csr_val,
                                                                        csr_row_ptr,
                                                                        csr_col_ind,
                                                                        info,
                                                                        buffer_size);
}

extern "C" rocsparse_status
    rocsparse_sprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const float*              A,
                                                 rocsparse_int             lda,
                                                 float                     percentage,
                                                 const rocsparse_mat_descr descr,
                                                 rocsparse_int*            csr_row_ptr,
                                                 rocsparse_int*            nnz_total_dev_host_ptr,
                                                 rocsparse_mat_info        info,
                                                 void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_nnz_by_percentage_template(handle,
                                                                m,
                                                                n,
                                                                A,
                                                                lda,
                                                                percentage,
                                                                descr,
                                                                csr_row_ptr,
                                                                nnz_total_dev_host_ptr,
                                                                info,
                                                                temp_buffer);
}

extern "C" rocsparse_status
    rocsparse_dprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const double*             A,
                                                 rocsparse_int             lda,
                                                 double                    percentage,
                                                 const rocsparse_mat_descr descr,
                                                 rocsparse_int*            csr_row_ptr,
                                                 rocsparse_int*            nnz_total_dev_host_ptr,
                                                 rocsparse_mat_info        info,
                                                 void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_nnz_by_percentage_template(handle,
                                                                m,
                                                                n,
                                                                A,
                                                                lda,
                                                                percentage,
                                                                descr,
                                                                csr_row_ptr,
                                                                nnz_total_dev_host_ptr,
                                                                info,
                                                                temp_buffer);
}

extern "C" rocsparse_status
    rocsparse_sprune_dense2csr_by_percentage(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const float*              A,
                                             rocsparse_int             lda,
                                             float                     percentage,
                                             const rocsparse_mat_descr descr,
                                             float*                    csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info        info,
                                             void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_by_percentage_template(handle,
                                                            m,
                                                            n,
                                                            A,
                                                            lda,
                                                            percentage,
                                                            descr,
                                                            csr_val,
                                                            csr_row_ptr,
                                                            csr_col_ind,
                                                            info,
                                                            temp_buffer);
}

extern "C" rocsparse_status
    rocsparse_dprune_dense2csr_by_percentage(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const double*             A,
                                             rocsparse_int             lda,
                                             double                    percentage,
                                             const rocsparse_mat_descr descr,
                                             double*                   csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info        info,
                                             void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_by_percentage_template(handle,
                                                            m,
                                                            n,
                                                            A,
                                                            lda,
                                                            percentage,
                                                            descr,
                                                            csr_val,
                                                            csr_row_ptr,
                                                            csr_col_ind,
                                                            info,
                                                            temp_buffer);
}
