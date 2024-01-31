/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_prune_csr2csr.h"
#include "definitions.h"
#include "rocsparse_nnz_compress.hpp"
#include "rocsparse_prune_csr2csr.hpp"
#include "utility.h"

#include "csr2csr_compress_device.h"
#include <rocprim/rocprim.hpp>

template <rocsparse_int BLOCK_SIZE,
          rocsparse_int SEGMENTS_PER_BLOCK,
          rocsparse_int SEGMENT_SIZE,
          rocsparse_int WF_SIZE,
          typename T,
          typename U>
ROCSPARSE_KERNEL(BLOCK_SIZE)
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
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
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
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
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
rocsparse_status
    rocsparse_prune_csr2csr_buffer_size_template(rocsparse_handle          handle, //0
                                                 rocsparse_int             m, //1
                                                 rocsparse_int             n, //2
                                                 rocsparse_int             nnz_A, //3
                                                 const rocsparse_mat_descr csr_descr_A, //4
                                                 const T*                  csr_val_A, //5
                                                 const rocsparse_int*      csr_row_ptr_A, //6
                                                 const rocsparse_int*      csr_col_ind_A, //7
                                                 const T*                  threshold, //8
                                                 const rocsparse_mat_descr csr_descr_C, //9
                                                 const T*                  csr_val_C, //10
                                                 const rocsparse_int*      csr_row_ptr_C, //11
                                                 const rocsparse_int*      csr_col_ind_C, //12
                                                 size_t*                   buffer_size) //13
{

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

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz_A);
    ROCSPARSE_CHECKARG_POINTER(4, csr_descr_A);
    ROCSPARSE_CHECKARG(4,
                       csr_descr_A,
                       (csr_descr_A->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz_A, csr_val_A);
    ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr_A);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz_A, csr_col_ind_A);
    ROCSPARSE_CHECKARG_POINTER(8, threshold);
    ROCSPARSE_CHECKARG_POINTER(9, csr_descr_C);
    ROCSPARSE_CHECKARG(9,
                       csr_descr_C,
                       (csr_descr_C->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(11, m, csr_row_ptr_C);
    //
    //    if (csr_val_C == nullptr || csr_col_ind_C == nullptr)
    //      {
    //	int64_t nnz_C;
    //	RETURN_IF_ROCSPARSE_ERROR(rocsparse_calculate_nnz(m,
    //							  rocsparse_get_indextype<rocsparse_int>(),
    //							  csr_row_ptr_C,
    //							  &nnz_C,
    //							  handle->stream));
    //
    //	ROCSPARSE_CHECKARG_ARRAY(10,nnz_C, csr_val_C);
    //	ROCSPARSE_CHECKARG_ARRAY(12,nnz_C, csr_col_ind_C);
    //      }
    //
    ROCSPARSE_CHECKARG_POINTER(13, buffer_size);

    *buffer_size = 0;
    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_prune_csr2csr_nnz_template(rocsparse_handle          handle, //0
                                                      rocsparse_int             m, //1
                                                      rocsparse_int             n, //2
                                                      rocsparse_int             nnz_A, //3
                                                      const rocsparse_mat_descr csr_descr_A, //4
                                                      const T*                  csr_val_A, //5
                                                      const rocsparse_int*      csr_row_ptr_A, //6
                                                      const rocsparse_int*      csr_col_ind_A, //7
                                                      const T*                  threshold, //8
                                                      const rocsparse_mat_descr csr_descr_C, //9
                                                      rocsparse_int*            csr_row_ptr_C, //10
                                                      rocsparse_int* nnz_total_dev_host_ptr, //11
                                                      void*          temp_buffer) //12
{

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

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz_A);
    ROCSPARSE_CHECKARG_POINTER(4, csr_descr_A);
    ROCSPARSE_CHECKARG(4,
                       csr_descr_A,
                       (csr_descr_A->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz_A, csr_val_A);
    ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr_A);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz_A, csr_col_ind_A);
    ROCSPARSE_CHECKARG_POINTER(9, csr_descr_C);
    ROCSPARSE_CHECKARG(9,
                       csr_descr_C,
                       (csr_descr_C->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(10, m, csr_row_ptr_C);

    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0 || nnz_A == 0)
    {
        if(nnz_total_dev_host_ptr != nullptr && csr_row_ptr_C != nullptr)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((set_array_to_value<256>),
                                               dim3(m / 256 + 1),
                                               dim3(256),
                                               0,
                                               stream,
                                               (m + 1),
                                               csr_row_ptr_C,
                                               static_cast<rocsparse_int>(csr_descr_C->base));

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
    ROCSPARSE_CHECKARG_POINTER(8, threshold);
    ROCSPARSE_CHECKARG_POINTER(11, nnz_total_dev_host_ptr);

    // Copy threshold to host
    T h_threshold;
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &h_threshold, threshold, sizeof(T), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    }
    else
    {
        h_threshold = *threshold;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_nnz_compress_template(handle,
                                                              m,
                                                              csr_descr_A,
                                                              csr_val_A,
                                                              csr_row_ptr_A,
                                                              &csr_row_ptr_C[1],
                                                              nnz_total_dev_host_ptr,
                                                              h_threshold));

    // Compute csr_row_ptr_C with the right index base.
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(csr_row_ptr_C,
                                       &csr_descr_C->base,
                                       sizeof(rocsparse_int),
                                       hipMemcpyHostToDevice,
                                       handle->stream));

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
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&temp_storage_ptr, temp_storage_size_bytes, handle->stream));
        temp_alloc = true;
    }

    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(temp_storage_ptr,
                                                temp_storage_size_bytes,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                m + 1,
                                                op,
                                                stream));

    if(temp_alloc)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_prune_csr2csr_template(rocsparse_handle          handle, //0
                                                  rocsparse_int             m, //1
                                                  rocsparse_int             n, //2
                                                  rocsparse_int             nnz_A, //3
                                                  const rocsparse_mat_descr csr_descr_A, //4
                                                  const T*                  csr_val_A, //5
                                                  const rocsparse_int*      csr_row_ptr_A, //6
                                                  const rocsparse_int*      csr_col_ind_A, //7
                                                  const T*                  threshold, //8
                                                  const rocsparse_mat_descr csr_descr_C, //9
                                                  T*                        csr_val_C, //10
                                                  const rocsparse_int*      csr_row_ptr_C, //11
                                                  rocsparse_int*            csr_col_ind_C, //12
                                                  void*                     temp_buffer) //13
{
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

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz_A);
    ROCSPARSE_CHECKARG_POINTER(4, csr_descr_A);
    ROCSPARSE_CHECKARG(4,
                       csr_descr_A,
                       (csr_descr_A->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz_A, csr_val_A);
    ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr_A);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz_A, csr_col_ind_A);
    ROCSPARSE_CHECKARG_POINTER(9, csr_descr_C);
    ROCSPARSE_CHECKARG(9,
                       csr_descr_C,
                       (csr_descr_C->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(11, m, csr_row_ptr_C);

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(8, threshold);

    if(csr_val_C == nullptr || csr_col_ind_C == nullptr)
    {
        int64_t nnz_C;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_calculate_nnz(
            m, rocsparse_get_indextype<rocsparse_int>(), csr_row_ptr_C, &nnz_C, handle->stream));

        ROCSPARSE_CHECKARG_ARRAY(10, nnz_C, csr_val_C);
        ROCSPARSE_CHECKARG_ARRAY(12, nnz_C, csr_col_ind_C);
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
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_arch_mismatch);
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
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_prune_csr2csr_buffer_size_template(handle,
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
                                                                           buffer_size));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
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
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_prune_csr2csr_buffer_size_template(handle,
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
                                                                           buffer_size));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
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
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_prune_csr2csr_nnz_template(handle,
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
                                                                   temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
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
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_prune_csr2csr_nnz_template(handle,
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
                                                                   temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
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
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_prune_csr2csr_template(handle,
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
                                                               temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
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
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_prune_csr2csr_template(handle,
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
                                                               temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
