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

#include "internal/conversion/rocsparse_csr2bsr.h"
#include "definitions.h"
#include "rocsparse_csr2bsr.hpp"
#include "utility.h"

#include "csr2bsr_device.h"

#include <rocprim/rocprim.hpp>

#define launch_csr2bsr_wavefront_per_row_multipass_kernel(blocksize, wfsize, blockdim)            \
    hipLaunchKernelGGL((csr2bsr_wavefront_per_row_multipass_kernel<blocksize, wfsize, blockdim>), \
                       dim3((mb - 1) / (blocksize / wfsize) + 1),                                 \
                       dim3(blocksize),                                                           \
                       0,                                                                         \
                       stream,                                                                    \
                       dir,                                                                       \
                       m,                                                                         \
                       n,                                                                         \
                       mb,                                                                        \
                       nb,                                                                        \
                       block_dim,                                                                 \
                       csr_descr->base,                                                           \
                       csr_val,                                                                   \
                       csr_row_ptr,                                                               \
                       csr_col_ind,                                                               \
                       bsr_descr->base,                                                           \
                       bsr_val,                                                                   \
                       bsr_row_ptr,                                                               \
                       bsr_col_ind);

#define launch_csr2bsr_block_per_row_multipass_kernel(blocksize, blockdim)            \
    hipLaunchKernelGGL((csr2bsr_block_per_row_multipass_kernel<blocksize, blockdim>), \
                       dim3(mb),                                                      \
                       dim3(blocksize),                                               \
                       0,                                                             \
                       stream,                                                        \
                       dir,                                                           \
                       m,                                                             \
                       n,                                                             \
                       mb,                                                            \
                       nb,                                                            \
                       block_dim,                                                     \
                       csr_descr->base,                                               \
                       csr_val,                                                       \
                       csr_row_ptr,                                                   \
                       csr_col_ind,                                                   \
                       bsr_descr->base,                                               \
                       bsr_val,                                                       \
                       bsr_row_ptr,                                                   \
                       bsr_col_ind);

template <typename T,
          typename std::enable_if<std::is_same<T, rocsparse_double_complex>::value, int>::type = 0>
static inline rocsparse_status csr2bsr_64_launcher(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             m,
                                                   rocsparse_int             n,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
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
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value
                                      || std::is_same<T, rocsparse_float_complex>::value,
                                  int>::type
          = 0>
static inline rocsparse_status csr2bsr_64_launcher(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             m,
                                                   rocsparse_int             n,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
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
    hipStream_t stream = handle->stream;

    launch_csr2bsr_block_per_row_multipass_kernel(256, 64);

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csr2bsr_template(rocsparse_handle          handle, //0
                                            rocsparse_direction       dir, //1
                                            rocsparse_int             m, //2
                                            rocsparse_int             n, //3
                                            const rocsparse_mat_descr csr_descr, //4
                                            const T*                  csr_val, //5
                                            const rocsparse_int*      csr_row_ptr, //6
                                            const rocsparse_int*      csr_col_ind, //7
                                            rocsparse_int             block_dim, //8
                                            const rocsparse_mat_descr bsr_descr, //9
                                            T*                        bsr_val, //10
                                            rocsparse_int*            bsr_row_ptr, //11
                                            rocsparse_int*            bsr_col_ind)
{

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2bsr"),
              dir,
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

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);
    ROCSPARSE_CHECKARG_POINTER(4, csr_descr);
    ROCSPARSE_CHECKARG(4,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_SIZE(8, block_dim);
    ROCSPARSE_CHECKARG(8, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(9, bsr_descr);
    ROCSPARSE_CHECKARG(9,
                       bsr_descr,
                       (bsr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(11, m, bsr_row_ptr);

    if(csr_val == nullptr || csr_col_ind == nullptr)
    {
        int64_t nnz;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_calculate_nnz(
            m, rocsparse_get_indextype<rocsparse_int>(), csr_row_ptr, &nnz, handle->stream));

        ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_col_ind);
    }

    const rocsparse_int mb = (m + block_dim - 1) / block_dim;
    const rocsparse_int nb = (n + block_dim - 1) / block_dim;

    rocsparse_int start = 0;
    rocsparse_int end   = 0;

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    const rocsparse_int nnzb = (end - start);

    ROCSPARSE_CHECKARG_ARRAY(10, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(12, nnzb, bsr_col_ind);

    if(nnzb == 0)
    {
        return rocsparse_status_success;
    }

    RETURN_IF_HIP_ERROR(
        hipMemsetAsync(bsr_val, 0, sizeof(T) * nnzb * block_dim * block_dim, handle->stream));
    // Stream
    hipStream_t stream = handle->stream;

    if(block_dim == 1)
    {
        hipLaunchKernelGGL((csr2bsr_block_dim_equals_one_kernel<256>),
                           dim3((mb - 1) / 256 + 1),
                           dim3(256),
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

    if(block_dim <= 4)
    {
        launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 16, 4);
    }
    else if(block_dim <= 8)
    {
        if(handle->wavefront_size == 64)
        {
            launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 64, 8);
        }
        else
        {
            launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 32, 8);
        }
    }
    else if(block_dim <= 16)
    {
        if(handle->wavefront_size == 64)
        {
            launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 64, 16);
        }
        else
        {
            launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 32, 16);
        }
    }
    else if(block_dim <= 32)
    {
        launch_csr2bsr_block_per_row_multipass_kernel(256, 32);
    }
    else if(block_dim <= 64 && !std::is_same<T, rocsparse_double_complex>())
    {
        csr2bsr_64_launcher(handle,
                            dir,
                            m,
                            n,
                            mb,
                            nb,
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
    else
    {
        // Use a blocksize of 32 to handle each block row
        constexpr rocsparse_int block_size       = 32;
        rocsparse_int           rows_per_segment = (block_dim + block_size - 1) / block_size;

        size_t buffer_size = 0;
        buffer_size += sizeof(rocsparse_int)
                       * ((size_t(mb) * block_size * 3 * rows_per_segment - 1) / 256 + 1) * 256;
        buffer_size
            += sizeof(T) * ((size_t(mb) * block_size * rows_per_segment - 1) / 256 + 1) * 256;

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= buffer_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&temp_storage_ptr, buffer_size, handle->stream));
            temp_alloc = true;
        }

        char*          ptr   = reinterpret_cast<char*>(temp_storage_ptr);
        rocsparse_int* temp1 = reinterpret_cast<rocsparse_int*>(ptr);
        ptr += sizeof(rocsparse_int)
               * ((size_t(mb) * block_size * 3 * rows_per_segment - 1) / 256 + 1) * 256;
        T* temp2 = reinterpret_cast<T*>(ptr);

        hipLaunchKernelGGL((csr2bsr_65_inf_kernel<block_size>),
                           dim3(mb),
                           dim3(block_size),
                           0,
                           handle->stream,
                           dir,
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
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
        }
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(blocksize, wfsize, blockdim) \
    hipLaunchKernelGGL(                                                                    \
        (csr2bsr_nnz_wavefront_per_row_multipass_kernel<blocksize, wfsize, blockdim>),     \
        dim3((mb - 1) / (blocksize / wfsize) + 1),                                         \
        dim3(blocksize),                                                                   \
        0,                                                                                 \
        handle->stream,                                                                    \
        m,                                                                                 \
        n,                                                                                 \
        mb,                                                                                \
        nb,                                                                                \
        block_dim,                                                                         \
        csr_descr->base,                                                                   \
        csr_row_ptr,                                                                       \
        csr_col_ind,                                                                       \
        bsr_descr->base,                                                                   \
        bsr_row_ptr);

#define launch_csr2bsr_nnz_block_per_row_multipass_kernel(blocksize, blockdim)            \
    hipLaunchKernelGGL((csr2bsr_nnz_block_per_row_multipass_kernel<blocksize, blockdim>), \
                       dim3(mb),                                                          \
                       dim3(blocksize),                                                   \
                       0,                                                                 \
                       handle->stream,                                                    \
                       m,                                                                 \
                       n,                                                                 \
                       mb,                                                                \
                       nb,                                                                \
                       block_dim,                                                         \
                       csr_descr->base,                                                   \
                       csr_row_ptr,                                                       \
                       csr_col_ind,                                                       \
                       bsr_descr->base,                                                   \
                       bsr_row_ptr);

extern "C" rocsparse_status rocsparse_csr2bsr_nnz(rocsparse_handle          handle, //0
                                                  rocsparse_direction       dir, //1
                                                  rocsparse_int             m, //2
                                                  rocsparse_int             n, //3
                                                  const rocsparse_mat_descr csr_descr, //4
                                                  const rocsparse_int*      csr_row_ptr, //5
                                                  const rocsparse_int*      csr_col_ind, //6
                                                  rocsparse_int             block_dim, //7
                                                  const rocsparse_mat_descr bsr_descr, //8
                                                  rocsparse_int*            bsr_row_ptr, //9
                                                  rocsparse_int*            bsr_nnz) //10
try
{

    // Logging
    log_trace(handle,
              "rocsparse_csr2bsr_nnz",
              dir,
              m,
              n,
              csr_descr,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              block_dim,
              bsr_descr,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_nnz);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);
    ROCSPARSE_CHECKARG_SIZE(7, block_dim);
    ROCSPARSE_CHECKARG(7, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(4, csr_descr);
    ROCSPARSE_CHECKARG(4,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_POINTER(8, bsr_descr);
    ROCSPARSE_CHECKARG(8,
                       bsr_descr,
                       (bsr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    const rocsparse_int mb = (m + block_dim - 1) / block_dim;
    const rocsparse_int nb = (n + block_dim - 1) / block_dim;

    // Quick return if possible
    if(m == 0 || n == 0)
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

        if(bsr_row_ptr != nullptr)
        {
            hipLaunchKernelGGL((set_array_to_value<256>),
                               dim3(((mb + 1) - 1) / 256 + 1),
                               dim3(256),
                               0,
                               handle->stream,
                               (mb + 1),
                               bsr_row_ptr,
                               static_cast<rocsparse_int>(bsr_descr->base));
        }

        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(9, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(10, bsr_nnz);

    if(csr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &csr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        const rocsparse_int nnz = (end - start);

        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    }

    // If block dimension is one then BSR is equal to CSR
    if(block_dim == 1)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL(csr2bsr_nnz_block_dim_equals_one_kernel<256>,
                               dim3(m / 256 + 1),
                               dim3(256),
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
            hipLaunchKernelGGL(csr2bsr_nnz_block_dim_equals_one_kernel<256>,
                               dim3(m / 256 + 1),
                               dim3(256),
                               0,
                               handle->stream,
                               m,
                               csr_descr->base,
                               csr_row_ptr,
                               bsr_descr->base,
                               bsr_row_ptr);

            rocsparse_int start = 0;
            rocsparse_int end   = 0;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                               &bsr_row_ptr[mb],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                               &bsr_row_ptr[0],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            *bsr_nnz = end - start;
        }

        return rocsparse_status_success;
    }

    if(block_dim <= 4)
    {
        launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 16, 4);
    }
    else if(block_dim <= 8)
    {
        if(handle->wavefront_size == 64)
        {
            launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 64, 8);
        }
        else
        {
            launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 32, 8);
        }
    }
    else if(block_dim <= 16)
    {
        if(handle->wavefront_size == 64)
        {
            launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 64, 16);
        }
        else
        {
            launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 32, 16);
        }
    }
    else if(block_dim <= 32)
    {
        launch_csr2bsr_nnz_block_per_row_multipass_kernel(256, 32);
    }
    else if(block_dim <= 64)
    {
        launch_csr2bsr_nnz_block_per_row_multipass_kernel(256, 64);
    }
    else
    {
        // Use a blocksize of 32 to handle each block row
        constexpr rocsparse_int block_size       = 32;
        rocsparse_int           rows_per_segment = (block_dim + block_size - 1) / block_size;

        size_t buffer_size = sizeof(rocsparse_int)
                             * ((size_t(mb) * block_size * 2 * rows_per_segment - 1) / 256 + 1)
                             * 256;

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= buffer_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&temp_storage_ptr, buffer_size, handle->stream));
            temp_alloc = true;
        }

        rocsparse_int* temp1 = reinterpret_cast<rocsparse_int*>(temp_storage_ptr);

        hipLaunchKernelGGL((csr2bsr_nnz_65_inf_kernel<block_size>),
                           dim3(mb),
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
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
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
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&temp_storage_ptr, temp_storage_size_bytes, handle->stream));
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
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
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
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        *bsr_nnz = end - start;
    }

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

#define CIMPL(NAME, T)                                                      \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_int             m,           \
                                     rocsparse_int             n,           \
                                     const rocsparse_mat_descr csr_descr,   \
                                     const T*                  csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     const rocsparse_mat_descr bsr_descr,   \
                                     T*                        bsr_val,     \
                                     rocsparse_int*            bsr_row_ptr, \
                                     rocsparse_int*            bsr_col_ind) \
    try                                                                     \
    {                                                                       \
        return rocsparse_csr2bsr_template(handle,                           \
                                          dir,                              \
                                          m,                                \
                                          n,                                \
                                          csr_descr,                        \
                                          csr_val,                          \
                                          csr_row_ptr,                      \
                                          csr_col_ind,                      \
                                          block_dim,                        \
                                          bsr_descr,                        \
                                          bsr_val,                          \
                                          bsr_row_ptr,                      \
                                          bsr_col_ind);                     \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

CIMPL(rocsparse_scsr2bsr, float);
CIMPL(rocsparse_dcsr2bsr, double);
CIMPL(rocsparse_ccsr2bsr, rocsparse_float_complex);
CIMPL(rocsparse_zcsr2bsr, rocsparse_double_complex);

#undef CIMPL
