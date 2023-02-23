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

#include "rocsparse_csr2gebsr.hpp"
#include "definitions.h"
#include "rocsparse_csr2bsr.hpp"
#include "utility.h"

#include "csr2gebsr_device.h"
#include <rocprim/rocprim.hpp>

#define launch_csr2gebsr_block_per_row_multipass_kernel(block_size, row_blockdim, col_blockdim) \
    hipLaunchKernelGGL(                                                                         \
        (csr2gebsr_block_per_row_multipass_kernel<block_size, row_blockdim, col_blockdim>),     \
        mb,                                                                                     \
        block_size,                                                                             \
        0,                                                                                      \
        stream,                                                                                 \
        direction,                                                                              \
        m,                                                                                      \
        n,                                                                                      \
        mb,                                                                                     \
        nb,                                                                                     \
        row_block_dim,                                                                          \
        col_block_dim,                                                                          \
        csr_descr->base,                                                                        \
        csr_val,                                                                                \
        csr_row_ptr,                                                                            \
        csr_col_ind,                                                                            \
        bsr_descr->base,                                                                        \
        bsr_val,                                                                                \
        bsr_row_ptr,                                                                            \
        bsr_col_ind);

#define launch_csr2gebsr_wavefront_per_row_multipass_kernel(                       \
    block_size, row_blockdim, col_blockdim, wf_size)                               \
    hipLaunchKernelGGL((csr2gebsr_wavefront_per_row_multipass_kernel<block_size,   \
                                                                     row_blockdim, \
                                                                     col_blockdim, \
                                                                     wf_size>),    \
                       dim3((mb - 1) / (block_size / wf_size) + 1),                \
                       block_size,                                                 \
                       0,                                                          \
                       stream,                                                     \
                       direction,                                                  \
                       m,                                                          \
                       n,                                                          \
                       mb,                                                         \
                       nb,                                                         \
                       row_block_dim,                                              \
                       col_block_dim,                                              \
                       csr_descr->base,                                            \
                       csr_val,                                                    \
                       csr_row_ptr,                                                \
                       csr_col_ind,                                                \
                       bsr_descr->base,                                            \
                       bsr_val,                                                    \
                       bsr_row_ptr,                                                \
                       bsr_col_ind);

template <typename T,
          typename std::enable_if<std::is_same<T, rocsparse_double_complex>::value, int>::type = 0>
static inline rocsparse_status csr2gebsr_64_64_launcher(rocsparse_handle          handle,
                                                        rocsparse_direction       direction,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        rocsparse_int             mb,
                                                        rocsparse_int             nb,
                                                        const rocsparse_mat_descr csr_descr,
                                                        const T*                  csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        const rocsparse_mat_descr bsr_descr,
                                                        T*                        bsr_val,
                                                        rocsparse_int*            bsr_row_ptr,
                                                        rocsparse_int*            bsr_col_ind,
                                                        rocsparse_int             row_block_dim,
                                                        rocsparse_int             col_block_dim)
{
    return rocsparse_status_internal_error;
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value
                                      || std::is_same<T, rocsparse_float_complex>::value,
                                  int>::type
          = 0>
static inline rocsparse_status csr2gebsr_64_64_launcher(rocsparse_handle          handle,
                                                        rocsparse_direction       direction,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        rocsparse_int             mb,
                                                        rocsparse_int             nb,
                                                        const rocsparse_mat_descr csr_descr,
                                                        const T*                  csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        const rocsparse_mat_descr bsr_descr,
                                                        T*                        bsr_val,
                                                        rocsparse_int*            bsr_row_ptr,
                                                        rocsparse_int*            bsr_col_ind,
                                                        rocsparse_int             row_block_dim,
                                                        rocsparse_int             col_block_dim)
{
    hipStream_t stream = handle->stream;

    launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 64);

    return rocsparse_status_success;
}

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
                                                          size_t*                   buffer_size)
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
              (const void*&)buffer_size);

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
    if(m < 0 || n < 0 || row_block_dim <= 0 || col_block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    //
    // Check buffer size argument
    //
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Quick return if possible
    //
    if(m == 0 || n == 0)
    {
        *buffer_size = sizeof(rocsparse_int) * 4;
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

    *buffer_size = sizeof(rocsparse_int) * 4;

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
                                              void*                     temp_buffer)
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
              (const void*&)temp_buffer);

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
    if(m < 0 || n < 0 || row_block_dim <= 0 || col_block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    //
    // Quick return if possible
    //
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    //
    // Check pointer arguments
    //
    if(csr_row_ptr == nullptr || bsr_row_ptr == nullptr || temp_buffer == nullptr)
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

    if(row_block_dim == col_block_dim)
    {
        return rocsparse_csr2bsr_template(handle,
                                          direction,
                                          m,
                                          n,
                                          csr_descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          row_block_dim,
                                          bsr_descr,
                                          bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind);
    }

    hipStream_t   stream = handle->stream;
    rocsparse_int mb     = (m + row_block_dim - 1) / row_block_dim;
    rocsparse_int nb     = (n + col_block_dim - 1) / col_block_dim;

    rocsparse_int start = 0;
    rocsparse_int end   = 0;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    rocsparse_int nnzb = (end - start);

    if(nnzb != 0 && (bsr_val == nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Set bsr val to zero.
    //
    RETURN_IF_HIP_ERROR(hipMemsetAsync(
        bsr_val, 0, sizeof(T) * nnzb * row_block_dim * col_block_dim, handle->stream));

    if(row_block_dim == 1)
    {
        hipLaunchKernelGGL((csr2gebsr_kernel_bm1<256>),
                           dim3(m / 256 + 1),
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
    // row_block_dim <= 64 && col_block_dim <= 64
    if(row_block_dim <= 2 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 2, 4);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 4, 8);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 8, 16);
        }
        else if(col_block_dim <= 16)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 16, 32);
        }
        else if(col_block_dim <= 32)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 32, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 32, 32);
            }
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 64, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 64, 32);
            }
        }
    }
    else if(row_block_dim <= 4 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 2, 8);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 4, 16);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 8, 32);
        }
        else if(col_block_dim <= 16)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 16, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 16, 32);
            }
        }
        else if(col_block_dim <= 32)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 32, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 32, 32);
            }
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 64, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 64, 32);
            }
        }
    }
    else if(row_block_dim <= 8 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 2, 16);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 4, 32);
        }
        else if(col_block_dim <= 8)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 8, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 8, 32);
            }
        }
        else if(col_block_dim <= 16)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 16, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 16, 32);
            }
        }
        else if(col_block_dim <= 32)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 32, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 32, 32);
            }
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 64, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(128, 8, 64, 32);
            }
        }
    }
    else if(row_block_dim <= 16 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 2, 32);
        }
        else if(col_block_dim <= 4)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 4, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 4, 32);
            }
        }
        else if(col_block_dim <= 8)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 8, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 8, 32);
            }
        }
        else if(col_block_dim <= 16)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 16, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 16, 32);
            }
        }
        else if(col_block_dim <= 32)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(128, 16, 32);
        }
        else
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(128, 16, 64);
        }
    }
    else if(row_block_dim <= 32 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 2, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 2, 32);
            }
        }
        else if(col_block_dim <= 4)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 4, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 4, 32);
            }
        }
        else if(col_block_dim <= 8)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 8, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 8, 32);
            }
        }
        else if(col_block_dim <= 16)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 32, 16);
        }
        else if(col_block_dim <= 32)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 32, 32);
        }
        else
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 32, 64);
        }
    }
    else if(row_block_dim <= 64 && col_block_dim <= 32)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(128, 64, 2);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 4);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 8);
        }
        else if(col_block_dim <= 16)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 16);
        }
        else
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 32);
        }
    }
    else if(row_block_dim <= 64 && col_block_dim <= 64
            && !std::is_same<T, rocsparse_double_complex>())
    {
        csr2gebsr_64_64_launcher(handle,
                                 direction,
                                 m,
                                 n,
                                 mb,
                                 nb,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim);
    }
    else
    {
        // Use a blocksize of 32 to handle each block row
        constexpr rocsparse_int block_size       = 32;
        rocsparse_int           rows_per_segment = (row_block_dim + block_size - 1) / block_size;

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

        hipLaunchKernelGGL((csr2gebsr_65_inf_kernel<block_size>),
                           dim3(mb),
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
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
        }
    }

    return rocsparse_status_success;
}

#define launch_csr2gebsr_nnz_block_per_row_multipass_kernel(block_size, blockdim)            \
    hipLaunchKernelGGL((csr2gebsr_nnz_block_per_row_multipass_kernel<block_size, blockdim>), \
                       dim3(mb),                                                             \
                       dim3(block_size),                                                     \
                       0,                                                                    \
                       handle->stream,                                                       \
                       m,                                                                    \
                       n,                                                                    \
                       mb,                                                                   \
                       nb,                                                                   \
                       row_block_dim,                                                        \
                       col_block_dim,                                                        \
                       csr_descr->base,                                                      \
                       csr_row_ptr,                                                          \
                       csr_col_ind,                                                          \
                       bsr_descr->base,                                                      \
                       bsr_row_ptr);

#define launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(block_size, row_blockdim, wf_size) \
    hipLaunchKernelGGL(                                                                            \
        (csr2gebsr_nnz_wavefront_per_row_multipass_kernel<block_size, row_blockdim, wf_size>),     \
        dim3((mb - 1) / (block_size / wf_size) + 1),                                               \
        dim3(block_size),                                                                          \
        0,                                                                                         \
        handle->stream,                                                                            \
        m,                                                                                         \
        n,                                                                                         \
        mb,                                                                                        \
        nb,                                                                                        \
        row_block_dim,                                                                             \
        col_block_dim,                                                                             \
        csr_descr->base,                                                                           \
        csr_row_ptr,                                                                               \
        csr_col_ind,                                                                               \
        bsr_descr->base,                                                                           \
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
                                                    void*                     temp_buffer)
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
              (const void*&)temp_buffer);

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
    if(m < 0 || n < 0 || row_block_dim <= 0 || col_block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    rocsparse_int mb = (m + row_block_dim - 1) / row_block_dim;
    rocsparse_int nb = (n + col_block_dim - 1) / col_block_dim;

    //
    // Quick return if possible, before checking pointer arguments.
    //
    if(m == 0 || n == 0)
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

    //
    // Check pointer arguments
    //
    if(csr_row_ptr == nullptr || bsr_row_ptr == nullptr || bsr_nnz_devhost == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_col_ind == nullptr)
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

    if(row_block_dim == col_block_dim)
    {
        return rocsparse_csr2bsr_nnz(handle,
                                     direction,
                                     m,
                                     n,
                                     csr_descr,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     row_block_dim,
                                     bsr_descr,
                                     bsr_row_ptr,
                                     bsr_nnz_devhost);
    }

    if(row_block_dim == 1)
    {
        hipLaunchKernelGGL(csr2gebsr_nnz_kernel_bm1<256>,
                           dim3((m / 256 + 1)),
                           dim3(256),
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
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                &temp_storage_ptr, temp_storage_size_bytes, handle->stream));
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
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hend,
                                               &bsr_row_ptr[mb],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hstart,
                                               &bsr_row_ptr[0],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            *bsr_nnz_devhost = hend - hstart;
        }

        return rocsparse_status_success;
    }

    if(row_block_dim <= 2)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 4);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 8);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 16);
        }
        else if(col_block_dim <= 16)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 32);
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 64);
            }
            else
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 32);
            }
        }
    }
    else if(row_block_dim <= 4)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 8);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 16);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 32);
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 64);
            }
            else
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 32);
            }
        }
    }
    else if(row_block_dim <= 8)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 8, 16);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 8, 32);
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 8, 64);
            }
            else
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 8, 32);
            }
        }
    }
    else if(row_block_dim <= 16)
    {
        launch_csr2gebsr_nnz_block_per_row_multipass_kernel(256, 16);
    }
    else if(row_block_dim <= 32)
    {
        launch_csr2gebsr_nnz_block_per_row_multipass_kernel(256, 32);
    }
    else if(row_block_dim <= 64)
    {
        launch_csr2gebsr_nnz_block_per_row_multipass_kernel(256, 64);
    }
    else
    {
        // Use a blocksize of 32 to handle each block row
        constexpr rocsparse_int block_size       = 32;
        rocsparse_int           rows_per_segment = (row_block_dim + block_size - 1) / block_size;

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

        hipLaunchKernelGGL((csr2gebsr_nnz_65_inf_kernel<block_size>),
                           dim3(mb),
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
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &hend, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hstart,
                                           &bsr_row_ptr[0],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

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
                                        size_t*                   buffer_size)   \
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
                                                        buffer_size);            \
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
                                     void*                     temp_buffer)      \
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
                                            temp_buffer);                        \
    }

C_IMPL(rocsparse_scsr2gebsr, float);
C_IMPL(rocsparse_dcsr2gebsr, double);
C_IMPL(rocsparse_ccsr2gebsr, rocsparse_float_complex);
C_IMPL(rocsparse_zcsr2gebsr, rocsparse_double_complex);
#undef C_IMPL
