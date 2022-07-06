/*! \file */
/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_gpsv_interleaved_batch.hpp"

#include "gpsv_interleaved_batch_device.h"

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_gpsv_interleaved_alg value_)
{
    switch(value_)
    {
    case rocsparse_gpsv_interleaved_alg_default:
    case rocsparse_gpsv_interleaved_alg_qr:
    {
        return false;
    }
    }
    return true;
};

template <typename T>
rocsparse_status
    rocsparse_gpsv_interleaved_batch_buffer_size_template(rocsparse_handle               handle,
                                                          rocsparse_gpsv_interleaved_alg alg,
                                                          rocsparse_int                  m,
                                                          const T*                       ds,
                                                          const T*                       dl,
                                                          const T*                       d,
                                                          const T*                       du,
                                                          const T*                       dw,
                                                          const T*                       x,
                                                          rocsparse_int batch_count,
                                                          rocsparse_int batch_stride,
                                                          size_t*       buffer_size)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgpsv_interleaved_batch_buffer_size"),
              alg,
              m,
              (const void*&)ds,
              (const void*&)dl,
              (const void*&)d,
              (const void*&)du,
              (const void*&)dw,
              (const void*&)x,
              batch_count,
              batch_stride,
              (const void*&)buffer_size);

    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 5 || batch_count < 0 || batch_stride < batch_count)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid buffer_size pointer
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(batch_count == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(ds == nullptr || dl == nullptr || d == nullptr || du == nullptr || dw == nullptr
       || x == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    *buffer_size = 0;

    if(std::is_same<T, float>() || std::is_same<T, double>())
    {
        *buffer_size += sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256; // dt1
        *buffer_size += sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256; // dt2
        *buffer_size += sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256; // dB
    }
    else
    {
        *buffer_size += sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256; // r3
        *buffer_size += sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256; // r4
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gpsv_interleaved_batch_template(rocsparse_handle               handle,
                                                           rocsparse_gpsv_interleaved_alg alg,
                                                           rocsparse_int                  m,
                                                           T*                             ds,
                                                           T*                             dl,
                                                           T*                             d,
                                                           T*                             du,
                                                           T*                             dw,
                                                           T*                             x,
                                                           rocsparse_int batch_count,
                                                           rocsparse_int batch_stride,
                                                           void*         temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgpsv_interleaved_batch"),
              alg,
              m,
              (const void*&)ds,
              (const void*&)dl,
              (const void*&)d,
              (const void*&)du,
              (const void*&)dw,
              (const void*&)x,
              batch_count,
              batch_stride,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f gpsv_interleaved_batch -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> ",
              "--batch_count",
              batch_count,
              "--batch_stride",
              batch_stride);

    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 5 || batch_count < 0 || batch_stride < batch_count)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(batch_count == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(ds == nullptr || dl == nullptr || d == nullptr || du == nullptr || dw == nullptr
       || x == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    if(std::is_same<T, float>() || std::is_same<T, double>())
    {
        T* dt1 = reinterpret_cast<T*>(ptr);
        ptr += sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256;

        T* dt2 = reinterpret_cast<T*>(ptr);
        ptr += sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256;

        T* B = reinterpret_cast<T*>(ptr);

        // Initialize buffers with zero
        RETURN_IF_HIP_ERROR(hipMemsetAsync(dt1, 0, sizeof(T) * m * batch_count, stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(dt2, 0, sizeof(T) * m * batch_count, stream));

#define GPSV_DIM 256
        dim3 gpsv_blocks((batch_count - 1) / GPSV_DIM + 1);
        dim3 gpsv_threads(GPSV_DIM);

        // Copy strided B into buffer
        hipLaunchKernelGGL((gpsv_strided_gather<GPSV_DIM>),
                           gpsv_blocks,
                           gpsv_threads,
                           0,
                           stream,
                           m,
                           batch_count,
                           batch_stride,
                           x,
                           B);

        // Launch kernel
        hipLaunchKernelGGL((gpsv_interleaved_batch_householder_qr_kernel<GPSV_DIM>),
                           gpsv_blocks,
                           gpsv_threads,
                           0,
                           stream,
                           m,
                           batch_count,
                           batch_stride,
                           ds,
                           dl,
                           d,
                           du,
                           dw,
                           x,
                           dt1,
                           dt2,
                           B);
#undef GPSV_DIM
    }
    else
    {
        T* r3 = reinterpret_cast<T*>(ptr);
        ptr += sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256;
        T* r4 = reinterpret_cast<T*>(ptr);
        ptr += sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256;

        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            r3, 0, sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            r4, 0, sizeof(T) * ((m * batch_count - 1) / 256 + 1) * 256, handle->stream));

        hipLaunchKernelGGL((gpsv_interleaved_batch_givens_qr_kernel<128>),
                           dim3(((batch_count - 1) / 128 + 1), 1, 1),
                           dim3(128, 1, 1),
                           0,
                           handle->stream,
                           m,
                           batch_count,
                           batch_stride,
                           ds,
                           dl,
                           d,
                           du,
                           dw,
                           r3,
                           r4,
                           x);
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                                 \
    extern "C" rocsparse_status NAME(rocsparse_handle               handle,                \
                                     rocsparse_gpsv_interleaved_alg alg,                   \
                                     rocsparse_int                  m,                     \
                                     const TYPE*                    ds,                    \
                                     const TYPE*                    dl,                    \
                                     const TYPE*                    d,                     \
                                     const TYPE*                    du,                    \
                                     const TYPE*                    dw,                    \
                                     const TYPE*                    x,                     \
                                     rocsparse_int                  batch_count,           \
                                     rocsparse_int                  batch_stride,          \
                                     size_t*                        buffer_size)           \
    {                                                                                      \
        return rocsparse_gpsv_interleaved_batch_buffer_size_template(                      \
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, buffer_size); \
    }

C_IMPL(rocsparse_sgpsv_interleaved_batch_buffer_size, float);
C_IMPL(rocsparse_dgpsv_interleaved_batch_buffer_size, double);
C_IMPL(rocsparse_cgpsv_interleaved_batch_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgpsv_interleaved_batch_buffer_size, rocsparse_double_complex);
#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                                 \
    extern "C" rocsparse_status NAME(rocsparse_handle               handle,                \
                                     rocsparse_gpsv_interleaved_alg alg,                   \
                                     rocsparse_int                  m,                     \
                                     TYPE*                          ds,                    \
                                     TYPE*                          dl,                    \
                                     TYPE*                          d,                     \
                                     TYPE*                          du,                    \
                                     TYPE*                          dw,                    \
                                     TYPE*                          x,                     \
                                     rocsparse_int                  batch_count,           \
                                     rocsparse_int                  batch_stride,          \
                                     void*                          temp_buffer)           \
    {                                                                                      \
        return rocsparse_gpsv_interleaved_batch_template(                                  \
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, temp_buffer); \
    }

C_IMPL(rocsparse_sgpsv_interleaved_batch, float);
C_IMPL(rocsparse_dgpsv_interleaved_batch, double);
C_IMPL(rocsparse_cgpsv_interleaved_batch, rocsparse_float_complex);
C_IMPL(rocsparse_zgpsv_interleaved_batch, rocsparse_double_complex);
#undef C_IMPL
