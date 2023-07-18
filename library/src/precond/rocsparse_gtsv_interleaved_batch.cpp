/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_gtsv_interleaved_batch.hpp"
#include "internal/precond/rocsparse_gtsv.h"

#include "gtsv_interleaved_batch_device.h"

template <typename T>
rocsparse_status
    rocsparse_gtsv_interleaved_batch_buffer_size_template(rocsparse_handle               handle,
                                                          rocsparse_gtsv_interleaved_alg alg,
                                                          rocsparse_int                  m,
                                                          const T*                       dl,
                                                          const T*                       d,
                                                          const T*                       du,
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
              replaceX<T>("rocsparse_Xgtsv_interleaved_batch_buffer_size"),
              alg,
              m,
              (const void*&)dl,
              (const void*&)d,
              (const void*&)du,
              (const void*&)x,
              batch_count,
              batch_stride,
              (const void*&)buffer_size);

    // Check algorithm
    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m <= 1 || batch_count < 0 || batch_stride < batch_count)
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
    if(dl == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(d == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(du == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(x == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    switch(alg)
    {
    case rocsparse_gtsv_interleaved_alg_thomas:
    {
        *buffer_size = 0;
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dc1
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dx1
        break;
    }
    case rocsparse_gtsv_interleaved_alg_lu:
    {
        *buffer_size = 0;
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // u2
        *buffer_size += ((sizeof(rocsparse_int) * m * batch_count - 1) / 256 + 1) * 256; // p
        break;
    }
    case rocsparse_gtsv_interleaved_alg_default:
    case rocsparse_gtsv_interleaved_alg_qr:
    {
        *buffer_size = 0;
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // r2
        break;
    }
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gtsv_interleaved_batch_thomas_template(rocsparse_handle handle,
                                                                  rocsparse_int    m,
                                                                  T*               dl,
                                                                  T*               d,
                                                                  T*               du,
                                                                  T*               x,
                                                                  rocsparse_int    batch_count,
                                                                  rocsparse_int    batch_stride,
                                                                  void*            temp_buffer)
{
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    T*    dc1 = reinterpret_cast<T*>(temp_buffer);
    ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
    T* dx1 = reinterpret_cast<T*>(reinterpret_cast<void*>(ptr));
    //  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;

    hipLaunchKernelGGL((gtsv_interleaved_batch_thomas_kernel<128>),
                       dim3(((batch_count - 1) / 128 + 1), 1, 1),
                       dim3(128, 1, 1),
                       0,
                       handle->stream,
                       m,
                       batch_count,
                       batch_stride,
                       dl,
                       d,
                       du,
                       dc1,
                       dx1,
                       x);

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gtsv_interleaved_batch_lu_template(rocsparse_handle handle,
                                                              rocsparse_int    m,
                                                              T*               dl,
                                                              T*               d,
                                                              T*               du,
                                                              T*               x,
                                                              rocsparse_int    batch_count,
                                                              rocsparse_int    batch_stride,
                                                              void*            temp_buffer)
{
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    T*    u2  = reinterpret_cast<T*>(temp_buffer);
    ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
    rocsparse_int* p = reinterpret_cast<rocsparse_int*>(reinterpret_cast<void*>(ptr));
    // ptr += ((sizeof(rocsparse_int) * m * batch_count - 1) / 256 + 1) * 256;

    RETURN_IF_HIP_ERROR(
        hipMemsetAsync(u2, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));

    hipLaunchKernelGGL((gtsv_interleaved_batch_lu_kernel<128>),
                       dim3(((batch_count - 1) / 128 + 1), 1, 1),
                       dim3(128, 1, 1),
                       0,
                       handle->stream,
                       m,
                       batch_count,
                       batch_stride,
                       dl,
                       d,
                       du,
                       u2,
                       p,
                       x);

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gtsv_interleaved_batch_qr_template(rocsparse_handle handle,
                                                              rocsparse_int    m,
                                                              T*               dl,
                                                              T*               d,
                                                              T*               du,
                                                              T*               x,
                                                              rocsparse_int    batch_count,
                                                              rocsparse_int    batch_stride,
                                                              void*            temp_buffer)
{
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    T*    r2  = reinterpret_cast<T*>(ptr);
    //   ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;

    RETURN_IF_HIP_ERROR(
        hipMemsetAsync(r2, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));

    hipLaunchKernelGGL((gtsv_interleaved_batch_qr_kernel<128>),
                       dim3(((batch_count - 1) / 128 + 1), 1, 1),
                       dim3(128, 1, 1),
                       0,
                       handle->stream,
                       m,
                       batch_count,
                       batch_stride,
                       dl,
                       d,
                       du,
                       r2,
                       x);

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gtsv_interleaved_batch_template(rocsparse_handle               handle,
                                                           rocsparse_gtsv_interleaved_alg alg,
                                                           rocsparse_int                  m,
                                                           T*                             dl,
                                                           T*                             d,
                                                           T*                             du,
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
              replaceX<T>("rocsparse_Xgtsv_interleaved_batch"),
              alg,
              m,
              (const void*&)dl,
              (const void*&)d,
              (const void*&)du,
              (const void*&)x,
              batch_count,
              batch_stride,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f gtsv_interleaved_batch -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> ",
              "--batch_count",
              batch_count,
              "--batch_stride",
              batch_stride);

    // Check algorithm
    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m <= 1 || batch_count < 0 || batch_stride < batch_count)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(batch_count == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(dl == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(d == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(du == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(x == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    switch(alg)
    {
    case rocsparse_gtsv_interleaved_alg_thomas:
        return rocsparse_gtsv_interleaved_batch_thomas_template(
            handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
    case rocsparse_gtsv_interleaved_alg_lu:
        return rocsparse_gtsv_interleaved_batch_lu_template(
            handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
    case rocsparse_gtsv_interleaved_alg_default:
    case rocsparse_gtsv_interleaved_alg_qr:
        return rocsparse_gtsv_interleaved_batch_qr_template(
            handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle               handle,        \
                                     rocsparse_gtsv_interleaved_alg alg,           \
                                     rocsparse_int                  m,             \
                                     const TYPE*                    dl,            \
                                     const TYPE*                    d,             \
                                     const TYPE*                    du,            \
                                     const TYPE*                    x,             \
                                     rocsparse_int                  batch_count,   \
                                     rocsparse_int                  batch_stride,  \
                                     size_t*                        buffer_size)   \
    try                                                                            \
    {                                                                              \
        return rocsparse_gtsv_interleaved_batch_buffer_size_template(              \
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, buffer_size); \
    }                                                                              \
    catch(...)                                                                     \
    {                                                                              \
        return exception_to_rocsparse_status();                                    \
    }

C_IMPL(rocsparse_sgtsv_interleaved_batch_buffer_size, float);
C_IMPL(rocsparse_dgtsv_interleaved_batch_buffer_size, double);
C_IMPL(rocsparse_cgtsv_interleaved_batch_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_interleaved_batch_buffer_size, rocsparse_double_complex);

#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle               handle,        \
                                     rocsparse_gtsv_interleaved_alg alg,           \
                                     rocsparse_int                  m,             \
                                     TYPE*                          dl,            \
                                     TYPE*                          d,             \
                                     TYPE*                          du,            \
                                     TYPE*                          x,             \
                                     rocsparse_int                  batch_count,   \
                                     rocsparse_int                  batch_stride,  \
                                     void*                          temp_buffer)   \
    try                                                                            \
    {                                                                              \
        return rocsparse_gtsv_interleaved_batch_template(                          \
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, temp_buffer); \
    }                                                                              \
    catch(...)                                                                     \
    {                                                                              \
        return exception_to_rocsparse_status();                                    \
    }

C_IMPL(rocsparse_sgtsv_interleaved_batch, float);
C_IMPL(rocsparse_dgtsv_interleaved_batch, double);
C_IMPL(rocsparse_cgtsv_interleaved_batch, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_interleaved_batch, rocsparse_double_complex);

#undef C_IMPL
