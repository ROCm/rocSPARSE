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

#include "rocsparse_gemvi.hpp"

#include "definitions.h"
#include "gemvi_device.h"
#include "utility.h"

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void gemvi_scale_kernel(I m, U scalar_device_host, T* x)
{
    auto scalar = load_scalar_device_host(scalar_device_host);

    if(scalar != static_cast<T>(1))
    {
        gemvi_scale_kernel(m, scalar, x);
    }
}

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void gemvi_kernel(I m,
                                                                I n,
                                                                U alpha_device_host,
                                                                const T* __restrict__ A,
                                                                I lda,
                                                                I nnz,
                                                                const T* __restrict__ x_val,
                                                                const I* __restrict__ x_ind,
                                                                U beta_device_host,
                                                                T* __restrict__ y,
                                                                rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
    {
        gemvi_device<BLOCKSIZE, WFSIZE>(m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base);
    }
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_gemvi_dispatch(rocsparse_handle     handle,
                                          rocsparse_operation  trans,
                                          I                    m,
                                          I                    n,
                                          U                    alpha_device_host,
                                          const T*             A,
                                          I                    lda,
                                          I                    nnz,
                                          const T*             x_val,
                                          const I*             x_ind,
                                          U                    beta_device_host,
                                          T*                   y,
                                          rocsparse_index_base idx_base,
                                          void*                temp_buffer)
{
#define GEMVI_DIM 1024
    // If nnz is zero, only compute beta * y
    if(nnz == 0)
    {
        hipLaunchKernelGGL((gemvi_scale_kernel<GEMVI_DIM>),
                           dim3((m - 1) / GEMVI_DIM + 1),
                           dim3(GEMVI_DIM),
                           0,
                           handle->stream,
                           m,
                           beta_device_host,
                           y);

        return rocsparse_status_success;
    }

    if(trans == rocsparse_operation_none)
    {
        if(handle->wavefront_size == 32)
        {
            dim3 gemvi_blocks((m - 1) / 32 + 1);
            dim3 gemvi_threads(GEMVI_DIM);

            hipLaunchKernelGGL((gemvi_kernel<GEMVI_DIM, 32>),
                               gemvi_blocks,
                               gemvi_threads,
                               0,
                               handle->stream,
                               m,
                               n,
                               alpha_device_host,
                               A,
                               lda,
                               nnz,
                               x_val,
                               x_ind,
                               beta_device_host,
                               y,
                               idx_base);
        }
        else
        {
            assert(handle->wavefront_size == 64);

            dim3 gemvi_blocks((m - 1) / 64 + 1);
            dim3 gemvi_threads(GEMVI_DIM);

            hipLaunchKernelGGL((gemvi_kernel<GEMVI_DIM, 64>),
                               gemvi_blocks,
                               gemvi_threads,
                               0,
                               handle->stream,
                               m,
                               n,
                               alpha_device_host,
                               A,
                               lda,
                               nnz,
                               x_val,
                               x_ind,
                               beta_device_host,
                               y,
                               idx_base);
        }
#undef GEMVI_DIM
    }
    else
    {
        return rocsparse_status_not_implemented;
    }

    return rocsparse_status_success;
}

template <typename I, typename T>
rocsparse_status rocsparse_gemvi_template(rocsparse_handle     handle,
                                          rocsparse_operation  trans,
                                          I                    m,
                                          I                    n,
                                          const T*             alpha_device_host,
                                          const T*             A,
                                          I                    lda,
                                          I                    nnz,
                                          const T*             x_val,
                                          const I*             x_ind,
                                          const T*             beta_device_host,
                                          T*                   y,
                                          rocsparse_index_base idx_base,
                                          void*                temp_buffer)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xgemvi"),
                  trans,
                  m,
                  n,
                  *alpha_device_host,
                  (const void*&)A,
                  lda,
                  nnz,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  *beta_device_host,
                  (const void*&)y,
                  idx_base,
                  (const void*&)temp_buffer);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xgemvi"),
                  trans,
                  m,
                  n,
                  (const void*&)alpha_device_host,
                  (const void*&)A,
                  lda,
                  nnz,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  (const void*&)beta_device_host,
                  (const void*&)y,
                  idx_base,
                  (const void*&)temp_buffer);
    }

    // Check operation mode
    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    // Check index base
    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if((m < 0) || (n < 0) || (nnz < 0))
    {
        return rocsparse_status_invalid_size;
    }

    // nnz of sparse vector cannot exceed its size
    if(nnz > n)
    {
        return rocsparse_status_invalid_size;
    }

    // Check leading dimension
    if(trans == rocsparse_operation_none)
    {
        if(lda < m)
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(lda < n)
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Check invalid pointers
    if(m > 0 && n > 0 && nnz > 0)
    {
        if(A == nullptr || x_val == nullptr || x_ind == nullptr || alpha_device_host == nullptr
           || temp_buffer == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
    }
    if(beta_device_host == nullptr || y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if there is no work to do - alpha can be (valid) nullptr!
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        if(alpha_device_host == nullptr && *beta_device_host == static_cast<T>(1))
        {
            return rocsparse_status_success;
        }

        if(alpha_device_host != nullptr)
        {
            if(*alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }
        }
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_gemvi_dispatch(handle,
                                        trans,
                                        m,
                                        n,
                                        alpha_device_host,
                                        A,
                                        lda,
                                        nnz,
                                        x_val,
                                        x_ind,
                                        beta_device_host,
                                        y,
                                        idx_base,
                                        temp_buffer);
    }
    else
    {
        return rocsparse_gemvi_dispatch(handle,
                                        trans,
                                        m,
                                        n,
                                        *alpha_device_host,
                                        A,
                                        lda,
                                        nnz,
                                        x_val,
                                        x_ind,
                                        *beta_device_host,
                                        y,
                                        idx_base,
                                        temp_buffer);
    }

    return rocsparse_status_internal_error;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

// Definition of the C-implementation

// rocsparse_xgemvi_buffer_size
#define CAPI_IMPL(name_, type_)                             \
    rocsparse_status name_(rocsparse_handle    handle,      \
                           rocsparse_operation trans,       \
                           rocsparse_int       m,           \
                           rocsparse_int       n,           \
                           rocsparse_int       nnz,         \
                           size_t*             buffer_size) \
    {                                                       \
        *buffer_size = 4;                                   \
        return rocsparse_status_success;                    \
    }

// C-implementations
CAPI_IMPL(rocsparse_sgemvi_buffer_size, float);
CAPI_IMPL(rocsparse_dgemvi_buffer_size, double);
CAPI_IMPL(rocsparse_cgemvi_buffer_size, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zgemvi_buffer_size, rocsparse_double_complex);

// Undefine the CAPI_IMPL macro
#undef CAPI_IMPL

// rocsparse_xgemvi
#define CAPI_IMPL(name_, type_)                              \
    rocsparse_status name_(rocsparse_handle     handle,      \
                           rocsparse_operation  trans,       \
                           rocsparse_int        m,           \
                           rocsparse_int        n,           \
                           const type_*         alpha,       \
                           const type_*         A,           \
                           rocsparse_int        lda,         \
                           rocsparse_int        nnz,         \
                           const type_*         x_val,       \
                           const rocsparse_int* x_ind,       \
                           const type_*         beta,        \
                           type_*               y,           \
                           rocsparse_index_base idx_base,    \
                           void*                temp_buffer) \
    {                                                        \
        try                                                  \
        {                                                    \
            return rocsparse_gemvi_template(handle,          \
                                            trans,           \
                                            m,               \
                                            n,               \
                                            alpha,           \
                                            A,               \
                                            lda,             \
                                            nnz,             \
                                            x_val,           \
                                            x_ind,           \
                                            beta,            \
                                            y,               \
                                            idx_base,        \
                                            temp_buffer);    \
        }                                                    \
        catch(...)                                           \
        {                                                    \
            return exception_to_rocsparse_status();          \
        }                                                    \
    }

// C-implementations
CAPI_IMPL(rocsparse_sgemvi, float);
CAPI_IMPL(rocsparse_dgemvi, double);
CAPI_IMPL(rocsparse_cgemvi, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zgemvi, rocsparse_double_complex);

// Undefine the CAPI_IMPL macro
#undef CAPI_IMPL
}
