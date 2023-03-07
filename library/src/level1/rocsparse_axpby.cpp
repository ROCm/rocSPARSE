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

#include "axpby_device.h"
#include "definitions.h"
#include "rocsparse_axpyi.hpp"
#include "utility.h"

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void axpby_scale_kernel(I size, U alpha_device_host, T* __restrict__ x)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != static_cast<T>(1))
    {
        axpby_scale_device<BLOCKSIZE>(size, alpha, x);
    }
}

template <typename I, typename T>
rocsparse_status rocsparse_axpby_template(rocsparse_handle            handle,
                                          const void*                 alpha,
                                          rocsparse_const_spvec_descr x,
                                          const void*                 beta,
                                          rocsparse_dnvec_descr       y)
{
    // Check for valid sizes
    if(y->size < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return
    if(y->size == 0)
    {
        return rocsparse_status_success;
    }

#define SCALE_DIM 256
    dim3 scale_blocks((y->size - 1) / SCALE_DIM + 1);
    dim3 scale_threads(SCALE_DIM);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((axpby_scale_kernel<SCALE_DIM>),
                           scale_blocks,
                           scale_threads,
                           0,
                           handle->stream,
                           (I)y->size,
                           (const T*)beta,
                           (T*)y->values);
    }
    else
    {
        const T* beta_ptr = (const T*)beta;

        if(*beta_ptr != static_cast<T>(1))
        {
            hipLaunchKernelGGL((axpby_scale_kernel<SCALE_DIM>),
                               scale_blocks,
                               scale_threads,
                               0,
                               handle->stream,
                               (I)y->size,
                               *beta_ptr,
                               (T*)y->values);
        }
    }
#undef SCALE_DIM

    return rocsparse_axpyi_template<I, T>(handle,
                                          (I)x->nnz,
                                          (const T*)alpha,
                                          (const T*)x->const_val_data,
                                          (const I*)x->const_idx_data,
                                          (T*)y->values,
                                          x->idx_base);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_axpby(rocsparse_handle            handle,
                                            const void*                 alpha,
                                            rocsparse_const_spvec_descr x,
                                            const void*                 beta,
                                            rocsparse_dnvec_descr       y)
try
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_axpby",
              (const void*&)alpha,
              (const void*&)x,
              (const void*&)beta,
              (const void*&)y);

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(x);
    RETURN_IF_NULLPTR(y);
    RETURN_IF_NULLPTR(alpha);
    RETURN_IF_NULLPTR(beta);

    // Check if descriptors are initialized
    if(x->init == false || y->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    // Check for matching types while we do not support mixed precision computation
    if(x->data_type != y->data_type)
    {
        return rocsparse_status_not_implemented;
    }

    // single real ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f32_r)
    {
        return rocsparse_axpby_template<int32_t, float>(handle, alpha, x, beta, y);
    }
    // double real ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f64_r)
    {
        return rocsparse_axpby_template<int32_t, double>(handle, alpha, x, beta, y);
    }
    // single complex ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f32_c)
    {
        return rocsparse_axpby_template<int32_t, rocsparse_float_complex>(
            handle, alpha, x, beta, y);
    }
    // double complex ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f64_c)
    {
        return rocsparse_axpby_template<int32_t, rocsparse_double_complex>(
            handle, alpha, x, beta, y);
    }
    // single real ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f32_r)
    {
        return rocsparse_axpby_template<int64_t, float>(handle, alpha, x, beta, y);
    }
    // double real ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f64_r)
    {
        return rocsparse_axpby_template<int64_t, double>(handle, alpha, x, beta, y);
    }
    // single complex ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f32_c)
    {
        return rocsparse_axpby_template<int64_t, rocsparse_float_complex>(
            handle, alpha, x, beta, y);
    }
    // double complex ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f64_c)
    {
        return rocsparse_axpby_template<int64_t, rocsparse_double_complex>(
            handle, alpha, x, beta, y);
    }

    return rocsparse_status_not_implemented;
}
catch(...)
{
    return exception_to_rocsparse_status();
}
