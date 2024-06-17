/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_ellmv.h"
#include "rocsparse_ellmv.hpp"

#include "control.h"
#include "ellmv_device.h"
#include "utility.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename I, typename A, typename X, typename Y, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellmvn_kernel(I m,
                       I n,
                       I ell_width,
                       U alpha_device_host,
                       const I* __restrict__ ell_col_ind,
                       const A* __restrict__ ell_val,
                       const X* __restrict__ x,
                       U beta_device_host,
                       Y* __restrict__ y,
                       rocsparse_index_base idx_base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        auto beta  = rocsparse::load_scalar_device_host(beta_device_host);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::ellmvn_device<BLOCKSIZE>(
                m, n, ell_width, alpha, ell_col_ind, ell_val, x, beta, y, idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename X, typename Y, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellmvt_kernel(rocsparse_operation trans,
                       I                   m,
                       I                   n,
                       I                   ell_width,
                       U                   alpha_device_host,
                       const I* __restrict__ ell_col_ind,
                       const A* __restrict__ ell_val,
                       const X* __restrict__ x,
                       Y* __restrict__ y,
                       rocsparse_index_base idx_base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        if(alpha != 0)
        {
            rocsparse::ellmvt_device<BLOCKSIZE>(
                trans, m, n, ell_width, alpha, ell_col_ind, ell_val, x, y, idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename Y, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellmvt_scale_kernel(I size, U scalar_device_host, Y* __restrict__ data)
    {
        auto scalar = rocsparse::load_scalar_device_host(scalar_device_host);
        if(scalar != 1)
        {
            rocsparse::ellmvt_scale_device(size, scalar, data);
        }
    }

    template <typename I, typename A, typename X, typename Y, typename U>
    rocsparse_status ellmv_dispatch(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    I                         m,
                                    I                         n,
                                    U                         alpha_device_host,
                                    const rocsparse_mat_descr descr,
                                    const A*                  ell_val,
                                    const I*                  ell_col_ind,
                                    I                         ell_width,
                                    const X*                  x,
                                    U                         beta_device_host,
                                    Y*                        y)
    {
        // Stream
        hipStream_t stream = handle->stream;

        // Run different ellmv kernels
        if(trans == rocsparse_operation_none)
        {
#define ELLMVN_DIM 512

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ellmvn_kernel<ELLMVN_DIM>),
                                               dim3((m - 1) / ELLMVN_DIM + 1),
                                               dim3(ELLMVN_DIM),
                                               0,
                                               stream,
                                               m,
                                               n,
                                               ell_width,
                                               alpha_device_host,
                                               ell_col_ind,
                                               ell_val,
                                               x,
                                               beta_device_host,
                                               y,
                                               descr->base);

#undef ELLMVN_DIM
        }
        else
        {
#define ELLMVT_DIM 1024
            // Scale y with beta
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ellmvt_scale_kernel<ELLMVT_DIM>),
                                               dim3((n - 1) / ELLMVT_DIM + 1),
                                               dim3(ELLMVT_DIM),
                                               0,
                                               stream,
                                               n,
                                               beta_device_host,
                                               y);

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ellmvt_kernel<ELLMVT_DIM>),
                                               dim3((m - 1) / ELLMVT_DIM + 1),
                                               dim3(ELLMVT_DIM),
                                               0,
                                               stream,
                                               trans,
                                               m,
                                               n,
                                               ell_width,
                                               alpha_device_host,
                                               ell_col_ind,
                                               ell_val,
                                               x,
                                               y,
                                               descr->base);
#undef ELLMVT_DIM
        }

        return rocsparse_status_success;
    }
}

template <typename T, typename I, typename A, typename X, typename Y>
rocsparse_status rocsparse::ellmv_template(rocsparse_handle          handle, // 0
                                           rocsparse_operation       trans, //1
                                           I                         m, //2
                                           I                         n, //3
                                           const T*                  alpha_device_host, //4
                                           const rocsparse_mat_descr descr, //5
                                           const A*                  ell_val, //6
                                           const I*                  ell_col_ind, //7
                                           I                         ell_width, //8
                                           const X*                  x, //9
                                           const T*                  beta_device_host, //10
                                           Y*                        y) //11
{
    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(5, descr);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xellmv"),
                         trans,
                         m,
                         n,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                         (const void*&)descr,
                         (const void*&)ell_val,
                         (const void*&)ell_col_ind,
                         ell_width,
                         (const void*&)x,
                         LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                         (const void*&)y);

    ROCSPARSE_CHECKARG_ENUM(1, trans);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        5, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);
    ROCSPARSE_CHECKARG_SIZE(8, ell_width);

    // Sanity check
    ROCSPARSE_CHECKARG(
        8, ell_width, ((m == 0 || n == 0) && ell_width != 0), rocsparse_status_invalid_size);

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        // matrix never accessed however still need to update y vector
        rocsparse_int ysize = (trans == rocsparse_operation_none) ? m : n;
        if(ysize > 0)
        {
            if(y == nullptr && beta_device_host == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::scale_array<256>),
                                                   dim3((ysize - 1) / 256 + 1),
                                                   dim3(256),
                                                   0,
                                                   handle->stream,
                                                   ysize,
                                                   y,
                                                   beta_device_host);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::scale_array<256>),
                                                   dim3((ysize - 1) / 256 + 1),
                                                   dim3(256),
                                                   0,
                                                   handle->stream,
                                                   ysize,
                                                   y,
                                                   *beta_device_host);
            }
        }

        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(4, alpha_device_host);
    ROCSPARSE_CHECKARG_POINTER(10, beta_device_host);

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    // Check the rest of the pointer arguments
    ROCSPARSE_CHECKARG_POINTER(6, ell_val);
    ROCSPARSE_CHECKARG_POINTER(7, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(9, x);
    ROCSPARSE_CHECKARG_POINTER(11, y);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellmv_dispatch(handle,
                                                            trans,
                                                            m,
                                                            n,
                                                            alpha_device_host,
                                                            descr,
                                                            ell_val,
                                                            ell_col_ind,
                                                            ell_width,
                                                            x,
                                                            beta_device_host,
                                                            y));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellmv_dispatch(handle,
                                                            trans,
                                                            m,
                                                            n,
                                                            *alpha_device_host,
                                                            descr,
                                                            ell_val,
                                                            ell_col_ind,
                                                            ell_width,
                                                            x,
                                                            *beta_device_host,
                                                            y));
        return rocsparse_status_success;
    }
    return rocsparse_status_success;
}

#define INSTANTIATE(TTYPE, ITYPE)                                                              \
    template rocsparse_status rocsparse::ellmv_template(rocsparse_handle          handle,      \
                                                        rocsparse_operation       trans,       \
                                                        ITYPE                     m,           \
                                                        ITYPE                     n,           \
                                                        const TTYPE*              alpha,       \
                                                        const rocsparse_mat_descr descr,       \
                                                        const TTYPE*              ell_val,     \
                                                        const ITYPE*              ell_col_ind, \
                                                        ITYPE                     ell_width,   \
                                                        const TTYPE*              x,           \
                                                        const TTYPE*              beta,        \
                                                        TTYPE*                    y);

INSTANTIATE(float, int32_t);
INSTANTIATE(float, int64_t);
INSTANTIATE(double, int32_t);
INSTANTIATE(double, int64_t);
INSTANTIATE(rocsparse_float_complex, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t);
#undef INSTANTIATE

#define INSTANTIATE_MIXED(TTYPE, ITYPE, ATYPE, XTYPE, YTYPE)                                   \
    template rocsparse_status rocsparse::ellmv_template(rocsparse_handle          handle,      \
                                                        rocsparse_operation       trans,       \
                                                        ITYPE                     m,           \
                                                        ITYPE                     n,           \
                                                        const TTYPE*              alpha,       \
                                                        const rocsparse_mat_descr descr,       \
                                                        const ATYPE*              ell_val,     \
                                                        const ITYPE*              ell_col_ind, \
                                                        ITYPE                     ell_width,   \
                                                        const XTYPE*              x,           \
                                                        const TTYPE*              beta,        \
                                                        YTYPE*                    y);

INSTANTIATE_MIXED(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(float, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(
    rocsparse_float_complex, int32_t, float, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE_MIXED(
    rocsparse_float_complex, int64_t, float, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE_MIXED(
    rocsparse_double_complex, int32_t, double, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE_MIXED(
    rocsparse_double_complex, int64_t, double, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE_MIXED(double, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, float, double, double);

INSTANTIATE_MIXED(rocsparse_double_complex,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

#undef INSTANTIATE_MIXED

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                        \
                                     rocsparse_operation       trans,                         \
                                     rocsparse_int             m,                             \
                                     rocsparse_int             n,                             \
                                     const TYPE*               alpha,                         \
                                     const rocsparse_mat_descr descr,                         \
                                     const TYPE*               ell_val,                       \
                                     const rocsparse_int*      ell_col_ind,                   \
                                     rocsparse_int             ell_width,                     \
                                     const TYPE*               x,                             \
                                     const TYPE*               beta,                          \
                                     TYPE*                     y)                             \
    try                                                                                       \
    {                                                                                         \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellmv_template(                                  \
            handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y)); \
        return rocsparse_status_success;                                                      \
    }                                                                                         \
    catch(...)                                                                                \
    {                                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                                         \
    }

C_IMPL(rocsparse_sellmv, float);
C_IMPL(rocsparse_dellmv, double);
C_IMPL(rocsparse_cellmv, rocsparse_float_complex);
C_IMPL(rocsparse_zellmv, rocsparse_double_complex);
#undef C_IMPL
