/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level1/rocsparse_roti.h"
#include "rocsparse_roti.hpp"

template <unsigned int BLOCKSIZE, typename I, typename T>
ROCSPARSE_DEVICE_ILF void
    roti_device(I nnz, T* x_val, const I* x_ind, T* y, T c, T s, rocsparse_index_base idx_base)
{
    I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

    if(idx >= nnz)
    {
        return;
    }

    I i = x_ind[idx] - idx_base;

    T xr = x_val[idx];
    T yr = y[i];

    x_val[idx] = c * xr + s * yr;
    y[i]       = c * yr - s * xr;
}

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void roti_kernel(I                    nnz,
                 T*                   x_val,
                 const I*             x_ind,
                 T*                   y,
                 U                    c_device_host,
                 U                    s_device_host,
                 rocsparse_index_base idx_base)
{
    auto c = load_scalar_device_host(c_device_host);
    auto s = load_scalar_device_host(s_device_host);
    if(c == static_cast<T>(1) && s == static_cast<T>(0))
    {
        return;
    }
    roti_device<BLOCKSIZE>(nnz, x_val, x_ind, y, c, s, idx_base);
}

template <typename I, typename T>
rocsparse_status rocsparse_roti_template(rocsparse_handle     handle, //0
                                         I                    nnz, //1
                                         T*                   x_val, //2
                                         const I*             x_ind, //3
                                         T*                   y, //4
                                         const T*             c, //5
                                         const T*             s, //6
                                         rocsparse_index_base idx_base) //7
{
    // Check for valid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging // TODO bench logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xroti"),
              nnz,
              (const void*&)x_val,
              (const void*&)x_ind,
              (const void*&)y,
              LOG_TRACE_SCALAR_VALUE(handle, c),
              LOG_TRACE_SCALAR_VALUE(handle, s),
              idx_base);

    // Check index base
    ROCSPARSE_CHECKARG_SIZE(1, nnz);
    ROCSPARSE_CHECKARG_ARRAY(2, nnz, x_val);
    ROCSPARSE_CHECKARG_ARRAY(3, nnz, x_ind);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, y);
    ROCSPARSE_CHECKARG_POINTER(5, c);
    ROCSPARSE_CHECKARG_POINTER(6, s);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define ROTI_DIM 512
    dim3 roti_blocks((nnz - 1) / ROTI_DIM + 1);
    dim3 roti_threads(ROTI_DIM);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((roti_kernel<ROTI_DIM>),
                           roti_blocks,
                           roti_threads,
                           0,
                           stream,
                           nnz,
                           x_val,
                           x_ind,
                           y,
                           c,
                           s,
                           idx_base);
    }
    else
    {
        if(*c == static_cast<T>(1) && *s == static_cast<T>(0))
        {
            return rocsparse_status_success;
        }

        hipLaunchKernelGGL((roti_kernel<ROTI_DIM>),
                           roti_blocks,
                           roti_threads,
                           0,
                           stream,
                           nnz,
                           x_val,
                           x_ind,
                           y,
                           *c,
                           *s,
                           idx_base);
    }
#undef ROTI_DIM
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sroti(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            float*               x_val,
                                            const rocsparse_int* x_ind,
                                            float*               y,
                                            const float*         c,
                                            const float*         s,
                                            rocsparse_index_base idx_base)
try
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_roti_template(handle, nnz, x_val, x_ind, y, c, s, idx_base));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_droti(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            double*              x_val,
                                            const rocsparse_int* x_ind,
                                            double*              y,
                                            const double*        c,
                                            const double*        s,
                                            rocsparse_index_base idx_base)
try
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_roti_template(handle, nnz, x_val, x_ind, y, c, s, idx_base));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

#define INSTANTIATE(I, T)                                                          \
    template rocsparse_status rocsparse_roti_template(rocsparse_handle     handle, \
                                                      I                    nnz,    \
                                                      T*                   x_val,  \
                                                      const I*             x_ind,  \
                                                      T*                   y,      \
                                                      const T*             c,      \
                                                      const T*             s,      \
                                                      rocsparse_index_base idx_base)

INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_double_complex);

#undef INSTANTIATE
