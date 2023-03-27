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

#include "rocsparse_coomv_aos.hpp"
#include "definitions.h"
#include "utility.h"

#include "coomv_device.h"

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_coomv_aos_alg value_)
{
    switch(value_)
    {
    case rocsparse_coomv_aos_alg_default:
    case rocsparse_coomv_aos_alg_segmented:
    case rocsparse_coomv_aos_alg_atomic:
    {
        return false;
    }
    }
    return true;
};

template <unsigned int BLOCKSIZE, typename I, typename Y, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void coomv_scale(I size, U beta_device_host, Y* __restrict__ data)
{
    auto beta = load_scalar_device_host(beta_device_host);
    if(beta != 1)
    {
        coomv_scale_device<BLOCKSIZE>(size, beta, data);
    }
}

template <unsigned int BLOCKSIZE,
          typename I,
          typename A,
          typename X,
          typename Y,
          typename T,
          typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void coomvn_aos_segmented_loops(int64_t nnz,
                                I       nloops,
                                U       alpha_device_host,
                                const I* __restrict__ coo_ind,
                                const A* __restrict__ coo_val,
                                const X* __restrict__ x,
                                Y* __restrict__ y,
                                I* __restrict__ row_block_red,
                                T* __restrict__ val_block_red,
                                rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != 0)
    {
        coomvn_aos_segmented_loops_device<BLOCKSIZE>(
            nnz, nloops, alpha, coo_ind, coo_val, x, y, row_block_red, val_block_red, idx_base);
    }
}

template <unsigned int BLOCKSIZE, typename I, typename Y, typename T, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void coomvn_segmented_loops_reduce(I nblocks,
                                   U alpha_device_host,
                                   const I* __restrict__ row_block_red,
                                   const T* __restrict__ val_block_red,
                                   Y* __restrict__ y)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != 0)
    {
        coomvn_segmented_loops_reduce_device<BLOCKSIZE>(nblocks, row_block_red, val_block_red, y);
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int LOOPS,
          typename I,
          typename A,
          typename X,
          typename Y,
          typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void coomvn_aos_atomic_loops(int64_t nnz,
                             U       alpha_device_host,
                             const I* __restrict__ coo_ind,
                             const A* __restrict__ coo_val,
                             const X* __restrict__ x,
                             Y* __restrict__ y,
                             rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != 0)
    {
        coomvn_aos_atomic_loops_device<BLOCKSIZE, LOOPS>(
            nnz, alpha, coo_ind, coo_val, x, y, idx_base);
    }
}

template <unsigned int BLOCKSIZE, typename I, typename A, typename X, typename Y, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void coomvt_aos_kernel(rocsparse_operation trans,
                       int64_t             nnz,
                       U                   alpha_device_host,
                       const I* __restrict__ coo_ind,
                       const A* __restrict__ coo_val,
                       const X* __restrict__ x,
                       Y* __restrict__ y,
                       rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != 0)
    {
        coomvt_aos_device(trans, nnz, alpha, coo_ind, coo_val, x, y, idx_base);
    }
}

template <typename I, typename A, typename X, typename Y, typename U>
rocsparse_status rocsparse_coomv_aos_atomic_dispatch(rocsparse_handle          handle,
                                                     rocsparse_operation       trans,
                                                     I                         m,
                                                     I                         n,
                                                     int64_t                   nnz,
                                                     U                         alpha_device_host,
                                                     const rocsparse_mat_descr descr,
                                                     const A*                  coo_val,
                                                     const I*                  coo_ind,
                                                     const X*                  x,
                                                     U                         beta_device_host,
                                                     Y*                        y)
{
    // Stream
    hipStream_t stream = handle->stream;

    I ysize = (trans == rocsparse_operation_none) ? m : n;

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // Scale y with beta
        hipLaunchKernelGGL((coomv_scale<1024>),
                           dim3((ysize - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           handle->stream,
                           ysize,
                           beta_device_host,
                           y);
    }
    else
    {
        auto beta = load_scalar_device_host(beta_device_host);
        // If beta == 0.0 we need to set y to 0
        if(beta == 0)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(y, 0, sizeof(Y) * ysize, handle->stream));
        }
        else if(beta != 1)
        {
            hipLaunchKernelGGL((coomv_scale<1024>),
                               dim3((ysize - 1) / 1024 + 1),
                               dim3(1024),
                               0,
                               handle->stream,
                               ysize,
                               beta,
                               y);
        }
    }

    // Run different coomv kernels
    switch(trans)
    {
    case rocsparse_operation_none:
    {
        hipLaunchKernelGGL((coomvn_aos_atomic_loops<256, 1>),
                           dim3((nnz - 1) / 256 + 1),
                           dim3(256),
                           0,
                           stream,
                           nnz,
                           alpha_device_host,
                           coo_ind,
                           coo_val,
                           x,
                           y,
                           descr->base);
        break;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        coomvt_aos_kernel<1024><<<(nnz - 1) / 1024 + 1, 1024, 0, handle->stream>>>(
            trans, nnz, alpha_device_host, coo_ind, coo_val, x, y, descr->base);
        break;
    }
    }

    return rocsparse_status_success;
}

template <typename T, typename I, typename A, typename X, typename Y, typename U>
rocsparse_status rocsparse_coomv_aos_segmented_dispatch(rocsparse_handle          handle,
                                                        rocsparse_operation       trans,
                                                        I                         m,
                                                        I                         n,
                                                        int64_t                   nnz,
                                                        U                         alpha_device_host,
                                                        const rocsparse_mat_descr descr,
                                                        const A*                  coo_val,
                                                        const I*                  coo_ind,
                                                        const X*                  x,
                                                        U                         beta_device_host,
                                                        Y*                        y)
{
    // Stream
    hipStream_t stream = handle->stream;

    I ysize = (trans == rocsparse_operation_none) ? m : n;

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // Scale y with beta
        hipLaunchKernelGGL((coomv_scale<1024>),
                           dim3((ysize - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           handle->stream,
                           ysize,
                           beta_device_host,
                           y);
    }
    else
    {
        auto beta = load_scalar_device_host(beta_device_host);
        // If beta == 0.0 we need to set y to 0
        if(beta == static_cast<T>(0))
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(y, 0, sizeof(Y) * ysize, handle->stream));
        }
        else if(beta != static_cast<T>(1))
        {
            hipLaunchKernelGGL((coomv_scale<1024>),
                               dim3((ysize - 1) / 1024 + 1),
                               dim3(1024),
                               0,
                               handle->stream,
                               ysize,
                               beta,
                               y);
        }
    }

    // Run different coomv kernels
    switch(trans)
    {
    case rocsparse_operation_none:
    {
#define COOMVN_DIM 256
        int maxthreads = handle->properties.maxThreadsPerBlock;
        int nprocs     = 2 * handle->properties.multiProcessorCount;
        int maxblocks  = (nprocs * maxthreads - 1) / COOMVN_DIM + 1;

        I minblocks = (nnz - 1) / COOMVN_DIM + 1;
        I nblocks   = maxblocks < minblocks ? maxblocks : minblocks;
        I nloops    = (nnz - 1) / (COOMVN_DIM * nblocks) + 1;

        // Buffer
        char* ptr = reinterpret_cast<char*>(handle->buffer);
        ptr += 256;

        // row block reduction buffer
        I* row_block_red = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * nblocks - 1) / 256 + 1) * 256;

        // val block reduction buffer
        T* val_block_red = reinterpret_cast<T*>(ptr);

        hipLaunchKernelGGL((coomvn_aos_segmented_loops<COOMVN_DIM>),
                           dim3(nblocks),
                           dim3(COOMVN_DIM),
                           0,
                           stream,
                           nnz,
                           nloops,
                           alpha_device_host,
                           coo_ind,
                           coo_val,
                           x,
                           y,
                           row_block_red,
                           val_block_red,
                           descr->base);

        hipLaunchKernelGGL((coomvn_segmented_loops_reduce<COOMVN_DIM>),
                           dim3(1),
                           dim3(COOMVN_DIM),
                           0,
                           stream,
                           nblocks,
                           alpha_device_host,
                           row_block_red,
                           val_block_red,
                           y);
#undef COOMVN_DIM
        break;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        coomvt_aos_kernel<1024><<<(nnz - 1) / 1024 + 1, 1024, 0, handle->stream>>>(
            trans, nnz, alpha_device_host, coo_ind, coo_val, x, y, descr->base);

        break;
    }
    }

    return rocsparse_status_success;
}

template <typename T, typename I, typename A, typename X, typename Y, typename U>
rocsparse_status rocsparse_coomv_aos_dispatch(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              rocsparse_coomv_aos_alg   alg,
                                              I                         m,
                                              I                         n,
                                              int64_t                   nnz,
                                              U                         alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const A*                  coo_val,
                                              const I*                  coo_ind,
                                              const X*                  x,
                                              U                         beta_device_host,
                                              Y*                        y)
{
    switch(alg)
    {
    case rocsparse_coomv_aos_alg_default:
    case rocsparse_coomv_aos_alg_atomic:
    {
        return rocsparse_coomv_aos_atomic_dispatch(handle,
                                                   trans,
                                                   m,
                                                   n,
                                                   nnz,
                                                   alpha_device_host,
                                                   descr,
                                                   coo_val,
                                                   coo_ind,
                                                   x,
                                                   beta_device_host,
                                                   y);
    }

    case rocsparse_coomv_aos_alg_segmented:
    {
        return rocsparse_coomv_aos_segmented_dispatch<T>(handle,
                                                         trans,
                                                         m,
                                                         n,
                                                         nnz,
                                                         alpha_device_host,
                                                         descr,
                                                         coo_val,
                                                         coo_ind,
                                                         x,
                                                         beta_device_host,
                                                         y);
    }
    }

    return rocsparse_status_invalid_value;
}

template <typename T, typename I, typename A, typename X, typename Y>
rocsparse_status rocsparse_coomv_aos_template(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              rocsparse_coomv_aos_alg   alg,
                                              I                         m,
                                              I                         n,
                                              int64_t                   nnz,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const A*                  coo_val,
                                              const I*                  coo_ind,
                                              const X*                  x,
                                              const T*                  beta_device_host,
                                              Y*                        y)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcoomv_aos"),
              trans,
              m,
              n,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_ind,
              (const void*&)x,
              LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
              (const void*&)y);

    log_bench(handle,
              "./rocsparse-bench -f coomv_aos -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> ",
              "--alpha",
              LOG_BENCH_SCALAR_VALUE(handle, alpha_device_host),
              "--beta",
              LOG_BENCH_SCALAR_VALUE(handle, beta_device_host));

    // Check index base
    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
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
                hipLaunchKernelGGL((scale_array<256>),
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
                hipLaunchKernelGGL((scale_array<256>),
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
    if(alpha_device_host == nullptr || beta_device_host == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    // Check the rest of the pointer arguments
    if(x == nullptr || y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((coo_val == nullptr && coo_ind != nullptr) || (coo_val != nullptr && coo_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_coomv_aos_dispatch<T>(handle,
                                               trans,
                                               alg,
                                               m,
                                               n,
                                               nnz,
                                               alpha_device_host,
                                               descr,
                                               coo_val,
                                               coo_ind,
                                               x,
                                               beta_device_host,
                                               y);
    }
    else
    {
        return rocsparse_coomv_aos_dispatch<T>(handle,
                                               trans,
                                               alg,
                                               m,
                                               n,
                                               nnz,
                                               *alpha_device_host,
                                               descr,
                                               coo_val,
                                               coo_ind,
                                               x,
                                               *beta_device_host,
                                               y);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(TTYPE, ITYPE)                                                                 \
    template rocsparse_status rocsparse_coomv_aos_template(rocsparse_handle        handle,        \
                                                           rocsparse_operation     trans,         \
                                                           rocsparse_coomv_aos_alg coomv_aos_alg, \
                                                           ITYPE                   m,             \
                                                           ITYPE                   n,             \
                                                           int64_t                 nnz,           \
                                                           const TTYPE* alpha_device_host,        \
                                                           const rocsparse_mat_descr descr,       \
                                                           const TTYPE*              coo_val,     \
                                                           const ITYPE*              coo_ind,     \
                                                           const TTYPE*              x,           \
                                                           const TTYPE* beta_device_host,         \
                                                           TTYPE*       y);

INSTANTIATE(float, int32_t);
INSTANTIATE(float, int64_t);
INSTANTIATE(double, int32_t);
INSTANTIATE(double, int64_t);
INSTANTIATE(rocsparse_float_complex, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t);
#undef INSTANTIATE

#define INSTANTIATE_MIXED(TTYPE, ITYPE, ATYPE, XTYPE, YTYPE)                                      \
    template rocsparse_status rocsparse_coomv_aos_template(rocsparse_handle        handle,        \
                                                           rocsparse_operation     trans,         \
                                                           rocsparse_coomv_aos_alg coomv_aos_alg, \
                                                           ITYPE                   m,             \
                                                           ITYPE                   n,             \
                                                           int64_t                 nnz,           \
                                                           const TTYPE* alpha_device_host,        \
                                                           const rocsparse_mat_descr descr,       \
                                                           const ATYPE*              coo_val,     \
                                                           const ITYPE*              coo_ind,     \
                                                           const XTYPE*              x,           \
                                                           const TTYPE* beta_device_host,         \
                                                           YTYPE*       y);

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
