/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_coomv.hpp"
#include "../conversion/rocsparse_coo2csr.hpp"
#include "definitions.h"
#include "utility.h"

#include "coomv_device.h"

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_coomv_alg value_)
{
    switch(value_)
    {
    case rocsparse_coomv_alg_default:
    case rocsparse_coomv_alg_segmented:
    case rocsparse_coomv_alg_atomic:
    {
        return false;
    }
    }
    return true;
};

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coomv_scale(I size, U beta_device_host, T* __restrict__ data)
{
    auto beta = load_scalar_device_host(beta_device_host);
    if(beta != static_cast<T>(1))
    {
        coomv_scale_device<BLOCKSIZE>(size, beta, data);
    }
}

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coomvn_segmented_loops(I nnz,
                                I nloops,
                                U alpha_device_host,
                                const I* __restrict__ coo_row_ind,
                                const I* __restrict__ coo_col_ind,
                                const T* __restrict__ coo_val,
                                const T* __restrict__ x,
                                T* __restrict__ y,
                                I* __restrict__ row_block_red,
                                T* __restrict__ val_block_red,
                                rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != static_cast<T>(0))
    {
        coomvn_segmented_loops_device<BLOCKSIZE>(nnz,
                                                 nloops,
                                                 alpha,
                                                 coo_row_ind,
                                                 coo_col_ind,
                                                 coo_val,
                                                 x,
                                                 y,
                                                 row_block_red,
                                                 val_block_red,
                                                 idx_base);
    }
}

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coomvn_segmented_loops_reduce(I nblocks,
                                       U alpha_device_host,
                                       const I* __restrict__ row_block_red,
                                       const T* __restrict__ val_block_red,
                                       T* __restrict__ y)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != static_cast<T>(0))
    {
        coomvn_segmented_loops_reduce_device<BLOCKSIZE>(nblocks, row_block_red, val_block_red, y);
    }
}

template <unsigned int BLOCKSIZE, unsigned int LOOPS, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coomvn_atomic_loops(I nnz,
                             U alpha_device_host,
                             const I* __restrict__ coo_row_ind,
                             const I* __restrict__ coo_col_ind,
                             const T* __restrict__ coo_val,
                             const T* __restrict__ x,
                             T* __restrict__ y,
                             rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != static_cast<T>(0))
    {
        coomvn_atomic_loops_device<BLOCKSIZE, LOOPS>(
            nnz, alpha, coo_row_ind, coo_col_ind, coo_val, x, y, idx_base);
    }
}

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void coomvt_kernel(rocsparse_operation trans,
                                                                 I                   nnz,
                                                                 U alpha_device_host,
                                                                 const I* __restrict__ coo_row_ind,
                                                                 const I* __restrict__ coo_col_ind,
                                                                 const T* __restrict__ coo_val,
                                                                 const T* __restrict__ x,
                                                                 T* __restrict__ y,
                                                                 rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != static_cast<T>(0))
    {
        coomvt_device(trans, nnz, alpha, coo_row_ind, coo_col_ind, coo_val, x, y, idx_base);
    }
}

template <typename I, typename T>
rocsparse_status rocsparse_coomv_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   rocsparse_coomv_alg       alg,
                                                   I                         m,
                                                   I                         n,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  coo_val,
                                                   const I*                  coo_row_ind,
                                                   const I*                  coo_col_ind)
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
              replaceX<T>("rocsparse_Xcoomv_analysis"),
              trans,
              alg,
              m,
              n,
              nnz,
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind);

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
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    switch(trans)
    {
    case rocsparse_operation_none:
    {
        I* max_nnz     = nullptr;
        I* csr_row_ptr = nullptr;
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&max_nnz, sizeof(I)));
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&csr_row_ptr, sizeof(I) * (m + 1)));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(max_nnz, 0, sizeof(I), handle->stream));

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_coo2csr_template(handle, coo_row_ind, nnz, m, csr_row_ptr, descr->base));

        hipLaunchKernelGGL((csr_max_nnz_per_row<256>),
                           dim3((m - 1) / 256 + 1),
                           dim3(256),
                           0,
                           handle->stream,
                           m,
                           csr_row_ptr,
                           max_nnz);

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &descr->max_nnz_per_row, max_nnz, sizeof(I), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        RETURN_IF_HIP_ERROR(hipFree(max_nnz));
        RETURN_IF_HIP_ERROR(hipFree(csr_row_ptr));
        break;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        break;
    }
    }

    return rocsparse_status_success;
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomv_atomic_dispatch(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 I                         m,
                                                 I                         n,
                                                 I                         nnz,
                                                 U                         alpha_device_host,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  coo_val,
                                                 const I*                  coo_row_ind,
                                                 const I*                  coo_col_ind,
                                                 const T*                  x,
                                                 U                         beta_device_host,
                                                 T*                        y)
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
            RETURN_IF_HIP_ERROR(hipMemsetAsync(y, 0, sizeof(T) * ysize, handle->stream));
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
        if(descr->max_nnz_per_row <= 10 * 256)
        {
            hipLaunchKernelGGL((coomvn_atomic_loops<256, 1>),
                               dim3((nnz - 1) / (1 * 256) + 1),
                               dim3(256),
                               0,
                               stream,
                               nnz,
                               alpha_device_host,
                               coo_row_ind,
                               coo_col_ind,
                               coo_val,
                               x,
                               y,
                               descr->base);
        }
        else
        {
            hipLaunchKernelGGL((coomvn_atomic_loops<256, 2>),
                               dim3((nnz - 1) / (2 * 256) + 1),
                               dim3(256),
                               0,
                               stream,
                               nnz,
                               alpha_device_host,
                               coo_row_ind,
                               coo_col_ind,
                               coo_val,
                               x,
                               y,
                               descr->base);
        }
        break;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        coomvt_kernel<1024><<<(nnz - 1) / 1024 + 1, 1024, 0, handle->stream>>>(
            trans, nnz, alpha_device_host, coo_row_ind, coo_col_ind, coo_val, x, y, descr->base);
        break;
    }
    }

    return rocsparse_status_success;
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomv_segmented_dispatch(rocsparse_handle          handle,
                                                    rocsparse_operation       trans,
                                                    I                         m,
                                                    I                         n,
                                                    I                         nnz,
                                                    U                         alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  coo_val,
                                                    const I*                  coo_row_ind,
                                                    const I*                  coo_col_ind,
                                                    const T*                  x,
                                                    U                         beta_device_host,
                                                    T*                        y)
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
            RETURN_IF_HIP_ERROR(hipMemsetAsync(y, 0, sizeof(T) * ysize, handle->stream));
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

        hipLaunchKernelGGL((coomvn_segmented_loops<COOMVN_DIM>),
                           dim3(nblocks),
                           dim3(COOMVN_DIM),
                           0,
                           stream,
                           nnz,
                           nloops,
                           alpha_device_host,
                           coo_row_ind,
                           coo_col_ind,
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
        coomvt_kernel<1024><<<(nnz - 1) / 1024 + 1, 1024, 0, handle->stream>>>(
            trans, nnz, alpha_device_host, coo_row_ind, coo_col_ind, coo_val, x, y, descr->base);
        break;
    }
    }

    return rocsparse_status_success;
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomv_dispatch(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          rocsparse_coomv_alg       alg,
                                          I                         m,
                                          I                         n,
                                          I                         nnz,
                                          U                         alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  coo_val,
                                          const I*                  coo_row_ind,
                                          const I*                  coo_col_ind,
                                          const T*                  x,
                                          U                         beta_device_host,
                                          T*                        y)
{
    switch(alg)
    {
    case rocsparse_coomv_alg_default:
    case rocsparse_coomv_alg_segmented:
    {
        return rocsparse_coomv_segmented_dispatch(handle,
                                                  trans,
                                                  m,
                                                  n,
                                                  nnz,
                                                  alpha_device_host,
                                                  descr,
                                                  coo_val,
                                                  coo_row_ind,
                                                  coo_col_ind,
                                                  x,
                                                  beta_device_host,
                                                  y);
    }
    case rocsparse_coomv_alg_atomic:
    {
        return rocsparse_coomv_atomic_dispatch(handle,
                                               trans,
                                               m,
                                               n,
                                               nnz,
                                               alpha_device_host,
                                               descr,
                                               coo_val,
                                               coo_row_ind,
                                               coo_col_ind,
                                               x,
                                               beta_device_host,
                                               y);
    }
    }

    return rocsparse_status_invalid_value;
}

template <typename I, typename T>
rocsparse_status rocsparse_coomv_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          rocsparse_coomv_alg       alg,
                                          I                         m,
                                          I                         n,
                                          I                         nnz,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  coo_val,
                                          const I*                  coo_row_ind,
                                          const I*                  coo_col_ind,
                                          const T*                  x,
                                          const T*                  beta_device_host,
                                          T*                        y)
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
              replaceX<T>("rocsparse_Xcoomv"),
              trans,
              alg,
              m,
              n,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)x,
              LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
              (const void*&)y);

    log_bench(handle,
              "./rocsparse-bench -f coomv -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> ",
              "--alpha",
              LOG_BENCH_SCALAR_VALUE(handle, alpha_device_host),
              "--beta",
              LOG_BENCH_SCALAR_VALUE(handle, beta_device_host));

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
    if(m == 0 || n == 0)
    {
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

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_coomv_dispatch(handle,
                                        trans,
                                        alg,
                                        m,
                                        n,
                                        nnz,
                                        alpha_device_host,
                                        descr,
                                        coo_val,
                                        coo_row_ind,
                                        coo_col_ind,
                                        x,
                                        beta_device_host,
                                        y);
    }
    else
    {
        return rocsparse_coomv_dispatch(handle,
                                        trans,
                                        alg,
                                        m,
                                        n,
                                        nnz,
                                        *alpha_device_host,
                                        descr,
                                        coo_val,
                                        coo_row_ind,
                                        coo_col_ind,
                                        x,
                                        *beta_device_host,
                                        y);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                              \
    template rocsparse_status rocsparse_coomv_analysis_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                      \
        rocsparse_operation       trans,                                       \
        rocsparse_coomv_alg       coomv_alg,                                   \
        ITYPE                     m,                                           \
        ITYPE                     n,                                           \
        ITYPE                     nnz,                                         \
        const rocsparse_mat_descr descr,                                       \
        const TTYPE*              coo_val,                                     \
        const ITYPE*              coo_row_ind,                                 \
        const ITYPE*              coo_col_ind);

INSTANTIATE(int32_t, float)
INSTANTIATE(int32_t, double)
INSTANTIATE(int32_t, rocsparse_float_complex)
INSTANTIATE(int32_t, rocsparse_double_complex)
INSTANTIATE(int64_t, float)
INSTANTIATE(int64_t, double)
INSTANTIATE(int64_t, rocsparse_float_complex)
INSTANTIATE(int64_t, rocsparse_double_complex)
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                                     \
    template rocsparse_status rocsparse_coomv_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                             \
        rocsparse_operation       trans,                              \
        rocsparse_coomv_alg       coomv_alg,                          \
        ITYPE                     m,                                  \
        ITYPE                     n,                                  \
        ITYPE                     nnz,                                \
        const TTYPE*              alpha_device_host,                  \
        const rocsparse_mat_descr descr,                              \
        const TTYPE*              coo_val,                            \
        const ITYPE*              coo_row_ind,                        \
        const ITYPE*              coo_col_ind,                        \
        const TTYPE*              x,                                  \
        const TTYPE*              beta_device_host,                   \
        TTYPE*                    y);

INSTANTIATE(int32_t, float)
INSTANTIATE(int32_t, double)
INSTANTIATE(int32_t, rocsparse_float_complex)
INSTANTIATE(int32_t, rocsparse_double_complex)
INSTANTIATE(int64_t, float)
INSTANTIATE(int64_t, double)
INSTANTIATE(int64_t, rocsparse_float_complex)
INSTANTIATE(int64_t, rocsparse_double_complex)
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             m,           \
                                     rocsparse_int             n,           \
                                     rocsparse_int             nnz,         \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               coo_val,     \
                                     const rocsparse_int*      coo_row_ind, \
                                     const rocsparse_int*      coo_col_ind, \
                                     const TYPE*               x,           \
                                     const TYPE*               beta,        \
                                     TYPE*                     y)           \
    {                                                                       \
        return rocsparse_coomv_template(handle,                             \
                                        trans,                              \
                                        rocsparse_coomv_alg_segmented,      \
                                        m,                                  \
                                        n,                                  \
                                        nnz,                                \
                                        alpha,                              \
                                        descr,                              \
                                        coo_val,                            \
                                        coo_row_ind,                        \
                                        coo_col_ind,                        \
                                        x,                                  \
                                        beta,                               \
                                        y);                                 \
    }

C_IMPL(rocsparse_scoomv, float);
C_IMPL(rocsparse_dcoomv, double);
C_IMPL(rocsparse_ccoomv, rocsparse_float_complex);
C_IMPL(rocsparse_zcoomv, rocsparse_double_complex);
#undef C_IMPL
