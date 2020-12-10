/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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
#include "definitions.h"
#include "utility.h"

#include "coomv_device.h"

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) __global__
    void coomv_scale(I size, U beta_device_host, T* __restrict__ data)
{
    auto beta = load_scalar_device_host(beta_device_host);
    if(beta != static_cast<T>(1))
    {
        coomv_scale_device(size, beta, data);
    }
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) __global__ void coomvn_wf(I nnz,
                                                       I loops,
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
    coomvn_general_wf_reduce<BLOCKSIZE, WF_SIZE>(nnz,
                                                 loops,
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

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomv_dispatch(rocsparse_handle          handle,
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

    // Run different coomv kernels
    if(trans == rocsparse_operation_none)
    {
#define COOMVN_DIM 128
        int maxthreads = handle->properties.maxThreadsPerBlock;
        int nprocs     = handle->properties.multiProcessorCount;
        int maxblocks  = (nprocs * maxthreads - 1) / COOMVN_DIM + 1;

        I minblocks = (nnz - 1) / COOMVN_DIM + 1;
        I nblocks   = maxblocks < minblocks ? maxblocks : minblocks;
        I nwfs      = nblocks * (COOMVN_DIM / handle->wavefront_size);
        I nloops    = (nnz / handle->wavefront_size + 1) / nwfs + 1;

        dim3 coomvn_blocks(nblocks);
        dim3 coomvn_threads(COOMVN_DIM);

        // Buffer
        char* ptr = reinterpret_cast<char*>(handle->buffer);
        ptr += 256;

        // row block reduction buffer
        I* row_block_red = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * nwfs - 1) / 256 + 1) * 256;

        // val block reduction buffer
        T* val_block_red = reinterpret_cast<T*>(ptr);

        if(handle->wavefront_size == 32)
        {
            hipLaunchKernelGGL((coomvn_wf<COOMVN_DIM, 32>),
                               coomvn_blocks,
                               coomvn_threads,
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
        }
        else if(handle->wavefront_size == 64)
        {
            hipLaunchKernelGGL((coomvn_wf<COOMVN_DIM, 64>),
                               coomvn_blocks,
                               coomvn_threads,
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
        }
        else
        {
            return rocsparse_status_arch_mismatch;
        }

        hipLaunchKernelGGL((coomvn_general_block_reduce<COOMVN_DIM>),
                           dim3(1),
                           coomvn_threads,
                           0,
                           stream,
                           nwfs,
                           row_block_red,
                           val_block_red,
                           y);
#undef COOMVN_DIM
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

template <typename I, typename T>
rocsparse_status rocsparse_coomv_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
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
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcoomv"),
                  trans,
                  m,
                  n,
                  nnz,
                  *alpha_device_host,
                  (const void*&)descr,
                  (const void*&)coo_val,
                  (const void*&)coo_row_ind,
                  (const void*&)coo_col_ind,
                  (const void*&)x,
                  *beta_device_host,
                  (const void*&)y);

        log_bench(handle,
                  "./rocsparse-bench -f coomv -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx> ",
                  "--alpha",
                  *alpha_device_host,
                  "--beta",
                  *beta_device_host);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcoomv"),
                  trans,
                  m,
                  n,
                  nnz,
                  (const void*&)alpha_device_host,
                  (const void*&)descr,
                  (const void*&)coo_val,
                  (const void*&)coo_row_ind,
                  (const void*&)coo_col_ind,
                  (const void*&)x,
                  (const void*&)beta_device_host,
                  (const void*&)y);
    }

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
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
    if(coo_val == nullptr || coo_row_ind == nullptr || coo_col_ind == nullptr || x == nullptr
       || y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // Scale y with beta
        hipLaunchKernelGGL((coomv_scale<1024>),
                           dim3((m - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           handle->stream,
                           m,
                           beta_device_host,
                           y);

        return rocsparse_coomv_dispatch(handle,
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
    else
    {

        // If beta == 0.0 we need to set y to 0
        if(*beta_device_host == static_cast<T>(0))
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(y, 0, sizeof(T) * m, handle->stream));
        }
        else if(*beta_device_host != static_cast<T>(1))
        {
            hipLaunchKernelGGL((coomv_scale<1024>),
                               dim3((m - 1) / 1024 + 1),
                               dim3(1024),
                               0,
                               handle->stream,
                               m,
                               *beta_device_host,
                               y);
        }
        return rocsparse_coomv_dispatch(handle,
                                        trans,
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

#define INSTANTIATE(ITYPE, TTYPE)                                     \
    template rocsparse_status rocsparse_coomv_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                             \
        rocsparse_operation       trans,                              \
        ITYPE                     m,                                  \
        ITYPE                     n,                                  \
        ITYPE                     nnz,                                \
        const TTYPE*              alpha,                              \
        const rocsparse_mat_descr descr,                              \
        const TTYPE*              coo_val,                            \
        const ITYPE*              coo_row_ind,                        \
        const ITYPE*              coo_col_ind,                        \
        const TTYPE*              x,                                  \
        const TTYPE*              beta,                               \
        TTYPE*                    y);

INSTANTIATE(int32_t, float)
INSTANTIATE(int32_t, double)
INSTANTIATE(int32_t, rocsparse_float_complex)
INSTANTIATE(int32_t, rocsparse_double_complex)
INSTANTIATE(int64_t, float)
INSTANTIATE(int64_t, double)
INSTANTIATE(int64_t, rocsparse_float_complex)
INSTANTIATE(int64_t, rocsparse_double_complex)

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
