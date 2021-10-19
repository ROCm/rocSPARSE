/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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

#include "definitions.h"
#include "rocsparse_coomv.hpp"
#include "utility.h"

#include "coomv_device.h"

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coomv_scale(I size, U beta_device_host, T* __restrict__ data)
{
    auto beta = load_scalar_device_host(beta_device_host);
    if(beta != static_cast<T>(1))
    {
        coomv_scale_device(size, beta, data);
    }
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void coomvn_aos_wf(I nnz,
                                                                 I loops,
                                                                 U alpha_device_host,
                                                                 const I* __restrict__ coo_ind,
                                                                 const T* __restrict__ coo_val,
                                                                 const T* __restrict__ x,
                                                                 T* __restrict__ y,
                                                                 I* __restrict__ row_block_red,
                                                                 T* __restrict__ val_block_red,
                                                                 rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    coomvn_aos_general_wf_reduce<BLOCKSIZE, WF_SIZE>(
        nnz, loops, alpha, coo_ind, coo_val, x, y, row_block_red, val_block_red, idx_base);
}

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void coomvt_aos_kernel(rocsparse_operation trans,
                                                                     I                   nnz,
                                                                     U alpha_device_host,
                                                                     const I* __restrict__ coo_ind,
                                                                     const T* __restrict__ coo_val,
                                                                     const T* __restrict__ x,
                                                                     T* __restrict__ y,
                                                                     rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    coomvt_aos_device(trans, nnz, alpha, coo_ind, coo_val, x, y, idx_base);
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomv_aos_dispatch(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              I                         m,
                                              I                         n,
                                              I                         nnz,
                                              U                         alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const T*                  coo_val,
                                              const I*                  coo_ind,
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
            // LCOV_EXCL_START
            hipLaunchKernelGGL((coomvn_aos_wf<COOMVN_DIM, 32>),
                               coomvn_blocks,
                               coomvn_threads,
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
            // LCOV_EXCL_STOP
        }
        else
        {
            assert(handle->wavefront_size == 64);
            hipLaunchKernelGGL((coomvn_aos_wf<COOMVN_DIM, 64>),
                               coomvn_blocks,
                               coomvn_threads,
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

template <typename I, typename T>
rocsparse_status rocsparse_coomv_aos_template(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              I                         m,
                                              I                         n,
                                              I                         nnz,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const T*                  coo_val,
                                              const I*                  coo_ind,
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
        return rocsparse_coomv_aos_dispatch(handle,
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
    else
    {
        return rocsparse_coomv_aos_dispatch(handle,
                                            trans,
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

#define INSTANTIATE(ITYPE, TTYPE)                                         \
    template rocsparse_status rocsparse_coomv_aos_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                 \
        rocsparse_operation       trans,                                  \
        ITYPE                     m,                                      \
        ITYPE                     n,                                      \
        ITYPE                     nnz,                                    \
        const TTYPE*              alpha_device_host,                      \
        const rocsparse_mat_descr descr,                                  \
        const TTYPE*              coo_val,                                \
        const ITYPE*              coo_ind,                                \
        const TTYPE*              x,                                      \
        const TTYPE*              beta_device_host,                       \
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
