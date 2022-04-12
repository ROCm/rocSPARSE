/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef ROCSPARSE_GEMMI_HPP
#define ROCSPARSE_GEMMI_HPP

#include "definitions.h"
#include "gemmi_device.h"
#include "utility.h"

template <unsigned int BLOCKSIZE, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void gemmi_scale_kernel(rocsparse_int size, U alpha_device_host, T* __restrict__ data)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    gemmi_scale_kernel<BLOCKSIZE>(size, alpha, data);
}

template <unsigned int BLOCKSIZE, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void gemmit_kernel(rocsparse_int m,
                       U             alpha_device_host,
                       const T* __restrict__ A,
                       rocsparse_int lda,
                       const rocsparse_int* __restrict__ csr_row_ptr,
                       const rocsparse_int* __restrict__ csr_col_ind,
                       const T* __restrict__ csr_val,
                       U beta_device_host,
                       T* __restrict__ C,
                       rocsparse_int        ldc,
                       rocsparse_index_base base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    gemmit_kernel<BLOCKSIZE>(
        m, alpha, A, lda, csr_row_ptr, csr_col_ind, csr_val, beta, C, ldc, base);
}

template <typename T>
rocsparse_status rocsparse_gemmi_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          rocsparse_int             k,
                                          rocsparse_int             nnz,
                                          const T*                  alpha,
                                          const T*                  A,
                                          rocsparse_int             lda,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          const T*                  beta,
                                          T*                        C,
                                          rocsparse_int             ldc)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgemmi"),
              trans_A,
              trans_B,
              m,
              n,
              k,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)A,
              lda,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              LOG_TRACE_SCALAR_VALUE(handle, beta),
              (const void*&)C,
              ldc);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }

    if(trans_B != rocsparse_operation_transpose)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments

    // beta and C is always required
    if(beta == nullptr || C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // A is only required if k != 0
    if(k != 0 && (alpha == nullptr || A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // B is only required if k != 0 and nnz != 0
    if(k != 0 && nnz != 0
       && (csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check leading dimensions
    if(lda < std::max(1, m) || ldc < std::max(1, m))
    {
        return rocsparse_status_invalid_size;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // If k == 0, scale C with beta
    if(k == 0)
    {
#define SCALE_DIM 256
        dim3 scale_blocks((m * n - 1) / SCALE_DIM + 1);
        dim3 scale_threads(SCALE_DIM);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL((gemmi_scale_kernel<SCALE_DIM>),
                               scale_blocks,
                               scale_threads,
                               0,
                               stream,
                               m * n,
                               beta,
                               C);
        }
        else
        {
            if(*beta == static_cast<T>(0))
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(C, 0, sizeof(T) * m * n, stream));
            }
            else if(*beta != static_cast<T>(1))
            {
                hipLaunchKernelGGL((gemmi_scale_kernel<SCALE_DIM>),
                                   scale_blocks,
                                   scale_threads,
                                   0,
                                   stream,
                                   m * n,
                                   *beta,
                                   C);
            }
        }
#undef SCALE_DIM

        return rocsparse_status_success;
    }

#define GEMMIT_DIM 256
    dim3 gemmit_blocks((m - 1) / GEMMIT_DIM + 1, n);
    dim3 gemmit_threads(GEMMIT_DIM);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((gemmit_kernel<GEMMIT_DIM>),
                           gemmit_blocks,
                           gemmit_threads,
                           0,
                           stream,
                           m,
                           alpha,
                           A,
                           lda,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           beta,
                           C,
                           ldc,
                           descr->base);
    }
    else
    {
        // Quick return
        if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
        {
            return rocsparse_status_success;
        }
        else if(*alpha == static_cast<T>(0))
        {
            if(*beta == static_cast<T>(0))
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(C, 0, sizeof(T) * m * n, stream));
            }
            else
            {
#define SCALE_DIM 256
                dim3 scale_blocks((m * n - 1) / SCALE_DIM + 1);
                dim3 scale_threads(SCALE_DIM);

                hipLaunchKernelGGL((gemmi_scale_kernel<SCALE_DIM>),
                                   scale_blocks,
                                   scale_threads,
                                   0,
                                   stream,
                                   m * n,
                                   *beta,
                                   C);
#undef SCALE_DIM
            }

            return rocsparse_status_success;
        }

        hipLaunchKernelGGL((gemmit_kernel<GEMMIT_DIM>),
                           gemmit_blocks,
                           gemmit_threads,
                           0,
                           stream,
                           m,
                           *alpha,
                           A,
                           lda,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           *beta,
                           C,
                           ldc,
                           descr->base);
    }
#undef GEMMIT_DIM

    return rocsparse_status_success;
}

#endif // ROCSPARSE_GEMMI_HPP
