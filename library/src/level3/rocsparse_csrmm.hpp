/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_CSRMM_HPP
#define ROCSPARSE_CSRMM_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "csrmm_device.h"

#include <hip/hip_runtime.h>

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__launch_bounds__(256) __global__
    void csrmmnn_kernel_host_pointer(rocsparse_int m,
                                     rocsparse_int n,
                                     rocsparse_int k,
                                     rocsparse_int nnz,
                                     T alpha,
                                     const rocsparse_int* __restrict__ csr_row_ptr,
                                     const rocsparse_int* __restrict__ csr_col_ind,
                                     const T* __restrict__ csr_val,
                                     const T* __restrict__ B,
                                     rocsparse_int ldb,
                                     T beta,
                                     T* __restrict__ C,
                                     rocsparse_int ldc,
                                     rocsparse_index_base idx_base)
{
    csrmmnn_general_device<T, BLOCKSIZE, WF_SIZE>(
        m, n, k, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, B, ldb, beta, C, ldc, idx_base);
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__launch_bounds__(256) __global__
    void csrmmnn_kernel_device_pointer(rocsparse_int m,
                                       rocsparse_int n,
                                       rocsparse_int k,
                                       rocsparse_int nnz,
                                       const T* alpha,
                                       const rocsparse_int* __restrict__ csr_row_ptr,
                                       const rocsparse_int* __restrict__ csr_col_ind,
                                       const T* __restrict__ csr_val,
                                       const T* __restrict__ B,
                                       rocsparse_int ldb,
                                       const T* beta,
                                       T* __restrict__ C,
                                       rocsparse_int ldc,
                                       rocsparse_index_base idx_base)
{
    if(*alpha == 0.0 && *beta == 1.0)
    {
        return;
    }

    csrmmnn_general_device<T, BLOCKSIZE, WF_SIZE>(
        m, n, k, nnz, *alpha, csr_row_ptr, csr_col_ind, csr_val, B, ldb, *beta, C, ldc, idx_base);
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__launch_bounds__(256) __global__
    void csrmmnt_kernel_host_pointer(rocsparse_int offset,
                                     rocsparse_int ncol,
                                     rocsparse_int m,
                                     rocsparse_int n,
                                     rocsparse_int k,
                                     rocsparse_int nnz,
                                     T alpha,
                                     const rocsparse_int* __restrict__ csr_row_ptr,
                                     const rocsparse_int* __restrict__ csr_col_ind,
                                     const T* __restrict__ csr_val,
                                     const T* __restrict__ B,
                                     rocsparse_int ldb,
                                     T beta,
                                     T* __restrict__ C,
                                     rocsparse_int ldc,
                                     rocsparse_index_base idx_base)
{
    csrmmnt_general_device<T, BLOCKSIZE, WF_SIZE>(offset,
                                                  ncol,
                                                  m,
                                                  n,
                                                  k,
                                                  nnz,
                                                  alpha,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  csr_val,
                                                  B,
                                                  ldb,
                                                  beta,
                                                  C,
                                                  ldc,
                                                  idx_base);
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__launch_bounds__(256) __global__
    void csrmmnt_kernel_device_pointer(rocsparse_int offset,
                                       rocsparse_int ncol,
                                       rocsparse_int m,
                                       rocsparse_int n,
                                       rocsparse_int k,
                                       rocsparse_int nnz,
                                       const T* alpha,
                                       const rocsparse_int* __restrict__ csr_row_ptr,
                                       const rocsparse_int* __restrict__ csr_col_ind,
                                       const T* __restrict__ csr_val,
                                       const T* __restrict__ B,
                                       rocsparse_int ldb,
                                       const T* beta,
                                       T* __restrict__ C,
                                       rocsparse_int ldc,
                                       rocsparse_index_base idx_base)
{
    if(*alpha == 0.0 && *beta == 1.0)
    {
        return;
    }

    csrmmnt_general_device<T, BLOCKSIZE, WF_SIZE>(offset,
                                                  ncol,
                                                  m,
                                                  n,
                                                  k,
                                                  nnz,
                                                  *alpha,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  csr_val,
                                                  B,
                                                  ldb,
                                                  *beta,
                                                  C,
                                                  ldc,
                                                  idx_base);
}

template <typename T>
rocsparse_status rocsparse_csrmm_template(rocsparse_handle handle,
                                          rocsparse_operation trans_A,
                                          rocsparse_operation trans_B,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int k,
                                          rocsparse_int nnz,
                                          const T* alpha,
                                          const rocsparse_mat_descr descr,
                                          const T* csr_val,
                                          const rocsparse_int* csr_row_ptr,
                                          const rocsparse_int* csr_col_ind,
                                          const T* B,
                                          rocsparse_int ldb,
                                          const T* beta,
                                          T* C,
                                          rocsparse_int ldc)
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

    // Logging TODO bench logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  nnz,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  *beta,
                  (const void*&)C,
                  ldc);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  nnz,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  (const void*&)beta,
                  (const void*&)C,
                  ldc);
    }

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(k < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check leading dimension of B
    rocsparse_int one = 1;
    if(trans_B == rocsparse_operation_none)
    {
        if(trans_A == rocsparse_operation_none)
        {
            if(ldb < std::max(one, k))
            {
                return rocsparse_status_invalid_size;
            }
        }
        else
        {
            if(ldb < std::max(one, m))
            {
                return rocsparse_status_invalid_size;
            }
        }
    }
    else
    {
        if(ldb < std::max(one, n))
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check leading dimension of C
    if(trans_A == rocsparse_operation_none)
    {
        if(ldc < std::max(one, m))
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(ldc < std::max(one, k))
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different csrmv kernels
    if(trans_A == rocsparse_operation_none)
    {
        if(trans_B == rocsparse_operation_none)
        {
#define CSRMMNN_DIM 256
#define SUB_WF_SIZE 8
            dim3 csrmmnn_blocks((SUB_WF_SIZE * m - 1) / CSRMMNN_DIM + 1, (n - 1) / SUB_WF_SIZE + 1);
            dim3 csrmmnn_threads(CSRMMNN_DIM);

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                hipLaunchKernelGGL((csrmmnn_kernel_device_pointer<T, CSRMMNN_DIM, SUB_WF_SIZE>),
                                   csrmmnn_blocks,
                                   csrmmnn_threads,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   nnz,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   B,
                                   ldb,
                                   beta,
                                   C,
                                   ldc,
                                   descr->base);
            }
            else
            {
                if(*alpha == 0.0 && *beta == 1.0)
                {
                    return rocsparse_status_success;
                }

                hipLaunchKernelGGL((csrmmnn_kernel_host_pointer<T, CSRMMNN_DIM, SUB_WF_SIZE>),
                                   csrmmnn_blocks,
                                   csrmmnn_threads,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   nnz,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   B,
                                   ldb,
                                   *beta,
                                   C,
                                   ldc,
                                   descr->base);
            }
#undef SUB_WF_SIZE
#undef CSRMMNN_DIM
        }
        else if(trans_B == rocsparse_operation_transpose)
        {
            // Average nnz per row of A
            rocsparse_int avg_row_nnz = (nnz - 1) / m + 1;

#define CSRMMNT_DIM 256
            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                // Computation is split into two parts, main and remainder
                // First step: Compute main, which is the maximum number of
                //             columns of B that is dividable by the next
                //             power of two of the average row nnz of A.
                // Second step: Compute remainder, which is the remaining
                //              columns of B.
                rocsparse_int main      = 0;
                rocsparse_int remainder = 0;

                // Launch appropriate kernel depending on row nnz of A
                if(avg_row_nnz < 16)
                {
                    remainder = n % 8;
                    main      = n - remainder;

                    // Launch main kernel if enough columns of B
                    if(main > 0)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_device_pointer<T, CSRMMNT_DIM, 8>),
                                           dim3((8 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           0,
                                           main,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                }
                else if(avg_row_nnz < 32)
                {
                    remainder = n % 16;
                    main      = n - remainder;

                    // Launch main kernel if enough columns of B
                    if(main > 0)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_device_pointer<T, CSRMMNT_DIM, 16>),
                                           dim3((16 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           0,
                                           main,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                }
                else if(avg_row_nnz < 64 || handle->wavefront_size == 32)
                {
                    remainder = n % 32;
                    main      = n - remainder;

                    // Launch main kernel if enough columns of B
                    if(main > 0)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_device_pointer<T, CSRMMNT_DIM, 32>),
                                           dim3((32 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           0,
                                           main,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                }
                else if(handle->wavefront_size == 64)
                {
                    remainder = n % 64;
                    main      = n - remainder;

                    // Launch main kernel if enough columns of B
                    if(main > 0)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_device_pointer<T, CSRMMNT_DIM, 64>),
                                           dim3((64 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           0,
                                           main,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                }
                else
                {
                    return rocsparse_status_arch_mismatch;
                }

                // Process remainder
                if(remainder > 0)
                {
                    if(remainder <= 8)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_device_pointer<T, CSRMMNT_DIM, 8>),
                                           dim3((8 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           main,
                                           n,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                    else if(remainder <= 16)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_device_pointer<T, CSRMMNT_DIM, 16>),
                                           dim3((16 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           main,
                                           n,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                    else if(remainder <= 32 || handle->wavefront_size == 32)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_device_pointer<T, CSRMMNT_DIM, 32>),
                                           dim3((32 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           main,
                                           n,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                    else if(remainder <= 64)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_device_pointer<T, CSRMMNT_DIM, 64>),
                                           dim3((64 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           main,
                                           n,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                    else
                    {
                        return rocsparse_status_arch_mismatch;
                    }
                }
            }
            else
            {
                // Quick return
                if(*alpha == 0.0 && *beta == 1.0)
                {
                    return rocsparse_status_success;
                }

                rocsparse_int main      = 0;
                rocsparse_int remainder = 0;

                // Launch appropriate kernel
                if(avg_row_nnz < 16)
                {
                    remainder = n % 8;
                    main      = n - remainder;

                    // Launch main kernel if enough columns of B
                    if(main > 0)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_host_pointer<T, CSRMMNT_DIM, 8>),
                                           dim3((8 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           0,
                                           main,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           *alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           *beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                }
                else if(avg_row_nnz < 32)
                {
                    remainder = n % 16;
                    main      = n - remainder;

                    // Launch main kernel if enough columns of B
                    if(main > 0)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_host_pointer<T, CSRMMNT_DIM, 16>),
                                           dim3((16 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           0,
                                           main,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           *alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           *beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                }
                else if(avg_row_nnz < 64 || handle->wavefront_size == 32)
                {
                    remainder = n % 32;
                    main      = n - remainder;

                    // Launch main kernel if enough columns of B
                    if(main > 0)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_host_pointer<T, CSRMMNT_DIM, 32>),
                                           dim3((32 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           0,
                                           main,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           *alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           *beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                }
                else if(handle->wavefront_size == 64)
                {
                    remainder = n % 64;
                    main      = n - remainder;

                    // Launch main kernel if enough columns of B
                    if(main > 0)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_host_pointer<T, CSRMMNT_DIM, 64>),
                                           dim3((64 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           0,
                                           main,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           *alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           *beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                }
                else
                {
                    return rocsparse_status_arch_mismatch;
                }

                // Process remainder
                if(remainder > 0)
                {
                    if(remainder <= 8)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_host_pointer<T, CSRMMNT_DIM, 8>),
                                           dim3((8 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           main,
                                           n,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           *alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           *beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                    else if(remainder <= 16)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_host_pointer<T, CSRMMNT_DIM, 16>),
                                           dim3((16 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           main,
                                           n,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           *alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           *beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                    else if(remainder <= 32 || handle->wavefront_size == 32)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_host_pointer<T, CSRMMNT_DIM, 32>),
                                           dim3((32 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           main,
                                           n,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           *alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           *beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                    else if(remainder <= 64)
                    {
                        hipLaunchKernelGGL((csrmmnt_kernel_host_pointer<T, CSRMMNT_DIM, 64>),
                                           dim3((64 * m - 1) / CSRMMNT_DIM + 1),
                                           dim3(CSRMMNT_DIM),
                                           0,
                                           stream,
                                           main,
                                           n,
                                           m,
                                           n,
                                           k,
                                           nnz,
                                           *alpha,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           *beta,
                                           C,
                                           ldc,
                                           descr->base);
                    }
                    else
                    {
                        return rocsparse_status_arch_mismatch;
                    }
                }
            }
#undef CSRMMNT_DIM
        }
        else
        {
            return rocsparse_status_not_implemented;
        }
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRMM_HPP
