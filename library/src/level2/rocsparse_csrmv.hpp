/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_CSRMV_HPP
#define ROCSPARSE_CSRMV_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "csrmv_device.h"

#include <hip/hip_runtime.h>

template <typename T, const rocsparse_int SUBWAVE_SIZE, const rocsparse_int WG_SIZE>
__global__ void csrmvn_kernel_host_pointer(rocsparse_int m,
                                           T alpha,
                                           const rocsparse_int* csr_row_ptr,
                                           const rocsparse_int* csr_col_ind,
                                           const T* csr_val,
                                           const T* x,
                                           T beta,
                                           T* y,
                                           rocsparse_index_base idx_base)
{
    csrmvn_general_device<T, SUBWAVE_SIZE, WG_SIZE>(
        m, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, idx_base);
}

template <typename T, const rocsparse_int SUBWAVE_SIZE, const rocsparse_int WG_SIZE>
__global__ void csrmvn_kernel_device_pointer(rocsparse_int m,
                                             const T* alpha,
                                             const rocsparse_int* csr_row_ptr,
                                             const rocsparse_int* csr_col_ind,
                                             const T* csr_val,
                                             const T* x,
                                             const T* beta,
                                             T* y,
                                             rocsparse_index_base idx_base)
{
    csrmvn_general_device<T, SUBWAVE_SIZE, WG_SIZE>(
        m, *alpha, csr_row_ptr, csr_col_ind, csr_val, x, *beta, y, idx_base);
}

template <typename T>
rocsparse_status rocsparse_csrmv_template(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int nnz,
                                          const T* alpha,
                                          const rocsparse_mat_descr descr,
                                          const T* csr_val,
                                          const rocsparse_int* csr_row_ptr,
                                          const rocsparse_int* csr_col_ind,
                                          const T* x,
                                          const T* beta,
                                          T* y)
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
                  replaceX<T>("rocsparse_Xcsrmv"),
                  trans,
                  m,
                  n,
                  nnz,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)x,
                  *beta,
                  (const void*&)y);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmv"),
                  trans,
                  m,
                  n,
                  nnz,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)x,
                  (const void*&)beta,
                  (const void*&)y);
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
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
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
    else if(x == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
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

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different csrmv kernels
    if(trans == rocsparse_operation_none)
    {
#define CSRMVN_DIM 512
        rocsparse_int nnz_per_row = nnz / m;

        dim3 csrmvn_blocks((m - 1) / CSRMVN_DIM + 1);
        dim3 csrmvn_threads(CSRMVN_DIM);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            if(handle->warp_size == 32)
            {
                if(nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 2, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 8)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 4, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 16)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 8, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 32)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 16, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
                else
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 32, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
            }
            else if(handle->warp_size == 64)
            {
                if(nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 2, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 8)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 4, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 16)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 8, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 32)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 16, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 64)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 32, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
                else
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 64, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       beta,
                                       y,
                                       descr->base);
                }
            }
            else
            {
                return rocsparse_status_arch_mismatch;
            }
        }
        else
        {
            if(*alpha == 0.0 && *beta == 1.0)
            {
                return rocsparse_status_success;
            }

            if(handle->warp_size == 32)
            {
                if(nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 2, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 8)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 4, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 16)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 8, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 32)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 16, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
                else
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 32, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
            }
            else if(handle->warp_size == 64)
            {
                if(nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 2, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 8)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 4, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 16)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 8, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 32)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 16, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
                else if(nnz_per_row < 64)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 32, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
                else
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 64, CSRMVN_DIM>),
                                       csrmvn_blocks,
                                       csrmvn_threads,
                                       0,
                                       stream,
                                       m,
                                       *alpha,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       x,
                                       *beta,
                                       y,
                                       descr->base);
                }
            }
            else
            {
                return rocsparse_status_arch_mismatch;
            }
        }
#undef CSRMVN_DIM
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRMV_HPP
