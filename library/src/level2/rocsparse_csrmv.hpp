/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_CSRMV_HPP
#define ROCSPARSE_CSRMV_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "csrmv_device.h"

#include <hip/hip_runtime.h>

#define BLOCKSIZE 1024
#define BLOCK_MULTIPLIER 3
#define ROWS_FOR_VECTOR 1
#define WG_BITS 24
#define ROW_BITS 32
#define WG_SIZE 256

template <typename T, rocsparse_int WF_SIZE>
__global__ void csrmvn_general_kernel_host_pointer(rocsparse_int m,
                                                   T alpha,
                                                   const rocsparse_int* __restrict__ csr_row_ptr,
                                                   const rocsparse_int* __restrict__ csr_col_ind,
                                                   const T* __restrict__ csr_val,
                                                   const T* __restrict__ x,
                                                   T beta,
                                                   T* __restrict__ y,
                                                   rocsparse_index_base idx_base)
{
    csrmvn_general_device<T, WF_SIZE>(
        m, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, idx_base);
}

template <typename T, rocsparse_int WF_SIZE>
__global__ void csrmvn_general_kernel_device_pointer(rocsparse_int m,
                                                     const T* alpha,
                                                     const rocsparse_int* __restrict__ csr_row_ptr,
                                                     const rocsparse_int* __restrict__ csr_col_ind,
                                                     const T* __restrict__ csr_val,
                                                     const T* __restrict__ x,
                                                     const T* beta,
                                                     T* __restrict__ y,
                                                     rocsparse_index_base idx_base)
{
    csrmvn_general_device<T, WF_SIZE>(
        m, *alpha, csr_row_ptr, csr_col_ind, csr_val, x, *beta, y, idx_base);
}

template <typename T>
__launch_bounds__(WG_SIZE) __global__
    void csrmvn_adaptive_kernel_host_pointer(unsigned long long* __restrict__ row_blocks,
                                             T alpha,
                                             const rocsparse_int* __restrict__ csr_row_ptr,
                                             const rocsparse_int* __restrict__ csr_col_ind,
                                             const T* __restrict__ csr_val,
                                             const T* __restrict__ x,
                                             T beta,
                                             T* __restrict__ y,
                                             rocsparse_index_base idx_base)
{
    csrmvn_adaptive_device<T,
                           BLOCKSIZE,
                           BLOCK_MULTIPLIER,
                           ROWS_FOR_VECTOR,
                           WG_BITS,
                           ROW_BITS,
                           WG_SIZE>(
        row_blocks, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, idx_base);
}

template <typename T>
__launch_bounds__(WG_SIZE) __global__
    void csrmvn_adaptive_kernel_device_pointer(unsigned long long* __restrict__ row_blocks,
                                               const T* alpha,
                                               const rocsparse_int* __restrict__ csr_row_ptr,
                                               const rocsparse_int* __restrict__ csr_col_ind,
                                               const T* __restrict__ csr_val,
                                               const T* __restrict__ x,
                                               const T* beta,
                                               T* __restrict__ y,
                                               rocsparse_index_base idx_base)
{
    csrmvn_adaptive_device<T,
                           BLOCKSIZE,
                           BLOCK_MULTIPLIER,
                           ROWS_FOR_VECTOR,
                           WG_BITS,
                           ROW_BITS,
                           WG_SIZE>(
        row_blocks, *alpha, csr_row_ptr, csr_col_ind, csr_val, x, *beta, y, idx_base);
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
                                          rocsparse_mat_info info,
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

    // Logging
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
                  (const void*&)y,
                  (const void*&)info);

        log_bench(handle,
                  "./rocsparse-bench -f csrmv -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx> "
                  "--alpha",
                  *alpha,
                  "--beta",
                  *beta);
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

    if(info == nullptr)
    {
        // If csrmv info is not available, call csrmv general
        return rocsparse_csrmv_general_template(
            handle, trans, m, n, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    }
    else if(info->csrmv_built == false)
    {
        // If csrmv info is not available, call csrmv general
        return rocsparse_csrmv_general_template(
            handle, trans, m, n, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    }
    else
    {
        // If csrmv info is available, call csrmv adaptive
        return rocsparse_csrmv_adaptive_template(handle,
                                                 trans,
                                                 m,
                                                 n,
                                                 nnz,
                                                 alpha,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 info->csrmv_info,
                                                 x,
                                                 beta,
                                                 y);
    }
}

template <typename T>
rocsparse_status rocsparse_csrmv_general_template(rocsparse_handle handle,
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
            if(handle->wavefront_size == 32)
            {
                if(nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 2>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 4>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 8>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 16>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 32>),
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
            else if(handle->wavefront_size == 64)
            {
                if(nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 2>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 4>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 8>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 16>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 32>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_device_pointer<T, 64>),
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

            if(handle->wavefront_size == 32)
            {
                if(nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 2>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 4>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 8>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 16>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 32>),
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
            else if(handle->wavefront_size == 64)
            {
                if(nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 2>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 4>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 8>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 16>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 32>),
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
                    hipLaunchKernelGGL((csrmvn_general_kernel_host_pointer<T, 64>),
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

template <typename T>
rocsparse_status rocsparse_csrmv_adaptive_template(rocsparse_handle handle,
                                                   rocsparse_operation trans,
                                                   rocsparse_int m,
                                                   rocsparse_int n,
                                                   rocsparse_int nnz,
                                                   const T* alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const T* csr_val,
                                                   const rocsparse_int* csr_row_ptr,
                                                   const rocsparse_int* csr_col_ind,
                                                   rocsparse_csrmv_info info,
                                                   const T* x,
                                                   const T* beta,
                                                   T* y)
{
    // Check if info matches current matrix and options
    if(info->trans != trans)
    {
        return rocsparse_status_invalid_value;
    }
    else if(info->m != m)
    {
        return rocsparse_status_invalid_size;
    }
    else if(info->n != n)
    {
        return rocsparse_status_invalid_size;
    }
    else if(info->nnz != nnz)
    {
        return rocsparse_status_invalid_size;
    }
    else if(info->descr != descr)
    {
        return rocsparse_status_invalid_value;
    }
    else if(info->csr_row_ptr != csr_row_ptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info->csr_col_ind != csr_col_ind)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different csrmv kernels
    if(trans == rocsparse_operation_none)
    {
        dim3 csrmvn_blocks((info->size / 2) - 1);
        dim3 csrmvn_threads(WG_SIZE);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL((csrmvn_adaptive_kernel_device_pointer<T>),
                               csrmvn_blocks,
                               csrmvn_threads,
                               0,
                               stream,
                               info->row_blocks,
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
            if(*alpha == 0.0 && *beta == 1.0)
            {
                return rocsparse_status_success;
            }

            hipLaunchKernelGGL((csrmvn_adaptive_kernel_host_pointer<T>),
                               csrmvn_blocks,
                               csrmvn_threads,
                               0,
                               stream,
                               info->row_blocks,
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
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRMV_HPP
