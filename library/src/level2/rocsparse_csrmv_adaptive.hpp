/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_CSRMV_ADAPTIVE_HPP
#define ROCSPARSE_CSRMV_ADAPTIVE_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "csrmv_adaptive_device.h"

#include <hip/hip_runtime.h>

#define BLOCKSIZE 1024
#define BLOCK_MULTIPLIER 3
#define ROWS_FOR_VECTOR 1
#define WG_BITS 24
#define ROW_BITS 32
#define WG_SIZE 256

template <typename T>
__launch_bounds__(WG_SIZE)
__global__ void csrmvn_adaptive_kernel_host_pointer(unsigned long long* __restrict__ row_blocks,
                                                    T alpha,
                                                    const rocsparse_int* __restrict__ csr_row_ptr,
                                                    const rocsparse_int* __restrict__ csr_col_ind,
                                                    const T* __restrict__ csr_val,
                                                    const T* __restrict__ x,
                                                    T beta,
                                                    T* __restrict__ y,
                                                    rocsparse_index_base idx_base)
{
    csrmvn_adaptive_device<T, BLOCKSIZE, BLOCK_MULTIPLIER, ROWS_FOR_VECTOR, WG_BITS, ROW_BITS, WG_SIZE>(
        row_blocks, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, idx_base);
}

template <typename T>
__launch_bounds__(WG_SIZE)
__global__ void csrmvn_adaptive_kernel_device_pointer(unsigned long long* __restrict__ row_blocks,
                                                      const T* alpha,
                                                      const rocsparse_int* __restrict__ csr_row_ptr,
                                                      const rocsparse_int* __restrict__ csr_col_ind,
                                                      const T* __restrict__ csr_val,
                                                      const T* __restrict__ x,
                                                      const T* beta,
                                                      T* __restrict__ y,
                                                      rocsparse_index_base idx_base)
{
    csrmvn_adaptive_device<T, BLOCKSIZE, BLOCK_MULTIPLIER, ROWS_FOR_VECTOR, WG_BITS, ROW_BITS, WG_SIZE>(
        row_blocks, *alpha, csr_row_ptr, csr_col_ind, csr_val, x, *beta, y, idx_base);
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
                                                   const T* x,
                                                   const T* beta,
                                                   T* y,
                                                   const rocsparse_csrmv_info info)
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
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging TODO bench logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmv_adaptive"),
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
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmv_adaptive"),
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
                  (const void*&)y,
                  (const void*&)info);
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
        dim3 csrmvn_adaptive_blocks((info->size / 2) - 1);
        dim3 csrmvn_adaptive_threads(WG_SIZE);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL((csrmvn_adaptive_kernel_device_pointer<T>),
                               csrmvn_adaptive_blocks,
                               csrmvn_adaptive_threads,
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
                               csrmvn_adaptive_blocks,
                               csrmvn_adaptive_threads,
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
        //TODO
        return rocsparse_status_not_implemented;
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRMV_ADAPTIVE_HPP
