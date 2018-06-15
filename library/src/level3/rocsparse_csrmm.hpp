/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_CSRMM_HPP
#define ROCSPARSE_CSRMM_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "csrmm_device.h"

#include <hip/hip_runtime.h>

template <typename T>
__global__ void csrmmn_kernel_host_pointer(rocsparse_int m,
                                           rocsparse_int n,
                                           rocsparse_int k,
                                           rocsparse_int nnz,
                                           T alpha,
                                           const rocsparse_int* csr_row_ptr,
                                           const rocsparse_int* csr_col_ind,
                                           const T* csr_val,
                                           const T* B,
                                           rocsparse_int ldb,
                                           T beta,
                                           T* C,
                                           rocsparse_int ldc,
                                           rocsparse_index_base idx_base)
{
    csrmmn_general_device<T>(m, n, k, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, B, ldb, beta, C, ldc, idx_base);
}

template <typename T>
__global__ void csrmmn_kernel_device_pointer(rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int k,
                                             rocsparse_int nnz,
                                             const T* alpha,
                                             const rocsparse_int* csr_row_ptr,
                                             const rocsparse_int* csr_col_ind,
                                             const T* csr_val,
                                             const T* B,
                                             rocsparse_int ldb,
                                             const T* beta,
                                             T* C,
                                             rocsparse_int ldc,
                                             rocsparse_index_base idx_base)
{
    csrmmn_general_device<T>(m, n, k, nnz, *alpha, csr_row_ptr, csr_col_ind, csr_val, B, ldb, *beta, C, ldc, idx_base);
}

template <typename T>
rocsparse_status rocsparse_csrmm_template(rocsparse_handle handle,
                                          rocsparse_operation trans,
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
                  trans,
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
                  trans,
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

    // Check leading dimensions
    if(trans == rocsparse_operation_none)
    {
        if(ldb < std::max(1, k))
        {
            return rocsparse_status_invalid_size;
        }
        else if(ldc < std::max(1, m))
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(ldb < std::max(1, m))
        {
            return rocsparse_status_invalid_size;
        }
        else if(ldc < std::max(1, k))
        {
            return rocsparse_status_invalid_size;
        }
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

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different csrmv kernels
    if(trans == rocsparse_operation_none)
    {
#define CSRMMN_DIM 512
        dim3 csrmmn_blocks((m - 1) / CSRMMN_DIM + 1);
        dim3 csrmmn_threads(CSRMMN_DIM);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
        }
        else
        {
            if(*alpha == 0.0 && *beta == 1.0)
            {
                return rocsparse_status_success;
            }
        }
#undef CSRMM_DIM
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRMM_HPP
