/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_ELLMV_HPP
#define ROCSPARSE_ELLMV_HPP

#include "rocsparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"
#include "ellmv_device.h"

#include <hip/hip_runtime.h>

template <typename T>
__global__ void ellmvn_kernel_host_pointer(rocsparse_int m,
                                           rocsparse_int n,
                                           rocsparse_int ell_width,
                                           T alpha,
                                           const rocsparse_int* ell_col_ind,
                                           const T* ell_val,
                                           const T* x,
                                           T beta,
                                           T* y,
                                           rocsparse_index_base idx_base)
{
    ellmvn_device(m, n, ell_width, alpha, ell_col_ind, ell_val, x, beta, y, idx_base);
}

template <typename T>
__global__ void ellmvn_kernel_device_pointer(rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int ell_width,
                                             const T* alpha,
                                             const rocsparse_int* ell_col_ind,
                                             const T* ell_val,
                                             const T* x,
                                             const T* beta,
                                             T* y,
                                             rocsparse_index_base idx_base)
{
    ellmvn_device(m, n, ell_width, *alpha, ell_col_ind, ell_val, x, *beta, y, idx_base);
}

/*! \brief SPARSE Level 2 API

    \details
    ellmv  multiplies the dense vector x[i] with scalar alpha and sparse m x n
    matrix A that is defined in ELL storage format and add the result to y[i]
    that is multiplied by beta, for  i = 1 , … , n

        y := alpha * op(A) * x + beta * y,

    @param[in]
    handle      rocsparse_handle.
                handle to the rocsparse library context queue.
    @param[in]
    trans       operation type of A.
    @param[in]
    m           number of rows of A.
    @param[in]
    n           number of columns of A.
    @param[in]
    alpha       scalar alpha.
    @param[in]
    descr       descriptor of A.
    @param[in]
    ell_val     array of nnz elements of A.
    @param[in]
    ell_col_ind array of nnz elements containing the column indices of A.
    @param[in]
    ell_width   ELL width that was pre-computed during format conversion.
    @param[in]
    x           array of n elements (op(A) = A) or m elements (op(A) = A^T or
                op(A) = A^H).
    @param[in]
    beta        scalar beta.
    @param[inout]
    y           array of m elements (op(A) = A) or n elements (op(A) = A^T or
                op(A) = A^H).

    ********************************************************************/
template <typename T>
rocsparse_status rocsparse_ellmv_template(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          const T* alpha,
                                          const rocsparse_mat_descr descr,
                                          const T* ell_val,
                                          const rocsparse_int* ell_col_ind,
                                          rocsparse_int ell_width,
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
                  replaceX<T>("rocsparse_Xellmv"),
                  trans,
                  m,
                  n,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)ell_val,
                  (const void*&)ell_col_ind,
                  ell_width,
                  (const void*&)x,
                  *beta,
                  (const void*&)y);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xellmv"),
                  trans,
                  m,
                  n,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)ell_val,
                  (const void*&)ell_col_ind,
                  ell_width,
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
    else if(ell_width < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(ell_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
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

    // Sanity check
    if((m == 0 || n == 0) && ell_width != 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different ellmv kernels
    if(trans == rocsparse_operation_none)
    {
#define ELLMVN_DIM 512
        dim3 ellmvn_blocks((m - 1) / ELLMVN_DIM + 1);
        dim3 ellmvn_threads(ELLMVN_DIM);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL((ellmvn_kernel_device_pointer<T>),
                               ellmvn_blocks,
                               ellmvn_threads,
                               0,
                               stream,
                               m,
                               n,
                               ell_width,
                               alpha,
                               ell_col_ind,
                               ell_val,
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

            hipLaunchKernelGGL((ellmvn_kernel_host_pointer<T>),
                               ellmvn_blocks,
                               ellmvn_threads,
                               0,
                               stream,
                               m,
                               n,
                               ell_width,
                               *alpha,
                               ell_col_ind,
                               ell_val,
                               x,
                               *beta,
                               y,
                               descr->base);
        }
#undef ELLMVN_DIM
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_ELLMV_HPP
