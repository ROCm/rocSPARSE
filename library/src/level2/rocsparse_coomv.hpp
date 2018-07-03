/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_COOMV_HPP
#define ROCSPARSE_COOMV_HPP

#include "rocsparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"
#include "coomv_device.h"

#include <hip/hip_runtime.h>

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WARPSIZE>
__launch_bounds__(128)
__global__ void coomvn_warp_host_pointer(rocsparse_int nnz,
                                         rocsparse_int loops,
                                         T alpha,
                                         const rocsparse_int* __restrict__ coo_row_ind,
                                         const rocsparse_int* __restrict__ coo_col_ind,
                                         const T* __restrict__ coo_val,
                                         const T* __restrict__ x,
                                         T* __restrict__ y,
                                         rocsparse_int* __restrict__ row_block_red,
                                         T* __restrict__ val_block_red,
                                         rocsparse_index_base idx_base)
{
    coomvn_general_warp_reduce<T, BLOCKSIZE, WARPSIZE>(nnz,
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

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WARPSIZE>
__launch_bounds__(128)
__global__ void coomvn_warp_device_pointer(rocsparse_int nnz,
                                           rocsparse_int loops,
                                           const T* alpha,
                                           const rocsparse_int* __restrict__ coo_row_ind,
                                           const rocsparse_int* __restrict__ coo_col_ind,
                                           const T* __restrict__ coo_val,
                                           const T* __restrict__ x,
                                           T* __restrict__ y,
                                           rocsparse_int* __restrict__ row_block_red,
                                           T* __restrict__ val_block_red,
                                           rocsparse_index_base idx_base)
{
    coomvn_general_warp_reduce<T, BLOCKSIZE, WARPSIZE>(nnz,
                                                       loops,
                                                       *alpha,
                                                       coo_row_ind,
                                                       coo_col_ind,
                                                       coo_val,
                                                       x,
                                                       y,
                                                       row_block_red,
                                                       val_block_red,
                                                       idx_base);
}

template <typename T>
rocsparse_status rocsparse_coomv_template(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int nnz,
                                          const T* alpha,
                                          const rocsparse_mat_descr descr,
                                          const T* coo_val,
                                          const rocsparse_int* coo_row_ind,
                                          const rocsparse_int* coo_col_ind,
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
                  replaceX<T>("rocsparse_Xcoomv"),
                  trans,
                  m,
                  n,
                  nnz,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)coo_val,
                  (const void*&)coo_row_ind,
                  (const void*&)coo_col_ind,
                  (const void*&)x,
                  *beta,
                  (const void*&)y);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcoomv"),
                  trans,
                  m,
                  n,
                  nnz,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)coo_val,
                  (const void*&)coo_row_ind,
                  (const void*&)coo_col_ind,
                  (const void*&)x,
                  (const void*&)beta,
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
    if(coo_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(coo_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(coo_col_ind == nullptr)
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

    // Run different coomv kernels
    if(trans == rocsparse_operation_none)
    {
#define COOMVN_DIM 128
        rocsparse_int maxthreads = handle->properties.maxThreadsPerBlock;
        rocsparse_int nprocs     = handle->properties.multiProcessorCount;
        rocsparse_int maxblocks  = (nprocs * maxthreads - 1) / COOMVN_DIM + 1;
        rocsparse_int minblocks  = (nnz - 1) / COOMVN_DIM + 1;

        rocsparse_int nblocks = maxblocks < minblocks ? maxblocks : minblocks;
        rocsparse_int nwarps  = nblocks * (COOMVN_DIM / handle->warp_size);
        rocsparse_int nloops  = (nnz / handle->warp_size + 1) / nwarps + 1;

        dim3 coomvn_blocks(nblocks);
        dim3 coomvn_threads(COOMVN_DIM);

        rocsparse_int* row_block_red = NULL;
        T* val_block_red             = NULL;

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&row_block_red, sizeof(rocsparse_int) * nwarps));
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&val_block_red, sizeof(T) * nwarps));

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            // We need a host copy of beta to avoid unneccessary kernel launch
            T h_beta;
            RETURN_IF_HIP_ERROR(hipMemcpy(&h_beta, beta, sizeof(T), hipMemcpyDeviceToHost));

            if(h_beta == static_cast<T>(0))
            {
                RETURN_IF_HIP_ERROR(hipMemset(y, 0, sizeof(T) * m));
            }
            else if(h_beta != static_cast<T>(1))
            {
                hipLaunchKernelGGL((coomv_scale<T>),
                                   dim3((m - 1) / COOMVN_DIM + 1),
                                   coomvn_threads,
                                   0,
                                   stream,
                                   m,
                                   h_beta,
                                   y);
            }

            if(handle->warp_size == 32)
            {
                hipLaunchKernelGGL((coomvn_warp_device_pointer<T, COOMVN_DIM, 32>),
                                   coomvn_blocks,
                                   coomvn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   nloops,
                                   alpha,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   x,
                                   y,
                                   row_block_red,
                                   val_block_red,
                                   descr->base);
            }
            else if(handle->warp_size == 64)
            {
                hipLaunchKernelGGL((coomvn_warp_device_pointer<T, COOMVN_DIM, 64>),
                                   coomvn_blocks,
                                   coomvn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   nloops,
                                   alpha,
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
        }
        else
        {
            if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }

            // If beta == 0.0 we need to set y to 0
            if(*beta == static_cast<T>(0))
            {
                RETURN_IF_HIP_ERROR(hipMemset(y, 0, sizeof(T) * m));
            }
            else if(*beta != static_cast<T>(1))
            {
                hipLaunchKernelGGL((coomv_scale<T>),
                                   dim3((m - 1) / COOMVN_DIM + 1),
                                   coomvn_threads,
                                   0,
                                   stream,
                                   m,
                                   *beta,
                                   y);
            }

            if(handle->warp_size == 32)
            {
                hipLaunchKernelGGL((coomvn_warp_host_pointer<T, COOMVN_DIM, 32>),
                                   coomvn_blocks,
                                   coomvn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   nloops,
                                   *alpha,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   x,
                                   y,
                                   row_block_red,
                                   val_block_red,
                                   descr->base);
            }
            else if(handle->warp_size == 64)
            {
                hipLaunchKernelGGL((coomvn_warp_host_pointer<T, COOMVN_DIM, 64>),
                                   coomvn_blocks,
                                   coomvn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   nloops,
                                   *alpha,
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
        }

        hipLaunchKernelGGL((coomvn_general_block_reduce<T, COOMVN_DIM>),
                           dim3(1),
                           coomvn_threads,
                           0,
                           stream,
                           nwarps,
                           row_block_red,
                           val_block_red,
                           y);

        RETURN_IF_HIP_ERROR(hipFree(row_block_red));
        RETURN_IF_HIP_ERROR(hipFree(val_block_red));
#undef COOMVN_DIM
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_COOMV_HPP
