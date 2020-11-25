/*! \file */
/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_CSRGEAM_HPP
#define ROCSPARSE_CSRGEAM_HPP

#include "csrgeam_device.h"
#include "definitions.h"
#include "utility.h"

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void csrgeam_fill_multipass_device_pointer(rocsparse_int m,
                                               rocsparse_int n,
                                               const T* __restrict__ alpha,
                                               const rocsparse_int* __restrict__ csr_row_ptr_A,
                                               const rocsparse_int* __restrict__ csr_col_ind_A,
                                               const T* __restrict__ csr_val_A,
                                               const T* __restrict__ beta,
                                               const rocsparse_int* __restrict__ csr_row_ptr_B,
                                               const rocsparse_int* __restrict__ csr_col_ind_B,
                                               const T* __restrict__ csr_val_B,
                                               const rocsparse_int* __restrict__ csr_row_ptr_C,
                                               rocsparse_int* __restrict__ csr_col_ind_C,
                                               T* __restrict__ csr_val_C,
                                               rocsparse_index_base idx_base_A,
                                               rocsparse_index_base idx_base_B,
                                               rocsparse_index_base idx_base_C)
{
    csrgeam_fill_multipass<T, BLOCKSIZE, WFSIZE>(m,
                                                 n,
                                                 *alpha,
                                                 csr_row_ptr_A,
                                                 csr_col_ind_A,
                                                 csr_val_A,
                                                 *beta,
                                                 csr_row_ptr_B,
                                                 csr_col_ind_B,
                                                 csr_val_B,
                                                 csr_row_ptr_C,
                                                 csr_col_ind_C,
                                                 csr_val_C,
                                                 idx_base_A,
                                                 idx_base_B,
                                                 idx_base_C);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void csrgeam_fill_multipass_host_pointer(rocsparse_int m,
                                             rocsparse_int n,
                                             T             alpha,
                                             const rocsparse_int* __restrict__ csr_row_ptr_A,
                                             const rocsparse_int* __restrict__ csr_col_ind_A,
                                             const T* __restrict__ csr_val_A,
                                             T beta,
                                             const rocsparse_int* __restrict__ csr_row_ptr_B,
                                             const rocsparse_int* __restrict__ csr_col_ind_B,
                                             const T* __restrict__ csr_val_B,
                                             const rocsparse_int* __restrict__ csr_row_ptr_C,
                                             rocsparse_int* __restrict__ csr_col_ind_C,
                                             T* __restrict__ csr_val_C,
                                             rocsparse_index_base idx_base_A,
                                             rocsparse_index_base idx_base_B,
                                             rocsparse_index_base idx_base_C)
{
    csrgeam_fill_multipass<T, BLOCKSIZE, WFSIZE>(m,
                                                 n,
                                                 alpha,
                                                 csr_row_ptr_A,
                                                 csr_col_ind_A,
                                                 csr_val_A,
                                                 beta,
                                                 csr_row_ptr_B,
                                                 csr_col_ind_B,
                                                 csr_val_B,
                                                 csr_row_ptr_C,
                                                 csr_col_ind_C,
                                                 csr_val_C,
                                                 idx_base_A,
                                                 idx_base_B,
                                                 idx_base_C);
}

template <typename T>
rocsparse_status rocsparse_csrgeam_template(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const T*                  csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const T*                  csr_val_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        csr_val_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C)
{
    // Check for valid handle, alpha, beta and descriptors
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(descr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(descr_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(descr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgeam"),
                  m,
                  n,
                  *alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_val_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  *beta,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_val_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  (const void*&)descr_C,
                  (const void*&)csr_val_C,
                  (const void*&)csr_row_ptr_C,
                  (const void*&)csr_col_ind_C);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgeam"),
                  m,
                  n,
                  (const void*&)alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_val_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  (const void*&)beta,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_val_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  (const void*&)descr_C,
                  (const void*&)csr_val_C,
                  (const void*&)csr_row_ptr_C,
                  (const void*&)csr_col_ind_C);
    }

    // Check index base
    if(descr_A->base != rocsparse_index_base_zero && descr_A->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr_B->base != rocsparse_index_base_zero && descr_B->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr_C->base != rocsparse_index_base_zero && descr_C->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_B->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_status_success;
    }

    // Check valid pointers
    if(csr_val_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_val_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_val_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Pointer mode device
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
#define CSRGEAM_DIM 256
        if(handle->wavefront_size == 32)
        {
            hipLaunchKernelGGL((csrgeam_fill_multipass_device_pointer<T, CSRGEAM_DIM, 32>),
                               dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                               dim3(CSRGEAM_DIM),
                               0,
                               stream,
                               m,
                               n,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               beta,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               descr_A->base,
                               descr_B->base,
                               descr_C->base);
        }
        else
        {
            hipLaunchKernelGGL((csrgeam_fill_multipass_device_pointer<T, CSRGEAM_DIM, 64>),
                               dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                               dim3(CSRGEAM_DIM),
                               0,
                               stream,
                               m,
                               n,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               beta,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               descr_A->base,
                               descr_B->base,
                               descr_C->base);
        }
    }
    else
    {
        if(handle->wavefront_size == 32)
        {
            hipLaunchKernelGGL((csrgeam_fill_multipass_host_pointer<T, CSRGEAM_DIM, 32>),
                               dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                               dim3(CSRGEAM_DIM),
                               0,
                               stream,
                               m,
                               n,
                               *alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               *beta,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               descr_A->base,
                               descr_B->base,
                               descr_C->base);
        }
        else
        {
            hipLaunchKernelGGL((csrgeam_fill_multipass_host_pointer<T, CSRGEAM_DIM, 64>),
                               dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                               dim3(CSRGEAM_DIM),
                               0,
                               stream,
                               m,
                               n,
                               *alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               *beta,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               descr_A->base,
                               descr_B->base,
                               descr_C->base);
        }
#undef CSRGEAM_DIM
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRGEAM_HPP
