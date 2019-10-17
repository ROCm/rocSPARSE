/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_CSRGEMM_HPP
#define ROCSPARSE_CSRGEMM_HPP

#include "csrgemm_device.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <type_traits>

#define CSRGEMM_MAXGROUPS 8
#define CSRGEMM_NNZ_HASH 79
#define CSRGEMM_FLL_HASH 137

template <typename T,
          unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int HASHSIZE,
          unsigned int HASHVAL>
__global__ void
    csrgemm_fill_wf_per_row_host_pointer(rocsparse_int m,
                                         rocsparse_int nk,
                                         const rocsparse_int* __restrict__ offset,
                                         const rocsparse_int* __restrict__ perm,
                                         T alpha,
                                         const rocsparse_int* __restrict__ csr_row_ptr_A,
                                         const rocsparse_int* __restrict__ csr_col_ind_A,
                                         const T* __restrict__ csr_val_A,
                                         const rocsparse_int* __restrict__ csr_row_ptr_B,
                                         const rocsparse_int* __restrict__ csr_col_ind_B,
                                         const T* __restrict__ csr_val_B,
                                         T beta,
                                         const rocsparse_int* __restrict__ csr_row_ptr_D,
                                         const rocsparse_int* __restrict__ csr_col_ind_D,
                                         const T* __restrict__ csr_val_D,
                                         const rocsparse_int* __restrict__ csr_row_ptr_C,
                                         rocsparse_int* __restrict__ csr_col_ind_C,
                                         T* __restrict__ csr_val_C,
                                         rocsparse_index_base idx_base_A,
                                         rocsparse_index_base idx_base_B,
                                         rocsparse_index_base idx_base_C,
                                         rocsparse_index_base idx_base_D,
                                         bool                 mul,
                                         bool                 add)
{
    csrgemm_fill_wf_per_row_device<T, BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(m,
                                                                            nk,
                                                                            offset,
                                                                            perm,
                                                                            alpha,
                                                                            csr_row_ptr_A,
                                                                            csr_col_ind_A,
                                                                            csr_val_A,
                                                                            csr_row_ptr_B,
                                                                            csr_col_ind_B,
                                                                            csr_val_B,
                                                                            beta,
                                                                            csr_row_ptr_D,
                                                                            csr_col_ind_D,
                                                                            csr_val_D,
                                                                            csr_row_ptr_C,
                                                                            csr_col_ind_C,
                                                                            csr_val_C,
                                                                            idx_base_A,
                                                                            idx_base_B,
                                                                            idx_base_C,
                                                                            idx_base_D,
                                                                            mul,
                                                                            add);
}

template <typename T,
          unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int HASHSIZE,
          unsigned int HASHVAL>
__global__ void
    csrgemm_fill_wf_per_row_device_pointer(rocsparse_int m,
                                           rocsparse_int nk,
                                           const rocsparse_int* __restrict__ offset,
                                           const rocsparse_int* __restrict__ perm,
                                           const T* __restrict__ alpha,
                                           const rocsparse_int* __restrict__ csr_row_ptr_A,
                                           const rocsparse_int* __restrict__ csr_col_ind_A,
                                           const T* __restrict__ csr_val_A,
                                           const rocsparse_int* __restrict__ csr_row_ptr_B,
                                           const rocsparse_int* __restrict__ csr_col_ind_B,
                                           const T* __restrict__ csr_val_B,
                                           const T* __restrict__ beta,
                                           const rocsparse_int* __restrict__ csr_row_ptr_D,
                                           const rocsparse_int* __restrict__ csr_col_ind_D,
                                           const T* __restrict__ csr_val_D,
                                           const rocsparse_int* __restrict__ csr_row_ptr_C,
                                           rocsparse_int* __restrict__ csr_col_ind_C,
                                           T* __restrict__ csr_val_C,
                                           rocsparse_index_base idx_base_A,
                                           rocsparse_index_base idx_base_B,
                                           rocsparse_index_base idx_base_C,
                                           rocsparse_index_base idx_base_D,
                                           bool                 mul,
                                           bool                 add)
{
    csrgemm_fill_wf_per_row_device<T, BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(
        m,
        nk,
        offset,
        perm,
        (mul == true) ? *alpha : static_cast<T>(0),
        csr_row_ptr_A,
        csr_col_ind_A,
        csr_val_A,
        csr_row_ptr_B,
        csr_col_ind_B,
        csr_val_B,
        (add == true) ? *beta : static_cast<T>(0),
        csr_row_ptr_D,
        csr_col_ind_D,
        csr_val_D,
        csr_row_ptr_C,
        csr_col_ind_C,
        csr_val_C,
        idx_base_A,
        idx_base_B,
        idx_base_C,
        idx_base_D,
        mul,
        add);
}

template <typename T,
          unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int HASHSIZE,
          unsigned int HASHVAL>
__attribute__((amdgpu_flat_work_group_size(128, 1024))) __global__ void
    csrgemm_fill_block_per_row_host_pointer(rocsparse_int nk,
                                            const rocsparse_int* __restrict__ offset,
                                            const rocsparse_int* __restrict__ perm,
                                            T alpha,
                                            const rocsparse_int* __restrict__ csr_row_ptr_A,
                                            const rocsparse_int* __restrict__ csr_col_ind_A,
                                            const T* __restrict__ csr_val_A,
                                            const rocsparse_int* __restrict__ csr_row_ptr_B,
                                            const rocsparse_int* __restrict__ csr_col_ind_B,
                                            const T* __restrict__ csr_val_B,
                                            T beta,
                                            const rocsparse_int* __restrict__ csr_row_ptr_D,
                                            const rocsparse_int* __restrict__ csr_col_ind_D,
                                            const T* __restrict__ csr_val_D,
                                            const rocsparse_int* __restrict__ csr_row_ptr_C,
                                            rocsparse_int* __restrict__ csr_col_ind_C,
                                            T* __restrict__ csr_val_C,
                                            rocsparse_index_base idx_base_A,
                                            rocsparse_index_base idx_base_B,
                                            rocsparse_index_base idx_base_C,
                                            rocsparse_index_base idx_base_D,
                                            bool                 mul,
                                            bool                 add)
{
    csrgemm_fill_block_per_row_device<T, BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(nk,
                                                                               offset,
                                                                               perm,
                                                                               alpha,
                                                                               csr_row_ptr_A,
                                                                               csr_col_ind_A,
                                                                               csr_val_A,
                                                                               csr_row_ptr_B,
                                                                               csr_col_ind_B,
                                                                               csr_val_B,
                                                                               beta,
                                                                               csr_row_ptr_D,
                                                                               csr_col_ind_D,
                                                                               csr_val_D,
                                                                               csr_row_ptr_C,
                                                                               csr_col_ind_C,
                                                                               csr_val_C,
                                                                               idx_base_A,
                                                                               idx_base_B,
                                                                               idx_base_C,
                                                                               idx_base_D,
                                                                               mul,
                                                                               add);
}

template <typename T,
          unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int HASHSIZE,
          unsigned int HASHVAL>
__attribute__((amdgpu_flat_work_group_size(128, 1024))) __global__ void
    csrgemm_fill_block_per_row_device_pointer(rocsparse_int nk,
                                              const rocsparse_int* __restrict__ offset,
                                              const rocsparse_int* __restrict__ perm,
                                              const T* __restrict__ alpha,
                                              const rocsparse_int* __restrict__ csr_row_ptr_A,
                                              const rocsparse_int* __restrict__ csr_col_ind_A,
                                              const T* __restrict__ csr_val_A,
                                              const rocsparse_int* __restrict__ csr_row_ptr_B,
                                              const rocsparse_int* __restrict__ csr_col_ind_B,
                                              const T* __restrict__ csr_val_B,
                                              const T* __restrict__ beta,
                                              const rocsparse_int* __restrict__ csr_row_ptr_D,
                                              const rocsparse_int* __restrict__ csr_col_ind_D,
                                              const T* __restrict__ csr_val_D,
                                              const rocsparse_int* __restrict__ csr_row_ptr_C,
                                              rocsparse_int* __restrict__ csr_col_ind_C,
                                              T* __restrict__ csr_val_C,
                                              rocsparse_index_base idx_base_A,
                                              rocsparse_index_base idx_base_B,
                                              rocsparse_index_base idx_base_C,
                                              rocsparse_index_base idx_base_D,
                                              bool                 mul,
                                              bool                 add)
{
    csrgemm_fill_block_per_row_device<T, BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(
        nk,
        offset,
        perm,
        (mul == true) ? *alpha : static_cast<T>(0),
        csr_row_ptr_A,
        csr_col_ind_A,
        csr_val_A,
        csr_row_ptr_B,
        csr_col_ind_B,
        csr_val_B,
        (add == true) ? *beta : static_cast<T>(0),
        csr_row_ptr_D,
        csr_col_ind_D,
        csr_val_D,
        csr_row_ptr_C,
        csr_col_ind_C,
        csr_val_C,
        idx_base_A,
        idx_base_B,
        idx_base_C,
        idx_base_D,
        mul,
        add);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int CHUNKSIZE>
__attribute__((amdgpu_flat_work_group_size(128, 1024))) __global__ void
    csrgemm_fill_block_per_row_multipass_host_pointer(
        rocsparse_int n,
        const rocsparse_int* __restrict__ offset,
        const rocsparse_int* __restrict__ perm,
        T alpha,
        const rocsparse_int* __restrict__ csr_row_ptr_A,
        const rocsparse_int* __restrict__ csr_col_ind_A,
        const T* __restrict__ csr_val_A,
        const rocsparse_int* __restrict__ csr_row_ptr_B,
        const rocsparse_int* __restrict__ csr_col_ind_B,
        const T* __restrict__ csr_val_B,
        T beta,
        const rocsparse_int* __restrict__ csr_row_ptr_D,
        const rocsparse_int* __restrict__ csr_col_ind_D,
        const T* __restrict__ csr_val_D,
        const rocsparse_int* __restrict__ csr_row_ptr_C,
        rocsparse_int* __restrict__ csr_col_ind_C,
        T* __restrict__ csr_val_C,
        rocsparse_int* __restrict__ workspace_B,
        rocsparse_index_base idx_base_A,
        rocsparse_index_base idx_base_B,
        rocsparse_index_base idx_base_C,
        rocsparse_index_base idx_base_D,
        bool                 mul,
        bool                 add)
{
    csrgemm_fill_block_per_row_multipass_device<T, BLOCKSIZE, WFSIZE, CHUNKSIZE>(n,
                                                                                 offset,
                                                                                 perm,
                                                                                 alpha,
                                                                                 csr_row_ptr_A,
                                                                                 csr_col_ind_A,
                                                                                 csr_val_A,
                                                                                 csr_row_ptr_B,
                                                                                 csr_col_ind_B,
                                                                                 csr_val_B,
                                                                                 beta,
                                                                                 csr_row_ptr_D,
                                                                                 csr_col_ind_D,
                                                                                 csr_val_D,
                                                                                 csr_row_ptr_C,
                                                                                 csr_col_ind_C,
                                                                                 csr_val_C,
                                                                                 workspace_B,
                                                                                 idx_base_A,
                                                                                 idx_base_B,
                                                                                 idx_base_C,
                                                                                 idx_base_D,
                                                                                 mul,
                                                                                 add);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int CHUNKSIZE>
__attribute__((amdgpu_flat_work_group_size(128, 1024))) __global__ void
    csrgemm_fill_block_per_row_multipass_device_pointer(
        rocsparse_int n,
        const rocsparse_int* __restrict__ offset,
        const rocsparse_int* __restrict__ perm,
        const T* __restrict__ alpha,
        const rocsparse_int* __restrict__ csr_row_ptr_A,
        const rocsparse_int* __restrict__ csr_col_ind_A,
        const T* __restrict__ csr_val_A,
        const rocsparse_int* __restrict__ csr_row_ptr_B,
        const rocsparse_int* __restrict__ csr_col_ind_B,
        const T* __restrict__ csr_val_B,
        const T* __restrict__ beta,
        const rocsparse_int* __restrict__ csr_row_ptr_D,
        const rocsparse_int* __restrict__ csr_col_ind_D,
        const T* __restrict__ csr_val_D,
        const rocsparse_int* __restrict__ csr_row_ptr_C,
        rocsparse_int* __restrict__ csr_col_ind_C,
        T* __restrict__ csr_val_C,
        rocsparse_int* __restrict__ workspace_B,
        rocsparse_index_base idx_base_A,
        rocsparse_index_base idx_base_B,
        rocsparse_index_base idx_base_C,
        rocsparse_index_base idx_base_D,
        bool                 mul,
        bool                 add)
{
    csrgemm_fill_block_per_row_multipass_device<T, BLOCKSIZE, WFSIZE, CHUNKSIZE>(
        n,
        offset,
        perm,
        (mul == true) ? *alpha : static_cast<T>(0),
        csr_row_ptr_A,
        csr_col_ind_A,
        csr_val_A,
        csr_row_ptr_B,
        csr_col_ind_B,
        csr_val_B,
        (add == true) ? *beta : static_cast<T>(0),
        csr_row_ptr_D,
        csr_col_ind_D,
        csr_val_D,
        csr_row_ptr_C,
        csr_col_ind_C,
        csr_val_C,
        workspace_B,
        idx_base_A,
        idx_base_B,
        idx_base_C,
        idx_base_D,
        mul,
        add);
}

template <typename T>
rocsparse_status rocsparse_csrgemm_mult_buffer_size_template(rocsparse_handle          handle,
                                                             rocsparse_operation       trans_A,
                                                             rocsparse_operation       trans_B,
                                                             rocsparse_int             m,
                                                             rocsparse_int             n,
                                                             rocsparse_int             k,
                                                             const T*                  alpha,
                                                             const rocsparse_mat_descr descr_A,
                                                             rocsparse_int             nnz_A,
                                                             const rocsparse_int* csr_row_ptr_A,
                                                             const rocsparse_int* csr_col_ind_A,
                                                             const rocsparse_mat_descr descr_B,
                                                             rocsparse_int             nnz_B,
                                                             const rocsparse_int* csr_row_ptr_B,
                                                             const rocsparse_int* csr_col_ind_B,
                                                             rocsparse_mat_info   info_C,
                                                             size_t*              buffer_size)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_internal_error;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || csr_row_ptr_A == nullptr || csr_col_ind_A == nullptr
       || descr_B == nullptr || csr_row_ptr_B == nullptr || csr_col_ind_B == nullptr
       || buffer_size == nullptr || alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
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

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_B->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;

        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // rocprim buffer
    size_t rocprim_size;
    size_t rocprim_max = 0;

    // rocprim::reduce
    RETURN_IF_HIP_ERROR(rocprim::reduce(nullptr,
                                        rocprim_size,
                                        csr_row_ptr_A,
                                        &nnz_A,
                                        0,
                                        m,
                                        rocprim::maximum<rocsparse_int>(),
                                        stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    // rocprim exclusive scan
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                rocprim_size,
                                                csr_row_ptr_A,
                                                &nnz_A,
                                                0,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    // rocprim::radix_sort_pairs
    rocprim::double_buffer<rocsparse_int> buf(&nnz_A, &nnz_B);
    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, rocprim_size, buf, buf, m, 0, 3, stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    *buffer_size = ((rocprim_max - 1) / 256 + 1) * 256;

    // Group arrays
    *buffer_size += sizeof(rocsparse_int) * 256 * CSRGEMM_MAXGROUPS;
    *buffer_size += sizeof(rocsparse_int) * 256;
    *buffer_size += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

    // Permutation arrays
    *buffer_size += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;
    *buffer_size += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;
    *buffer_size += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrgemm_scal_buffer_size_template(rocsparse_handle          handle,
                                                             rocsparse_int             m,
                                                             rocsparse_int             n,
                                                             const T*                  beta,
                                                             const rocsparse_mat_descr descr_D,
                                                             rocsparse_int             nnz_D,
                                                             const rocsparse_int* csr_row_ptr_D,
                                                             const rocsparse_int* csr_col_ind_D,
                                                             rocsparse_mat_info   info_C,
                                                             size_t*              buffer_size)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_internal_error;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_D == nullptr || csr_row_ptr_D == nullptr || csr_col_ind_D == nullptr
       || buffer_size == nullptr || beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr_D->base != rocsparse_index_base_zero && descr_D->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr_D->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // No buffer requirements for matrix scaling
    *buffer_size = 4;

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrgemm_buffer_size_template(rocsparse_handle          handle,
                                                        rocsparse_operation       trans_A,
                                                        rocsparse_operation       trans_B,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        rocsparse_int             k,
                                                        const T*                  alpha,
                                                        const rocsparse_mat_descr descr_A,
                                                        rocsparse_int             nnz_A,
                                                        const rocsparse_int*      csr_row_ptr_A,
                                                        const rocsparse_int*      csr_col_ind_A,
                                                        const rocsparse_mat_descr descr_B,
                                                        rocsparse_int             nnz_B,
                                                        const rocsparse_int*      csr_row_ptr_B,
                                                        const rocsparse_int*      csr_col_ind_B,
                                                        const T*                  beta,
                                                        const rocsparse_mat_descr descr_D,
                                                        rocsparse_int             nnz_D,
                                                        const rocsparse_int*      csr_row_ptr_D,
                                                        const rocsparse_int*      csr_col_ind_D,
                                                        rocsparse_mat_info        info_C,
                                                        size_t*                   buffer_size)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgemm_buffer_size"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  *alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  *beta,
                  (const void*&)descr_D,
                  nnz_D,
                  (const void*&)csr_row_ptr_D,
                  (const void*&)csr_col_ind_D,
                  (const void*&)info_C,
                  (const void*&)buffer_size);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgemm_buffer_size"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  (const void*&)alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  (const void*&)beta,
                  (const void*&)descr_D,
                  nnz_D,
                  (const void*&)csr_row_ptr_D,
                  (const void*&)csr_col_ind_D,
                  (const void*&)info_C,
                  (const void*&)buffer_size);
    }

    // Check for valid rocsparse_mat_info
    if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Clear csrgemm info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrgemm_info(info_C->csrgemm_info));

    // Create csrgemm info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrgemm_info(&info_C->csrgemm_info));

    // Set info parameters
    info_C->csrgemm_info->mul = (alpha != nullptr);
    info_C->csrgemm_info->add = (beta != nullptr);

    // Either alpha or beta can be nullptr
    if(alpha != nullptr && beta != nullptr)
    {
        // alpha != nullptr && beta != nullptr
        // TODO
        // rocsparse_csrgemm_multadd_template(...)
        return rocsparse_status_not_implemented;
    }
    else if(alpha != nullptr && beta == nullptr)
    {
        // alpha != nullptr && beta == nullptr
        return rocsparse_csrgemm_mult_buffer_size_template<T>(handle,
                                                              trans_A,
                                                              trans_B,
                                                              m,
                                                              n,
                                                              k,
                                                              alpha,
                                                              descr_A,
                                                              nnz_A,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              descr_B,
                                                              nnz_B,
                                                              csr_row_ptr_B,
                                                              csr_col_ind_B,
                                                              info_C,
                                                              buffer_size);
    }
    else if(alpha == nullptr && beta != nullptr)
    {
        // alpha == nullptr && beta != nullptr
        return rocsparse_csrgemm_scal_buffer_size_template<T>(
            handle, m, n, beta, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, buffer_size);
    }
    else
    {
        // alpha == nullptr && beta == nullptr
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_success;
}

// Disable for rocsparse_double_complex, as required size would exceed available memory
template <typename T,
          typename std::enable_if<std::is_same<T, rocsparse_double_complex>::value, int>::type = 0>
inline rocsparse_status csrgemm_launcher(rocsparse_handle     handle,
                                         rocsparse_int        group_size,
                                         const rocsparse_int* group_offset,
                                         const rocsparse_int* perm,
                                         rocsparse_int        m,
                                         rocsparse_int        n,
                                         rocsparse_int        k,
                                         const T*             alpha,
                                         const rocsparse_int* csr_row_ptr_A,
                                         const rocsparse_int* csr_col_ind_A,
                                         const T*             csr_val_A,
                                         const rocsparse_int* csr_row_ptr_B,
                                         const rocsparse_int* csr_col_ind_B,
                                         const T*             csr_val_B,
                                         const T*             beta,
                                         const rocsparse_int* csr_row_ptr_D,
                                         const rocsparse_int* csr_col_ind_D,
                                         const T*             csr_val_D,
                                         const rocsparse_int* csr_row_ptr_C,
                                         rocsparse_int*       csr_col_ind_C,
                                         T*                   csr_val_C,
                                         rocsparse_index_base base_A,
                                         rocsparse_index_base base_B,
                                         rocsparse_index_base base_C,
                                         rocsparse_index_base base_D,
                                         bool                 mul,
                                         bool                 add)
{
    return rocsparse_status_internal_error;
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value
                                      || std::is_same<T, rocsparse_float_complex>::value,
                                  int>::type
          = 0>
inline rocsparse_status csrgemm_launcher(rocsparse_handle     handle,
                                         rocsparse_int        group_size,
                                         const rocsparse_int* group_offset,
                                         const rocsparse_int* perm,
                                         rocsparse_int        m,
                                         rocsparse_int        n,
                                         rocsparse_int        k,
                                         const T*             alpha,
                                         const rocsparse_int* csr_row_ptr_A,
                                         const rocsparse_int* csr_col_ind_A,
                                         const T*             csr_val_A,
                                         const rocsparse_int* csr_row_ptr_B,
                                         const rocsparse_int* csr_col_ind_B,
                                         const T*             csr_val_B,
                                         const T*             beta,
                                         const rocsparse_int* csr_row_ptr_D,
                                         const rocsparse_int* csr_col_ind_D,
                                         const T*             csr_val_D,
                                         const rocsparse_int* csr_row_ptr_C,
                                         rocsparse_int*       csr_col_ind_C,
                                         T*                   csr_val_C,
                                         rocsparse_index_base base_A,
                                         rocsparse_index_base base_B,
                                         rocsparse_index_base base_C,
                                         rocsparse_index_base base_D,
                                         bool                 mul,
                                         bool                 add)
{
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 64
#define CSRGEMM_HASHSIZE 4096
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<T,
                                                                      CSRGEMM_DIM,
                                                                      CSRGEMM_SUB,
                                                                      CSRGEMM_HASHSIZE,
                                                                      CSRGEMM_FLL_HASH>),
                           dim3(group_size),
                           dim3(CSRGEMM_DIM),
                           0,
                           handle->stream,
                           std::max(k, n),
                           group_offset,
                           perm,
                           alpha,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_val_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_val_B,
                           beta,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_val_D,
                           csr_row_ptr_C,
                           csr_col_ind_C,
                           csr_val_C,
                           base_A,
                           base_B,
                           base_C,
                           base_D,
                           mul,
                           add);
    }
    else
    {
        hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<T,
                                                                    CSRGEMM_DIM,
                                                                    CSRGEMM_SUB,
                                                                    CSRGEMM_HASHSIZE,
                                                                    CSRGEMM_FLL_HASH>),
                           dim3(group_size),
                           dim3(CSRGEMM_DIM),
                           0,
                           handle->stream,
                           std::max(k, n),
                           group_offset,
                           perm,
                           mul ? *alpha : static_cast<T>(0),
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_val_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_val_B,
                           add ? *beta : static_cast<T>(0),
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_val_D,
                           csr_row_ptr_C,
                           csr_col_ind_C,
                           csr_val_C,
                           base_A,
                           base_B,
                           base_C,
                           base_D,
                           mul,
                           add);
    }
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrgemm_calc_template(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 rocsparse_int             k,
                                                 const T*                  alpha,
                                                 const rocsparse_mat_descr descr_A,
                                                 rocsparse_int             nnz_A,
                                                 const T*                  csr_val_A,
                                                 const rocsparse_int*      csr_row_ptr_A,
                                                 const rocsparse_int*      csr_col_ind_A,
                                                 const rocsparse_mat_descr descr_B,
                                                 rocsparse_int             nnz_B,
                                                 const T*                  csr_val_B,
                                                 const rocsparse_int*      csr_row_ptr_B,
                                                 const rocsparse_int*      csr_col_ind_B,
                                                 const T*                  beta,
                                                 const rocsparse_mat_descr descr_D,
                                                 rocsparse_int             nnz_D,
                                                 const T*                  csr_val_D,
                                                 const rocsparse_int*      csr_row_ptr_D,
                                                 const rocsparse_int*      csr_col_ind_D,
                                                 const rocsparse_mat_descr descr_C,
                                                 T*                        csr_val_C,
                                                 const rocsparse_int*      csr_row_ptr_C,
                                                 rocsparse_int*            csr_col_ind_C,
                                                 const rocsparse_mat_info  info_C,
                                                 void*                     temp_buffer)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Index base
    rocsparse_index_base base_A
        = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
    rocsparse_index_base base_B
        = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
    rocsparse_index_base base_D
        = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

    // Flag for double complex
    constexpr bool is_double_complex = std::is_same<T, rocsparse_double_complex>::value;

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // rocprim buffer
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Determine maximum non-zero entries per row of all rows
    rocsparse_int* workspace = reinterpret_cast<rocsparse_int*>(buffer);

#define CSRGEMM_DIM 256
    hipLaunchKernelGGL((csrgemm_max_row_nnz_part1<CSRGEMM_DIM>),
                       dim3(CSRGEMM_DIM),
                       dim3(CSRGEMM_DIM),
                       0,
                       stream,
                       m,
                       csr_row_ptr_C,
                       workspace);

    hipLaunchKernelGGL(
        (csrgemm_max_row_nnz_part2<CSRGEMM_DIM>), dim3(1), dim3(CSRGEMM_DIM), 0, stream, workspace);
#undef CSRGEMM_DIM

    rocsparse_int nnz_max;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&nnz_max, workspace, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Group offset buffer
    rocsparse_int* d_group_offset = reinterpret_cast<rocsparse_int*>(buffer);
    buffer += sizeof(rocsparse_int) * 256;

    // Group size buffer
    rocsparse_int h_group_size[CSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(rocsparse_int) * CSRGEMM_MAXGROUPS);

    // Permutation array
    rocsparse_int* d_perm = nullptr;

    // If maximum of row nnz exceeds 16, we process the rows in groups of
    // similar sized row nnz
    if(nnz_max > 16)
    {
        // Group size buffer
        rocsparse_int* d_group_size = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += sizeof(rocsparse_int) * 256 * CSRGEMM_MAXGROUPS;

        // Permutation temporary arrays
        rocsparse_int* tmp_vals = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        rocsparse_int* tmp_keys = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        rocsparse_int* tmp_groups = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        hipLaunchKernelGGL(
            (csrgemm_group_reduce_part2<CSRGEMM_DIM, CSRGEMM_MAXGROUPS, is_double_complex>),
            dim3(CSRGEMM_DIM),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            m,
            csr_row_ptr_C,
            d_group_size,
            tmp_groups);

        hipLaunchKernelGGL((csrgemm_group_reduce_part3<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
                           dim3(1),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           d_group_size);
#undef CSRGEMM_DIM

        // Exclusive sum to obtain group offsets
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<rocsparse_int>(),
                                                    stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<rocsparse_int>(),
                                                    stream));

        // Copy group sizes to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_group_size,
                                           d_group_size,
                                           sizeof(rocsparse_int) * CSRGEMM_MAXGROUPS,
                                           hipMemcpyDeviceToHost,
                                           stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, m, tmp_perm));

        rocprim::double_buffer<rocsparse_int> d_keys(tmp_groups, tmp_keys);
        rocprim::double_buffer<rocsparse_int> d_vals(tmp_perm, tmp_vals);

        // Sort pairs (by groups)
        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, rocprim_size, d_keys, d_vals, m, 0, 3, stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            rocprim_buffer, rocprim_size, d_keys, d_vals, m, 0, 3, stream));

        d_perm = d_vals.current();

        // Release tmp_groups buffer
        buffer -= ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        // Release tmp_keys buffer
        buffer -= ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(rocsparse_int), stream));
    }

    // Compute columns and accumulate values for each group

    // pointer mode device
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // Group 0: 0 - 16 non-zeros per row
        if(h_group_size[0] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 16
            hipLaunchKernelGGL((csrgemm_fill_wf_per_row_device_pointer<T,
                                                                       CSRGEMM_DIM,
                                                                       CSRGEMM_SUB,
                                                                       CSRGEMM_HASHSIZE,
                                                                       CSRGEMM_FLL_HASH>),
                               dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               h_group_size[0],
                               std::max(k, n),
                               &d_group_offset[0],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 1: 17 - 32 non-zeros per row
        if(h_group_size[1] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 32
            hipLaunchKernelGGL((csrgemm_fill_wf_per_row_device_pointer<T,
                                                                       CSRGEMM_DIM,
                                                                       CSRGEMM_SUB,
                                                                       CSRGEMM_HASHSIZE,
                                                                       CSRGEMM_FLL_HASH>),
                               dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               h_group_size[1],
                               std::max(k, n),
                               &d_group_offset[1],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 2: 33 - 256 non-zeros per row
        if(h_group_size[2] > 0)
        {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 256
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<T,
                                                                          CSRGEMM_DIM,
                                                                          CSRGEMM_SUB,
                                                                          CSRGEMM_HASHSIZE,
                                                                          CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[2]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[2],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 3: 257 - 512 non-zeros per row
        if(h_group_size[3] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 512
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<T,
                                                                          CSRGEMM_DIM,
                                                                          CSRGEMM_SUB,
                                                                          CSRGEMM_HASHSIZE,
                                                                          CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[3]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[3],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 4: 513 - 1024 non-zeros per row
        if(h_group_size[4] > 0)
        {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 1024
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<T,
                                                                          CSRGEMM_DIM,
                                                                          CSRGEMM_SUB,
                                                                          CSRGEMM_HASHSIZE,
                                                                          CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[4]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[4],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 5: 1025 - 2048 non-zeros per row
        if(h_group_size[5] > 0)
        {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 2048
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<T,
                                                                          CSRGEMM_DIM,
                                                                          CSRGEMM_SUB,
                                                                          CSRGEMM_HASHSIZE,
                                                                          CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[5]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[5],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

#ifndef rocsparse_ILP64
        // Group 6: 2049 - 4096 non-zeros per row
        if(h_group_size[6] > 0 && !is_double_complex)
        {
            RETURN_IF_ROCSPARSE_ERROR(csrgemm_launcher<T>(handle,
                                                          h_group_size[6],
                                                          &d_group_offset[6],
                                                          d_perm,
                                                          m,
                                                          n,
                                                          k,
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_val_A,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          csr_val_B,
                                                          beta,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          csr_val_D,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          csr_val_C,
                                                          base_A,
                                                          base_B,
                                                          descr_C->base,
                                                          base_D,
                                                          info_C->csrgemm_info->mul,
                                                          info_C->csrgemm_info->add));
        }
#endif

        // Group 7: more than 4096 non-zeros per row
        if(h_group_size[7] > 0)
        {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
            rocsparse_int* workspace_B = nullptr;

            if(info_C->csrgemm_info->mul == true)
            {
                // Allocate additional buffer for C = alpha * A * B
                RETURN_IF_HIP_ERROR(hipMalloc((void**)&workspace_B, sizeof(rocsparse_int) * nnz_A));
            }

            hipLaunchKernelGGL(
                (csrgemm_fill_block_per_row_multipass_device_pointer<T,
                                                                     CSRGEMM_DIM,
                                                                     CSRGEMM_SUB,
                                                                     CSRGEMM_CHUNKSIZE>),
                dim3(h_group_size[7]),
                dim3(CSRGEMM_DIM),
                0,
                stream,
                n,
                &d_group_offset[7],
                d_perm,
                alpha,
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_val_B,
                beta,
                csr_row_ptr_D,
                csr_col_ind_D,
                csr_val_D,
                csr_row_ptr_C,
                csr_col_ind_C,
                csr_val_C,
                workspace_B,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add);

            if(info_C->csrgemm_info->mul == true)
            {
                RETURN_IF_HIP_ERROR(hipFree(workspace_B));
            }
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }
    }
    else
    {
        // Group 0: 0 - 16 non-zeros per row
        if(h_group_size[0] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 16
            hipLaunchKernelGGL((csrgemm_fill_wf_per_row_host_pointer<T,
                                                                     CSRGEMM_DIM,
                                                                     CSRGEMM_SUB,
                                                                     CSRGEMM_HASHSIZE,
                                                                     CSRGEMM_FLL_HASH>),
                               dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               h_group_size[0],
                               std::max(k, n),
                               &d_group_offset[0],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 1: 17 - 32 non-zeros per row
        if(h_group_size[1] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 32
            hipLaunchKernelGGL((csrgemm_fill_wf_per_row_host_pointer<T,
                                                                     CSRGEMM_DIM,
                                                                     CSRGEMM_SUB,
                                                                     CSRGEMM_HASHSIZE,
                                                                     CSRGEMM_FLL_HASH>),
                               dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               h_group_size[1],
                               std::max(k, n),
                               &d_group_offset[1],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 2: 33 - 256 non-zeros per row
        if(h_group_size[2] > 0)
        {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 256
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<T,
                                                                        CSRGEMM_DIM,
                                                                        CSRGEMM_SUB,
                                                                        CSRGEMM_HASHSIZE,
                                                                        CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[2]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[2],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 3: 257 - 512 non-zeros per row
        if(h_group_size[3] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 512
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<T,
                                                                        CSRGEMM_DIM,
                                                                        CSRGEMM_SUB,
                                                                        CSRGEMM_HASHSIZE,
                                                                        CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[3]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[3],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 4: 513 - 1024 non-zeros per row
        if(h_group_size[4] > 0)
        {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 1024
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<T,
                                                                        CSRGEMM_DIM,
                                                                        CSRGEMM_SUB,
                                                                        CSRGEMM_HASHSIZE,
                                                                        CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[4]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[4],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 5: 1025 - 2048 non-zeros per row
        if(h_group_size[5] > 0)
        {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 2048
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<T,
                                                                        CSRGEMM_DIM,
                                                                        CSRGEMM_SUB,
                                                                        CSRGEMM_HASHSIZE,
                                                                        CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[5]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[5],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

#ifndef rocsparse_ILP64
        // Group 6: 2049 - 4096 non-zeros per row
        if(h_group_size[6] > 0 && !is_double_complex)
        {
            RETURN_IF_ROCSPARSE_ERROR(csrgemm_launcher<T>(handle,
                                                          h_group_size[6],
                                                          &d_group_offset[6],
                                                          d_perm,
                                                          m,
                                                          n,
                                                          k,
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_val_A,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          csr_val_B,
                                                          beta,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          csr_val_D,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          csr_val_C,
                                                          base_A,
                                                          base_B,
                                                          descr_C->base,
                                                          base_D,
                                                          info_C->csrgemm_info->mul,
                                                          info_C->csrgemm_info->add));
        }
#endif

        // Group 7: more than 4096 non-zeros per row
        if(h_group_size[7] > 0)
        {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
            rocsparse_int* workspace_B = nullptr;

            if(info_C->csrgemm_info->mul == true)
            {
                // Allocate additional buffer for C = alpha * A * B
                RETURN_IF_HIP_ERROR(hipMalloc((void**)&workspace_B, sizeof(rocsparse_int) * nnz_A));
            }

            hipLaunchKernelGGL(
                (csrgemm_fill_block_per_row_multipass_host_pointer<T,
                                                                   CSRGEMM_DIM,
                                                                   CSRGEMM_SUB,
                                                                   CSRGEMM_CHUNKSIZE>),
                dim3(h_group_size[7]),
                dim3(CSRGEMM_DIM),
                0,
                stream,
                n,
                &d_group_offset[7],
                d_perm,
                (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_val_B,
                (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                csr_row_ptr_D,
                csr_col_ind_D,
                csr_val_D,
                csr_row_ptr_C,
                csr_col_ind_C,
                csr_val_C,
                workspace_B,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add);

            if(info_C->csrgemm_info->mul == true)
            {
                RETURN_IF_HIP_ERROR(hipFree(workspace_B));
            }
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrgemm_mult_template(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 rocsparse_int             k,
                                                 const T*                  alpha,
                                                 const rocsparse_mat_descr descr_A,
                                                 rocsparse_int             nnz_A,
                                                 const T*                  csr_val_A,
                                                 const rocsparse_int*      csr_row_ptr_A,
                                                 const rocsparse_int*      csr_col_ind_A,
                                                 const rocsparse_mat_descr descr_B,
                                                 rocsparse_int             nnz_B,
                                                 const T*                  csr_val_B,
                                                 const rocsparse_int*      csr_row_ptr_B,
                                                 const rocsparse_int*      csr_col_ind_B,
                                                 const T*                  beta,
                                                 const rocsparse_mat_descr descr_D,
                                                 rocsparse_int             nnz_D,
                                                 const T*                  csr_val_D,
                                                 const rocsparse_int*      csr_row_ptr_D,
                                                 const rocsparse_int*      csr_col_ind_D,
                                                 const rocsparse_mat_descr descr_C,
                                                 T*                        csr_val_C,
                                                 const rocsparse_int*      csr_row_ptr_C,
                                                 rocsparse_int*            csr_col_ind_C,
                                                 const rocsparse_mat_info  info_C,
                                                 void*                     temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || csr_val_A == nullptr || csr_row_ptr_A == nullptr
       || csr_col_ind_A == nullptr || descr_B == nullptr || csr_val_B == nullptr
       || csr_row_ptr_B == nullptr || csr_col_ind_B == nullptr || descr_C == nullptr
       || csr_val_C == nullptr || csr_row_ptr_C == nullptr || csr_col_ind_C == nullptr
       || temp_buffer == nullptr || alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
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

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_status_success;
    }

    // Perform gemm calculation
    return rocsparse_csrgemm_calc_template<T>(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_val_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_val_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              beta,
                                              descr_D,
                                              nnz_D,
                                              csr_val_D,
                                              csr_row_ptr_D,
                                              csr_col_ind_D,
                                              descr_C,
                                              csr_val_C,
                                              csr_row_ptr_C,
                                              csr_col_ind_C,
                                              info_C,
                                              temp_buffer);
}

template <typename T>
__global__ void csrgemm_copy_scale_host_pointer(rocsparse_int size,
                                                T             alpha,
                                                const T* __restrict__ in,
                                                T* __restrict__ out)
{
    csrgemm_copy_scale_device(size, alpha, in, out);
}

template <typename T>
__global__ void csrgemm_copy_scale_device_pointer(rocsparse_int size,
                                                  const T* __restrict__ alpha,
                                                  const T* __restrict__ in,
                                                  T* __restrict__ out)
{
    csrgemm_copy_scale_device(size, *alpha, in, out);
}

template <typename T>
rocsparse_status rocsparse_csrgemm_scal_template(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const T*                  beta,
                                                 const rocsparse_mat_descr descr_D,
                                                 rocsparse_int             nnz_D,
                                                 const T*                  csr_val_D,
                                                 const rocsparse_int*      csr_row_ptr_D,
                                                 const rocsparse_int*      csr_col_ind_D,
                                                 const rocsparse_mat_descr descr_C,
                                                 T*                        csr_val_C,
                                                 const rocsparse_int*      csr_row_ptr_C,
                                                 rocsparse_int*            csr_col_ind_C,
                                                 const rocsparse_mat_info  info_C,
                                                 void*                     temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_D == nullptr || csr_val_D == nullptr || csr_row_ptr_D == nullptr
       || csr_col_ind_D == nullptr || descr_C == nullptr || csr_val_C == nullptr
       || csr_row_ptr_C == nullptr || csr_col_ind_C == nullptr || temp_buffer == nullptr
       || beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr_C->base != rocsparse_index_base_zero && descr_C->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr_D->base != rocsparse_index_base_zero && descr_D->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_D->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz_D == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Copy column entries
#define CSRGEMM_DIM 1024
    dim3 csrgemm_blocks((nnz_D - 1) / CSRGEMM_DIM + 1);
    dim3 csrgemm_threads(CSRGEMM_DIM);

    hipLaunchKernelGGL((csrgemm_copy),
                       csrgemm_blocks,
                       csrgemm_threads,
                       0,
                       stream,
                       nnz_D,
                       csr_col_ind_D,
                       csr_col_ind_C,
                       descr_D->base,
                       descr_C->base);

    // Scale the matrix
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((csrgemm_copy_scale_device_pointer),
                           csrgemm_blocks,
                           csrgemm_threads,
                           0,
                           stream,
                           nnz_D,
                           beta,
                           csr_val_D,
                           csr_val_C);
    }
    else
    {
        hipLaunchKernelGGL((csrgemm_copy_scale_host_pointer),
                           csrgemm_blocks,
                           csrgemm_threads,
                           0,
                           stream,
                           nnz_D,
                           *beta,
                           csr_val_D,
                           csr_val_C);
    }
#undef CSRGEMM_DIM

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrgemm_template(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            rocsparse_int             k,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const T*                  csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const T*                  csr_val_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_D,
                                            rocsparse_int             nnz_D,
                                            const T*                  csr_val_D,
                                            const rocsparse_int*      csr_row_ptr_D,
                                            const rocsparse_int*      csr_col_ind_D,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        csr_val_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C,
                                            const rocsparse_mat_info  info_C,
                                            void*                     temp_buffer)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgemm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  *alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_val_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_val_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  *beta,
                  (const void*&)descr_D,
                  nnz_D,
                  (const void*&)csr_val_D,
                  (const void*&)csr_row_ptr_D,
                  (const void*&)csr_col_ind_D,
                  (const void*&)descr_C,
                  (const void*&)csr_val_C,
                  (const void*&)csr_row_ptr_C,
                  (const void*&)csr_col_ind_C,
                  (const void*&)info_C,
                  (const void*&)temp_buffer);

        log_bench(handle,
                  "./rocsparse-bench -f csrgemm -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx> ",
                  "--alpha",
                  *alpha,
                  "--beta",
                  *beta);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgemm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  (const void*&)alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_val_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_val_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  (const void*&)beta,
                  (const void*&)descr_D,
                  nnz_D,
                  (const void*&)csr_val_D,
                  (const void*&)csr_row_ptr_D,
                  (const void*&)csr_col_ind_D,
                  (const void*&)descr_C,
                  (const void*&)csr_val_C,
                  (const void*&)csr_row_ptr_C,
                  (const void*&)csr_col_ind_C,
                  (const void*&)info_C,
                  (const void*&)temp_buffer);
    }

    // Check for valid rocsparse_mat_info
    if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for valid rocsparse_csrgemm_info
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Either mult, add or multadd need to be performed
    if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == true)
    {
        // C = alpha * A * B + beta * D
        // TODO
        return rocsparse_status_not_implemented;
    }
    else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == false)
    {
        // C = alpha * A * B
        return rocsparse_csrgemm_mult_template<T>(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_val_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_val_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_val_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  descr_C,
                                                  csr_val_C,
                                                  csr_row_ptr_C,
                                                  csr_col_ind_C,
                                                  info_C,
                                                  temp_buffer);
    }
    else if(info_C->csrgemm_info->mul == false && info_C->csrgemm_info->add == true)
    {
        // C = beta * D
        return rocsparse_csrgemm_scal_template<T>(handle,
                                                  m,
                                                  n,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_val_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  descr_C,
                                                  csr_val_C,
                                                  csr_row_ptr_C,
                                                  csr_col_ind_C,
                                                  info_C,
                                                  temp_buffer);
    }
    else
    {
        // C = 0
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRGEMM_HPP
