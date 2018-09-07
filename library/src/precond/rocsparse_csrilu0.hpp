/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_CSRILU0_HPP
#define ROCSPARSE_CSRILU0_HPP

#include "rocsparse.h"
#include "utility.h"
#include "csrilu0_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_csrilu0_template(rocsparse_handle handle,
                                            rocsparse_int m,
                                            rocsparse_int nnz,
                                            const rocsparse_mat_descr descr,
                                            T* csr_val,
                                            const rocsparse_int* csr_row_ptr,
                                            const rocsparse_int* csr_col_ind,
                                            rocsparse_mat_info info,
                                            rocsparse_solve_policy policy,
                                            void* temp_buffer)
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

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrilu0"),
              m,
              nnz,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              policy,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f csrilu0 -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> ");

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
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;



    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    ptr += 256;
    ptr += 256;
    ptr += 256;

    // done array
    rocsparse_int* d_done_array = reinterpret_cast<rocsparse_int*>(ptr);

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemset(d_done_array, 0, sizeof(rocsparse_int) * m));

#define CSRILU0_DIM 256
    dim3 csrilu0_blocks((m * handle->wavefront_size - 1) / CSRILU0_DIM + 1);
    dim3 csrilu0_threads(CSRILU0_DIM);

    if(handle->wavefront_size == 32)
    {
        hipLaunchKernelGGL((csrilu0_binsearch_kernel<T, CSRILU0_DIM, 32>),
                           csrilu0_blocks,
                           csrilu0_threads,
                           0,
                           stream,
                           m,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           info->csrilu0_info->csr_diag_ind,
                           d_done_array,
                           info->csrilu0_info->row_map,
                           info->csrilu0_info->zero_pivot,
                           descr->base);
    }
    else if(handle->wavefront_size == 64)
    {
        if(info->csrilu0_info->max_nnz <= 64)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 64, 1>),
                               csrilu0_blocks,
                               csrilu0_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               info->csrilu0_info->csr_diag_ind,
                               d_done_array,
                               info->csrilu0_info->row_map,
                               info->csrilu0_info->zero_pivot,
                               descr->base);
        }
        else if(info->csrilu0_info->max_nnz <= 128)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 64, 2>),
                               csrilu0_blocks,
                               csrilu0_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               info->csrilu0_info->csr_diag_ind,
                               d_done_array,
                               info->csrilu0_info->row_map,
                               info->csrilu0_info->zero_pivot,
                               descr->base);
        }
        else if(info->csrilu0_info->max_nnz <= 256)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 64, 4>),
                               csrilu0_blocks,
                               csrilu0_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               info->csrilu0_info->csr_diag_ind,
                               d_done_array,
                               info->csrilu0_info->row_map,
                               info->csrilu0_info->zero_pivot,
                               descr->base);
        }
        else if(info->csrilu0_info->max_nnz <= 512)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 64, 8>),
                               csrilu0_blocks,
                               csrilu0_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               info->csrilu0_info->csr_diag_ind,
                               d_done_array,
                               info->csrilu0_info->row_map,
                               info->csrilu0_info->zero_pivot,
                               descr->base);
        }
        else if(info->csrilu0_info->max_nnz <= 1024)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 64, 16>),
                               csrilu0_blocks,
                               csrilu0_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               info->csrilu0_info->csr_diag_ind,
                               d_done_array,
                               info->csrilu0_info->row_map,
                               info->csrilu0_info->zero_pivot,
                               descr->base);
        }
        else
        {
            printf("standard kernel\n");
            hipLaunchKernelGGL((csrilu0_binsearch_kernel<T, CSRILU0_DIM, 64>),
                               csrilu0_blocks,
                               csrilu0_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               info->csrilu0_info->csr_diag_ind,
                               d_done_array,
                               info->csrilu0_info->row_map,
                               info->csrilu0_info->zero_pivot,
                               descr->base);
        }
    }
    else
    {
        return rocsparse_status_arch_mismatch;
    }
#undef CSRILU0_DIM

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRILU0_HPP
