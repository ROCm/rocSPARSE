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
#ifndef ROCSPARSE_CSRILU0_HPP
#define ROCSPARSE_CSRILU0_HPP

#include "definitions.h"
#include "rocsparse.h"
#include "utility.h"
#include "csrilu0_device.h"
#include "../level2/rocsparse_csrsv.hpp"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_csrilu0_analysis_template(rocsparse_handle handle,
                                                     rocsparse_int m,
                                                     rocsparse_int nnz,
                                                     const rocsparse_mat_descr descr,
                                                     const T* csr_val,
                                                     const rocsparse_int* csr_row_ptr,
                                                     const rocsparse_int* csr_col_ind,
                                                     rocsparse_mat_info info,
                                                     rocsparse_analysis_policy analysis,
                                                     rocsparse_solve_policy solve,
                                                     void* temp_buffer)
{
    // Check for valid handle
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
              replaceX<T>("rocsparse_Xcsrilu0_analysis"),
              m,
              nnz,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              solve,
              analysis);

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

    // Check analysis policy
    if(analysis != rocsparse_analysis_policy_reuse && analysis != rocsparse_analysis_policy_force)
    {
        return rocsparse_status_invalid_value;
    }

    // Check solve policy
    if(solve != rocsparse_solve_policy_auto)
    {
        return rocsparse_status_invalid_value;
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

    // Quick return if possible
    if(m == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Differentiate the analysis policies
    if(analysis == rocsparse_analysis_policy_reuse)
    {
        // We try to re-use already analyzed lower part, if available.
        // It is the user's responsibility that this data is still valid,
        // since he passed the 'reuse' flag.

        // If csrilu0 meta data is already available, do nothing
        if(info->csrilu0_info != nullptr)
        {
            return rocsparse_status_success;
        }

        // Check for other lower analysis meta data
        rocsparse_csrtr_info reuse = nullptr;

        // csrsv_lower meta data
        if(info->csrsv_lower_info != nullptr)
        {
            reuse = info->csrsv_lower_info;
        }

        // TODO add more crossover data here

        // If data has been found, use it
        if(reuse != nullptr)
        {
            info->csrilu0_info = reuse;

            return rocsparse_status_success;
        }
    }

    // User is explicitly asking to force a re-analysis, or no valid data has been
    // found to be re-used.

    // Clear csrilu0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrilu0_info));

    // Create csrilu0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrtr_info(&info->csrilu0_info));

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrtr_analysis(handle,
                                                       rocsparse_operation_none,
                                                       m,
                                                       nnz,
                                                       descr,
                                                       csr_row_ptr,
                                                       csr_col_ind,
                                                       info->csrilu0_info,
                                                       temp_buffer));

    return rocsparse_status_success;
}

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

    log_bench(handle, "./rocsparse-bench -f csrilu0 -r", replaceX<T>("X"), "--mtx <matrix.mtx> ");

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

    // Quick return if possible
    if(m == 0 || nnz == 0)
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
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    ptr += 256;

    // done array
    int* d_done_array = reinterpret_cast<int*>(ptr);

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_done_array, 0, sizeof(int) * m, stream));

    // Max nnz per row
    rocsparse_int max_nnz = info->csrilu0_info->max_nnz;

#define CSRILU0_DIM 256
    dim3 csrilu0_blocks((m * handle->wavefront_size - 1) / CSRILU0_DIM + 1);
    dim3 csrilu0_threads(CSRILU0_DIM);

    if(handle->wavefront_size == 32)
    {
        if(max_nnz <= 32)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 32, 1>),
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
        else if(max_nnz <= 64)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 32, 2>),
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
        else if(max_nnz <= 128)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 32, 4>),
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
        else if(max_nnz <= 256)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 32, 8>),
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
        else if(max_nnz <= 512)
        {
            hipLaunchKernelGGL((csrilu0_hash_kernel<T, CSRILU0_DIM, 32, 16>),
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
    }
    else if(handle->wavefront_size == 64)
    {
        if(max_nnz <= 64)
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
        else if(max_nnz <= 128)
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
        else if(max_nnz <= 256)
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
        else if(max_nnz <= 512)
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
        else if(max_nnz <= 1024)
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
