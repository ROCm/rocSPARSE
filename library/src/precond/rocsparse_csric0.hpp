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
#ifndef ROCSPARSE_CSRIC0_HPP
#define ROCSPARSE_CSRIC0_HPP

#include "../level2/rocsparse_csrsv.hpp"
#include "csric0_device.h"
#include "definitions.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_csric0_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    rocsparse_int             nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  csr_val,
                                                    const rocsparse_int*      csr_row_ptr,
                                                    const rocsparse_int*      csr_col_ind,
                                                    rocsparse_mat_info        info,
                                                    rocsparse_analysis_policy analysis,
                                                    rocsparse_solve_policy    solve,
                                                    void*                     temp_buffer)
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
              replaceX<T>("rocsparse_Xcsric0_analysis"),
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

        // If csric0 meta data is already available, do nothing
        if(info->csric0_info != nullptr)
        {
            return rocsparse_status_success;
        }

        // Check for other lower analysis meta data

        if(info->csrilu0_info != nullptr)
        {
            // csrilu0 meta data
            info->csric0_info = info->csrilu0_info;
            return rocsparse_status_success;
        }
        else if(info->csrsv_lower_info != nullptr)
        {
            // csrsv meta data
            info->csric0_info = info->csrsv_lower_info;
            return rocsparse_status_success;
        }
        else if(info->csrsm_lower_info != nullptr)
        {
            // csrsm meta data
            info->csric0_info = info->csrsm_lower_info;
            return rocsparse_status_success;
        }
    }

    // User is explicitly asking to force a re-analysis, or no valid data has been
    // found to be re-used.

    // Clear csric0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csric0_info));

    // Create csric0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&info->csric0_info));

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(handle,
                                                     rocsparse_operation_none,
                                                     m,
                                                     nnz,
                                                     descr,
                                                     csr_val,
                                                     csr_row_ptr,
                                                     csr_col_ind,
                                                     info->csric0_info,
                                                     &info->zero_pivot,
                                                     temp_buffer));

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csric0_template(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           T*                        csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info,
                                           rocsparse_solve_policy    policy,
                                           void*                     temp_buffer)
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
              replaceX<T>("rocsparse_Xcsric0"),
              m,
              nnz,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              policy,
              (const void*&)temp_buffer);

    log_bench(handle, "./rocsparse-bench -f csric0 -r", replaceX<T>("X"), "--mtx <matrix.mtx> ");

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

    // Check for analysis call
    if(info->csric0_info == nullptr)
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
    rocsparse_int max_nnz = info->csric0_info->max_nnz;

    // Determine gcnArch
    int gcnArch = handle->properties.gcnArch;

#define CSRIC0_DIM 256
    dim3 csric0_blocks((m * handle->wavefront_size - 1) / CSRIC0_DIM + 1);
    dim3 csric0_threads(CSRIC0_DIM);

    if(gcnArch == 908)
    {
        hipLaunchKernelGGL((csric0_binsearch_kernel<T, CSRIC0_DIM, 64, true>),
                           csric0_blocks,
                           csric0_threads,
                           0,
                           stream,
                           m,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           info->csric0_info->trm_diag_ind,
                           d_done_array,
                           info->csric0_info->row_map,
                           info->zero_pivot,
                           descr->base);
    }
    else
    {
        if(handle->wavefront_size == 32)
        {
            if(max_nnz <= 32)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 32, 1>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else if(max_nnz <= 64)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 32, 2>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else if(max_nnz <= 128)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 32, 4>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else if(max_nnz <= 256)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 32, 8>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else if(max_nnz <= 512)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 32, 16>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else
            {
                hipLaunchKernelGGL((csric0_binsearch_kernel<T, CSRIC0_DIM, 32, false>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
        }
        else if(handle->wavefront_size == 64)
        {
            if(max_nnz <= 64)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 64, 1>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else if(max_nnz <= 128)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 64, 2>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else if(max_nnz <= 256)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 64, 4>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else if(max_nnz <= 512)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 64, 8>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else if(max_nnz <= 1024)
            {
                hipLaunchKernelGGL((csric0_hash_kernel<T, CSRIC0_DIM, 64, 16>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
            else
            {
                hipLaunchKernelGGL((csric0_binsearch_kernel<T, CSRIC0_DIM, 64, false>),
                                   csric0_blocks,
                                   csric0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   info->csric0_info->trm_diag_ind,
                                   d_done_array,
                                   info->csric0_info->row_map,
                                   info->zero_pivot,
                                   descr->base);
            }
        }
        else
        {
            return rocsparse_status_arch_mismatch;
        }
    }
#undef CSRIC0_DIM

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRIC0_HPP
