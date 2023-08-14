/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../level2/rocsparse_csrsv.hpp"
#include "csrilu0_device.h"
#include "definitions.h"
#include "utility.h"

template <typename T, typename U>
rocsparse_status rocsparse_csrilu0_numeric_boost_template(rocsparse_handle   handle,
                                                          rocsparse_mat_info info,
                                                          int                enable_boost,
                                                          const U*           boost_tol,
                                                          const T*           boost_val)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrilu0_numeric_boost"),
              (const void*&)info,
              enable_boost,
              (const void*&)boost_tol,
              (const void*&)boost_val);

    // Reset boost
    info->boost_enable        = 0;
    info->use_double_prec_tol = 0;

    // Numeric boost
    if(enable_boost)
    {
        // Check pointer arguments
        if(boost_tol == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
        else if(boost_val == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }

        info->boost_enable        = enable_boost;
        info->use_double_prec_tol = std::is_same<U, double>();
        info->boost_tol           = reinterpret_cast<const void*>(boost_tol);
        info->boost_val           = reinterpret_cast<const void*>(boost_val);
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrilu0_analysis_template(rocsparse_handle          handle,
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

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_requires_sorted_storage;
    }

    // Check analysis policy
    if(rocsparse_enum_utils::is_invalid(analysis))
    {
        return rocsparse_status_invalid_value;
    }

    // Check solve policy
    if(rocsparse_enum_utils::is_invalid(solve))
    {
        return rocsparse_status_invalid_value;
    }

    if(solve != rocsparse_solve_policy_auto)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (csr_val == nullptr && csr_col_ind == nullptr))
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

        if(info->csric0_info != nullptr)
        {
            // csric0 meta data
            info->csrilu0_info = info->csric0_info;
            return rocsparse_status_success;
        }
        else if(info->csrsv_lower_info != nullptr)
        {
            // csrsv meta data
            info->csrilu0_info = info->csrsv_lower_info;
            return rocsparse_status_success;
        }
        else if(info->csrsvt_upper_info != nullptr)
        {
            // csrsvt meta data
            info->csrilu0_info = info->csrsvt_upper_info;
            return rocsparse_status_success;
        }
        else if(info->csrsm_lower_info != nullptr)
        {
            // csrsm meta data
            info->csrilu0_info = info->csrsm_lower_info;
            return rocsparse_status_success;
        }
        else if(info->csrsmt_upper_info != nullptr)
        {
            // csrsmt meta data
            info->csrilu0_info = info->csrsmt_upper_info;
            return rocsparse_status_success;
        }
    }

    // User is explicitly asking to force a re-analysis, or no valid data has been
    // found to be re-used.

    // Clear csrilu0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrilu0_info));

    // Create csrilu0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&info->csrilu0_info));

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(handle,
                                                     rocsparse_operation_none,
                                                     m,
                                                     nnz,
                                                     descr,
                                                     csr_val,
                                                     csr_row_ptr,
                                                     csr_col_ind,
                                                     info->csrilu0_info,
                                                     (rocsparse_int**)&info->zero_pivot,
                                                     temp_buffer));

    return rocsparse_status_success;
}

template <unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          bool         SLEEP,
          typename T,
          typename U,
          typename V>
ROCSPARSE_KERNEL(BLOCKSIZE)
void csrilu0_binsearch(rocsparse_int        m,
                       const rocsparse_int* csr_row_ptr,
                       const rocsparse_int* csr_col_ind,
                       T*                   csr_val,
                       const rocsparse_int* csr_diag_ind,
                       int*                 done,
                       const rocsparse_int* map,
                       rocsparse_int*       zero_pivot,
                       rocsparse_index_base idx_base,
                       int                  enable_boost,
                       U                    boost_tol_device_host,
                       V                    boost_val_device_host)
{
    auto boost_tol = (enable_boost) ? load_scalar_device_host(boost_tol_device_host)
                                    : zero_scalar_device_host(boost_tol_device_host);

    auto boost_val = (enable_boost) ? load_scalar_device_host(boost_val_device_host)
                                    : zero_scalar_device_host(boost_val_device_host);

    csrilu0_binsearch_kernel<BLOCKSIZE, WFSIZE, SLEEP>(m,
                                                       csr_row_ptr,
                                                       csr_col_ind,
                                                       csr_val,
                                                       csr_diag_ind,
                                                       done,
                                                       map,
                                                       zero_pivot,
                                                       idx_base,
                                                       enable_boost,
                                                       boost_tol,
                                                       boost_val);
}

template <unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int HASH,
          typename T,
          typename U,
          typename V>
ROCSPARSE_KERNEL(BLOCKSIZE)
void csrilu0_hash(rocsparse_int        m,
                  const rocsparse_int* csr_row_ptr,
                  const rocsparse_int* csr_col_ind,
                  T*                   csr_val,
                  const rocsparse_int* csr_diag_ind,
                  int*                 done,
                  const rocsparse_int* map,
                  rocsparse_int*       zero_pivot,
                  rocsparse_index_base idx_base,
                  int                  enable_boost,
                  U                    boost_tol_device_host,
                  V                    boost_val_device_host)
{
    auto boost_tol = (enable_boost) ? load_scalar_device_host(boost_tol_device_host)
                                    : zero_scalar_device_host(boost_tol_device_host);

    auto boost_val = (enable_boost) ? load_scalar_device_host(boost_val_device_host)
                                    : zero_scalar_device_host(boost_val_device_host);

    csrilu0_hash_kernel<BLOCKSIZE, WFSIZE, HASH>(m,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 csr_val,
                                                 csr_diag_ind,
                                                 done,
                                                 map,
                                                 zero_pivot,
                                                 idx_base,
                                                 enable_boost,
                                                 boost_tol,
                                                 boost_val);
}

template <typename T, typename U, typename V>
rocsparse_status rocsparse_csrilu0_dispatch(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             nnz,
                                            const rocsparse_mat_descr descr,
                                            T*                        csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            rocsparse_mat_info        info,
                                            rocsparse_solve_policy    policy,
                                            void*                     temp_buffer,
                                            U                         boost_tol_device_host,
                                            V                         boost_val_device_host)
{
    // Check for valid handle and matrix descriptor
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

    // Determine gcnArch and ASIC revision
    const std::string gcn_arch_name = rocsparse_handle_get_arch_name(handle);
    const int         asicRev       = handle->asic_rev;

#define CSRILU0_DIM 256
    dim3 csrilu0_blocks((m * handle->wavefront_size - 1) / CSRILU0_DIM + 1);
    dim3 csrilu0_threads(CSRILU0_DIM);

    if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
    {
        hipLaunchKernelGGL((csrilu0_binsearch<CSRILU0_DIM, 64, true>),
                           csrilu0_blocks,
                           csrilu0_threads,
                           0,
                           stream,
                           m,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                           d_done_array,
                           (rocsparse_int*)info->csrilu0_info->row_map,
                           (rocsparse_int*)info->zero_pivot,
                           descr->base,
                           info->boost_enable,
                           boost_tol_device_host,
                           boost_val_device_host);
    }
    else
    {
        if(handle->wavefront_size == 32)
        {
            if(max_nnz < 32)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 32, 1>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else if(max_nnz < 64)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 32, 2>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else if(max_nnz < 128)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 32, 4>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else if(max_nnz < 256)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 32, 8>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else if(max_nnz < 512)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 32, 16>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else
            {
                hipLaunchKernelGGL((csrilu0_binsearch<CSRILU0_DIM, 32, false>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
        }
        else if(handle->wavefront_size == 64)
        {
            if(max_nnz < 64)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 64, 1>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else if(max_nnz < 128)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 64, 2>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else if(max_nnz < 256)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 64, 4>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else if(max_nnz < 512)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 64, 8>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else if(max_nnz < 1024)
            {
                hipLaunchKernelGGL((csrilu0_hash<CSRILU0_DIM, 64, 16>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
            else
            {
                hipLaunchKernelGGL((csrilu0_binsearch<CSRILU0_DIM, 64, false>),
                                   csrilu0_blocks,
                                   csrilu0_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                                   d_done_array,
                                   (rocsparse_int*)info->csrilu0_info->row_map,
                                   (rocsparse_int*)info->zero_pivot,
                                   descr->base,
                                   info->boost_enable,
                                   boost_tol_device_host,
                                   boost_val_device_host);
            }
        }
        else
        {
            return rocsparse_status_arch_mismatch;
        }
    }
#undef CSRILU0_DIM

    return rocsparse_status_success;
}

template <typename T, typename U>
rocsparse_status rocsparse_csrilu0_template(rocsparse_handle          handle,
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

    // Check solve policy
    if(rocsparse_enum_utils::is_invalid(policy))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_requires_sorted_storage;
    }

    // Check sizes
    if(m < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (csr_val == nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for analysis call
    if(info->csrilu0_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_csrilu0_dispatch(handle,
                                          m,
                                          nnz,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          policy,
                                          temp_buffer,
                                          reinterpret_cast<const U*>(info->boost_tol),
                                          reinterpret_cast<const T*>(info->boost_val));
    }
    else
    {
        return rocsparse_csrilu0_dispatch(
            handle,
            m,
            nnz,
            descr,
            csr_val,
            csr_row_ptr,
            csr_col_ind,
            info,
            policy,
            temp_buffer,
            (info->boost_enable != 0) ? *reinterpret_cast<const U*>(info->boost_tol)
                                      : static_cast<U>(0),
            (info->boost_enable != 0) ? *reinterpret_cast<const T*>(info->boost_val)
                                      : static_cast<T>(0));
    }
}
