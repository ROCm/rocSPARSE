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
#ifndef ROCSPARSE_CSRSM_HPP
#define ROCSPARSE_CSRSM_HPP

#include "../level2/rocsparse_csrsv.hpp"
#include "csrsm_device.h"
#include "definitions.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <limits>
#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_csrsm_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_B,
                                                      rocsparse_int             m,
                                                      rocsparse_int             nrhs,
                                                      rocsparse_int             nnz,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  csr_val,
                                                      const rocsparse_int*      csr_row_ptr,
                                                      const rocsparse_int*      csr_col_ind,
                                                      const T*                  B,
                                                      rocsparse_int             ldb,
                                                      rocsparse_mat_info        info,
                                                      rocsparse_solve_policy    policy,
                                                      size_t*                   buffer_size)
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
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrsm_buffer_size"),
                  trans_A,
                  trans_B,
                  m,
                  nrhs,
                  nnz,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  (const void*&)info,
                  policy,
                  (const void*&)buffer_size);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrsm_buffer_size"),
                  trans_A,
                  trans_B,
                  m,
                  nrhs,
                  nnz,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  (const void*&)info,
                  policy,
                  (const void*&)buffer_size);
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

    // Check operation type
    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }
    if(trans_B != rocsparse_operation_none && trans_B != rocsparse_operation_transpose)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nrhs < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // TODO check ldb

    // Check for valid buffer_size pointer
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || nrhs == 0 || nnz == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;
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
    else if(B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // rocsparse_int max_nnz
    *buffer_size = 256;

    // Each thread block performs at most blockdim columns of the
    // rhs matrix. Therefore, the number of blocks depend on nrhs
    // and the blocksize.
    // Because of this, we might need a larger done_array compared
    // to csrsv.
    int blockdim = 512;
    while(nrhs <= blockdim && blockdim > 32)
    {
        blockdim >>= 1;
    }

    blockdim <<= 1;
    int narrays = (nrhs - 1) / blockdim + 1;

    // int done_array
    *buffer_size += sizeof(int) * ((m * narrays - 1) / 256 + 1) * 256;

    // rocsparse_int workspace
    *buffer_size += sizeof(rocsparse_int) * ((m - 1) / 256 + 1) * 256;

    // int workspace2
    *buffer_size += sizeof(int) * ((m - 1) / 256 + 1) * 256;

    size_t         rocprim_size;
    rocsparse_int* ptr  = reinterpret_cast<rocsparse_int*>(buffer_size);
    int*           ptr2 = reinterpret_cast<int*>(buffer_size);

    rocprim::double_buffer<rocsparse_int> dummy(ptr, ptr);
    rocprim::double_buffer<int>           dummy2(ptr2, ptr2);

    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, rocprim_size, dummy2, dummy, m, 0, 32, stream));

    // rocprim buffer
    *buffer_size += rocprim_size;

    // Additional buffer to store transpose of B, if trans_B == rocsparse_operation_none
    if(trans_B == rocsparse_operation_none)
    {
        *buffer_size += sizeof(T) * ((m * nrhs - 1) / 256 + 1) * 256;
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrsm_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nrhs,
                                                   rocsparse_int             nnz,
                                                   const T*                  alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   const T*                  B,
                                                   rocsparse_int             ldb,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_analysis_policy analysis,
                                                   rocsparse_solve_policy    solve,
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
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrsm_analysis"),
                  trans_A,
                  trans_B,
                  m,
                  nrhs,
                  nnz,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  (const void*&)info,
                  analysis,
                  solve,
                  (const void*&)temp_buffer);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrsm_analysis"),
                  trans_A,
                  trans_B,
                  m,
                  nrhs,
                  nnz,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  (const void*&)info,
                  analysis,
                  solve,
                  (const void*&)temp_buffer);
    }

    // Check operation type
    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }
    if(trans_B != rocsparse_operation_none && trans_B != rocsparse_operation_transpose)
    {
        return rocsparse_status_not_implemented;
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
    else if(nrhs < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || nrhs == 0 || nnz == 0)
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
    else if(B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Switch between lower and upper triangular analysis
    if(descr->fill_mode == rocsparse_fill_mode_upper)
    {
        // Differentiate the analysis policies
        if(analysis == rocsparse_analysis_policy_reuse)
        {
            // We try to re-use already analyzed upper part, if available.
            // It is the user's responsibility that this data is still valid,
            // since he passed the 'reuse' flag.

            // If csrsm meta data is already available, do nothing
            if(trans_A == rocsparse_operation_none && info->csrsm_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }
            //            else if(trans_A == rocsparse_operation_transpose && info->csrsmt_upper_info != nullptr)
            //            {
            //                return rocsparse_status_success;
            //            }

            // Check for other upper analysis meta data

            if(trans_A == rocsparse_operation_none && info->csrsv_upper_info != nullptr)
            {
                // csrsv meta data
                info->csrsm_upper_info = info->csrsv_upper_info;
                return rocsparse_status_success;
            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used

        // Clear csrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsm_upper_info));

        // Create csrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&info->csrsm_upper_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(handle,
                                                         trans_A,
                                                         m,
                                                         nnz,
                                                         descr,
                                                         csr_val,
                                                         csr_row_ptr,
                                                         csr_col_ind,
                                                         info->csrsm_upper_info,
                                                         &info->zero_pivot,
                                                         temp_buffer));
    }
    else
    {
        // Differentiate the analysis policies
        if(analysis == rocsparse_analysis_policy_reuse)
        {
            // We try to re-use already analyzed lower part, if available.
            // It is the user's responsibility that this data is still valid,
            // since he passed the 'reuse' flag.

            // If csrsm meta data is already available, do nothing
            if(trans_A == rocsparse_operation_none && info->csrsm_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }
            //            else if(trans_A == rocsparse_operation_transpose && info->csrsmt_lower_info != nullptr)
            //            {
            //                return rocsparse_status_success;
            //            }

            // Check for other lower analysis meta data

            if(trans_A == rocsparse_operation_none && info->csrilu0_info != nullptr)
            {
                // csrilu0 meta data
                info->csrsm_lower_info = info->csrilu0_info;
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_none && info->csric0_info != nullptr)
            {
                // csric0 meta data
                info->csrsm_lower_info = info->csric0_info;
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_none && info->csrsv_lower_info != nullptr)
            {
                // csrsv meta data
                info->csrsm_lower_info = info->csrsv_lower_info;
                return rocsparse_status_success;
            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used

        // Clear csrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsm_lower_info));

        // Create csrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&info->csrsm_lower_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(handle,
                                                         trans_A,
                                                         m,
                                                         nnz,
                                                         descr,
                                                         csr_val,
                                                         csr_row_ptr,
                                                         csr_col_ind,
                                                         info->csrsm_lower_info,
                                                         &info->zero_pivot,
                                                         temp_buffer));
    }

    return rocsparse_status_success;
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void csrsm_host_pointer(rocsparse_int m,
                            rocsparse_int nrhs,
                            T             alpha,
                            const rocsparse_int* __restrict__ csr_row_ptr,
                            const rocsparse_int* __restrict__ csr_col_ind,
                            const T* __restrict__ csr_val,
                            T* __restrict__ B,
                            rocsparse_int ldb,
                            int* __restrict__ done_array,
                            rocsparse_int* __restrict__ map,
                            rocsparse_int* __restrict__ zero_pivot,
                            rocsparse_index_base idx_base,
                            rocsparse_fill_mode  fill_mode,
                            rocsparse_diag_type  diag_type)
{
    csrsm_device<T, BLOCKSIZE, WFSIZE, SLEEP>(m,
                                              nrhs,
                                              alpha,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              csr_val,
                                              B,
                                              ldb,
                                              done_array,
                                              map,
                                              zero_pivot,
                                              idx_base,
                                              fill_mode,
                                              diag_type);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void csrsm_device_pointer(rocsparse_int m,
                              rocsparse_int nrhs,
                              const T*      alpha,
                              const rocsparse_int* __restrict__ csr_row_ptr,
                              const rocsparse_int* __restrict__ csr_col_ind,
                              const T* __restrict__ csr_val,
                              T* __restrict__ B,
                              rocsparse_int ldb,
                              int* __restrict__ done_array,
                              rocsparse_int* __restrict__ map,
                              rocsparse_int* __restrict__ zero_pivot,
                              rocsparse_index_base idx_base,
                              rocsparse_fill_mode  fill_mode,
                              rocsparse_diag_type  diag_type)
{
    csrsm_device<T, BLOCKSIZE, WFSIZE, SLEEP>(m,
                                              nrhs,
                                              *alpha,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              csr_val,
                                              B,
                                              ldb,
                                              done_array,
                                              map,
                                              zero_pivot,
                                              idx_base,
                                              fill_mode,
                                              diag_type);
}

template <typename T>
rocsparse_status rocsparse_csrsm_solve_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_int             m,
                                                rocsparse_int             nrhs,
                                                rocsparse_int             nnz,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                T*                        B,
                                                rocsparse_int             ldb,
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
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrsm_solve"),
                  trans_A,
                  trans_B,
                  m,
                  nrhs,
                  nnz,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  (const void*&)info,
                  policy,
                  (const void*&)temp_buffer);

        log_bench(handle,
                  "./rocsparse-bench -f csrsm -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx> ",
                  "--alpha",
                  *alpha);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrsm_solve"),
                  trans_A,
                  trans_B,
                  m,
                  nrhs,
                  nnz,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  (const void*&)info,
                  policy,
                  (const void*&)temp_buffer);
    }

    // Check operation type
    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }
    if(trans_B != rocsparse_operation_none && trans_B != rocsparse_operation_transpose)
    {
        return rocsparse_status_not_implemented;
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
    else if(nrhs < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || nrhs == 0 || nnz == 0)
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
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(B == nullptr)
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

    // Each thread block performs at most blockdim columns of the
    // rhs matrix. Therefore, the number of blocks depend on nrhs
    // and the blocksize.
    // Because of this, we might need a larger done_array compared
    // to csrsv.
    int blockdim = 512;
    while(nrhs <= blockdim && blockdim > 32)
    {
        blockdim >>= 1;
    }
    blockdim <<= 1;

    int narrays = (nrhs - 1) / blockdim + 1;

    // done array
    int* done_array = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * ((m * narrays - 1) / 256 + 1) * 256;

    // Temporary array to store transpoe of B
    T* Bt = (trans_B == rocsparse_operation_none) ? reinterpret_cast<T*>(ptr) : B;

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, sizeof(int) * m * narrays, stream));

    rocsparse_trm_info csrsm = (descr->fill_mode == rocsparse_fill_mode_upper)
                                   ? info->csrsm_upper_info
                                   : info->csrsm_lower_info;

    if(csrsm == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // If diag type is unit, re-initialize zero pivot to remove structural zeros
    if(descr->diag_type == rocsparse_diag_type_unit)
    {
        rocsparse_int max = std::numeric_limits<rocsparse_int>::max();
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            info->zero_pivot, &max, sizeof(rocsparse_int), hipMemcpyHostToDevice, stream));

        // Wait for device transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }

    // Leading dimension
    rocsparse_int ldimB = ldb;

    // Transpose B if B is not transposed yet to improve performance
    if(trans_B == rocsparse_operation_none)
    {
        // Leading dimension for transposed B
        ldimB = nrhs;

#define CSRSM_DIM_X 32
#define CSRSM_DIM_Y 8
        dim3 csrsm_blocks((m - 1) / CSRSM_DIM_X + 1);
        dim3 csrsm_threads(CSRSM_DIM_X * CSRSM_DIM_Y);

        hipLaunchKernelGGL((csrsm_transpose<T, CSRSM_DIM_X, CSRSM_DIM_Y>),
                           csrsm_blocks,
                           csrsm_threads,
                           0,
                           stream,
                           m,
                           nrhs,
                           B,
                           ldb,
                           Bt,
                           ldimB);
#undef CSRSM_DIM_X
#undef CSRSM_DIM_Y
    }

    dim3 csrsm_blocks(((nrhs - 1) / blockdim + 1) * m);
    dim3 csrsm_threads(blockdim);

    // Determine gcnArch
    int gcnArch = handle->properties.gcnArch;

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // rocsparse_pointer_mode_device

        if(blockdim == 64)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 64, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 64, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 128)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 128, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 128, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 256)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 256, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 256, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 512)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 512, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 512, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 1024)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 1024, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_device_pointer<T, 1024, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else
        {
            return rocsparse_status_internal_error;
        }
    }
    else
    {
        // rocsparse_pointer_mode_host

        if(blockdim == 64)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 64, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 64, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 128)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 128, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 128, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 256)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 256, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 256, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 512)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 512, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 512, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 1024)
        {
            if(gcnArch == 908)
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 1024, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm_host_pointer<T, 1024, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   m,
                                   nrhs,
                                   *alpha,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   csrsm->row_map,
                                   info->zero_pivot,
                                   descr->base,
                                   descr->fill_mode,
                                   descr->diag_type);
            }
        }
        else
        {
            return rocsparse_status_internal_error;
        }
    }

    // Transpose B back if B was not initially transposed
    if(trans_B == rocsparse_operation_none)
    {
#define CSRSM_DIM_X 32
#define CSRSM_DIM_Y 8
        dim3 csrsm_blocks((m - 1) / CSRSM_DIM_X + 1);
        dim3 csrsm_threads(CSRSM_DIM_X * CSRSM_DIM_Y);

        hipLaunchKernelGGL((csrsm_transpose_back<T, CSRSM_DIM_X, CSRSM_DIM_Y>),
                           csrsm_blocks,
                           csrsm_threads,
                           0,
                           stream,
                           m,
                           nrhs,
                           Bt,
                           ldimB,
                           B,
                           ldb);
#undef CSRSM_DIM_X
#undef CSRSM_DIM_Y
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRSM_HPP
