/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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
#include "rocsparse_csrsv.hpp"

#include "../conversion/rocsparse_coo2csr.hpp"
#include "../conversion/rocsparse_csr2coo.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "../level1/rocsparse_gthr.hpp"
#include "csrsv_device.h"
#include "definitions.h"
#include "utility.h"
#include <rocprim/rocprim.hpp>

template <typename I, typename J, typename T>
rocsparse_status rocsparse_trm_analysis(rocsparse_handle          handle,
                                        rocsparse_operation       trans,
                                        J                         m,
                                        I                         nnz,
                                        const rocsparse_mat_descr descr,
                                        const T*                  csr_val,
                                        const I*                  csr_row_ptr,
                                        const J*                  csr_col_ind,
                                        rocsparse_trm_info        info,
                                        J**                       zero_pivot,
                                        void*                     temp_buffer)
{
    // Stream
    hipStream_t stream = handle->stream;

    // If analyzing transposed, allocate some info memory to hold the transposed matrix
    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
    {
        // TODO: this need to be changed.
        // LCOV_EXCL_START
        if(info->trmt_perm != nullptr || info->trmt_row_ptr != nullptr
           || info->trmt_col_ind != nullptr)
        {
            return rocsparse_status_internal_error;
        }
        // LCOV_EXCL_STOP

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);

        // work1 buffer
        J* tmp_work1 = reinterpret_cast<J*>(ptr);
        ptr += sizeof(J) * ((nnz - 1) / 256 + 1) * 256;

        // work2 buffer
        I* tmp_work2 = reinterpret_cast<I*>(ptr);
        ptr += sizeof(I) * ((nnz - 1) / 256 + 1) * 256;

        // rocprim buffer
        void* rocprim_buffer = reinterpret_cast<void*>(ptr);

        // Load CSR column indices into work1 buffer
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            tmp_work1, csr_col_ind, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->trmt_perm, sizeof(I) * nnz));
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->trmt_row_ptr, sizeof(I) * (m + 1)));
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->trmt_col_ind, sizeof(J) * nnz));

        if(nnz > 0)
        {
            // Create identity permutation
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse_create_identity_permutation_template(handle, nnz, (I*)info->trmt_perm));

            // Stable sort COO by columns
            rocprim::double_buffer<J> keys(tmp_work1, (J*)info->trmt_col_ind);
            rocprim::double_buffer<I> vals((I*)info->trmt_perm, tmp_work2);

            unsigned int startbit = 0;
            unsigned int endbit   = rocsparse_clz(m);

            size_t rocprim_size;

            RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
                nullptr, rocprim_size, keys, vals, nnz, startbit, endbit, stream));
            RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
                rocprim_buffer, rocprim_size, keys, vals, nnz, startbit, endbit, stream));

            // Copy permutation vector, if not already available
            if(vals.current() != info->trmt_perm)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(info->trmt_perm,
                                                   vals.current(),
                                                   sizeof(I) * nnz,
                                                   hipMemcpyDeviceToDevice,
                                                   stream));
            }

            // Create column pointers
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_coo2csr_template(
                handle, keys.current(), nnz, m, (I*)info->trmt_row_ptr, descr->base));

            // Create row indices
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse_csr2coo_template(handle, csr_row_ptr, nnz, m, tmp_work1, descr->base));

            // Permute column indices
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_gthr_template(handle,
                                                              nnz,
                                                              tmp_work1,
                                                              (J*)info->trmt_col_ind,
                                                              (const I*)info->trmt_perm,
                                                              rocsparse_index_base_zero));
        }
        else
        {
            hipLaunchKernelGGL((set_array_to_value<256, J, I>),
                               dim3(m / 256 + 1),
                               dim3(256),
                               0,
                               stream,
                               (m + 1),
                               (I*)info->trmt_row_ptr,
                               static_cast<I>(descr->base));
        }
    }

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // Initialize temporary buffer with 0
    size_t buffer_size = 256 + sizeof(int) * ((m - 1) / 256 + 1) * 256;
    RETURN_IF_HIP_ERROR(hipMemsetAsync(ptr, 0, sizeof(char) * buffer_size, stream));

    // max_nnz
    I* d_max_nnz = reinterpret_cast<I*>(ptr);
    ptr += 256;

    // done array
    int* done_array = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * ((m - 1) / 256 + 1) * 256;

    // workspace
    J* workspace = reinterpret_cast<J*>(ptr);
    ptr += sizeof(J) * ((m - 1) / 256 + 1) * 256;

    // workspace2
    int* workspace2 = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * ((m - 1) / 256 + 1) * 256;

    // rocprim buffer
    void* rocprim_buffer = reinterpret_cast<void*>(ptr);

    // Allocate buffer to hold diagonal entry point
    RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->trm_diag_ind, sizeof(I) * m));

    // Allocate buffer to hold zero pivot
    RETURN_IF_HIP_ERROR(hipMalloc((void**)zero_pivot, sizeof(J)));

    // Allocate buffer to hold row map
    RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->row_map, sizeof(J) * m));

    // Initialize zero pivot
    J max = std::numeric_limits<J>::max();
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(*zero_pivot, &max, sizeof(J), hipMemcpyHostToDevice, stream));

    // Wait for device transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Determine gcnArch and ASIC revision
    int gcnArch = handle->properties.gcnArch;
    int asicRev = handle->asic_rev;

// Run analysis
#define CSRSV_DIM 1024
    dim3 csrsv_blocks((handle->wavefront_size * m - 1) / CSRSV_DIM + 1);
    dim3 csrsv_threads(CSRSV_DIM);

    if(trans == rocsparse_operation_none)
    {
        if(gcnArch == 908 && asicRev < 2)
        {
            // LCOV_EXCL_START
            if(descr->fill_mode == rocsparse_fill_mode_upper)
            {
                hipLaunchKernelGGL((csrsv_analysis_upper_kernel<CSRSV_DIM, 64, true>),
                                   csrsv_blocks,
                                   csrsv_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   (I*)info->trm_diag_ind,
                                   done_array,
                                   d_max_nnz,
                                   (J*)*zero_pivot,
                                   descr->base,
                                   descr->diag_type);
            }
            else if(descr->fill_mode == rocsparse_fill_mode_lower)
            {
                hipLaunchKernelGGL((csrsv_analysis_lower_kernel<CSRSV_DIM, 64, true>),
                                   csrsv_blocks,
                                   csrsv_threads,
                                   0,
                                   stream,
                                   m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   (I*)info->trm_diag_ind,
                                   done_array,
                                   d_max_nnz,
                                   (J*)*zero_pivot,
                                   descr->base,
                                   descr->diag_type);
            }
            // LCOV_EXCL_STOP
        }
        else
        {
            if(handle->wavefront_size == 32)
            {
                // LCOV_EXCL_START
                if(descr->fill_mode == rocsparse_fill_mode_upper)
                {
                    hipLaunchKernelGGL((csrsv_analysis_upper_kernel<CSRSV_DIM, 32, false>),
                                       csrsv_blocks,
                                       csrsv_threads,
                                       0,
                                       stream,
                                       m,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       (I*)info->trm_diag_ind,
                                       done_array,
                                       d_max_nnz,
                                       (J*)*zero_pivot,
                                       descr->base,
                                       descr->diag_type);
                }
                else if(descr->fill_mode == rocsparse_fill_mode_lower)
                {
                    hipLaunchKernelGGL((csrsv_analysis_lower_kernel<CSRSV_DIM, 32, false>),
                                       csrsv_blocks,
                                       csrsv_threads,
                                       0,
                                       stream,
                                       m,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       (I*)info->trm_diag_ind,
                                       done_array,
                                       d_max_nnz,
                                       (J*)*zero_pivot,
                                       descr->base,
                                       descr->diag_type);
                }
                // LCOV_EXCL_STOP
            }
            else
            {
                assert(handle->wavefront_size == 64);
                if(descr->fill_mode == rocsparse_fill_mode_upper)
                {
                    hipLaunchKernelGGL((csrsv_analysis_upper_kernel<CSRSV_DIM, 64, false>),
                                       csrsv_blocks,
                                       csrsv_threads,
                                       0,
                                       stream,
                                       m,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       (I*)info->trm_diag_ind,
                                       done_array,
                                       d_max_nnz,
                                       (J*)*zero_pivot,
                                       descr->base,
                                       descr->diag_type);
                }
                else if(descr->fill_mode == rocsparse_fill_mode_lower)
                {
                    hipLaunchKernelGGL((csrsv_analysis_lower_kernel<CSRSV_DIM, 64, false>),
                                       csrsv_blocks,
                                       csrsv_threads,
                                       0,
                                       stream,
                                       m,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       (I*)info->trm_diag_ind,
                                       done_array,
                                       d_max_nnz,
                                       (J*)*zero_pivot,
                                       descr->base,
                                       descr->diag_type);
                }
            }
        }
    }
    else if(trans == rocsparse_operation_transpose
            || trans == rocsparse_operation_conjugate_transpose)
    {
        if(gcnArch == 908 && asicRev < 2)
        {
            // LCOV_EXCL_START
            if(descr->fill_mode == rocsparse_fill_mode_upper)
            {
                hipLaunchKernelGGL((csrsv_analysis_lower_kernel<CSRSV_DIM, 64, true>),
                                   csrsv_blocks,
                                   csrsv_threads,
                                   0,
                                   stream,
                                   m,
                                   (const I*)info->trmt_row_ptr,
                                   (const J*)info->trmt_col_ind,
                                   (I*)info->trm_diag_ind,
                                   done_array,
                                   d_max_nnz,
                                   (J*)*zero_pivot,
                                   descr->base,
                                   descr->diag_type);
            }
            else if(descr->fill_mode == rocsparse_fill_mode_lower)
            {
                hipLaunchKernelGGL((csrsv_analysis_upper_kernel<CSRSV_DIM, 64, true>),
                                   csrsv_blocks,
                                   csrsv_threads,
                                   0,
                                   stream,
                                   m,
                                   (const I*)info->trmt_row_ptr,
                                   (const J*)info->trmt_col_ind,
                                   (I*)info->trm_diag_ind,
                                   done_array,
                                   d_max_nnz,
                                   (J*)*zero_pivot,
                                   descr->base,
                                   descr->diag_type);
            }
            // LCOV_EXCL_STOP
        }
        else
        {
            if(handle->wavefront_size == 32)
            {
                // LCOV_EXCL_START
                if(descr->fill_mode == rocsparse_fill_mode_upper)
                {
                    hipLaunchKernelGGL((csrsv_analysis_lower_kernel<CSRSV_DIM, 32, false>),
                                       csrsv_blocks,
                                       csrsv_threads,
                                       0,
                                       stream,
                                       m,
                                       (const I*)info->trmt_row_ptr,
                                       (const J*)info->trmt_col_ind,
                                       (I*)info->trm_diag_ind,
                                       done_array,
                                       d_max_nnz,
                                       (J*)*zero_pivot,
                                       descr->base,
                                       descr->diag_type);
                }
                else if(descr->fill_mode == rocsparse_fill_mode_lower)
                {
                    hipLaunchKernelGGL((csrsv_analysis_upper_kernel<CSRSV_DIM, 32, false>),
                                       csrsv_blocks,
                                       csrsv_threads,
                                       0,
                                       stream,
                                       m,
                                       (const I*)info->trmt_row_ptr,
                                       (const J*)info->trmt_col_ind,
                                       (I*)info->trm_diag_ind,
                                       done_array,
                                       d_max_nnz,
                                       (J*)*zero_pivot,
                                       descr->base,
                                       descr->diag_type);
                }
                // LCOV_EXCL_STOP
            }
            else
            {
                assert(handle->wavefront_size == 64);
                if(descr->fill_mode == rocsparse_fill_mode_upper)
                {
                    hipLaunchKernelGGL((csrsv_analysis_lower_kernel<CSRSV_DIM, 64, false>),
                                       csrsv_blocks,
                                       csrsv_threads,
                                       0,
                                       stream,
                                       m,
                                       (const I*)info->trmt_row_ptr,
                                       (const J*)info->trmt_col_ind,
                                       (I*)info->trm_diag_ind,
                                       done_array,
                                       d_max_nnz,
                                       (J*)*zero_pivot,
                                       descr->base,
                                       descr->diag_type);
                }
                else if(descr->fill_mode == rocsparse_fill_mode_lower)
                {
                    hipLaunchKernelGGL((csrsv_analysis_upper_kernel<CSRSV_DIM, 64, false>),
                                       csrsv_blocks,
                                       csrsv_threads,
                                       0,
                                       stream,
                                       m,
                                       (const I*)info->trmt_row_ptr,
                                       (const J*)info->trmt_col_ind,
                                       (I*)info->trm_diag_ind,
                                       done_array,
                                       d_max_nnz,
                                       (J*)*zero_pivot,
                                       descr->base,
                                       descr->diag_type);
                }
            }
        }
    }
    else
    {
        // LCOV_EXCL_START
        return rocsparse_status_internal_error;
        // LCOV_EXCL_STOP
    }
#undef CSRSV_DIM

    // Post processing
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&info->max_nnz, d_max_nnz, sizeof(I), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_identity_permutation_template(handle, m, workspace));

    size_t rocprim_size;

    unsigned int startbit = 0;
    unsigned int endbit   = rocsparse_clz(m);

    rocprim::double_buffer<int> keys(done_array, workspace2);
    rocprim::double_buffer<J>   vals(workspace, (J*)info->row_map);

    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, rocprim_size, keys, vals, m, startbit, endbit, stream));

    RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
        rocprim_buffer, rocprim_size, keys, vals, m, startbit, endbit, stream));

    if(vals.current() != info->row_map)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            info->row_map, vals.current(), sizeof(J) * m, hipMemcpyDeviceToDevice, stream));
    }

    // Store some pointers to verify correct execution
    info->m           = m;
    info->nnz         = nnz;
    info->descr       = descr;
    info->trm_row_ptr = (trans == rocsparse_operation_none) ? csr_row_ptr : info->trmt_row_ptr;
    info->trm_col_ind = (trans == rocsparse_operation_none) ? csr_col_ind : info->trmt_col_ind;

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrsv_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   J                         m,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
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
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrsv_analysis"),
              trans,
              m,
              nnz,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              solve,
              analysis,
              (const void*&)temp_buffer);

    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(analysis))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(solve))
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

    if(nnz != 0 && (csr_col_ind == nullptr && csr_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Switch between lower and upper triangular analysis
    if(descr->fill_mode == rocsparse_fill_mode_upper)
    {
        // Differentiate the analysis policies
        if(analysis == rocsparse_analysis_policy_reuse)
        {
            // We try to re-use already analyzed lower part, if available.
            // It is the user's responsibility that this data is still valid,
            // since he passed the 'reuse' flag.

            // If csrsv meta data is already available, do nothing
            if(trans == rocsparse_operation_none && info->csrsv_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_transpose && info->csrsvt_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_conjugate_transpose
                    && info->csrsvt_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other lower analysis meta data

            if(trans == rocsparse_operation_none && info->csrsm_upper_info != nullptr)
            {
                // csrsm meta data
                info->csrsv_upper_info = info->csrsm_upper_info;
                return rocsparse_status_success;
            }

            if(trans == rocsparse_operation_transpose && info->csrsmt_upper_info != nullptr)
            {
                // csrsv meta data
                info->csrsvt_upper_info = info->csrsmt_upper_info;
                return rocsparse_status_success;
            }

            if(trans == rocsparse_operation_conjugate_transpose
               && info->csrsmt_upper_info != nullptr)
            {
                // csrsv meta data
                info->csrsvt_upper_info = info->csrsmt_upper_info;
                return rocsparse_status_success;
            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used.

        // Clear csrsv info

        // Clear csrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info((trans == rocsparse_operation_none)
                                                                 ? info->csrsv_upper_info
                                                                 : info->csrsvt_upper_info));

        // Create csrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info((trans == rocsparse_operation_none)
                                                                ? &info->csrsv_upper_info
                                                                : &info->csrsvt_upper_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(
            handle,
            trans,
            m,
            nnz,
            descr,
            csr_val,
            csr_row_ptr,
            csr_col_ind,
            (trans == rocsparse_operation_none) ? info->csrsv_upper_info : info->csrsvt_upper_info,
            (J**)&info->zero_pivot,
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

            // If csrsv meta data is already available, do nothing
            if(trans == rocsparse_operation_none && info->csrsv_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_transpose && info->csrsvt_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_conjugate_transpose
                    && info->csrsvt_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other lower analysis meta data

            if(trans == rocsparse_operation_none && info->csrilu0_info != nullptr)
            {
                // csrilu0 meta data
                info->csrsv_lower_info = info->csrilu0_info;
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_none && info->csric0_info != nullptr)
            {
                // csric0 meta data
                info->csrsv_lower_info = info->csric0_info;
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_none && info->csrsm_lower_info != nullptr)
            {
                // csrsm meta data
                info->csrsv_lower_info = info->csrsm_lower_info;
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_transpose && info->csrsmt_lower_info != nullptr)
            {
                // csrsm meta data
                info->csrsvt_lower_info = info->csrsmt_lower_info;
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_conjugate_transpose
                    && info->csrsmt_lower_info != nullptr)
            {
                // csrsm meta data
                info->csrsvt_lower_info = info->csrsmt_lower_info;
                return rocsparse_status_success;
            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used.

        // Clear csrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info((trans == rocsparse_operation_none)
                                                                 ? info->csrsv_lower_info
                                                                 : info->csrsvt_lower_info));

        // Create csrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info((trans == rocsparse_operation_none)
                                                                ? &info->csrsv_lower_info
                                                                : &info->csrsvt_lower_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(
            handle,
            trans,
            m,
            nnz,
            descr,
            csr_val,
            csr_row_ptr,
            csr_col_ind,
            (trans == rocsparse_operation_none) ? info->csrsv_lower_info : info->csrsvt_lower_info,
            (J**)&info->zero_pivot,
            temp_buffer));
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                    \
    template rocsparse_status rocsparse_trm_analysis(rocsparse_handle          handle,      \
                                                     rocsparse_operation       trans,       \
                                                     JTYPE                     m,           \
                                                     ITYPE                     nnz,         \
                                                     const rocsparse_mat_descr descr,       \
                                                     const TTYPE*              csr_val,     \
                                                     const ITYPE*              csr_row_ptr, \
                                                     const JTYPE*              csr_col_ind, \
                                                     rocsparse_trm_info        info,        \
                                                     JTYPE**                   zero_pivot,  \
                                                     void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                         \
    template rocsparse_status rocsparse_csrsv_analysis_template( \
        rocsparse_handle          handle,                        \
        rocsparse_operation       trans,                         \
        JTYPE                     m,                             \
        ITYPE                     nnz,                           \
        const rocsparse_mat_descr descr,                         \
        const TTYPE*              csr_val,                       \
        const ITYPE*              csr_row_ptr,                   \
        const JTYPE*              csr_col_ind,                   \
        rocsparse_mat_info        info,                          \
        rocsparse_analysis_policy analysis,                      \
        rocsparse_solve_policy    solve,                         \
        void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             m,           \
                                     rocsparse_int             nnz,         \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     rocsparse_mat_info        info,        \
                                     rocsparse_analysis_policy analysis,    \
                                     rocsparse_solve_policy    solve,       \
                                     void*                     temp_buffer) \
    {                                                                       \
        return rocsparse_csrsv_analysis_template(handle,                    \
                                                 trans,                     \
                                                 m,                         \
                                                 nnz,                       \
                                                 descr,                     \
                                                 csr_val,                   \
                                                 csr_row_ptr,               \
                                                 csr_col_ind,               \
                                                 info,                      \
                                                 analysis,                  \
                                                 solve,                     \
                                                 temp_buffer);              \
    }

C_IMPL(rocsparse_scsrsv_analysis, float);
C_IMPL(rocsparse_dcsrsv_analysis, double);
C_IMPL(rocsparse_ccsrsv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsv_analysis, rocsparse_double_complex);

#undef C_IMPL
