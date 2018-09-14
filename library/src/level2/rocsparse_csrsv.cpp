/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_csrsv.hpp"

#include <limits>

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle handle,
                                                        rocsparse_operation trans,
                                                        rocsparse_int m,
                                                        rocsparse_int nnz,
                                                        const rocsparse_mat_descr descr,
                                                        const rocsparse_int* csr_row_ptr,
                                                        const rocsparse_int* csr_col_ind,
                                                        rocsparse_mat_info info,
                                                        size_t* buffer_size)
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
              "rocsparse_csrsv_buffer_size",
              trans,
              m,
              nnz,
              (const void*&)descr,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              (const void*&)buffer_size);

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
    if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || nnz == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    // rocsparse_int max depth
    *buffer_size = 256;

    // unsigned long long total_spin
    *buffer_size += 256;

    // rocsparse_int max_nnz
    *buffer_size += 256;

    // rocsparse_int done_array[m]
    *buffer_size += sizeof(rocsparse_int) * ((m - 1) / 256 + 1) * 256;

    // rocsparse_int rows_per_level[m]
    *buffer_size += sizeof(rocsparse_int) * ((m - 1) / 256 + 1) * 256;

    size_t hipcub_size = 0;
    rocsparse_int* ptr = nullptr;
    RETURN_IF_HIP_ERROR(hipcub::DeviceScan::InclusiveSum(nullptr, hipcub_size, ptr, ptr, m));

    // hipcub buffer
    *buffer_size += hipcub_size;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle handle,
                                                     rocsparse_operation trans,
                                                     rocsparse_int m,
                                                     rocsparse_int nnz,
                                                     const rocsparse_mat_descr descr,
                                                     const rocsparse_int* csr_row_ptr,
                                                     const rocsparse_int* csr_col_ind,
                                                     rocsparse_mat_info info,
                                                     rocsparse_analysis_policy analysis,
                                                     rocsparse_solve_policy solve,
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
              "rocsparse_csrsv_analysis",
              trans,
              m,
              nnz,
              (const void*&)descr,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              solve,
              analysis,
              (const void*&)temp_buffer);

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
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
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
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Switch between lower and upper triangular analysis
    if(descr->fill_mode == rocsparse_fill_mode_upper)
    {
        // This is currently the only case where we need upper triangular analysis,
        // therefore we ignore the analysis policy

        // Clear csrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrsv_upper_info));

        // Create csrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrtr_info(&info->csrsv_upper_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrtr_analysis(handle,
                                                           trans,
                                                           m,
                                                           nnz,
                                                           descr,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           info->csrsv_upper_info,
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
            if(info->csrsv_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other lower analysis meta data
            rocsparse_csrtr_info reuse = nullptr;

            // csrilu0 meta data
            if(info->csrilu0_info != nullptr)
            {
                reuse = info->csrilu0_info;
            }

            // TODO add more crossover data here

            // If data has been found, use it
            if(reuse != nullptr)
            {
                info->csrsv_lower_info = reuse;

                return rocsparse_status_success;
            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used.

        // Clear csrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrsv_lower_info));

        // Create csrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrtr_info(&info->csrsv_lower_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrtr_analysis(handle,
                                                           trans,
                                                           m,
                                                           nnz,
                                                           descr,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           info->csrsv_lower_info,
                                                           temp_buffer));
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_csrsv_clear(rocsparse_handle handle,
                                                  const rocsparse_mat_descr descr,
                                                  rocsparse_mat_info info)
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
    log_trace(handle, "rocsparse_csrsv_clear", (const void*&)descr, (const void*&)info);

    // Determine which info meta data should be deleted
    if(descr->fill_mode == rocsparse_fill_mode_lower)
    {
        // If meta data is shared, do not delete anything
        if(info->csrilu0_info == info->csrsv_lower_info)
        {
            info->csrsv_lower_info = nullptr;

            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrsv_lower_info));
        info->csrsv_lower_info = nullptr;
    }
    else if(descr->fill_mode == rocsparse_fill_mode_upper)
    {
        // Upper info has no shares (yet)
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrsv_upper_info));
        info->csrsv_upper_info = nullptr;
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsrsv_solve(rocsparse_handle handle,
                                                   rocsparse_operation trans,
                                                   rocsparse_int m,
                                                   rocsparse_int nnz,
                                                   const float* alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const float* csr_val,
                                                   const rocsparse_int* csr_row_ind,
                                                   const rocsparse_int* csr_col_ind,
                                                   rocsparse_mat_info info,
                                                   const float* x,
                                                   float* y,
                                                   rocsparse_solve_policy policy,
                                                   void* temp_buffer)
{
    return rocsparse_csrsv_solve_template<float>(handle,
                                                 trans,
                                                 m,
                                                 nnz,
                                                 alpha,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ind,
                                                 csr_col_ind,
                                                 info,
                                                 x,
                                                 y,
                                                 policy,
                                                 temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsrsv_solve(rocsparse_handle handle,
                                                   rocsparse_operation trans,
                                                   rocsparse_int m,
                                                   rocsparse_int nnz,
                                                   const double* alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const double* csr_val,
                                                   const rocsparse_int* csr_row_ind,
                                                   const rocsparse_int* csr_col_ind,
                                                   rocsparse_mat_info info,
                                                   const double* x,
                                                   double* y,
                                                   rocsparse_solve_policy policy,
                                                   void* temp_buffer)
{
    return rocsparse_csrsv_solve_template<double>(handle,
                                                  trans,
                                                  m,
                                                  nnz,
                                                  alpha,
                                                  descr,
                                                  csr_val,
                                                  csr_row_ind,
                                                  csr_col_ind,
                                                  info,
                                                  x,
                                                  y,
                                                  policy,
                                                  temp_buffer);
}

extern "C" rocsparse_status rocsparse_csrsv_zero_pivot(rocsparse_handle handle,
                                                       const rocsparse_mat_descr descr,
                                                       rocsparse_mat_info info,
                                                       rocsparse_int* position)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle, "rocsparse_csrsv_zero_pivot", (const void*&)info, (const void*&)position);

    // Check pointer arguments
    if(position == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Determine the info meta data place
    rocsparse_csrtr_info csrsv = nullptr;

    // For hipSPARSE compatibility mode, we allow descr == nullptr
    // In this case, only lower OR upper is populated and we can use the right
    // info meta data
    if(descr == nullptr)
    {
        if(info->csrsv_lower_info != nullptr)
        {
            csrsv = info->csrsv_lower_info;
        }
        else
        {
            csrsv = info->csrsv_upper_info;
        }
    }
    else
    {
        // Switch between upper and lower triangular
        if(descr->fill_mode == rocsparse_fill_mode_lower)
        {
            csrsv = info->csrsv_lower_info;
        }
        else
        {
            csrsv = info->csrsv_upper_info;
        }
    }

    // If m == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(csrsv == nullptr)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 255, sizeof(rocsparse_int), stream));
        }
        else
        {
            *position = -1;
        }

        return rocsparse_status_success;
    }

    // Differentiate between pointer modes
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // rocsparse_pointer_mode_device
        rocsparse_int pivot;

        RETURN_IF_HIP_ERROR(
            hipMemcpy(&pivot, csrsv->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        if(pivot == std::numeric_limits<rocsparse_int>::max())
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 255, sizeof(rocsparse_int), stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(
                position, csrsv->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToDevice));

            return rocsparse_status_zero_pivot;
        }
    }
    else
    {
        // rocsparse_pointer_mode_host
        RETURN_IF_HIP_ERROR(
            hipMemcpy(position, csrsv->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // If no zero pivot is found, set -1
        if(*position == std::numeric_limits<rocsparse_int>::max())
        {
            *position = -1;
        }
        else
        {
            return rocsparse_status_zero_pivot;
        }
    }

    return rocsparse_status_success;
}
