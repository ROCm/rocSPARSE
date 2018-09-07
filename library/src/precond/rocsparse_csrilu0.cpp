/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "definitions.h"
#include "rocsparse.h"
#include "rocsparse_csrilu0.hpp"

#include "../level2/rocsparse_csrsv.hpp"

#include <hipcub/hipcub.hpp>

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle handle,
                                                          rocsparse_int m,
                                                          rocsparse_int nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_mat_info info,
                                                          size_t* buffer_size)
{
    return rocsparse_csrsv_buffer_size(handle,
                                       rocsparse_operation_none,
                                       m,
                                       nnz,
                                       descr,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       info,
                                       buffer_size);
}

extern "C" rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle handle,
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
              "rocsparse_csrilu0_analysis",
              m,
              nnz,
              (const void*&)descr,
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

extern "C" rocsparse_status rocsparse_csrilu0_clear(rocsparse_handle handle,
                                                    rocsparse_mat_info info)
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
    log_trace(handle,
              "rocsparse_csrilu0_clear",
              (const void*&)info);

    // If meta data is shared, do not delete anything
    if(info->csrilu0_info == info->csrsv_lower_info)
    {
        info->csrilu0_info = nullptr;

        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrilu0_info));
    info->csrilu0_info = nullptr;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsrilu0(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int nnz,
                                               const rocsparse_mat_descr descr,
                                               float* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_mat_info info,
                                               rocsparse_solve_policy policy,
                                               void* temp_buffer)
{
    return rocsparse_csrilu0_template<float>(handle,
                                             m,
                                             nnz,
                                             descr,
                                             csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             info,
                                             policy,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsrilu0(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int nnz,
                                               const rocsparse_mat_descr descr,
                                               double* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_mat_info info,
                                               rocsparse_solve_policy policy,
                                               void* temp_buffer)
{
    return rocsparse_csrilu0_template<double>(handle,
                                              m,
                                              nnz,
                                              descr,
                                              csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              policy,
                                              temp_buffer);
}

extern "C" rocsparse_status rocsparse_csrilu0_zero_pivot(rocsparse_handle handle,
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
    log_trace(handle,
              "rocsparse_csrilu0_zero_pivot",
              (const void*&)info,
              (const void*&)position);

    // Check pointer arguments
    if(position == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
//    hipStream_t stream = handle->stream;

    // Synchronize stream TODO should not be required...
//    hipStreamSynchronize(stream);

    // If m == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->csrilu0_info == nullptr)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemset(position, 255, sizeof(rocsparse_int)));
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

        RETURN_IF_HIP_ERROR(hipMemcpy(&pivot, info->csrilu0_info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        if(pivot == std::numeric_limits<rocsparse_int>::max())
        {
            RETURN_IF_HIP_ERROR(hipMemset(position, 255, sizeof(rocsparse_int)));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(position, info->csrilu0_info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToDevice));

            return rocsparse_status_zero_pivot;
        }
    }
    else
    {
        // rocsparse_pointer_mode_host
        RETURN_IF_HIP_ERROR(hipMemcpy(position, info->csrilu0_info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // If no zero pivot is found, set -1
        *position = (*position == std::numeric_limits<rocsparse_int>::max()) ? -1 : *position;

        if(*position != -1)
        {
            return rocsparse_status_zero_pivot;
        }
    }

    return rocsparse_status_success;
}
