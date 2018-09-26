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

#include "rocsparse.h"
#include "rocsparse_csrsv.hpp"

#include <limits>

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrsv_buffer_size(rocsparse_handle handle,
                                                         rocsparse_operation trans,
                                                         rocsparse_int m,
                                                         rocsparse_int nnz,
                                                         const rocsparse_mat_descr descr,
                                                         const float* csr_val,
                                                         const rocsparse_int* csr_row_ptr,
                                                         const rocsparse_int* csr_col_ind,
                                                         rocsparse_mat_info info,
                                                         size_t* buffer_size)
{
    return rocsparse_csrsv_buffer_size_template<float>(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

extern "C" rocsparse_status rocsparse_dcsrsv_buffer_size(rocsparse_handle handle,
                                                         rocsparse_operation trans,
                                                         rocsparse_int m,
                                                         rocsparse_int nnz,
                                                         const rocsparse_mat_descr descr,
                                                         const double* csr_val,
                                                         const rocsparse_int* csr_row_ptr,
                                                         const rocsparse_int* csr_col_ind,
                                                         rocsparse_mat_info info,
                                                         size_t* buffer_size)
{
    return rocsparse_csrsv_buffer_size_template<double>(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

extern "C" rocsparse_status rocsparse_scsrsv_analysis(rocsparse_handle handle,
                                                      rocsparse_operation trans,
                                                      rocsparse_int m,
                                                      rocsparse_int nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const float* csr_val,
                                                      const rocsparse_int* csr_row_ptr,
                                                      const rocsparse_int* csr_col_ind,
                                                      rocsparse_mat_info info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy solve,
                                                      void* temp_buffer)
{
    return rocsparse_csrsv_analysis_template<float>(handle,
                                                    trans,
                                                    m,
                                                    nnz,
                                                    descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    info,
                                                    analysis,
                                                    solve,
                                                    temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsrsv_analysis(rocsparse_handle handle,
                                                      rocsparse_operation trans,
                                                      rocsparse_int m,
                                                      rocsparse_int nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const double* csr_val,
                                                      const rocsparse_int* csr_row_ptr,
                                                      const rocsparse_int* csr_col_ind,
                                                      rocsparse_mat_info info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy solve,
                                                      void* temp_buffer)
{
    return rocsparse_csrsv_analysis_template<double>(handle,
                                                     trans,
                                                     m,
                                                     nnz,
                                                     descr,
                                                     csr_val,
                                                     csr_row_ptr,
                                                     csr_col_ind,
                                                     info,
                                                     analysis,
                                                     solve,
                                                     temp_buffer);
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
