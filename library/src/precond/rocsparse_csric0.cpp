/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/precond/rocsparse_csric0.h"
#include "rocsparse_csric0.hpp"

#include "internal/level2/rocsparse_csrsv.h"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsric0_buffer_size(rocsparse_handle          handle,
                                                          rocsparse_int             m,
                                                          rocsparse_int             nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const float*              csr_val,
                                                          const rocsparse_int*      csr_row_ptr,
                                                          const rocsparse_int*      csr_col_ind,
                                                          rocsparse_mat_info        info,
                                                          size_t*                   buffer_size)
try
{
    return rocsparse_scsrsv_buffer_size(handle,
                                        rocsparse_operation_none,
                                        m,
                                        nnz,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        buffer_size);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_dcsric0_buffer_size(rocsparse_handle          handle,
                                                          rocsparse_int             m,
                                                          rocsparse_int             nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const double*             csr_val,
                                                          const rocsparse_int*      csr_row_ptr,
                                                          const rocsparse_int*      csr_col_ind,
                                                          rocsparse_mat_info        info,
                                                          size_t*                   buffer_size)
try
{
    return rocsparse_dcsrsv_buffer_size(handle,
                                        rocsparse_operation_none,
                                        m,
                                        nnz,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        buffer_size);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_ccsric0_buffer_size(rocsparse_handle               handle,
                                                          rocsparse_int                  m,
                                                          rocsparse_int                  nnz,
                                                          const rocsparse_mat_descr      descr,
                                                          const rocsparse_float_complex* csr_val,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_mat_info   info,
                                                          size_t*              buffer_size)
try
{
    return rocsparse_ccsrsv_buffer_size(handle,
                                        rocsparse_operation_none,
                                        m,
                                        nnz,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        buffer_size);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_zcsric0_buffer_size(rocsparse_handle                handle,
                                                          rocsparse_int                   m,
                                                          rocsparse_int                   nnz,
                                                          const rocsparse_mat_descr       descr,
                                                          const rocsparse_double_complex* csr_val,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_mat_info   info,
                                                          size_t*              buffer_size)
try
{
    return rocsparse_zcsrsv_buffer_size(handle,
                                        rocsparse_operation_none,
                                        m,
                                        nnz,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        buffer_size);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_scsric0_analysis(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const float*              csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       const rocsparse_int*      csr_col_ind,
                                                       rocsparse_mat_info        info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy    solve,
                                                       void*                     temp_buffer)
try
{
    return rocsparse_csric0_analysis_template(handle,
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
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_dcsric0_analysis(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const double*             csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       const rocsparse_int*      csr_col_ind,
                                                       rocsparse_mat_info        info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy    solve,
                                                       void*                     temp_buffer)
try
{
    return rocsparse_csric0_analysis_template(handle,
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
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_ccsric0_analysis(rocsparse_handle               handle,
                                                       rocsparse_int                  m,
                                                       rocsparse_int                  nnz,
                                                       const rocsparse_mat_descr      descr,
                                                       const rocsparse_float_complex* csr_val,
                                                       const rocsparse_int*           csr_row_ptr,
                                                       const rocsparse_int*           csr_col_ind,
                                                       rocsparse_mat_info             info,
                                                       rocsparse_analysis_policy      analysis,
                                                       rocsparse_solve_policy         solve,
                                                       void*                          temp_buffer)
try
{
    return rocsparse_csric0_analysis_template(handle,
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
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_zcsric0_analysis(rocsparse_handle                handle,
                                                       rocsparse_int                   m,
                                                       rocsparse_int                   nnz,
                                                       const rocsparse_mat_descr       descr,
                                                       const rocsparse_double_complex* csr_val,
                                                       const rocsparse_int*            csr_row_ptr,
                                                       const rocsparse_int*            csr_col_ind,
                                                       rocsparse_mat_info              info,
                                                       rocsparse_analysis_policy       analysis,
                                                       rocsparse_solve_policy          solve,
                                                       void*                           temp_buffer)
try
{
    return rocsparse_csric0_analysis_template(handle,
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
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_csric0_clear(rocsparse_handle handle, rocsparse_mat_info info)
try
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
    log_trace(handle, "rocsparse_csric0_clear", (const void*&)info);

    // If meta data is not shared, delete it
    if(!rocsparse_check_trm_shared(info, info->csric0_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csric0_info));
    }

    info->csric0_info = nullptr;

    return rocsparse_status_success;
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_scsric0(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              float*                    csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
try
{
    return rocsparse_csric0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_dcsric0(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              double*                   csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
try
{
    return rocsparse_csric0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_ccsric0(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              rocsparse_float_complex*  csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
try
{
    return rocsparse_csric0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_zcsric0(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              rocsparse_double_complex* csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
try
{
    return rocsparse_csric0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_csric0_zero_pivot(rocsparse_handle   handle,
                                                        rocsparse_mat_info info,
                                                        rocsparse_int*     position)
try
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
    log_trace(handle, "rocsparse_csric0_zero_pivot", (const void*&)info, (const void*&)position);

    // Check pointer arguments
    if(position == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // If m == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->csric0_info == nullptr)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 0xFF, sizeof(rocsparse_int), stream));
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

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &pivot, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        if(pivot == std::numeric_limits<rocsparse_int>::max())
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 0xFF, sizeof(rocsparse_int), stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(position,
                                               info->zero_pivot,
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToDevice,
                                               stream));

            return rocsparse_status_zero_pivot;
        }
    }
    else
    {
        // rocsparse_pointer_mode_host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            position, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

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
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_csric0_negative_pivot(rocsparse_handle   handle,
                                                            rocsparse_mat_info info,
                                                            rocsparse_int*     position)
try
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
    log_trace(
        handle, "rocsparse_csric0_negative_pivot", (const void*&)info, (const void*&)position);

    // Check pointer arguments
    if(position == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // If m == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->csric0_info == nullptr)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 0xFF, sizeof(rocsparse_int), stream));
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
        rocsparse_int pivot = std::numeric_limits<rocsparse_int>::max();

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &pivot, info->negative_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        if(pivot == std::numeric_limits<rocsparse_int>::max())
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 0xFF, sizeof(rocsparse_int), stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(position,
                                               info->negative_pivot,
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToDevice,
                                               stream));

            // ----------------------------------
            // TODO: no official enum entry for
            //       rocsparse_status_negative_pivot
            // ----------------------------------
            return rocsparse_status_negative_pivot;
        }
    }
    else
    {
        // rocsparse_pointer_mode_host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            position, info->negative_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // If no negative pivot is found, set -1
        if(*position == std::numeric_limits<rocsparse_int>::max())
        {
            *position = -1;
        }
        else
        {
            // ----------------------------------
            // TODO: no official enum entry for
            //       rocsparse_status_negative_pivot
            // ----------------------------------
            return rocsparse_status_negative_pivot;
        }
    }

    return rocsparse_status_success;
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status
    rocsparse_csric0_set_tol(rocsparse_handle handle, rocsparse_mat_info info, double tol)
try
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle, "rocsparse_csric0_set_tol", (const void*&)info, tol);

    // Stream
    hipStream_t stream = handle->stream;

    {
        double h_singularity_tol[1];
        h_singularity_tol[0] = tol;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&(info->singularity_tol),
                                           h_singularity_tol,
                                           sizeof(double),
                                           hipMemcpyHostToDevice,
                                           stream));
    }

    return rocsparse_status_success;
}
catch(...)
{
    return exception_to_rocsparse_status();
}
