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

#include "rocsparse_csrilu0.hpp"
#include "definitions.h"
#include "rocsparse.h"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrilu0_buffer_size(rocsparse_handle          handle,
                                                           rocsparse_int             m,
                                                           rocsparse_int             nnz,
                                                           const rocsparse_mat_descr descr,
                                                           const float*              csr_val,
                                                           const rocsparse_int*      csr_row_ptr,
                                                           const rocsparse_int*      csr_col_ind,
                                                           rocsparse_mat_info        info,
                                                           size_t*                   buffer_size)
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

extern "C" rocsparse_status rocsparse_dcsrilu0_buffer_size(rocsparse_handle          handle,
                                                           rocsparse_int             m,
                                                           rocsparse_int             nnz,
                                                           const rocsparse_mat_descr descr,
                                                           const double*             csr_val,
                                                           const rocsparse_int*      csr_row_ptr,
                                                           const rocsparse_int*      csr_col_ind,
                                                           rocsparse_mat_info        info,
                                                           size_t*                   buffer_size)
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

extern "C" rocsparse_status rocsparse_ccsrilu0_buffer_size(rocsparse_handle               handle,
                                                           rocsparse_int                  m,
                                                           rocsparse_int                  nnz,
                                                           const rocsparse_mat_descr      descr,
                                                           const rocsparse_float_complex* csr_val,
                                                           const rocsparse_int* csr_row_ptr,
                                                           const rocsparse_int* csr_col_ind,
                                                           rocsparse_mat_info   info,
                                                           size_t*              buffer_size)
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

extern "C" rocsparse_status rocsparse_zcsrilu0_buffer_size(rocsparse_handle                handle,
                                                           rocsparse_int                   m,
                                                           rocsparse_int                   nnz,
                                                           const rocsparse_mat_descr       descr,
                                                           const rocsparse_double_complex* csr_val,
                                                           const rocsparse_int* csr_row_ptr,
                                                           const rocsparse_int* csr_col_ind,
                                                           rocsparse_mat_info   info,
                                                           size_t*              buffer_size)
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

extern "C" rocsparse_status rocsparse_scsrilu0_analysis(rocsparse_handle          handle,
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
{
    return rocsparse_csrilu0_analysis_template(handle,
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

extern "C" rocsparse_status rocsparse_dcsrilu0_analysis(rocsparse_handle          handle,
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
{
    return rocsparse_csrilu0_analysis_template(handle,
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

extern "C" rocsparse_status rocsparse_ccsrilu0_analysis(rocsparse_handle               handle,
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
{
    return rocsparse_csrilu0_analysis_template(handle,
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

extern "C" rocsparse_status rocsparse_zcsrilu0_analysis(rocsparse_handle                handle,
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
{
    return rocsparse_csrilu0_analysis_template(handle,
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

extern "C" rocsparse_status rocsparse_csrilu0_clear(rocsparse_handle   handle,
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
    log_trace(handle, "rocsparse_csrilu0_clear", (const void*&)info);

    // If meta data is shared, do not delete anything
    if(info->csrilu0_info == info->csrsv_lower_info || info->csrilu0_info == info->csrsv_upper_info)
    {
        info->csrilu0_info = nullptr;

        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrilu0_info));
    info->csrilu0_info = nullptr;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               float*                    csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
{
    return rocsparse_csrilu0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               double*                   csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
{
    return rocsparse_csrilu0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

extern "C" rocsparse_status rocsparse_ccsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_float_complex*  csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
{
    return rocsparse_csrilu0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

extern "C" rocsparse_status rocsparse_zcsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_double_complex* csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
{
    return rocsparse_csrilu0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

extern "C" rocsparse_status rocsparse_csrilu0_zero_pivot(rocsparse_handle   handle,
                                                         rocsparse_mat_info info,
                                                         rocsparse_int*     position)
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
    log_trace(handle, "rocsparse_csrilu0_zero_pivot", (const void*&)info, (const void*&)position);

    // Check pointer arguments
    if(position == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // If m == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->csrilu0_info == nullptr)
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

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&pivot,
                                           info->csrilu0_info->zero_pivot,
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        if(pivot == std::numeric_limits<rocsparse_int>::max())
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 255, sizeof(rocsparse_int), stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(position,
                                               info->csrilu0_info->zero_pivot,
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToDevice,
                                               stream));

            return rocsparse_status_zero_pivot;
        }
    }
    else
    {
        // rocsparse_pointer_mode_host
        RETURN_IF_HIP_ERROR(hipMemcpy(position,
                                      info->csrilu0_info->zero_pivot,
                                      sizeof(rocsparse_int),
                                      hipMemcpyDeviceToHost));

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
