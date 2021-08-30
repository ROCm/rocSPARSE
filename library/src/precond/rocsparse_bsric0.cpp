/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_bsric0.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sbsric0_buffer_size(rocsparse_handle          handle,
                                                          rocsparse_direction       dir,
                                                          rocsparse_int             mb,
                                                          rocsparse_int             nnzb,
                                                          const rocsparse_mat_descr descr,
                                                          const float*              bsr_val,
                                                          const rocsparse_int*      bsr_row_ptr,
                                                          const rocsparse_int*      bsr_col_ind,
                                                          rocsparse_int             block_dim,
                                                          rocsparse_mat_info        info,
                                                          size_t*                   buffer_size)
{
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    if(block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    return rocsparse_scsrsv_buffer_size(handle,
                                        rocsparse_operation_none,
                                        mb,
                                        nnzb,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        info,
                                        buffer_size);
}

extern "C" rocsparse_status rocsparse_dbsric0_buffer_size(rocsparse_handle          handle,
                                                          rocsparse_direction       dir,
                                                          rocsparse_int             mb,
                                                          rocsparse_int             nnzb,
                                                          const rocsparse_mat_descr descr,
                                                          const double*             bsr_val,
                                                          const rocsparse_int*      bsr_row_ptr,
                                                          const rocsparse_int*      bsr_col_ind,
                                                          rocsparse_int             block_dim,
                                                          rocsparse_mat_info        info,
                                                          size_t*                   buffer_size)
{
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    if(block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    return rocsparse_dcsrsv_buffer_size(handle,
                                        rocsparse_operation_none,
                                        mb,
                                        nnzb,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        info,
                                        buffer_size);
}

extern "C" rocsparse_status rocsparse_cbsric0_buffer_size(rocsparse_handle               handle,
                                                          rocsparse_direction            dir,
                                                          rocsparse_int                  mb,
                                                          rocsparse_int                  nnzb,
                                                          const rocsparse_mat_descr      descr,
                                                          const rocsparse_float_complex* bsr_val,
                                                          const rocsparse_int* bsr_row_ptr,
                                                          const rocsparse_int* bsr_col_ind,
                                                          rocsparse_int        block_dim,
                                                          rocsparse_mat_info   info,
                                                          size_t*              buffer_size)
{
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    if(block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    return rocsparse_ccsrsv_buffer_size(handle,
                                        rocsparse_operation_none,
                                        mb,
                                        nnzb,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        info,
                                        buffer_size);
}

extern "C" rocsparse_status rocsparse_zbsric0_buffer_size(rocsparse_handle                handle,
                                                          rocsparse_direction             dir,
                                                          rocsparse_int                   mb,
                                                          rocsparse_int                   nnzb,
                                                          const rocsparse_mat_descr       descr,
                                                          const rocsparse_double_complex* bsr_val,
                                                          const rocsparse_int* bsr_row_ptr,
                                                          const rocsparse_int* bsr_col_ind,
                                                          rocsparse_int        block_dim,
                                                          rocsparse_mat_info   info,
                                                          size_t*              buffer_size)
{
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    if(block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    return rocsparse_zcsrsv_buffer_size(handle,
                                        rocsparse_operation_none,
                                        mb,
                                        nnzb,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        info,
                                        buffer_size);
}

extern "C" rocsparse_status rocsparse_sbsric0_analysis(rocsparse_handle          handle,
                                                       rocsparse_direction       dir,
                                                       rocsparse_int             mb,
                                                       rocsparse_int             nnzb,
                                                       const rocsparse_mat_descr descr,
                                                       const float*              bsr_val,
                                                       const rocsparse_int*      bsr_row_ptr,
                                                       const rocsparse_int*      bsr_col_ind,
                                                       rocsparse_int             block_dim,
                                                       rocsparse_mat_info        info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy    solve,
                                                       void*                     temp_buffer)
{
    return rocsparse_bsric0_analysis_template(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              block_dim,
                                              info,
                                              analysis,
                                              solve,
                                              temp_buffer);
}

extern "C" rocsparse_status rocsparse_dbsric0_analysis(rocsparse_handle          handle,
                                                       rocsparse_direction       dir,
                                                       rocsparse_int             mb,
                                                       rocsparse_int             nnzb,
                                                       const rocsparse_mat_descr descr,
                                                       const double*             bsr_val,
                                                       const rocsparse_int*      bsr_row_ptr,
                                                       const rocsparse_int*      bsr_col_ind,
                                                       rocsparse_int             block_dim,
                                                       rocsparse_mat_info        info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy    solve,
                                                       void*                     temp_buffer)
{
    return rocsparse_bsric0_analysis_template(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              block_dim,
                                              info,
                                              analysis,
                                              solve,
                                              temp_buffer);
}

extern "C" rocsparse_status rocsparse_cbsric0_analysis(rocsparse_handle               handle,
                                                       rocsparse_direction            dir,
                                                       rocsparse_int                  mb,
                                                       rocsparse_int                  nnzb,
                                                       const rocsparse_mat_descr      descr,
                                                       const rocsparse_float_complex* bsr_val,
                                                       const rocsparse_int*           bsr_row_ptr,
                                                       const rocsparse_int*           bsr_col_ind,
                                                       rocsparse_int                  block_dim,
                                                       rocsparse_mat_info             info,
                                                       rocsparse_analysis_policy      analysis,
                                                       rocsparse_solve_policy         solve,
                                                       void*                          temp_buffer)
{
    return rocsparse_bsric0_analysis_template(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              block_dim,
                                              info,
                                              analysis,
                                              solve,
                                              temp_buffer);
}

extern "C" rocsparse_status rocsparse_zbsric0_analysis(rocsparse_handle                handle,
                                                       rocsparse_direction             dir,
                                                       rocsparse_int                   mb,
                                                       rocsparse_int                   nnzb,
                                                       const rocsparse_mat_descr       descr,
                                                       const rocsparse_double_complex* bsr_val,
                                                       const rocsparse_int*            bsr_row_ptr,
                                                       const rocsparse_int*            bsr_col_ind,
                                                       rocsparse_int                   block_dim,
                                                       rocsparse_mat_info              info,
                                                       rocsparse_analysis_policy       analysis,
                                                       rocsparse_solve_policy          solve,
                                                       void*                           temp_buffer)
{
    return rocsparse_bsric0_analysis_template(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              block_dim,
                                              info,
                                              analysis,
                                              solve,
                                              temp_buffer);
}

extern "C" rocsparse_status rocsparse_bsric0_clear(rocsparse_handle handle, rocsparse_mat_info info)
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
    log_trace(handle, "rocsparse_bsric0_clear", (const void*&)info);

    // If meta data is not shared, delete it
    if(!rocsparse_check_trm_shared(info, info->bsric0_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsric0_info));
    }

    info->bsric0_info = nullptr;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_sbsric0(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_int             mb,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              float*                    bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
{
    return rocsparse_bsric0_template(handle,
                                     dir,
                                     mb,
                                     nnzb,
                                     descr,
                                     bsr_val,
                                     bsr_row_ptr,
                                     bsr_col_ind,
                                     block_dim,
                                     info,
                                     policy,
                                     temp_buffer);
}

extern "C" rocsparse_status rocsparse_dbsric0(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_int             mb,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              double*                   bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
{
    return rocsparse_bsric0_template(handle,
                                     dir,
                                     mb,
                                     nnzb,
                                     descr,
                                     bsr_val,
                                     bsr_row_ptr,
                                     bsr_col_ind,
                                     block_dim,
                                     info,
                                     policy,
                                     temp_buffer);
}

extern "C" rocsparse_status rocsparse_cbsric0(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_int             mb,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              rocsparse_float_complex*  bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
{
    return rocsparse_bsric0_template(handle,
                                     dir,
                                     mb,
                                     nnzb,
                                     descr,
                                     bsr_val,
                                     bsr_row_ptr,
                                     bsr_col_ind,
                                     block_dim,
                                     info,
                                     policy,
                                     temp_buffer);
}

extern "C" rocsparse_status rocsparse_zbsric0(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_int             mb,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              rocsparse_double_complex* bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
{
    return rocsparse_bsric0_template(handle,
                                     dir,
                                     mb,
                                     nnzb,
                                     descr,
                                     bsr_val,
                                     bsr_row_ptr,
                                     bsr_col_ind,
                                     block_dim,
                                     info,
                                     policy,
                                     temp_buffer);
}

extern "C" rocsparse_status rocsparse_bsric0_zero_pivot(rocsparse_handle   handle,
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
    log_trace(handle, "rocsparse_bsric0_zero_pivot", (const void*&)info, (const void*&)position);

    // Check pointer arguments
    if(position == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // If mb == 0 || nnzb == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->bsric0_info == nullptr)
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

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &pivot, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        if(pivot == std::numeric_limits<rocsparse_int>::max())
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 255, sizeof(rocsparse_int), stream));
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
        RETURN_IF_HIP_ERROR(
            hipMemcpy(position, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

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
