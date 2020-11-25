/*! \file */
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

#include "rocsparse_bsrsv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sbsrsv_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_direction       dir,
                                                         rocsparse_operation       trans,
                                                         rocsparse_int             mb,
                                                         rocsparse_int             nnzb,
                                                         const rocsparse_mat_descr descr,
                                                         const float*              bsr_val,
                                                         const rocsparse_int*      bsr_row_ptr,
                                                         const rocsparse_int*      bsr_col_ind,
                                                         rocsparse_int             bsr_dim,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size)
{
    // Check direction
    if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes that are not checked by csrsv
    if(bsr_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    rocsparse_status stat = rocsparse_scsrsv_buffer_size(
        handle, trans, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, info, buffer_size);

    // Need additional buffer when using transposed
    if(trans == rocsparse_operation_transpose)
    {
        // Remove additional CSR buffer
        *buffer_size -= sizeof(float) * ((nnzb - 1) / 256 + 1) * 256;

        // Add BSR buffer instead
        *buffer_size += sizeof(float) * ((nnzb * bsr_dim * bsr_dim - 1) / 256 + 1) * 256;
    }

    return stat;
}

extern "C" rocsparse_status rocsparse_dbsrsv_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_direction       dir,
                                                         rocsparse_operation       trans,
                                                         rocsparse_int             mb,
                                                         rocsparse_int             nnzb,
                                                         const rocsparse_mat_descr descr,
                                                         const double*             bsr_val,
                                                         const rocsparse_int*      bsr_row_ptr,
                                                         const rocsparse_int*      bsr_col_ind,
                                                         rocsparse_int             bsr_dim,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size)
{
    // Check direction
    if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes that are not checked by csrsv
    if(bsr_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    rocsparse_status stat = rocsparse_dcsrsv_buffer_size(
        handle, trans, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, info, buffer_size);

    // Need additional buffer when using transposed
    if(trans == rocsparse_operation_transpose)
    {
        // Remove additional CSR buffer
        *buffer_size -= sizeof(double) * ((nnzb - 1) / 256 + 1) * 256;

        // Add BSR buffer instead
        *buffer_size += sizeof(double) * ((nnzb * bsr_dim * bsr_dim - 1) / 256 + 1) * 256;
    }

    return stat;
}

extern "C" rocsparse_status rocsparse_cbsrsv_buffer_size(rocsparse_handle               handle,
                                                         rocsparse_direction            dir,
                                                         rocsparse_operation            trans,
                                                         rocsparse_int                  mb,
                                                         rocsparse_int                  nnzb,
                                                         const rocsparse_mat_descr      descr,
                                                         const rocsparse_float_complex* bsr_val,
                                                         const rocsparse_int*           bsr_row_ptr,
                                                         const rocsparse_int*           bsr_col_ind,
                                                         rocsparse_int                  bsr_dim,
                                                         rocsparse_mat_info             info,
                                                         size_t*                        buffer_size)
{
    // Check direction
    if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes that are not checked by csrsv
    if(bsr_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    rocsparse_status stat = rocsparse_ccsrsv_buffer_size(
        handle, trans, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, info, buffer_size);

    // Need additional buffer when using transposed
    if(trans == rocsparse_operation_transpose)
    {
        // Remove additional CSR buffer
        *buffer_size -= sizeof(rocsparse_float_complex) * ((nnzb - 1) / 256 + 1) * 256;

        // Add BSR buffer instead
        *buffer_size
            += sizeof(rocsparse_float_complex) * ((nnzb * bsr_dim * bsr_dim - 1) / 256 + 1) * 256;
    }

    return stat;
}

extern "C" rocsparse_status rocsparse_zbsrsv_buffer_size(rocsparse_handle                handle,
                                                         rocsparse_direction             dir,
                                                         rocsparse_operation             trans,
                                                         rocsparse_int                   mb,
                                                         rocsparse_int                   nnzb,
                                                         const rocsparse_mat_descr       descr,
                                                         const rocsparse_double_complex* bsr_val,
                                                         const rocsparse_int* bsr_row_ptr,
                                                         const rocsparse_int* bsr_col_ind,
                                                         rocsparse_int        bsr_dim,
                                                         rocsparse_mat_info   info,
                                                         size_t*              buffer_size)
{
    // Check direction
    if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes that are not checked by csrsv
    if(bsr_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    rocsparse_status stat = rocsparse_zcsrsv_buffer_size(
        handle, trans, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, info, buffer_size);

    // Need additional buffer when using transposed
    if(trans == rocsparse_operation_transpose)
    {
        // Remove additional CSR buffer
        *buffer_size -= sizeof(rocsparse_double_complex) * ((nnzb - 1) / 256 + 1) * 256;

        // Add BSR buffer instead
        *buffer_size
            += sizeof(rocsparse_double_complex) * ((nnzb * bsr_dim * bsr_dim - 1) / 256 + 1) * 256;
    }

    return stat;
}

extern "C" rocsparse_status rocsparse_sbsrsv_analysis(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nnzb,
                                                      const rocsparse_mat_descr descr,
                                                      const float*              bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             bsr_dim,
                                                      rocsparse_mat_info        info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy    solve,
                                                      void*                     temp_buffer)
{
    return rocsparse_bsrsv_analysis_template(handle,
                                             dir,
                                             trans,
                                             mb,
                                             nnzb,
                                             descr,
                                             bsr_val,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             bsr_dim,
                                             info,
                                             analysis,
                                             solve,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_dbsrsv_analysis(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nnzb,
                                                      const rocsparse_mat_descr descr,
                                                      const double*             bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             bsr_dim,
                                                      rocsparse_mat_info        info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy    solve,
                                                      void*                     temp_buffer)
{
    return rocsparse_bsrsv_analysis_template(handle,
                                             dir,
                                             trans,
                                             mb,
                                             nnzb,
                                             descr,
                                             bsr_val,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             bsr_dim,
                                             info,
                                             analysis,
                                             solve,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_cbsrsv_analysis(rocsparse_handle               handle,
                                                      rocsparse_direction            dir,
                                                      rocsparse_operation            trans,
                                                      rocsparse_int                  mb,
                                                      rocsparse_int                  nnzb,
                                                      const rocsparse_mat_descr      descr,
                                                      const rocsparse_float_complex* bsr_val,
                                                      const rocsparse_int*           bsr_row_ptr,
                                                      const rocsparse_int*           bsr_col_ind,
                                                      rocsparse_int                  bsr_dim,
                                                      rocsparse_mat_info             info,
                                                      rocsparse_analysis_policy      analysis,
                                                      rocsparse_solve_policy         solve,
                                                      void*                          temp_buffer)
{
    return rocsparse_bsrsv_analysis_template(handle,
                                             dir,
                                             trans,
                                             mb,
                                             nnzb,
                                             descr,
                                             bsr_val,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             bsr_dim,
                                             info,
                                             analysis,
                                             solve,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_zbsrsv_analysis(rocsparse_handle                handle,
                                                      rocsparse_direction             dir,
                                                      rocsparse_operation             trans,
                                                      rocsparse_int                   mb,
                                                      rocsparse_int                   nnzb,
                                                      const rocsparse_mat_descr       descr,
                                                      const rocsparse_double_complex* bsr_val,
                                                      const rocsparse_int*            bsr_row_ptr,
                                                      const rocsparse_int*            bsr_col_ind,
                                                      rocsparse_int                   bsr_dim,
                                                      rocsparse_mat_info              info,
                                                      rocsparse_analysis_policy       analysis,
                                                      rocsparse_solve_policy          solve,
                                                      void*                           temp_buffer)
{
    return rocsparse_bsrsv_analysis_template(handle,
                                             dir,
                                             trans,
                                             mb,
                                             nnzb,
                                             descr,
                                             bsr_val,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             bsr_dim,
                                             info,
                                             analysis,
                                             solve,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_bsrsv_clear(rocsparse_handle handle, rocsparse_mat_info info)
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
    log_trace(handle, "rocsparse_bsrsv_clear", (const void*&)info);

    // Clear bsrsv meta data (this includes lower, upper and their transposed equivalents
    if(!rocsparse_check_trm_shared(info, info->bsrsv_lower_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsv_lower_info));
    }
    if(!rocsparse_check_trm_shared(info, info->bsrsvt_lower_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsvt_lower_info));
    }
    if(!rocsparse_check_trm_shared(info, info->bsrsv_upper_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsv_upper_info));
    }
    if(!rocsparse_check_trm_shared(info, info->bsrsvt_upper_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsvt_upper_info));
    }

    info->bsrsv_lower_info  = nullptr;
    info->bsrsvt_lower_info = nullptr;
    info->bsrsv_upper_info  = nullptr;
    info->bsrsvt_upper_info = nullptr;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_sbsrsv_solve(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nnzb,
                                                   const float*              alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const float*              bsr_val,
                                                   const rocsparse_int*      bsr_row_ptr,
                                                   const rocsparse_int*      bsr_col_ind,
                                                   rocsparse_int             bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   const float*              x,
                                                   float*                    y,
                                                   rocsparse_solve_policy    policy,
                                                   void*                     temp_buffer)
{
    return rocsparse_bsrsv_solve_template(handle,
                                          dir,
                                          trans,
                                          mb,
                                          nnzb,
                                          alpha,
                                          descr,
                                          bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          bsr_dim,
                                          info,
                                          x,
                                          y,
                                          policy,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_dbsrsv_solve(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nnzb,
                                                   const double*             alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const double*             bsr_val,
                                                   const rocsparse_int*      bsr_row_ptr,
                                                   const rocsparse_int*      bsr_col_ind,
                                                   rocsparse_int             bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   const double*             x,
                                                   double*                   y,
                                                   rocsparse_solve_policy    policy,
                                                   void*                     temp_buffer)
{
    return rocsparse_bsrsv_solve_template(handle,
                                          dir,
                                          trans,
                                          mb,
                                          nnzb,
                                          alpha,
                                          descr,
                                          bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          bsr_dim,
                                          info,
                                          x,
                                          y,
                                          policy,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_cbsrsv_solve(rocsparse_handle               handle,
                                                   rocsparse_direction            dir,
                                                   rocsparse_operation            trans,
                                                   rocsparse_int                  mb,
                                                   rocsparse_int                  nnzb,
                                                   const rocsparse_float_complex* alpha,
                                                   const rocsparse_mat_descr      descr,
                                                   const rocsparse_float_complex* bsr_val,
                                                   const rocsparse_int*           bsr_row_ptr,
                                                   const rocsparse_int*           bsr_col_ind,
                                                   rocsparse_int                  bsr_dim,
                                                   rocsparse_mat_info             info,
                                                   const rocsparse_float_complex* x,
                                                   rocsparse_float_complex*       y,
                                                   rocsparse_solve_policy         policy,
                                                   void*                          temp_buffer)
{
    return rocsparse_bsrsv_solve_template(handle,
                                          dir,
                                          trans,
                                          mb,
                                          nnzb,
                                          alpha,
                                          descr,
                                          bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          bsr_dim,
                                          info,
                                          x,
                                          y,
                                          policy,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_zbsrsv_solve(rocsparse_handle                handle,
                                                   rocsparse_direction             dir,
                                                   rocsparse_operation             trans,
                                                   rocsparse_int                   mb,
                                                   rocsparse_int                   nnzb,
                                                   const rocsparse_double_complex* alpha,
                                                   const rocsparse_mat_descr       descr,
                                                   const rocsparse_double_complex* bsr_val,
                                                   const rocsparse_int*            bsr_row_ptr,
                                                   const rocsparse_int*            bsr_col_ind,
                                                   rocsparse_int                   bsr_dim,
                                                   rocsparse_mat_info              info,
                                                   const rocsparse_double_complex* x,
                                                   rocsparse_double_complex*       y,
                                                   rocsparse_solve_policy          policy,
                                                   void*                           temp_buffer)
{
    return rocsparse_bsrsv_solve_template(handle,
                                          dir,
                                          trans,
                                          mb,
                                          nnzb,
                                          alpha,
                                          descr,
                                          bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          bsr_dim,
                                          info,
                                          x,
                                          y,
                                          policy,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_bsrsv_zero_pivot(rocsparse_handle   handle,
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
    log_trace(handle, "rocsparse_bsrsv_zero_pivot", (const void*&)info, (const void*&)position);

    // Check pointer arguments
    if(position == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // If mb == 0 || nnzb == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->zero_pivot == nullptr)
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
        rocsparse_int zero_pivot;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &zero_pivot, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        if(zero_pivot == std::numeric_limits<rocsparse_int>::max())
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
