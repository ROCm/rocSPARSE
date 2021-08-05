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

#include "rocsparse_bsrsv.hpp"

#include "definitions.h"
#include "utility.h"

#include "../level2/rocsparse_csrsv.hpp"

template <typename T>
rocsparse_status rocsparse_bsrsv_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  bsr_val,
                                                   const rocsparse_int*      bsr_row_ptr,
                                                   const rocsparse_int*      bsr_col_ind,
                                                   rocsparse_int             block_dim,
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

    if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrsv_analysis"),
              dir,
              trans,
              mb,
              nnzb,
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              (const void*&)info,
              solve,
              analysis,
              (const void*&)temp_buffer);

    // Check operation
    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    // Check direction
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    // Check analysis
    if(rocsparse_enum_utils::is_invalid(analysis))
    {
        return rocsparse_status_invalid_value;
    }

    // Check solve
    if(rocsparse_enum_utils::is_invalid(solve))
    {
        return rocsparse_status_invalid_value;
    }

    // Check operation type
    if(trans != rocsparse_operation_none && trans != rocsparse_operation_transpose)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0 || nnzb < 0 || block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nnzb == 0 || block_dim == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_val == nullptr)
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

            // If bsrsv meta data is already available, do nothing
            if(trans == rocsparse_operation_none && info->bsrsv_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_transpose && info->bsrsvt_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }

            //            // Check for other upper analysis meta data that could be used
            //            if(trans == rocsparse_operation_none && info->bsrsm_upper_info != nullptr)
            //            {
            //                // bsrsm meta data
            //                info->bsrsv_upper_info = info->bsrsm_upper_info;
            //                return rocsparse_status_success;
            //            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used.

        // Clear bsrsv
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info((trans == rocsparse_operation_none)
                                                                 ? info->bsrsv_upper_info
                                                                 : info->bsrsvt_upper_info));

        // Create bsrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info((trans == rocsparse_operation_none)
                                                                ? &info->bsrsv_upper_info
                                                                : &info->bsrsvt_upper_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(
            handle,
            trans,
            mb,
            nnzb,
            descr,
            bsr_val,
            bsr_row_ptr,
            bsr_col_ind,
            (trans == rocsparse_operation_none) ? info->bsrsv_upper_info : info->bsrsvt_upper_info,
            (rocsparse_int**)&info->zero_pivot,
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

            // If bsrsv meta data is already available, do nothing
            if(trans == rocsparse_operation_none && info->bsrsv_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_transpose && info->bsrsvt_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other lower analysis meta data that could be used
            if(trans == rocsparse_operation_none && info->bsric0_info != nullptr)
            {
                // bsric0 meta data
                info->bsrsv_lower_info = info->bsric0_info;
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_none && info->bsrilu0_info != nullptr)
            {
                // bsrilu0 meta data
                info->bsrsv_lower_info = info->bsrilu0_info;
                return rocsparse_status_success;
            }
            // else if(trans == rocsparse_operation_none && info->bsrsm_lower_info != nullptr)
            // {
            //     // bsrsm meta data
            //     info->bsrsv_lower_info = info->bsrsm_lower_info;
            //     return rocsparse_status_success;
            // }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used.

        // Clear bsrsv
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info((trans == rocsparse_operation_none)
                                                                 ? info->bsrsv_lower_info
                                                                 : info->bsrsvt_lower_info));

        // Create bsrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info((trans == rocsparse_operation_none)
                                                                ? &info->bsrsv_lower_info
                                                                : &info->bsrsvt_lower_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(
            handle,
            trans,
            mb,
            nnzb,
            descr,
            bsr_val,
            bsr_row_ptr,
            bsr_col_ind,
            (trans == rocsparse_operation_none) ? info->bsrsv_lower_info : info->bsrsvt_lower_info,
            (rocsparse_int**)&info->zero_pivot,
            temp_buffer));
    }

    return rocsparse_status_success;
}

// bsrsv_analysis
#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             nnzb,        \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     rocsparse_mat_info        info,        \
                                     rocsparse_analysis_policy analysis,    \
                                     rocsparse_solve_policy    solve,       \
                                     void*                     temp_buffer) \
    {                                                                       \
        return rocsparse_bsrsv_analysis_template(handle,                    \
                                                 dir,                       \
                                                 trans,                     \
                                                 mb,                        \
                                                 nnzb,                      \
                                                 descr,                     \
                                                 bsr_val,                   \
                                                 bsr_row_ptr,               \
                                                 bsr_col_ind,               \
                                                 block_dim,                 \
                                                 info,                      \
                                                 analysis,                  \
                                                 solve,                     \
                                                 temp_buffer);              \
    }

C_IMPL(rocsparse_sbsrsv_analysis, float);
C_IMPL(rocsparse_dbsrsv_analysis, double);
C_IMPL(rocsparse_cbsrsv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsv_analysis, rocsparse_double_complex);
#undef C_IMPL
