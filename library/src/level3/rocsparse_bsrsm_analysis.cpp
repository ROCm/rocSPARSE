/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "definitions.h"
#include "utility.h"

#include "../level2/rocsparse_csrsv.hpp"

template <typename T>
rocsparse_status rocsparse_bsrsm_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_X,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nrhs,
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
    // Check for valid handle, matrix descriptor and info
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrsm_analysis"),
              dir,
              trans_A,
              trans_X,
              mb,
              nrhs,
              nnzb,
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              (const void*&)info,
              analysis,
              solve,
              (const void*&)temp_buffer);

    if(rocsparse_enum_utils::is_invalid(dir) || rocsparse_enum_utils::is_invalid(trans_A)
       || rocsparse_enum_utils::is_invalid(trans_X) || rocsparse_enum_utils::is_invalid(analysis)
       || rocsparse_enum_utils::is_invalid(solve))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0 || nrhs < 0 || nnzb < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((bsr_val == nullptr && bsr_col_ind != nullptr)
       || (bsr_val != nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnzb != 0 && (bsr_val == nullptr && bsr_col_ind == nullptr))
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

            // If bsrsm meta data is already available, do nothing
            if(trans_A == rocsparse_operation_none && info->bsrsm_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_transpose && info->bsrsmt_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other upper analysis meta data

            if(trans_A == rocsparse_operation_none && info->bsrsv_upper_info != nullptr)
            {
                // bsrsv meta data
                info->bsrsm_upper_info = info->bsrsv_upper_info;
                return rocsparse_status_success;
            }

            if(trans_A == rocsparse_operation_transpose && info->bsrsvt_upper_info != nullptr)
            {
                // bsrsv meta data
                info->bsrsmt_upper_info = info->bsrsvt_upper_info;
                return rocsparse_status_success;
            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used

        // Clear bsrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info((trans_A == rocsparse_operation_none)
                                                                 ? info->bsrsm_upper_info
                                                                 : info->bsrsmt_upper_info));

        // Create bsrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info((trans_A == rocsparse_operation_none)
                                                                ? &info->bsrsm_upper_info
                                                                : &info->bsrsmt_upper_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(handle,
                                                         trans_A,
                                                         mb,
                                                         nnzb,
                                                         descr,
                                                         bsr_val,
                                                         bsr_row_ptr,
                                                         bsr_col_ind,
                                                         (trans_A == rocsparse_operation_none)
                                                             ? info->bsrsm_upper_info
                                                             : info->bsrsmt_upper_info,
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

            // If bsrsm meta data is already available, do nothing
            if(trans_A == rocsparse_operation_none && info->bsrsm_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_transpose && info->bsrsmt_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other lower analysis meta data

            if(trans_A == rocsparse_operation_none && info->bsrilu0_info != nullptr)
            {
                // bsrilu0 meta data
                info->bsrsm_lower_info = info->bsrilu0_info;
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_none && info->bsric0_info != nullptr)
            {
                // bsric0 meta data
                info->bsrsm_lower_info = info->bsric0_info;
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_none && info->bsrsv_lower_info != nullptr)
            {
                // bsrsv meta data
                info->bsrsm_lower_info = info->bsrsv_lower_info;
                return rocsparse_status_success;
            }

            if(trans_A == rocsparse_operation_transpose && info->bsrsvt_lower_info != nullptr)
            {
                // bsrsv meta data
                info->bsrsm_upper_info = info->bsrsvt_lower_info;
                return rocsparse_status_success;
            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used

        // Clear bsrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info((trans_A == rocsparse_operation_none)
                                                                 ? info->bsrsm_lower_info
                                                                 : info->bsrsmt_lower_info));

        // Create bsrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info((trans_A == rocsparse_operation_none)
                                                                ? &info->bsrsm_lower_info
                                                                : &info->bsrsmt_lower_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(handle,
                                                         trans_A,
                                                         mb,
                                                         nnzb,
                                                         descr,
                                                         bsr_val,
                                                         bsr_row_ptr,
                                                         bsr_col_ind,
                                                         (trans_A == rocsparse_operation_none)
                                                             ? info->bsrsm_lower_info
                                                             : info->bsrsmt_lower_info,
                                                         (rocsparse_int**)&info->zero_pivot,
                                                         temp_buffer));
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_operation       trans_A,     \
                                     rocsparse_operation       trans_X,     \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             nrhs,        \
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
    try                                                                     \
    {                                                                       \
        return rocsparse_bsrsm_analysis_template(handle,                    \
                                                 dir,                       \
                                                 trans_A,                   \
                                                 trans_X,                   \
                                                 mb,                        \
                                                 nrhs,                      \
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
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        return exception_to_rocsparse_status();                             \
    }

C_IMPL(rocsparse_sbsrsm_analysis, float);
C_IMPL(rocsparse_dbsrsm_analysis, double);
C_IMPL(rocsparse_cbsrsm_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsm_analysis, rocsparse_double_complex);

#undef C_IMPL
