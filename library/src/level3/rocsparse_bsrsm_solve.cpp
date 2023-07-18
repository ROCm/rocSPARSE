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

#include "../level2/rocsparse_bsrsv.hpp"
#include "internal/level3/rocsparse_bsrsm.h"
#include "utility.h"

template <typename T, typename U>
rocsparse_status rocsparse_bsrsm_solve_template_large(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_X,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nrhs,
                                                      rocsparse_int             nnzb,
                                                      U                         alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             block_dim,
                                                      rocsparse_mat_info        info,
                                                      const T*                  B,
                                                      rocsparse_int             ldb,
                                                      T*                        X,
                                                      rocsparse_int             ldx,
                                                      void*                     temp_buffer);

template <typename T, typename U>
rocsparse_status rocsparse_bsrsm_solve_template_dispatch(rocsparse_handle          handle,
                                                         rocsparse_direction       dir,
                                                         rocsparse_operation       trans_A,
                                                         rocsparse_operation       trans_X,
                                                         rocsparse_int             mb,
                                                         rocsparse_int             nrhs,
                                                         rocsparse_int             nnzb,
                                                         U                         alpha,
                                                         const rocsparse_mat_descr descr,
                                                         const T*                  bsr_val,
                                                         const rocsparse_int*      bsr_row_ptr,
                                                         const rocsparse_int*      bsr_col_ind,
                                                         rocsparse_int             block_dim,
                                                         rocsparse_mat_info        info,
                                                         const T*                  B,
                                                         rocsparse_int             ldb,
                                                         T*                        X,
                                                         rocsparse_int             ldx,
                                                         rocsparse_solve_policy    policy,
                                                         void*                     temp_buffer)
{
    return rocsparse_bsrsm_solve_template_large(handle,
                                                dir,
                                                trans_A,
                                                trans_X,
                                                mb,
                                                nrhs,
                                                nnzb,
                                                alpha,
                                                descr,
                                                bsr_val,
                                                bsr_row_ptr,
                                                bsr_col_ind,
                                                block_dim,
                                                info,
                                                B,
                                                ldb,
                                                X,
                                                ldx,
                                                temp_buffer);
}

template <typename T>
rocsparse_status rocsparse_bsrsm_solve_template(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_X,
                                                rocsparse_int             mb,
                                                rocsparse_int             nrhs,
                                                rocsparse_int             nnzb,
                                                const T*                  alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                const T*                  B,
                                                rocsparse_int             ldb,
                                                T*                        X,
                                                rocsparse_int             ldx,
                                                rocsparse_solve_policy    policy,
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
              replaceX<T>("rocsparse_Xbsrsm_solve"),
              dir,
              trans_A,
              trans_X,
              mb,
              nrhs,
              nnzb,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              (const void*&)info,
              (const void*&)B,
              ldb,
              (const void*&)X,
              ldx,
              policy,
              (const void*&)temp_buffer);

    if(rocsparse_enum_utils::is_invalid(dir) || rocsparse_enum_utils::is_invalid(trans_A)
       || rocsparse_enum_utils::is_invalid(trans_X) || rocsparse_enum_utils::is_invalid(policy))
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
        return rocsparse_status_requires_sorted_storage;
    }

    // Check sizes
    if(mb < 0 || nrhs < 0 || nnzb < 0 || block_dim <= 0 || ldb < 0 || ldx < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check leading dimensions
    if(trans_X == rocsparse_operation_none)
    {
        if(std::min(ldb, ldx) < mb * block_dim)
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(std::min(ldb, ldx) < nrhs)
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Quick return if possible
    if(mb == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr == nullptr || alpha_device_host == nullptr || B == nullptr || X == nullptr
       || temp_buffer == nullptr)
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

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_bsrsm_solve_template_dispatch(handle,
                                                       dir,
                                                       trans_A,
                                                       trans_X,
                                                       mb,
                                                       nrhs,
                                                       nnzb,
                                                       alpha_device_host,
                                                       descr,
                                                       bsr_val,
                                                       bsr_row_ptr,
                                                       bsr_col_ind,
                                                       block_dim,
                                                       info,
                                                       B,
                                                       ldb,
                                                       X,
                                                       ldx,
                                                       policy,
                                                       temp_buffer);
    }
    else
    {
        return rocsparse_bsrsm_solve_template_dispatch(handle,
                                                       dir,
                                                       trans_A,
                                                       trans_X,
                                                       mb,
                                                       nrhs,
                                                       nnzb,
                                                       *alpha_device_host,
                                                       descr,
                                                       bsr_val,
                                                       bsr_row_ptr,
                                                       bsr_col_ind,
                                                       block_dim,
                                                       info,
                                                       B,
                                                       ldb,
                                                       X,
                                                       ldx,
                                                       policy,
                                                       temp_buffer);
    }
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
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     rocsparse_mat_info        info,        \
                                     const TYPE*               B,           \
                                     rocsparse_int             ldb,         \
                                     TYPE*                     X,           \
                                     rocsparse_int             ldx,         \
                                     rocsparse_solve_policy    policy,      \
                                     void*                     temp_buffer) \
    try                                                                     \
    {                                                                       \
        return rocsparse_bsrsm_solve_template(handle,                       \
                                              dir,                          \
                                              trans_A,                      \
                                              trans_X,                      \
                                              mb,                           \
                                              nrhs,                         \
                                              nnzb,                         \
                                              alpha,                        \
                                              descr,                        \
                                              bsr_val,                      \
                                              bsr_row_ptr,                  \
                                              bsr_col_ind,                  \
                                              block_dim,                    \
                                              info,                         \
                                              B,                            \
                                              ldb,                          \
                                              X,                            \
                                              ldx,                          \
                                              policy,                       \
                                              temp_buffer);                 \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        return exception_to_rocsparse_status();                             \
    }

C_IMPL(rocsparse_sbsrsm_solve, float);
C_IMPL(rocsparse_dbsrsm_solve, double);
C_IMPL(rocsparse_cbsrsm_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsm_solve, rocsparse_double_complex);

#undef C_IMPL
