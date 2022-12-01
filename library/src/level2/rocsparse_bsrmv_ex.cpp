/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_bsrmv_ex.hpp"
#include "rocsparse_bsrxmv_spzl.hpp"
#include "rocsparse_csrmv.hpp"

template <typename T, typename I, typename J>
rocsparse_status rocsparse_bsrmv_ex_analysis_template(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      J                         mb,
                                                      J                         nb,
                                                      I                         nnzb,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const I*                  bsr_row_ptr,
                                                      const J*                  bsr_col_ind,
                                                      J                         block_dim,
                                                      rocsparse_mat_info        info)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check for valid matrix descriptor and info struct
    if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrmv_analysis"),
              dir,
              trans,
              mb,
              nb,
              nnzb,
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              (const void*&)info);

    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    if(trans != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0 || nb < 0 || nnzb < 0 || block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0 || block_dim == 0)
    {
        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    if(mb > 0 && bsr_row_ptr == nullptr)
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

    if(descr->storage_mode == rocsparse_storage_mode_sorted)
    {
        if(block_dim == 1)
        {
            return rocsparse_csrmv_analysis_template(
                handle, trans, mb, nb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, info);
        }
    }

    return rocsparse_status_success;
}

template <typename T, typename I, typename J, typename U>
rocsparse_status rocsparse_bsrmv_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans,
                                                   J                         mb,
                                                   J                         nb,
                                                   I                         nnzb,
                                                   U                         alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  bsr_val,
                                                   const I*                  bsr_row_ptr,
                                                   const J*                  bsr_col_ind,
                                                   J                         block_dim,
                                                   const T*                  x,
                                                   U                         beta_device_host,
                                                   T*                        y)
{
    if(trans != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }

    //
    // block_dim == 1 is the CSR case
    //
    if(block_dim == 1)
    {
        return rocsparse_csrmv_template_dispatch(handle,
                                                 trans,
                                                 mb,
                                                 nb,
                                                 nnzb,
                                                 alpha_device_host,
                                                 descr,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_row_ptr + 1,
                                                 bsr_col_ind,
                                                 x,
                                                 beta_device_host,
                                                 y,
                                                 false);
    }

    // LCOV_EXCL_START
    // Run different bsrmv kernels
    if(handle->wavefront_size == 32)
    {
        bsrxmvn_general<T, I, J>(handle,
                                 dir,
                                 mb,
                                 alpha_device_host,
                                 0,
                                 nullptr,
                                 bsr_row_ptr,
                                 nullptr,
                                 bsr_col_ind,
                                 bsr_val,
                                 block_dim,
                                 x,
                                 beta_device_host,
                                 y,
                                 descr->base);
        return rocsparse_status_success;
    }
    // LCOV_EXCL_STOP

    if(block_dim == 2)
    {
        bsrxmvn_2x2<T, I, J>(handle,
                             dir,
                             mb,
                             nnzb,
                             alpha_device_host,
                             0,
                             nullptr,
                             bsr_row_ptr,
                             nullptr,

                             bsr_col_ind,
                             bsr_val,
                             x,
                             beta_device_host,
                             y,
                             descr->base);
    }
    else if(block_dim == 3)
    {
        bsrxmvn_3x3<T, I, J>(handle,
                             dir,
                             mb,
                             nnzb,
                             alpha_device_host,
                             0,
                             nullptr,
                             bsr_row_ptr,
                             nullptr,

                             bsr_col_ind,
                             bsr_val,
                             x,
                             beta_device_host,
                             y,
                             descr->base);
    }
    else if(block_dim == 4)
    {
        bsrxmvn_4x4<T, I, J>(handle,
                             dir,
                             mb,
                             nnzb,
                             alpha_device_host,
                             0,
                             nullptr,
                             bsr_row_ptr,
                             nullptr,

                             bsr_col_ind,
                             bsr_val,
                             x,
                             beta_device_host,
                             y,
                             descr->base);
    }
    else if(block_dim == 5)
    {
        bsrxmvn_5x5<T, I, J>(handle,
                             dir,
                             mb,
                             nnzb,
                             alpha_device_host,
                             0,
                             nullptr,
                             bsr_row_ptr,
                             nullptr,

                             bsr_col_ind,
                             bsr_val,
                             x,
                             beta_device_host,
                             y,
                             descr->base);
    }
    else if(block_dim == 8)
    {
        bsrxmvn_8x8<T, I, J>(handle,
                             dir,
                             mb,
                             nnzb,
                             alpha_device_host,
                             0,
                             nullptr,
                             bsr_row_ptr,
                             nullptr,

                             bsr_col_ind,
                             bsr_val,
                             x,
                             beta_device_host,
                             y,
                             descr->base);
    }
    else if(block_dim == 16)
    {
        bsrxmvn_16x16<T, I, J>(handle,
                               dir,
                               mb,
                               nnzb,
                               alpha_device_host,
                               0,
                               nullptr,
                               bsr_row_ptr,
                               nullptr,

                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta_device_host,
                               y,
                               descr->base);
    }
    else if(block_dim > 16 && block_dim <= 32)
    {

        bsrxmvn_17_32<T, I, J>(handle,
                               dir,
                               mb,
                               nnzb,
                               alpha_device_host,
                               0,
                               nullptr,
                               bsr_row_ptr,
                               nullptr,

                               bsr_col_ind,
                               bsr_val,
                               block_dim,
                               x,
                               beta_device_host,
                               y,
                               descr->base);
    }
    else
    {
        bsrxmvn_general<T, I, J>(handle,
                                 dir,
                                 mb,
                                 alpha_device_host,
                                 0,
                                 nullptr,
                                 bsr_row_ptr,
                                 nullptr,
                                 bsr_col_ind,
                                 bsr_val,
                                 block_dim,
                                 x,
                                 beta_device_host,
                                 y,
                                 descr->base);
    }

    return rocsparse_status_success;
}

template <typename T, typename I, typename J, typename U>
rocsparse_status rocsparse_bsrmv_adaptive_template_dispatch(rocsparse_handle    handle,
                                                            rocsparse_direction dir,
                                                            rocsparse_operation trans,
                                                            J                   mb,
                                                            J                   nb,
                                                            I                   nnzb,
                                                            U                   alpha_device_host,
                                                            const rocsparse_mat_descr descr,
                                                            const T*                  bsr_val,
                                                            const I*                  bsr_row_ptr,
                                                            const J*                  bsr_col_ind,
                                                            J                         block_dim,
                                                            rocsparse_csrmv_info      info,
                                                            const T*                  x,
                                                            U  beta_device_host,
                                                            T* y)
{
    if(trans != rocsparse_operation_none || descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // block_dim == 1 is the CSR case
    if(block_dim == 1)
    {
        return rocsparse_csrmv_adaptive_template_dispatch(handle,
                                                          trans,
                                                          mb,
                                                          nb,
                                                          nnzb,
                                                          alpha_device_host,
                                                          descr,
                                                          bsr_val,
                                                          bsr_row_ptr,
                                                          bsr_col_ind,
                                                          info,
                                                          x,
                                                          beta_device_host,
                                                          y,
                                                          false);
    }

    return rocsparse_bsrmv_template_dispatch(handle,
                                             dir,
                                             trans,
                                             mb,
                                             nb,
                                             nnzb,
                                             alpha_device_host,
                                             descr,
                                             bsr_val,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             block_dim,
                                             x,
                                             beta_device_host,
                                             y);
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_bsrmv_ex_template(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans,
                                             J                         mb,
                                             J                         nb,
                                             I                         nnzb,
                                             const T*                  alpha_device_host,
                                             const rocsparse_mat_descr descr,
                                             const T*                  bsr_val,
                                             const I*                  bsr_row_ptr,
                                             const J*                  bsr_col_ind,
                                             J                         block_dim,
                                             rocsparse_mat_info        info,
                                             const T*                  x,
                                             const T*                  beta_device_host,
                                             T*                        y)
{
    //
    // Check for valid handle and matrix descriptor
    //
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Logging
    //
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrmv"),
              dir,
              trans,
              mb,
              nb,
              nnzb,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              (const void*&)x,
              LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
              (const void*&)y);

    log_bench(handle,
              "./rocsparse-bench -f bsrmv -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> "
              "--blockdim",
              block_dim,
              "--alpha",
              LOG_BENCH_SCALAR_VALUE(handle, alpha_device_host),
              "--beta",
              LOG_BENCH_SCALAR_VALUE(handle, beta_device_host));

    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    if(trans != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    //
    // Check sizes
    //
    if(mb < 0 || nb < 0 || nnzb < 0 || block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    //
    // Quick return if possible
    //
    if(mb == 0 || nb == 0 || block_dim == 0)
    {
        return rocsparse_status_success;
    }

    //
    // Check pointer arguments
    //
    if(alpha_device_host == nullptr || beta_device_host == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Another quick return.
    //
    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    //
    // Check the rest of pointer arguments
    //
    if((mb > 0 && bsr_row_ptr == nullptr) || x == nullptr || y == nullptr)
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

    if(info == nullptr || info->csrmv_info == nullptr || trans != rocsparse_operation_none
       || descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        // If bsrmv info is not available, call bsrmv general
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            return rocsparse_bsrmv_template_dispatch(handle,
                                                     dir,
                                                     trans,
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     alpha_device_host,
                                                     descr,
                                                     bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     block_dim,
                                                     x,
                                                     beta_device_host,
                                                     y);
        }
        else
        {
            return rocsparse_bsrmv_template_dispatch(handle,
                                                     dir,
                                                     trans,
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     *alpha_device_host,
                                                     descr,
                                                     bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     block_dim,
                                                     x,
                                                     *beta_device_host,
                                                     y);
        }
    }
    else
    {
        // If bsrmv info is available, call bsrmv adaptive
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            return rocsparse_bsrmv_adaptive_template_dispatch(handle,
                                                              dir,
                                                              trans,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              alpha_device_host,
                                                              descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              info->csrmv_info,
                                                              x,
                                                              beta_device_host,
                                                              y);
        }
        else
        {
            return rocsparse_bsrmv_adaptive_template_dispatch(handle,
                                                              dir,
                                                              trans,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              *alpha_device_host,
                                                              descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              info->csrmv_info,
                                                              x,
                                                              *beta_device_host,
                                                              y);
        }
    }
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE)                                                 \
    template rocsparse_status rocsparse_bsrmv_ex_analysis_template<TTYPE, ITYPE, JTYPE>( \
        rocsparse_handle          handle,                                                \
        rocsparse_direction       dir,                                                   \
        rocsparse_operation       trans,                                                 \
        JTYPE                     mb,                                                    \
        JTYPE                     nb,                                                    \
        ITYPE                     nnzb,                                                  \
        const rocsparse_mat_descr descr,                                                 \
        const TTYPE*              bsr_val,                                               \
        const ITYPE*              bsr_row_ptr,                                           \
        const JTYPE*              bsr_col_ind,                                           \
        JTYPE                     block_dim,                                             \
        rocsparse_mat_info        info)

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);

INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);

INSTANTIATE(float, int64_t, int64_t);
INSTANTIATE(double, int64_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);

#undef INSTANTIATE

#define INSTANTIATE(TTYPE, ITYPE, JTYPE)                                        \
    template rocsparse_status rocsparse_bsrmv_ex_template<TTYPE, ITYPE, JTYPE>( \
        rocsparse_handle          handle,                                       \
        rocsparse_direction       dir,                                          \
        rocsparse_operation       trans,                                        \
        JTYPE                     mb,                                           \
        JTYPE                     nb,                                           \
        ITYPE                     nnzb,                                         \
        const TTYPE*              alpha_device_host,                            \
        const rocsparse_mat_descr descr,                                        \
        const TTYPE*              bsr_val,                                      \
        const ITYPE*              bsr_row_ptr,                                  \
        const JTYPE*              bsr_col_ind,                                  \
        JTYPE                     block_dim,                                    \
        rocsparse_mat_info        info,                                         \
        const TTYPE*              x,                                            \
        const TTYPE*              beta_device_host,                             \
        TTYPE*                    y)

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);

INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);

INSTANTIATE(float, int64_t, int64_t);
INSTANTIATE(double, int64_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);

#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

// rocsparse_xbsrmv_analysis
#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             nb,          \
                                     rocsparse_int             nnzb,        \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     rocsparse_mat_info        info)        \
    {                                                                       \
        return rocsparse_bsrmv_ex_analysis_template(handle,                 \
                                                    dir,                    \
                                                    trans,                  \
                                                    mb,                     \
                                                    nb,                     \
                                                    nnzb,                   \
                                                    descr,                  \
                                                    bsr_val,                \
                                                    bsr_row_ptr,            \
                                                    bsr_col_ind,            \
                                                    block_dim,              \
                                                    info);                  \
    }

C_IMPL(rocsparse_sbsrmv_ex_analysis, float);
C_IMPL(rocsparse_dbsrmv_ex_analysis, double);
C_IMPL(rocsparse_cbsrmv_ex_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmv_ex_analysis, rocsparse_double_complex);

#undef C_IMPL

// rocsparse_xbsrmv_ex
#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             nb,          \
                                     rocsparse_int             nnzb,        \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     rocsparse_mat_info        info,        \
                                     const TYPE*               x,           \
                                     const TYPE*               beta,        \
                                     TYPE*                     y)           \
    {                                                                       \
        return rocsparse_bsrmv_ex_template(handle,                          \
                                           dir,                             \
                                           trans,                           \
                                           mb,                              \
                                           nb,                              \
                                           nnzb,                            \
                                           alpha,                           \
                                           descr,                           \
                                           bsr_val,                         \
                                           bsr_row_ptr,                     \
                                           bsr_col_ind,                     \
                                           block_dim,                       \
                                           info,                            \
                                           x,                               \
                                           beta,                            \
                                           y);                              \
    }

C_IMPL(rocsparse_sbsrmv_ex, float);
C_IMPL(rocsparse_dbsrmv_ex, double);
C_IMPL(rocsparse_cbsrmv_ex, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmv_ex, rocsparse_double_complex);

#undef C_IMPL

extern "C" rocsparse_status rocsparse_bsrmv_ex_clear(rocsparse_handle   handle,
                                                     rocsparse_mat_info info)
{
    return rocsparse_csrmv_clear(handle, info);
}
