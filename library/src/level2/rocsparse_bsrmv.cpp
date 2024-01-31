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

#include "internal/level2/rocsparse_bsrmv.h"
#include "internal/level2/rocsparse_csrmv.h"
#include "rocsparse_bsrmv.hpp"
#include "rocsparse_bsrxmv_spzl.hpp"
#include "rocsparse_csrmv.hpp"

template <typename I, typename J, typename A>
rocsparse_status rocsparse_bsrmv_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans,
                                                   J                         mb,
                                                   J                         nb,
                                                   I                         nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   const A*                  bsr_val,
                                                   const I*                  bsr_row_ptr,
                                                   const J*                  bsr_col_ind,
                                                   J                         block_dim,
                                                   rocsparse_mat_info        info)
{
    // Check for valid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    // Check for valid matrix descriptor and info struct
    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG_POINTER(11, info);

    // Logging
    log_trace(handle,
              replaceX<A>("rocsparse_Xbsrmv_analysis"),
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

    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_ENUM(2, trans);

    ROCSPARSE_CHECKARG(
        2, trans, (trans != rocsparse_operation_none), rocsparse_status_not_implemented);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        6, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check sizes

    ROCSPARSE_CHECKARG_SIZE(3, mb);
    ROCSPARSE_CHECKARG_SIZE(4, nb);
    ROCSPARSE_CHECKARG_SIZE(5, nnzb);
    ROCSPARSE_CHECKARG_SIZE(10, block_dim);
    ROCSPARSE_CHECKARG(10, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

    // Quick return if possible
    if(mb == 0 || nb == 0 || nnzb == 0)
    {
        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(8, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsr_col_ind);

    if(descr->storage_mode == rocsparse_storage_mode_sorted)
    {
        if(block_dim == 1)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse_csrmv_analysis_template(handle,
                                                  trans,
                                                  rocsparse_csrmv_alg_adaptive,
                                                  mb,
                                                  nb,
                                                  nnzb,
                                                  descr,
                                                  bsr_val,
                                                  bsr_row_ptr,
                                                  bsr_col_ind,
                                                  info));
            return rocsparse_status_success;
        }
    }

    return rocsparse_status_success;
}

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
rocsparse_status rocsparse_bsrmv_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans,
                                                   J                         mb,
                                                   J                         nb,
                                                   I                         nnzb,
                                                   U                         alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const A*                  bsr_val,
                                                   const I*                  bsr_row_ptr,
                                                   const J*                  bsr_col_ind,
                                                   J                         block_dim,
                                                   const X*                  x,
                                                   U                         beta_device_host,
                                                   Y*                        y)
{
    if(trans != rocsparse_operation_none)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    //
    // block_dim == 1 is the CSR case
    //
    if(block_dim == 1)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrmv_stream_template_dispatch<T>(handle,
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
                                                                              false));
        return rocsparse_status_success;
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

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
rocsparse_status rocsparse_bsrmv_adaptive_template_dispatch(rocsparse_handle    handle,
                                                            rocsparse_direction dir,
                                                            rocsparse_operation trans,
                                                            J                   mb,
                                                            J                   nb,
                                                            I                   nnzb,
                                                            U                   alpha_device_host,
                                                            const rocsparse_mat_descr descr,
                                                            const A*                  bsr_val,
                                                            const I*                  bsr_row_ptr,
                                                            const J*                  bsr_col_ind,
                                                            J                         block_dim,
                                                            rocsparse_csrmv_info      info,
                                                            const X*                  x,
                                                            U  beta_device_host,
                                                            Y* y)
{
    if(trans != rocsparse_operation_none)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_requires_sorted_storage;
    }

    // block_dim == 1 is the CSR case
    if(block_dim == 1)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrmv_adaptive_template_dispatch<T>(handle,
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
                                                                                false));
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_bsrmv_template_dispatch<T>(handle,
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
                                                                   y));
    return rocsparse_status_success;
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse_bsrmv_template(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans,
                                          J                         mb,
                                          J                         nb,
                                          I                         nnzb,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const A*                  bsr_val,
                                          const I*                  bsr_row_ptr,
                                          const J*                  bsr_col_ind,
                                          J                         block_dim,
                                          rocsparse_mat_info        info,
                                          const X*                  x,
                                          const T*                  beta_device_host,
                                          Y*                        y)
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(7, descr);

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

    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_ENUM(2, trans);

    ROCSPARSE_CHECKARG(
        2, trans, (trans != rocsparse_operation_none), rocsparse_status_not_implemented);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        6, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    //
    // Check sizes
    //
    ROCSPARSE_CHECKARG_SIZE(3, mb);
    ROCSPARSE_CHECKARG_SIZE(4, nb);
    ROCSPARSE_CHECKARG_SIZE(5, nnzb);
    ROCSPARSE_CHECKARG_SIZE(11, block_dim);
    ROCSPARSE_CHECKARG(11, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

    //
    // Quick return if possible
    //
    if(mb == 0 || nb == 0 || nnzb == 0)
    {
        // matrix never accessed however still need to update y vector
        rocsparse_int ysize = (trans == rocsparse_operation_none) ? block_dim * mb : block_dim * nb;
        if(ysize > 0)
        {
            if(y == nullptr && beta_device_host == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((scale_array<256>),
                                                   dim3((ysize - 1) / 256 + 1),
                                                   dim3(256),
                                                   0,
                                                   handle->stream,
                                                   ysize,
                                                   y,
                                                   beta_device_host);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((scale_array<256>),
                                                   dim3((ysize - 1) / 256 + 1),
                                                   dim3(256),
                                                   0,
                                                   handle->stream,
                                                   ysize,
                                                   y,
                                                   *beta_device_host);
            }
        }

        return rocsparse_status_success;
    }

    //
    // Check pointer arguments
    //
    ROCSPARSE_CHECKARG_POINTER(6, alpha_device_host);
    ROCSPARSE_CHECKARG_POINTER(14, beta_device_host);

    //
    // Another quick return.
    //
    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(8, int64_t(nnzb) * block_dim * block_dim, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(9, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(10, nnzb, bsr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(13, x);
    ROCSPARSE_CHECKARG_POINTER(15, y);

    if(info == nullptr || info->csrmv_info == nullptr || trans != rocsparse_operation_none
       || descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        // If bsrmv info is not available, call bsrmv general
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_bsrmv_template_dispatch<T>(handle,
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
                                                                           y));
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_bsrmv_template_dispatch<T>(handle,
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
                                                                           y));
            return rocsparse_status_success;
        }
    }
    else
    {
        // If bsrmv info is available, call bsrmv adaptive
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse_bsrmv_adaptive_template_dispatch<T>(handle,
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
                                                              y));
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse_bsrmv_adaptive_template_dispatch<T>(handle,
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
                                                              y));
            return rocsparse_status_success;
        }
    }
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE)                                                            \
    template rocsparse_status rocsparse_bsrmv_analysis_template(rocsparse_handle          handle,   \
                                                                rocsparse_direction       dir,      \
                                                                rocsparse_operation       trans,    \
                                                                JTYPE                     mb,       \
                                                                JTYPE                     nb,       \
                                                                ITYPE                     nnzb,     \
                                                                const rocsparse_mat_descr descr,    \
                                                                const TTYPE*              bsr_val,  \
                                                                const ITYPE*       bsr_row_ptr,     \
                                                                const JTYPE*       bsr_col_ind,     \
                                                                JTYPE              block_dim,       \
                                                                rocsparse_mat_info info);           \
    template rocsparse_status rocsparse_bsrmv_template(rocsparse_handle          handle,            \
                                                       rocsparse_direction       dir,               \
                                                       rocsparse_operation       trans,             \
                                                       JTYPE                     mb,                \
                                                       JTYPE                     nb,                \
                                                       ITYPE                     nnzb,              \
                                                       const TTYPE*              alpha_device_host, \
                                                       const rocsparse_mat_descr descr,             \
                                                       const TTYPE*              bsr_val,           \
                                                       const ITYPE*              bsr_row_ptr,       \
                                                       const JTYPE*              bsr_col_ind,       \
                                                       JTYPE                     block_dim,         \
                                                       rocsparse_mat_info        info,              \
                                                       const TTYPE*              x,                 \
                                                       const TTYPE*              beta_device_host,  \
                                                       TTYPE*                    y);

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(float, int64_t, int64_t);
INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(double, int64_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE_MIXED_ANALYSIS(ITYPE, JTYPE, ATYPE)                                            \
    template rocsparse_status rocsparse_bsrmv_analysis_template(rocsparse_handle          handle,  \
                                                                rocsparse_direction       dir,     \
                                                                rocsparse_operation       trans,   \
                                                                JTYPE                     mb,      \
                                                                JTYPE                     nb,      \
                                                                ITYPE                     nnzb,    \
                                                                const rocsparse_mat_descr descr,   \
                                                                const ATYPE*              bsr_val, \
                                                                const ITYPE*       bsr_row_ptr,    \
                                                                const JTYPE*       bsr_col_ind,    \
                                                                JTYPE              block_dim,      \
                                                                rocsparse_mat_info info)

INSTANTIATE_MIXED_ANALYSIS(int32_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int64_t, int8_t);
#undef INSTANTIATE_MIXED_ANALYSIS

#define INSTANTIATE_MIXED(TTYPE, ITYPE, JTYPE, ATYPE, XTYPE, YTYPE)                                 \
    template rocsparse_status rocsparse_bsrmv_template(rocsparse_handle          handle,            \
                                                       rocsparse_direction       dir,               \
                                                       rocsparse_operation       trans,             \
                                                       JTYPE                     mb,                \
                                                       JTYPE                     nb,                \
                                                       ITYPE                     nnzb,              \
                                                       const TTYPE*              alpha_device_host, \
                                                       const rocsparse_mat_descr descr,             \
                                                       const ATYPE*              bsr_val,           \
                                                       const ITYPE*              bsr_row_ptr,       \
                                                       const JTYPE*              bsr_col_ind,       \
                                                       JTYPE                     block_dim,         \
                                                       rocsparse_mat_info        info,              \
                                                       const XTYPE*              x,                 \
                                                       const TTYPE*              beta_device_host,  \
                                                       YTYPE*                    y)

INSTANTIATE_MIXED(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(rocsparse_float_complex,
                  int32_t,
                  int32_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(rocsparse_float_complex,
                  int64_t,
                  int32_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(rocsparse_float_complex,
                  int64_t,
                  int64_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int32_t,
                  int32_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int32_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int64_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

INSTANTIATE_MIXED(double, int32_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int64_t, float, double, double);

INSTANTIATE_MIXED(rocsparse_double_complex,
                  int32_t,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int64_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

#undef INSTANTIATE_MIXED

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

// rocsparse_xbsrmv_analysis
#define C_IMPL(NAME, TYPE)                                                       \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,           \
                                     rocsparse_direction       dir,              \
                                     rocsparse_operation       trans,            \
                                     rocsparse_int             mb,               \
                                     rocsparse_int             nb,               \
                                     rocsparse_int             nnzb,             \
                                     const rocsparse_mat_descr descr,            \
                                     const TYPE*               bsr_val,          \
                                     const rocsparse_int*      bsr_row_ptr,      \
                                     const rocsparse_int*      bsr_col_ind,      \
                                     rocsparse_int             block_dim,        \
                                     rocsparse_mat_info        info)             \
    try                                                                          \
    {                                                                            \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_bsrmv_analysis_template(handle,      \
                                                                    dir,         \
                                                                    trans,       \
                                                                    mb,          \
                                                                    nb,          \
                                                                    nnzb,        \
                                                                    descr,       \
                                                                    bsr_val,     \
                                                                    bsr_row_ptr, \
                                                                    bsr_col_ind, \
                                                                    block_dim,   \
                                                                    info));      \
        return rocsparse_status_success;                                         \
    }                                                                            \
    catch(...)                                                                   \
    {                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                            \
    }

C_IMPL(rocsparse_sbsrmv_analysis, float);
C_IMPL(rocsparse_dbsrmv_analysis, double);
C_IMPL(rocsparse_cbsrmv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmv_analysis, rocsparse_double_complex);

#undef C_IMPL

// rocsparse_xbsrmv
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
    try                                                                     \
    {                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_bsrmv_template(handle,          \
                                                           dir,             \
                                                           trans,           \
                                                           mb,              \
                                                           nb,              \
                                                           nnzb,            \
                                                           alpha,           \
                                                           descr,           \
                                                           bsr_val,         \
                                                           bsr_row_ptr,     \
                                                           bsr_col_ind,     \
                                                           block_dim,       \
                                                           info,            \
                                                           x,               \
                                                           beta,            \
                                                           y));             \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

C_IMPL(rocsparse_sbsrmv, float);
C_IMPL(rocsparse_dbsrmv, double);
C_IMPL(rocsparse_cbsrmv, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmv, rocsparse_double_complex);

#undef C_IMPL

extern "C" rocsparse_status rocsparse_bsrmv_clear(rocsparse_handle handle, rocsparse_mat_info info)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrmv_clear(handle, info));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
