/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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

#include "rocsparse_bsrxmv.hpp"
#include "rocsparse_bsrxmv_spzl.hpp"

template <typename T, typename U>
rocsparse_status rocsparse_bsrxmv_template_dispatch(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_operation       trans,
                                                    rocsparse_int             size_of_mask,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nb,
                                                    rocsparse_int             nnzb,
                                                    U                         alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  bsr_val,
                                                    const rocsparse_int*      bsr_mask_ptr,
                                                    const rocsparse_int*      bsr_row_ptr,
                                                    const rocsparse_int*      bsr_end_ptr,
                                                    const rocsparse_int*      bsr_col_ind,
                                                    rocsparse_int             block_dim,
                                                    const T*                  x,
                                                    U                         beta_device_host,
                                                    T*                        y)
{
    // LCOV_EXCL_START
    // Run different bsrxmv kernels
    if(handle->wavefront_size == 32)
    {
        bsrxmvn_general(handle,
                        dir,
                        mb,
                        alpha_device_host,
                        size_of_mask,
                        bsr_mask_ptr,
                        bsr_row_ptr,
                        bsr_end_ptr,
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
        bsrxmvn_2x2(handle,
                    dir,
                    mb,
                    nnzb,
                    alpha_device_host,
                    size_of_mask,
                    bsr_mask_ptr,
                    bsr_row_ptr,
                    bsr_end_ptr,
                    bsr_col_ind,
                    bsr_val,
                    x,
                    beta_device_host,
                    y,
                    descr->base);
    }
    else if(block_dim == 3)
    {
        bsrxmvn_3x3(handle,
                    dir,
                    mb,
                    nnzb,
                    alpha_device_host,
                    size_of_mask,
                    bsr_mask_ptr,
                    bsr_row_ptr,
                    bsr_end_ptr,
                    bsr_col_ind,
                    bsr_val,
                    x,
                    beta_device_host,
                    y,
                    descr->base);
    }
    else if(block_dim == 4)
    {
        bsrxmvn_4x4(handle,
                    dir,
                    mb,
                    nnzb,
                    alpha_device_host,
                    size_of_mask,
                    bsr_mask_ptr,
                    bsr_row_ptr,
                    bsr_end_ptr,
                    bsr_col_ind,
                    bsr_val,
                    x,
                    beta_device_host,
                    y,
                    descr->base);
    }
    else if(block_dim == 5)
    {
        bsrxmvn_5x5(handle,
                    dir,
                    mb,
                    nnzb,
                    alpha_device_host,
                    size_of_mask,
                    bsr_mask_ptr,
                    bsr_row_ptr,
                    bsr_end_ptr,
                    bsr_col_ind,
                    bsr_val,
                    x,
                    beta_device_host,
                    y,
                    descr->base);
    }
    else if(block_dim == 8)
    {
        bsrxmvn_8x8(handle,
                    dir,
                    mb,
                    nnzb,
                    alpha_device_host,
                    size_of_mask,
                    bsr_mask_ptr,
                    bsr_row_ptr,
                    bsr_end_ptr,
                    bsr_col_ind,
                    bsr_val,
                    x,
                    beta_device_host,
                    y,
                    descr->base);
    }
    else if(block_dim == 16)
    {
        bsrxmvn_16x16(handle,
                      dir,
                      mb,
                      nnzb,
                      alpha_device_host,
                      size_of_mask,
                      bsr_mask_ptr,
                      bsr_row_ptr,
                      bsr_end_ptr,
                      bsr_col_ind,
                      bsr_val,
                      x,
                      beta_device_host,
                      y,
                      descr->base);
    }
    else if(block_dim > 16 && block_dim <= 32)
    {
        bsrxmvn_17_32(handle,
                      dir,
                      mb,
                      nnzb,
                      alpha_device_host,
                      size_of_mask,
                      bsr_mask_ptr,
                      bsr_row_ptr,
                      bsr_end_ptr,
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
        bsrxmvn_general(handle,
                        dir,
                        mb,
                        alpha_device_host,
                        size_of_mask,
                        bsr_mask_ptr,
                        bsr_row_ptr,
                        bsr_end_ptr,
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

template <typename T>
rocsparse_status rocsparse_bsrxmv_template(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans,
                                           rocsparse_int             size_of_mask,
                                           rocsparse_int             mb,
                                           rocsparse_int             nb,
                                           rocsparse_int             nnzb,
                                           const T*                  alpha_device_host,
                                           const rocsparse_mat_descr descr,
                                           const T*                  bsr_val,
                                           const rocsparse_int*      bsr_mask_ptr,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_end_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             block_dim,
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
              replaceX<T>("rocsparse_Xbsrxmv"),
              dir,
              trans,
              size_of_mask,
              mb,
              nb,
              nnzb,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_mask_ptr,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_end_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              (const void*&)x,
              LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
              (const void*&)y);

    log_bench(handle,
              "./rocsparse-bench -f bsrxmv -r",
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

    if(block_dim == 1)
    {
        return rocsparse_status_not_implemented;
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

    //
    // Check sizes
    //
    if(mb < 0 || nb < 0 || nnzb < 0 || block_dim < 0 || size_of_mask < 0)
    {
        return rocsparse_status_invalid_size;
    }

    //
    // Quick return if possible
    //
    if(mb == 0 || nb == 0 || block_dim == 0 || size_of_mask == 0)
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
    if(bsr_mask_ptr == nullptr || bsr_row_ptr == nullptr || bsr_end_ptr == nullptr || x == nullptr
       || y == nullptr)
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
        return rocsparse_bsrxmv_template_dispatch(handle,
                                                  dir,
                                                  trans,
                                                  size_of_mask,
                                                  mb,
                                                  nb,
                                                  nnzb,
                                                  alpha_device_host,
                                                  descr,
                                                  bsr_val,
                                                  bsr_mask_ptr,
                                                  bsr_row_ptr,
                                                  bsr_end_ptr,
                                                  bsr_col_ind,
                                                  block_dim,
                                                  x,
                                                  beta_device_host,
                                                  y);
    }
    else
    {
        return rocsparse_bsrxmv_template_dispatch(handle,
                                                  dir,
                                                  trans,
                                                  size_of_mask,
                                                  mb,
                                                  nb,
                                                  nnzb,
                                                  *alpha_device_host,
                                                  descr,
                                                  bsr_val,
                                                  bsr_mask_ptr,
                                                  bsr_row_ptr,
                                                  bsr_end_ptr,
                                                  bsr_col_ind,
                                                  block_dim,
                                                  x,
                                                  *beta_device_host,
                                                  y);
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,       \
                                     rocsparse_direction       dir,          \
                                     rocsparse_operation       trans,        \
                                     rocsparse_int             size_of_mask, \
                                     rocsparse_int             mb,           \
                                     rocsparse_int             nb,           \
                                     rocsparse_int             nnzb,         \
                                     const TYPE*               alpha,        \
                                     const rocsparse_mat_descr descr,        \
                                     const TYPE*               bsr_val,      \
                                     const rocsparse_int*      bsr_mask_ptr, \
                                     const rocsparse_int*      bsr_row_ptr,  \
                                     const rocsparse_int*      bsr_end_ptr,  \
                                     const rocsparse_int*      bsr_col_ind,  \
                                     rocsparse_int             block_dim,    \
                                     const TYPE*               x,            \
                                     const TYPE*               beta,         \
                                     TYPE*                     y)            \
    {                                                                        \
        return rocsparse_bsrxmv_template(handle,                             \
                                         dir,                                \
                                         trans,                              \
                                         size_of_mask,                       \
                                         mb,                                 \
                                         nb,                                 \
                                         nnzb,                               \
                                         alpha,                              \
                                         descr,                              \
                                         bsr_val,                            \
                                         bsr_mask_ptr,                       \
                                         bsr_row_ptr,                        \
                                         bsr_end_ptr,                        \
                                         bsr_col_ind,                        \
                                         block_dim,                          \
                                         x,                                  \
                                         beta,                               \
                                         y);                                 \
    }

C_IMPL(rocsparse_sbsrxmv, float);
C_IMPL(rocsparse_dbsrxmv, double);
C_IMPL(rocsparse_cbsrxmv, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrxmv, rocsparse_double_complex);

#undef C_IMPL
