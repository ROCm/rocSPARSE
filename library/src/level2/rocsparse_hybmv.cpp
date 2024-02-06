/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_hybmv.h"
#include "control.h"
#include "rocsparse_coomv.hpp"
#include "rocsparse_ellmv.hpp"
#include "rocsparse_hybmv.hpp"
#include "utility.h"

template <typename T>
rocsparse_status rocsparse::hybmv_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           const T*                  alpha_device_host,
                                           const rocsparse_mat_descr descr,
                                           const rocsparse_hyb_mat   hyb,
                                           const T*                  x,
                                           const T*                  beta_device_host,
                                           T*                        y)
{
    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(3, descr);
    ROCSPARSE_CHECKARG_POINTER(4, hyb);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xhybmv"),
                         trans,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                         (const void*&)descr,
                         (const void*&)hyb,
                         (const void*&)x,
                         LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                         (const void*&)y);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        3, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    // Check matrix sorting mode

    ROCSPARSE_CHECKARG(3,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ENUM(1, trans);

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(2, alpha_device_host);
    ROCSPARSE_CHECKARG_POINTER(6, beta_device_host);

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    ROCSPARSE_CHECKARG_POINTER(5, x);
    ROCSPARSE_CHECKARG_POINTER(7, y);

    // LCOV_EXCL_START
    // Check sizes
    ROCSPARSE_CHECKARG(4,
                       hyb,
                       (hyb->m < 0 || hyb->n < 0 || hyb->ell_nnz + hyb->coo_nnz < 0),
                       rocsparse_status_invalid_size);

    // Check ELL-HYB structure
    ROCSPARSE_CHECKARG(
        4, hyb, ((hyb->ell_nnz > 0) && (hyb->ell_width < 0)), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(
        4,
        hyb,
        ((hyb->ell_nnz > 0) && (hyb->ell_col_ind == nullptr || hyb->ell_val == nullptr)),
        rocsparse_status_invalid_pointer);
    // Check COO-HYB structure
    ROCSPARSE_CHECKARG(4,
                       hyb,
                       ((hyb->coo_nnz > 0)
                        && (hyb->coo_row_ind == nullptr || hyb->coo_col_ind == nullptr
                            || hyb->coo_val == nullptr)),
                       rocsparse_status_invalid_pointer);

    // LCOV_EXCL_STOP

    // Quick return if possible
    if(hyb->m == 0 || hyb->n == 0 || hyb->ell_nnz + hyb->coo_nnz == 0)
    {
        return rocsparse_status_success;
    }

    // ELL part
    if(hyb->ell_nnz > 0)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellmv_template(handle,
                                                            trans,
                                                            hyb->m,
                                                            hyb->n,
                                                            alpha_device_host,
                                                            descr,
                                                            (T*)hyb->ell_val,
                                                            hyb->ell_col_ind,
                                                            hyb->ell_width,
                                                            x,
                                                            beta_device_host,
                                                            y));
    }

    // COO part
    if(hyb->coo_nnz > 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            // Beta is applied by ELL part, IF ell_nnz > 0
            if(hyb->ell_nnz > 0)
            {
                T* coo_beta = nullptr;
                rocsparse::one(handle, &coo_beta);

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomv_template(handle,
                                                                    trans,
                                                                    rocsparse_coomv_alg_segmented,
                                                                    hyb->m,
                                                                    hyb->n,
                                                                    hyb->coo_nnz,
                                                                    alpha_device_host,
                                                                    descr,
                                                                    (T*)hyb->coo_val,
                                                                    hyb->coo_row_ind,
                                                                    hyb->coo_col_ind,
                                                                    x,
                                                                    coo_beta,
                                                                    y));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomv_template(handle,
                                                                    trans,
                                                                    rocsparse_coomv_alg_segmented,
                                                                    hyb->m,
                                                                    hyb->n,
                                                                    hyb->coo_nnz,
                                                                    alpha_device_host,
                                                                    descr,
                                                                    (T*)hyb->coo_val,
                                                                    hyb->coo_row_ind,
                                                                    hyb->coo_col_ind,
                                                                    x,
                                                                    beta_device_host,
                                                                    y));
            }
        }
        else
        {
            // Beta is applied by ELL part, IF ell_nnz > 0
            T coo_beta = (hyb->ell_nnz > 0) ? static_cast<T>(1) : *beta_device_host;

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomv_template(handle,
                                                                trans,
                                                                rocsparse_coomv_alg_segmented,
                                                                hyb->m,
                                                                hyb->n,
                                                                hyb->coo_nnz,
                                                                alpha_device_host,
                                                                descr,
                                                                (T*)hyb->coo_val,
                                                                hyb->coo_row_ind,
                                                                hyb->coo_col_ind,
                                                                x,
                                                                &coo_beta,
                                                                y));
        }
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                            \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                \
                                     rocsparse_operation       trans,                 \
                                     const TYPE*               alpha,                 \
                                     const rocsparse_mat_descr descr,                 \
                                     const rocsparse_hyb_mat   hyb,                   \
                                     const TYPE*               x,                     \
                                     const TYPE*               beta,                  \
                                     TYPE*                     y)                     \
    try                                                                               \
    {                                                                                 \
        RETURN_IF_ROCSPARSE_ERROR(                                                    \
            rocsparse::hybmv_template(handle, trans, alpha, descr, hyb, x, beta, y)); \
        return rocsparse_status_success;                                              \
    }                                                                                 \
    catch(...)                                                                        \
    {                                                                                 \
        RETURN_ROCSPARSE_EXCEPTION();                                                 \
    }

C_IMPL(rocsparse_shybmv, float);
C_IMPL(rocsparse_dhybmv, double);
C_IMPL(rocsparse_chybmv, rocsparse_float_complex);
C_IMPL(rocsparse_zhybmv, rocsparse_double_complex);
#undef C_IMPL
