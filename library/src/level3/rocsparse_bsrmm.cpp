/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level3/rocsparse_bsrmm.h"
#include "rocsparse_bsrmm.hpp"
#include "rocsparse_csrmm.hpp"

#include "../level2/rocsparse_bsrmv.hpp"

#include "common.h"
#include "templates.h"
#include "utility.h"
#include <hip/hip_runtime.h>

namespace rocsparse
{
    template <typename T, typename U>
    rocsparse_status bsrmm_template_small(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_int             mb,
                                          rocsparse_int             n,
                                          rocsparse_int             kb,
                                          rocsparse_int             nnzb,
                                          U                         alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  bsr_val,
                                          const rocsparse_int*      bsr_row_ptr,
                                          const rocsparse_int*      bsr_col_ind,
                                          rocsparse_int             block_dim,
                                          const T*                  B,
                                          int64_t                   ldb,
                                          U                         beta,
                                          T*                        C,
                                          int64_t                   ldc);

    template <typename T, typename U>
    rocsparse_status bsrmm_template_large_ext(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              rocsparse_int             mb,
                                              rocsparse_int             n,
                                              rocsparse_int             kb,
                                              rocsparse_int             nnzb,
                                              U                         alpha,
                                              const rocsparse_mat_descr descr,
                                              const T*                  bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              const T*                  B,
                                              int64_t                   ldb,
                                              U                         beta,
                                              T*                        C,
                                              int64_t                   ldc);

    template <typename T, typename U>
    rocsparse_status bsrmm_template_general(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             mb,
                                            rocsparse_int             n,
                                            rocsparse_int             kb,
                                            rocsparse_int             nnzb,
                                            U                         alpha,
                                            const rocsparse_mat_descr descr,
                                            const T*                  bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             block_dim,
                                            const T*                  B,
                                            int64_t                   ldb,
                                            U                         beta,
                                            T*                        C,
                                            int64_t                   ldc);
}

template <typename T, typename U>
rocsparse_status rocsparse::bsrmm_template_dispatch(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             n,
                                                    rocsparse_int             kb,
                                                    rocsparse_int             nnzb,
                                                    U                         alpha,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  bsr_val,
                                                    const rocsparse_int*      bsr_row_ptr,
                                                    const rocsparse_int*      bsr_col_ind,
                                                    rocsparse_int             block_dim,
                                                    const T*                  B,
                                                    int64_t                   ldb,
                                                    U                         beta,
                                                    T*                        C,
                                                    int64_t                   ldc)
{

    // If n is only 1 and B are non-transposed, then call bsrmv
    if(n == 1)
    {
        if(trans_B == rocsparse_operation_none)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_template_dispatch<T>(handle,
                                                                            dir,
                                                                            trans_A,
                                                                            mb,
                                                                            kb,
                                                                            nnzb,
                                                                            alpha,
                                                                            descr,
                                                                            bsr_val,
                                                                            bsr_row_ptr,
                                                                            bsr_col_ind,
                                                                            block_dim,
                                                                            B,
                                                                            beta,
                                                                            C));
            return rocsparse_status_success;
        }
    }

    // If block dimension is one we can simply call csrmm
    if(block_dim == 1)
    {
        rocsparse_int nnz = nnzb * block_dim;
        rocsparse_int m   = mb * block_dim;
        rocsparse_int k   = kb * block_dim;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_template_dispatch<T>(handle,
                                                                        trans_A,
                                                                        trans_B,
                                                                        rocsparse_csrmm_alg_default,
                                                                        m,
                                                                        n,
                                                                        k,
                                                                        nnz,
                                                                        1,
                                                                        0,
                                                                        0,
                                                                        alpha,
                                                                        descr,
                                                                        bsr_val,
                                                                        bsr_row_ptr,
                                                                        bsr_col_ind,
                                                                        B,
                                                                        ldb,
                                                                        1,
                                                                        0,
                                                                        rocsparse_order_column,
                                                                        beta,
                                                                        C,
                                                                        ldc,
                                                                        1,
                                                                        0,
                                                                        rocsparse_order_column,
                                                                        nullptr,
                                                                        false));
        return rocsparse_status_success;
    }

    if(block_dim == 2)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_template_small(handle,
                                                                  dir,
                                                                  trans_A,
                                                                  trans_B,
                                                                  mb,
                                                                  n,
                                                                  kb,
                                                                  nnzb,
                                                                  alpha,
                                                                  descr,
                                                                  bsr_val,
                                                                  bsr_row_ptr,
                                                                  bsr_col_ind,
                                                                  block_dim,
                                                                  B,
                                                                  ldb,
                                                                  beta,
                                                                  C,
                                                                  ldc));
        return rocsparse_status_success;
    }

    if(block_dim <= 32)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_template_large_ext(handle,
                                                                      dir,
                                                                      trans_A,
                                                                      trans_B,
                                                                      mb,
                                                                      n,
                                                                      kb,
                                                                      nnzb,
                                                                      alpha,
                                                                      descr,
                                                                      bsr_val,
                                                                      bsr_row_ptr,
                                                                      bsr_col_ind,
                                                                      block_dim,
                                                                      B,
                                                                      ldb,
                                                                      beta,
                                                                      C,
                                                                      ldc));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_template_general(handle,
                                                                    dir,
                                                                    trans_A,
                                                                    trans_B,
                                                                    mb,
                                                                    n,
                                                                    kb,
                                                                    nnzb,
                                                                    alpha,
                                                                    descr,
                                                                    bsr_val,
                                                                    bsr_row_ptr,
                                                                    bsr_col_ind,

                                                                    block_dim,
                                                                    B,
                                                                    ldb,
                                                                    beta,
                                                                    C,
                                                                    ldc));
        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::bsrmm_quickreturn(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              rocsparse_int             mb,
                                              rocsparse_int             n,
                                              rocsparse_int             kb,
                                              rocsparse_int             nnzb,
                                              const T*                  alpha,
                                              const rocsparse_mat_descr descr,
                                              const T*                  bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              const T*                  B,
                                              int64_t                   ldb,
                                              const T*                  beta,
                                              T*                        C,
                                              int64_t                   ldc)
{
    if(mb == 0 || n == 0 || kb == 0)
    {
        // matrix never accessed however still need to update C matrix
        rocsparse_int m      = block_dim * mb;
        rocsparse_int C_size = m * n;
        if(C_size > 0)
        {
            if(C == nullptr && beta == nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
            }

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::scale_array_2d<256>),
                                                   dim3((C_size - 1) / 256 + 1),
                                                   dim3(256),
                                                   0,
                                                   handle->stream,
                                                   m,
                                                   n,
                                                   ldc,
                                                   0,
                                                   C,
                                                   beta,
                                                   rocsparse_order_column);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::scale_array_2d<256>),
                                                   dim3((C_size - 1) / 256 + 1),
                                                   dim3(256),
                                                   0,
                                                   handle->stream,
                                                   m,
                                                   n,
                                                   ldc,
                                                   0,
                                                   C,
                                                   *beta,
                                                   rocsparse_order_column);
            }
        }

        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

namespace rocsparse
{
    template <typename T>
    static rocsparse_status bsrmm_checkarg(rocsparse_handle          handle, //0
                                           rocsparse_direction       dir, //1
                                           rocsparse_operation       trans_A, //2
                                           rocsparse_operation       trans_B, //3
                                           rocsparse_int             mb, //4
                                           rocsparse_int             n, //5
                                           rocsparse_int             kb, //6
                                           rocsparse_int             nnzb, //7
                                           const T*                  alpha, //8
                                           const rocsparse_mat_descr descr, //9
                                           const T*                  bsr_val, //10
                                           const rocsparse_int*      bsr_row_ptr, //11
                                           const rocsparse_int*      bsr_col_ind, //12
                                           rocsparse_int             block_dim, //13
                                           const T*                  B, //14
                                           int64_t                   ldb, //15
                                           const T*                  beta, //16
                                           T*                        C, //17
                                           int64_t                   ldc) //18
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_ENUM(2, trans_A);

        ROCSPARSE_CHECKARG(
            2, trans_A, (trans_A != rocsparse_operation_none), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG_ENUM(3, trans_B);
        ROCSPARSE_CHECKARG(
            3,
            trans_B,
            (trans_B != rocsparse_operation_none && trans_B != rocsparse_operation_transpose),
            rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG_SIZE(4, mb);
        ROCSPARSE_CHECKARG_SIZE(5, n);
        ROCSPARSE_CHECKARG_SIZE(6, kb);
        ROCSPARSE_CHECKARG_SIZE(7, nnzb);

        ROCSPARSE_CHECKARG_POINTER(9, descr);
        ROCSPARSE_CHECKARG(9,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(9,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(10, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(11, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(12, nnzb, bsr_col_ind);

        ROCSPARSE_CHECKARG_SIZE(13, block_dim);
        ROCSPARSE_CHECKARG(13, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

        const rocsparse_status status = rocsparse::bsrmm_quickreturn(handle,
                                                                     dir,
                                                                     trans_A,
                                                                     trans_B,
                                                                     mb,
                                                                     n,
                                                                     kb,
                                                                     nnzb,
                                                                     alpha,
                                                                     descr,
                                                                     bsr_val,
                                                                     bsr_row_ptr,
                                                                     bsr_col_ind,
                                                                     block_dim,
                                                                     B,
                                                                     ldb,
                                                                     beta,
                                                                     C,
                                                                     ldc);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(8, alpha);
        ROCSPARSE_CHECKARG_POINTER(14, B);
        ROCSPARSE_CHECKARG_SIZE(15, ldb);
        ROCSPARSE_CHECKARG_POINTER(16, beta);
        ROCSPARSE_CHECKARG_POINTER(17, C);
        ROCSPARSE_CHECKARG_SIZE(18, ldc);

        static constexpr rocsparse_int s_one = static_cast<rocsparse_int>(1);

        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            ROCSPARSE_CHECKARG(
                18, ldc, (ldc < std::max(s_one, mb * block_dim)), rocsparse_status_invalid_size);

            // Check leading dimension of B
            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(15,
                                   ldb,
                                   (ldb < std::max(s_one, kb * block_dim)),
                                   rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    15, ldb, (ldb < std::max(s_one, n)), rocsparse_status_invalid_size);
                break;
            }
            }
            break;
        }

        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            ROCSPARSE_CHECKARG(
                18, ldc, (ldc < std::max(s_one, kb * block_dim)), rocsparse_status_invalid_size);

            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(15,
                                   ldb,
                                   (ldb < std::max(s_one, mb * block_dim)),
                                   rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    15, ldb, (ldb < std::max(s_one, n)), rocsparse_status_invalid_size);
                break;
            }
            }
            break;
        }
        }
        return rocsparse_status_continue;
    }
}

template <typename T>
rocsparse_status rocsparse::bsrmm_core(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             mb,
                                       rocsparse_int             n,
                                       rocsparse_int             kb,
                                       rocsparse_int             nnzb,
                                       const T*                  alpha,
                                       const rocsparse_mat_descr descr,
                                       const T*                  bsr_val,
                                       const rocsparse_int*      bsr_row_ptr,
                                       const rocsparse_int*      bsr_col_ind,
                                       rocsparse_int             block_dim,
                                       const T*                  B,
                                       int64_t                   ldb,
                                       const T*                  beta,
                                       T*                        C,
                                       int64_t                   ldc)
{
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_template_dispatch(handle,
                                                                     dir,
                                                                     trans_A,
                                                                     trans_B,
                                                                     mb,
                                                                     n,
                                                                     kb,
                                                                     nnzb,
                                                                     alpha,
                                                                     descr,
                                                                     bsr_val,
                                                                     bsr_row_ptr,
                                                                     bsr_col_ind,
                                                                     block_dim,
                                                                     B,
                                                                     ldb,
                                                                     beta,
                                                                     C,
                                                                     ldc));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_template_dispatch(handle,
                                                                     dir,
                                                                     trans_A,
                                                                     trans_B,
                                                                     mb,
                                                                     n,
                                                                     kb,
                                                                     nnzb,
                                                                     *alpha,
                                                                     descr,
                                                                     bsr_val,
                                                                     bsr_row_ptr,
                                                                     bsr_col_ind,
                                                                     block_dim,
                                                                     B,
                                                                     ldb,
                                                                     *beta,
                                                                     C,
                                                                     ldc));
        return rocsparse_status_success;
    }
}

namespace rocsparse
{
    template <typename T>
    rocsparse_status bsrmm_impl(rocsparse_handle          handle,
                                rocsparse_direction       dir,
                                rocsparse_operation       trans_A,
                                rocsparse_operation       trans_B,
                                rocsparse_int             mb,
                                rocsparse_int             n,
                                rocsparse_int             kb,
                                rocsparse_int             nnzb,
                                const T*                  alpha,
                                const rocsparse_mat_descr descr,
                                const T*                  bsr_val,
                                const rocsparse_int*      bsr_row_ptr,
                                const rocsparse_int*      bsr_col_ind,
                                rocsparse_int             block_dim,
                                const T*                  B,
                                rocsparse_int             ldb,
                                const T*                  beta,
                                T*                        C,
                                rocsparse_int             ldc)
    {

        log_trace(handle,
                  replaceX<T>("rocsparse_Xbsrmm"),
                  dir,
                  trans_A,
                  trans_B,
                  mb,
                  n,
                  kb,
                  nnzb,
                  LOG_TRACE_SCALAR_VALUE(handle, alpha),
                  (const void*&)descr,
                  (const void*&)bsr_val,
                  (const void*&)bsr_row_ptr,
                  (const void*&)bsr_col_ind,
                  block_dim,
                  (const void*&)B,
                  ldb,
                  LOG_TRACE_SCALAR_VALUE(handle, beta),
                  (const void*&)C,
                  ldc);

        const rocsparse_status status = rocsparse::bsrmm_checkarg(handle,
                                                                  dir,
                                                                  trans_A,
                                                                  trans_B,
                                                                  mb,
                                                                  n,
                                                                  kb,
                                                                  nnzb,
                                                                  alpha,
                                                                  descr,
                                                                  bsr_val,
                                                                  bsr_row_ptr,
                                                                  bsr_col_ind,
                                                                  block_dim,
                                                                  B,
                                                                  ldb,
                                                                  beta,
                                                                  C,
                                                                  ldc);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_core(handle,
                                                        dir,
                                                        trans_A,
                                                        trans_B,
                                                        mb,
                                                        n,
                                                        kb,
                                                        nnzb,
                                                        alpha,
                                                        descr,
                                                        bsr_val,
                                                        bsr_row_ptr,
                                                        bsr_col_ind,
                                                        block_dim,
                                                        B,
                                                        ldb,
                                                        beta,
                                                        C,
                                                        ldc));
        return rocsparse_status_success;
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
                                     rocsparse_operation       trans_B,     \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             n,           \
                                     rocsparse_int             kb,          \
                                     rocsparse_int             nnzb,        \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     const TYPE*               B,           \
                                     rocsparse_int             ldb,         \
                                     const TYPE*               beta,        \
                                     TYPE*                     C,           \
                                     rocsparse_int             ldc)         \
    try                                                                     \
    {                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_impl<TYPE>(handle,       \
                                                              dir,          \
                                                              trans_A,      \
                                                              trans_B,      \
                                                              mb,           \
                                                              n,            \
                                                              kb,           \
                                                              nnzb,         \
                                                              alpha,        \
                                                              descr,        \
                                                              bsr_val,      \
                                                              bsr_row_ptr,  \
                                                              bsr_col_ind,  \
                                                              block_dim,    \
                                                              B,            \
                                                              ldb,          \
                                                              beta,         \
                                                              C,            \
                                                              ldc));        \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

C_IMPL(rocsparse_sbsrmm, float);
C_IMPL(rocsparse_dbsrmm, double);
C_IMPL(rocsparse_cbsrmm, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmm, rocsparse_double_complex);
#undef C_IMPL
