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

#pragma once
#ifndef TEMPLATES_H
#define TEMPLATES_H

#include "rocsparse.h"

//
// For reusing without recompiling.
// e.g. call rocsparse_bsrmv rather than rocsparse_bsrmv_template.
//

// bsrmv
template <typename T>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             nnzb,
                                        const T*                  alpha,
                                        const rocsparse_mat_descr descr,
                                        const T*                  bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             bsr_dim,
                                        const T*                  x,
                                        const T*                  beta,
                                        T*                        y);

template <>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             nnzb,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             bsr_dim,
                                        const float*              x,
                                        const float*              beta,
                                        float*                    y)
{
    return rocsparse_sbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            x,
                            beta,
                            y);
}

template <>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             nnzb,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             bsr_dim,
                                        const double*             x,
                                        const double*             beta,
                                        double*                   y)
{
    return rocsparse_dbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            x,
                            beta,
                            y);
}

template <>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle               handle,
                                        rocsparse_direction            dir,
                                        rocsparse_operation            trans,
                                        rocsparse_int                  mb,
                                        rocsparse_int                  nb,
                                        rocsparse_int                  nnzb,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* bsr_val,
                                        const rocsparse_int*           bsr_row_ptr,
                                        const rocsparse_int*           bsr_col_ind,
                                        rocsparse_int                  bsr_dim,
                                        const rocsparse_float_complex* x,
                                        const rocsparse_float_complex* beta,
                                        rocsparse_float_complex*       y)
{
    return rocsparse_cbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            x,
                            beta,
                            y);
}

template <>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle                handle,
                                        rocsparse_direction             dir,
                                        rocsparse_operation             trans,
                                        rocsparse_int                   mb,
                                        rocsparse_int                   nb,
                                        rocsparse_int                   nnzb,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* bsr_val,
                                        const rocsparse_int*            bsr_row_ptr,
                                        const rocsparse_int*            bsr_col_ind,
                                        rocsparse_int                   bsr_dim,
                                        const rocsparse_double_complex* x,
                                        const rocsparse_double_complex* beta,
                                        rocsparse_double_complex*       y)
{
    return rocsparse_zbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            x,
                            beta,
                            y);
}

// csrmm
template <typename T>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_int             m,
                                        rocsparse_int             n,
                                        rocsparse_int             k,
                                        rocsparse_int             nnz,
                                        const T*                  alpha,
                                        const rocsparse_mat_descr descr,
                                        const T*                  csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        const T*                  B,
                                        rocsparse_int             ldb,
                                        const T*                  beta,
                                        T*                        C,
                                        rocsparse_int             ldc);

template <>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_int             m,
                                        rocsparse_int             n,
                                        rocsparse_int             k,
                                        rocsparse_int             nnz,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        const float*              B,
                                        rocsparse_int             ldb,
                                        const float*              beta,
                                        float*                    C,
                                        rocsparse_int             ldc)
{
    return rocsparse_scsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_int             m,
                                        rocsparse_int             n,
                                        rocsparse_int             k,
                                        rocsparse_int             nnz,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        const double*             B,
                                        rocsparse_int             ldb,
                                        const double*             beta,
                                        double*                   C,
                                        rocsparse_int             ldc)
{
    return rocsparse_dcsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle               handle,
                                        rocsparse_operation            trans_A,
                                        rocsparse_operation            trans_B,
                                        rocsparse_int                  m,
                                        rocsparse_int                  n,
                                        rocsparse_int                  k,
                                        rocsparse_int                  nnz,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* csr_val,
                                        const rocsparse_int*           csr_row_ptr,
                                        const rocsparse_int*           csr_col_ind,
                                        const rocsparse_float_complex* B,
                                        rocsparse_int                  ldb,
                                        const rocsparse_float_complex* beta,
                                        rocsparse_float_complex*       C,
                                        rocsparse_int                  ldc)
{
    return rocsparse_ccsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle                handle,
                                        rocsparse_operation             trans_A,
                                        rocsparse_operation             trans_B,
                                        rocsparse_int                   m,
                                        rocsparse_int                   n,
                                        rocsparse_int                   k,
                                        rocsparse_int                   nnz,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* csr_val,
                                        const rocsparse_int*            csr_row_ptr,
                                        const rocsparse_int*            csr_col_ind,
                                        const rocsparse_double_complex* B,
                                        rocsparse_int                   ldb,
                                        const rocsparse_double_complex* beta,
                                        rocsparse_double_complex*       C,
                                        rocsparse_int                   ldc)
{
    return rocsparse_zcsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

#endif
