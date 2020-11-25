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

#include "rocsparse_bsrmm.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_sbsrmm(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             rocsparse_int             mb,
                                             rocsparse_int             n,
                                             rocsparse_int             kb,
                                             rocsparse_int             nnzb,
                                             const float*              alpha,
                                             const rocsparse_mat_descr descr,
                                             const float*              bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             block_dim,
                                             const float*              B,
                                             rocsparse_int             ldb,
                                             const float*              beta,
                                             float*                    C,
                                             rocsparse_int             ldc)
{
    return rocsparse_bsrmm_template(handle,
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
}

extern "C" rocsparse_status rocsparse_dbsrmm(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             rocsparse_int             mb,
                                             rocsparse_int             n,
                                             rocsparse_int             kb,
                                             rocsparse_int             nnzb,
                                             const double*             alpha,
                                             const rocsparse_mat_descr descr,
                                             const double*             bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             block_dim,
                                             const double*             B,
                                             rocsparse_int             ldb,
                                             const double*             beta,
                                             double*                   C,
                                             rocsparse_int             ldc)
{
    return rocsparse_bsrmm_template(handle,
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
}

extern "C" rocsparse_status rocsparse_cbsrmm(rocsparse_handle               handle,
                                             rocsparse_direction            dir,
                                             rocsparse_operation            trans_A,
                                             rocsparse_operation            trans_B,
                                             rocsparse_int                  mb,
                                             rocsparse_int                  n,
                                             rocsparse_int                  kb,
                                             rocsparse_int                  nnzb,
                                             const rocsparse_float_complex* alpha,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* bsr_val,
                                             const rocsparse_int*           bsr_row_ptr,
                                             const rocsparse_int*           bsr_col_ind,
                                             rocsparse_int                  block_dim,
                                             const rocsparse_float_complex* B,
                                             rocsparse_int                  ldb,
                                             const rocsparse_float_complex* beta,
                                             rocsparse_float_complex*       C,
                                             rocsparse_int                  ldc)
{
    return rocsparse_bsrmm_template(handle,
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
}

extern "C" rocsparse_status rocsparse_zbsrmm(rocsparse_handle                handle,
                                             rocsparse_direction             dir,
                                             rocsparse_operation             trans_A,
                                             rocsparse_operation             trans_B,
                                             rocsparse_int                   mb,
                                             rocsparse_int                   n,
                                             rocsparse_int                   kb,
                                             rocsparse_int                   nnzb,
                                             const rocsparse_double_complex* alpha,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* bsr_val,
                                             const rocsparse_int*            bsr_row_ptr,
                                             const rocsparse_int*            bsr_col_ind,
                                             rocsparse_int                   block_dim,
                                             const rocsparse_double_complex* B,
                                             rocsparse_int                   ldb,
                                             const rocsparse_double_complex* beta,
                                             rocsparse_double_complex*       C,
                                             rocsparse_int                   ldc)
{
    return rocsparse_bsrmm_template(handle,
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
}
