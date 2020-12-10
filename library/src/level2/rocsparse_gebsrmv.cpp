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

#include "rocsparse_gebsrmv.hpp"
#include "rocsparse.h"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sgebsrmv(rocsparse_handle          handle,
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
                                               rocsparse_int             row_block_dim,
                                               rocsparse_int             col_block_dim,
                                               const float*              x,
                                               const float*              beta,
                                               float*                    y)
{
    return rocsparse_gebsrmv_template(handle,
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
                                      row_block_dim,
                                      col_block_dim,
                                      x,
                                      beta,
                                      y);
}

extern "C" rocsparse_status rocsparse_dgebsrmv(rocsparse_handle          handle,
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
                                               rocsparse_int             row_block_dim,
                                               rocsparse_int             col_block_dim,
                                               const double*             x,
                                               const double*             beta,
                                               double*                   y)
{
    return rocsparse_gebsrmv_template(handle,
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
                                      row_block_dim,
                                      col_block_dim,
                                      x,
                                      beta,
                                      y);
}

extern "C" rocsparse_status rocsparse_cgebsrmv(rocsparse_handle               handle,
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
                                               rocsparse_int                  row_block_dim,
                                               rocsparse_int                  col_block_dim,
                                               const rocsparse_float_complex* x,
                                               const rocsparse_float_complex* beta,
                                               rocsparse_float_complex*       y)
{
    return rocsparse_gebsrmv_template(handle,
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
                                      row_block_dim,
                                      col_block_dim,
                                      x,
                                      beta,
                                      y);
}

extern "C" rocsparse_status rocsparse_zgebsrmv(rocsparse_handle                handle,
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
                                               rocsparse_int                   row_block_dim,
                                               rocsparse_int                   col_block_dim,
                                               const rocsparse_double_complex* x,
                                               const rocsparse_double_complex* beta,
                                               rocsparse_double_complex*       y)
{
    return rocsparse_gebsrmv_template(handle,
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
                                      row_block_dim,
                                      col_block_dim,
                                      x,
                                      beta,
                                      y);
}
