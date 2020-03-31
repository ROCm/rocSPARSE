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
#include "rocsparse.h"

#include "rocsparse_bsr2csr.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sbsr2csr(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nb,
                                               const rocsparse_mat_descr bsr_descr,
                                               const float*              bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               const rocsparse_mat_descr csr_descr,
                                               float*                    csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind)
{
    return rocsparse_bsr2csr_template(handle,
                                      dir,
                                      mb,
                                      nb,
                                      bsr_descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      block_dim,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind);
}

extern "C" rocsparse_status rocsparse_dbsr2csr(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nb,
                                               const rocsparse_mat_descr bsr_descr,
                                               const double*             bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               const rocsparse_mat_descr csr_descr,
                                               double*                   csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind)
{
    return rocsparse_bsr2csr_template(handle,
                                      dir,
                                      mb,
                                      nb,
                                      bsr_descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      block_dim,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind);
}

extern "C" rocsparse_status rocsparse_cbsr2csr(rocsparse_handle               handle,
                                               rocsparse_direction            dir,
                                               rocsparse_int                  mb,
                                               rocsparse_int                  nb,
                                               const rocsparse_mat_descr      bsr_descr,
                                               const rocsparse_float_complex* bsr_val,
                                               const rocsparse_int*           bsr_row_ptr,
                                               const rocsparse_int*           bsr_col_ind,
                                               rocsparse_int                  block_dim,
                                               const rocsparse_mat_descr      csr_descr,
                                               rocsparse_float_complex*       csr_val,
                                               rocsparse_int*                 csr_row_ptr,
                                               rocsparse_int*                 csr_col_ind)
{
    return rocsparse_bsr2csr_template(handle,
                                      dir,
                                      mb,
                                      nb,
                                      bsr_descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      block_dim,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind);
}

extern "C" rocsparse_status rocsparse_zbsr2csr(rocsparse_handle                handle,
                                               rocsparse_direction             dir,
                                               rocsparse_int                   mb,
                                               rocsparse_int                   nb,
                                               const rocsparse_mat_descr       bsr_descr,
                                               const rocsparse_double_complex* bsr_val,
                                               const rocsparse_int*            bsr_row_ptr,
                                               const rocsparse_int*            bsr_col_ind,
                                               rocsparse_int                   block_dim,
                                               const rocsparse_mat_descr       csr_descr,
                                               rocsparse_double_complex*       csr_val,
                                               rocsparse_int*                  csr_row_ptr,
                                               rocsparse_int*                  csr_col_ind)
{
    return rocsparse_bsr2csr_template(handle,
                                      dir,
                                      mb,
                                      nb,
                                      bsr_descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      block_dim,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind);
}
