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

#include "rocsparse_dense2coo.hpp"

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status rocsparse_sdense2coo(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              A,
                                                 rocsparse_int             ld,
                                                 const rocsparse_int*      nnz_per_rows,
                                                 float*                    coo_val,
                                                 rocsparse_int*            coo_row_ind,
                                                 rocsparse_int*            coo_col_ind)
{
    return rocsparse_dense2coo_template(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
}

extern "C" rocsparse_status rocsparse_ddense2coo(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             A,
                                                 rocsparse_int             ld,
                                                 const rocsparse_int*      nnz_per_rows,
                                                 double*                   coo_val,
                                                 rocsparse_int*            coo_row_ind,
                                                 rocsparse_int*            coo_col_ind)
{
    return rocsparse_dense2coo_template(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
}

extern "C" rocsparse_status rocsparse_cdense2coo(rocsparse_handle               handle,
                                                 rocsparse_int                  m,
                                                 rocsparse_int                  n,
                                                 const rocsparse_mat_descr      descr,
                                                 const rocsparse_float_complex* A,
                                                 rocsparse_int                  ld,
                                                 const rocsparse_int*           nnz_per_rows,
                                                 rocsparse_float_complex*       coo_val,
                                                 rocsparse_int*                 coo_row_ind,
                                                 rocsparse_int*                 coo_col_ind)
{
    return rocsparse_dense2coo_template(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
}

extern "C" rocsparse_status rocsparse_zdense2coo(rocsparse_handle                handle,
                                                 rocsparse_int                   m,
                                                 rocsparse_int                   n,
                                                 const rocsparse_mat_descr       descr,
                                                 const rocsparse_double_complex* A,
                                                 rocsparse_int                   ld,
                                                 const rocsparse_int*            nnz_per_rows,
                                                 rocsparse_double_complex*       coo_val,
                                                 rocsparse_int*                  coo_row_ind,
                                                 rocsparse_int*                  coo_col_ind)
{
    return rocsparse_dense2coo_template(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
}
