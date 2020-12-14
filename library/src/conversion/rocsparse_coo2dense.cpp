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

#include "rocsparse_coo2dense.hpp"

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status rocsparse_scoo2dense(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 rocsparse_int             nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              coo_val,
                                                 const rocsparse_int*      coo_row_ind,
                                                 const rocsparse_int*      coo_col_ind,
                                                 float*                    A,
                                                 rocsparse_int             ld)
{
    return rocsparse_coo2dense_template(
        handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld);
}

extern "C" rocsparse_status rocsparse_dcoo2dense(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 rocsparse_int             nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             coo_val,
                                                 const rocsparse_int*      coo_row_ind,
                                                 const rocsparse_int*      coo_col_ind,
                                                 double*                   A,
                                                 rocsparse_int             ld)
{
    return rocsparse_coo2dense_template(
        handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld);
}

extern "C" rocsparse_status rocsparse_ccoo2dense(rocsparse_handle               handle,
                                                 rocsparse_int                  m,
                                                 rocsparse_int                  n,
                                                 rocsparse_int                  nnz,
                                                 const rocsparse_mat_descr      descr,
                                                 const rocsparse_float_complex* coo_val,
                                                 const rocsparse_int*           coo_row_ind,
                                                 const rocsparse_int*           coo_col_ind,
                                                 rocsparse_float_complex*       A,
                                                 rocsparse_int                  ld)
{
    return rocsparse_coo2dense_template(
        handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld);
}

extern "C" rocsparse_status rocsparse_zcoo2dense(rocsparse_handle                handle,
                                                 rocsparse_int                   m,
                                                 rocsparse_int                   n,
                                                 rocsparse_int                   nnz,
                                                 const rocsparse_mat_descr       descr,
                                                 const rocsparse_double_complex* coo_val,
                                                 const rocsparse_int*            coo_row_ind,
                                                 const rocsparse_int*            coo_col_ind,
                                                 rocsparse_double_complex*       A,
                                                 rocsparse_int                   ld)
{
    return rocsparse_coo2dense_template(
        handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld);
}
