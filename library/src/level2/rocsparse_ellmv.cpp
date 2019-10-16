/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include "rocsparse_ellmv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sellmv(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const float*              alpha,
                                             const rocsparse_mat_descr descr,
                                             const float*              ell_val,
                                             const rocsparse_int*      ell_col_ind,
                                             rocsparse_int             ell_width,
                                             const float*              x,
                                             const float*              beta,
                                             float*                    y)
{
    return rocsparse_ellmv_template(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

extern "C" rocsparse_status rocsparse_dellmv(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const double*             alpha,
                                             const rocsparse_mat_descr descr,
                                             const double*             ell_val,
                                             const rocsparse_int*      ell_col_ind,
                                             rocsparse_int             ell_width,
                                             const double*             x,
                                             const double*             beta,
                                             double*                   y)
{
    return rocsparse_ellmv_template(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

extern "C" rocsparse_status rocsparse_cellmv(rocsparse_handle               handle,
                                             rocsparse_operation            trans,
                                             rocsparse_int                  m,
                                             rocsparse_int                  n,
                                             const rocsparse_float_complex* alpha,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* ell_val,
                                             const rocsparse_int*           ell_col_ind,
                                             rocsparse_int                  ell_width,
                                             const rocsparse_float_complex* x,
                                             const rocsparse_float_complex* beta,
                                             rocsparse_float_complex*       y)
{
    return rocsparse_ellmv_template(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

extern "C" rocsparse_status rocsparse_zellmv(rocsparse_handle                handle,
                                             rocsparse_operation             trans,
                                             rocsparse_int                   m,
                                             rocsparse_int                   n,
                                             const rocsparse_double_complex* alpha,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* ell_val,
                                             const rocsparse_int*            ell_col_ind,
                                             rocsparse_int                   ell_width,
                                             const rocsparse_double_complex* x,
                                             const rocsparse_double_complex* beta,
                                             rocsparse_double_complex*       y)
{
    return rocsparse_ellmv_template(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}
