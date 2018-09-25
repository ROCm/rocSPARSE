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
#include "rocsparse_coomv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scoomv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int nnz,
                                             const float* alpha,
                                             const rocsparse_mat_descr descr,
                                             const float* coo_val,
                                             const rocsparse_int* coo_row_ind,
                                             const rocsparse_int* coo_col_ind,
                                             const float* x,
                                             const float* beta,
                                             float* y)
{
    return rocsparse_coomv_template<float>(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

extern "C" rocsparse_status rocsparse_dcoomv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int nnz,
                                             const double* alpha,
                                             const rocsparse_mat_descr descr,
                                             const double* coo_val,
                                             const rocsparse_int* coo_row_ind,
                                             const rocsparse_int* coo_col_ind,
                                             const double* x,
                                             const double* beta,
                                             double* y)
{
    return rocsparse_coomv_template<double>(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}
