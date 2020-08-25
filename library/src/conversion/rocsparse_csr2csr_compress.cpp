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

#include "rocsparse_csr2csr_compress.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsr2csr_compress(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const rocsparse_mat_descr descr_A,
                                                        const float*              csr_val_A,
                                                        const rocsparse_int*      csr_row_ptr_A,
                                                        const rocsparse_int*      csr_col_ind_A,
                                                        rocsparse_int             nnz_A,
                                                        const rocsparse_int*      nnz_per_row,
                                                        float*                    csr_val_C,
                                                        rocsparse_int*            csr_row_ptr_C,
                                                        rocsparse_int*            csr_col_ind_C,
                                                        float                     tol)
{
    return rocsparse_csr2csr_compress_template(handle,
                                               m,
                                               n,
                                               descr_A,
                                               csr_val_A,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               nnz_A,
                                               nnz_per_row,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               tol);
}

extern "C" rocsparse_status rocsparse_dcsr2csr_compress(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const rocsparse_mat_descr descr_A,
                                                        const double*             csr_val_A,
                                                        const rocsparse_int*      csr_row_ptr_A,
                                                        const rocsparse_int*      csr_col_ind_A,
                                                        rocsparse_int             nnz_A,
                                                        const rocsparse_int*      nnz_per_row,
                                                        double*                   csr_val_C,
                                                        rocsparse_int*            csr_row_ptr_C,
                                                        rocsparse_int*            csr_col_ind_C,
                                                        double                    tol)
{
    return rocsparse_csr2csr_compress_template(handle,
                                               m,
                                               n,
                                               descr_A,
                                               csr_val_A,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               nnz_A,
                                               nnz_per_row,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               tol);
}

extern "C" rocsparse_status rocsparse_ccsr2csr_compress(rocsparse_handle               handle,
                                                        rocsparse_int                  m,
                                                        rocsparse_int                  n,
                                                        const rocsparse_mat_descr      descr_A,
                                                        const rocsparse_float_complex* csr_val_A,
                                                        const rocsparse_int*     csr_row_ptr_A,
                                                        const rocsparse_int*     csr_col_ind_A,
                                                        rocsparse_int            nnz_A,
                                                        const rocsparse_int*     nnz_per_row,
                                                        rocsparse_float_complex* csr_val_C,
                                                        rocsparse_int*           csr_row_ptr_C,
                                                        rocsparse_int*           csr_col_ind_C,
                                                        rocsparse_float_complex  tol)
{
    return rocsparse_csr2csr_compress_template(handle,
                                               m,
                                               n,
                                               descr_A,
                                               csr_val_A,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               nnz_A,
                                               nnz_per_row,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               tol);
}

extern "C" rocsparse_status rocsparse_zcsr2csr_compress(rocsparse_handle                handle,
                                                        rocsparse_int                   m,
                                                        rocsparse_int                   n,
                                                        const rocsparse_mat_descr       descr_A,
                                                        const rocsparse_double_complex* csr_val_A,
                                                        const rocsparse_int*      csr_row_ptr_A,
                                                        const rocsparse_int*      csr_col_ind_A,
                                                        rocsparse_int             nnz_A,
                                                        const rocsparse_int*      nnz_per_row,
                                                        rocsparse_double_complex* csr_val_C,
                                                        rocsparse_int*            csr_row_ptr_C,
                                                        rocsparse_int*            csr_col_ind_C,
                                                        rocsparse_double_complex  tol)
{
    return rocsparse_csr2csr_compress_template(handle,
                                               m,
                                               n,
                                               descr_A,
                                               csr_val_A,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               nnz_A,
                                               nnz_per_row,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               tol);
}
