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

#include "rocsparse_csrmm.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrmm(rocsparse_handle          handle,
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
    return rocsparse_csrmm_template<float>(handle,
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

extern "C" rocsparse_status rocsparse_dcsrmm(rocsparse_handle          handle,
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
    return rocsparse_csrmm_template<double>(handle,
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
