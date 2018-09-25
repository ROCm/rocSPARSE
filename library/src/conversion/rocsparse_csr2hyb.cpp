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
#include "rocsparse_csr2hyb.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsr2hyb(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               const rocsparse_mat_descr descr,
                                               const float* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_hyb_mat hyb,
                                               rocsparse_int user_ell_width,
                                               rocsparse_hyb_partition partition_type)
{
    return rocsparse_csr2hyb_template(handle,
                                      m,
                                      n,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      hyb,
                                      user_ell_width,
                                      partition_type);
}

extern "C" rocsparse_status rocsparse_dcsr2hyb(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               const rocsparse_mat_descr descr,
                                               const double* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_hyb_mat hyb,
                                               rocsparse_int user_ell_width,
                                               rocsparse_hyb_partition partition_type)
{
    return rocsparse_csr2hyb_template(handle,
                                      m,
                                      n,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      hyb,
                                      user_ell_width,
                                      partition_type);
}
