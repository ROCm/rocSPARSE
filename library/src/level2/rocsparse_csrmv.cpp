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
#include "definitions.h"
#include "rocsparse.h"

#include "rocsparse_csrmv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrmv_analysis(rocsparse_handle          handle,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             m,
                                                      rocsparse_int             n,
                                                      rocsparse_int             nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const float*              csr_val,
                                                      const rocsparse_int*      csr_row_ptr,
                                                      const rocsparse_int*      csr_col_ind,
                                                      rocsparse_mat_info        info)
{
    return rocsparse_csrmv_analysis_template(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

extern "C" rocsparse_status rocsparse_dcsrmv_analysis(rocsparse_handle          handle,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             m,
                                                      rocsparse_int             n,
                                                      rocsparse_int             nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const double*             csr_val,
                                                      const rocsparse_int*      csr_row_ptr,
                                                      const rocsparse_int*      csr_col_ind,
                                                      rocsparse_mat_info        info)
{
    return rocsparse_csrmv_analysis_template(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

extern "C" rocsparse_status rocsparse_ccsrmv_analysis(rocsparse_handle               handle,
                                                      rocsparse_operation            trans,
                                                      rocsparse_int                  m,
                                                      rocsparse_int                  n,
                                                      rocsparse_int                  nnz,
                                                      const rocsparse_mat_descr      descr,
                                                      const rocsparse_float_complex* csr_val,
                                                      const rocsparse_int*           csr_row_ptr,
                                                      const rocsparse_int*           csr_col_ind,
                                                      rocsparse_mat_info             info)
{
    return rocsparse_csrmv_analysis_template(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

extern "C" rocsparse_status rocsparse_zcsrmv_analysis(rocsparse_handle                handle,
                                                      rocsparse_operation             trans,
                                                      rocsparse_int                   m,
                                                      rocsparse_int                   n,
                                                      rocsparse_int                   nnz,
                                                      const rocsparse_mat_descr       descr,
                                                      const rocsparse_double_complex* csr_val,
                                                      const rocsparse_int*            csr_row_ptr,
                                                      const rocsparse_int*            csr_col_ind,
                                                      rocsparse_mat_info              info)
{
    return rocsparse_csrmv_analysis_template(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

extern "C" rocsparse_status rocsparse_csrmv_clear(rocsparse_handle handle, rocsparse_mat_info info)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle, "rocsparse_csrmv_clear", (const void*&)info);

    // Destroy csrmv info struct
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrmv_info(info->csrmv_info));
    info->csrmv_info = nullptr;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsrmv(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             rocsparse_int             nnz,
                                             const float*              alpha,
                                             const rocsparse_mat_descr descr,
                                             const float*              csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             const float*              x,
                                             const float*              beta,
                                             float*                    y)
{
    return rocsparse_csrmv_template(handle,
                                    trans,
                                    m,
                                    n,
                                    nnz,
                                    alpha,
                                    descr,
                                    csr_val,
                                    csr_row_ptr,
                                    csr_col_ind,
                                    info,
                                    x,
                                    beta,
                                    y);
}

extern "C" rocsparse_status rocsparse_dcsrmv(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             rocsparse_int             nnz,
                                             const double*             alpha,
                                             const rocsparse_mat_descr descr,
                                             const double*             csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             const double*             x,
                                             const double*             beta,
                                             double*                   y)
{
    return rocsparse_csrmv_template(handle,
                                    trans,
                                    m,
                                    n,
                                    nnz,
                                    alpha,
                                    descr,
                                    csr_val,
                                    csr_row_ptr,
                                    csr_col_ind,
                                    info,
                                    x,
                                    beta,
                                    y);
}

extern "C" rocsparse_status rocsparse_ccsrmv(rocsparse_handle               handle,
                                             rocsparse_operation            trans,
                                             rocsparse_int                  m,
                                             rocsparse_int                  n,
                                             rocsparse_int                  nnz,
                                             const rocsparse_float_complex* alpha,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* csr_val,
                                             const rocsparse_int*           csr_row_ptr,
                                             const rocsparse_int*           csr_col_ind,
                                             rocsparse_mat_info             info,
                                             const rocsparse_float_complex* x,
                                             const rocsparse_float_complex* beta,
                                             rocsparse_float_complex*       y)
{
    return rocsparse_csrmv_template(handle,
                                    trans,
                                    m,
                                    n,
                                    nnz,
                                    alpha,
                                    descr,
                                    csr_val,
                                    csr_row_ptr,
                                    csr_col_ind,
                                    info,
                                    x,
                                    beta,
                                    y);
}

extern "C" rocsparse_status rocsparse_zcsrmv(rocsparse_handle                handle,
                                             rocsparse_operation             trans,
                                             rocsparse_int                   m,
                                             rocsparse_int                   n,
                                             rocsparse_int                   nnz,
                                             const rocsparse_double_complex* alpha,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* csr_val,
                                             const rocsparse_int*            csr_row_ptr,
                                             const rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info              info,
                                             const rocsparse_double_complex* x,
                                             const rocsparse_double_complex* beta,
                                             rocsparse_double_complex*       y)
{
    return rocsparse_csrmv_template(handle,
                                    trans,
                                    m,
                                    n,
                                    nnz,
                                    alpha,
                                    descr,
                                    csr_val,
                                    csr_row_ptr,
                                    csr_col_ind,
                                    info,
                                    x,
                                    beta,
                                    y);
}
