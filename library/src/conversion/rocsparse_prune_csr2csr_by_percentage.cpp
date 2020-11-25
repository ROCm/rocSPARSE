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

#include "rocsparse_prune_csr2csr_by_percentage.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status
    rocsparse_sprune_csr2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             nnz_A,
                                                       const rocsparse_mat_descr csr_descr_A,
                                                       const float*              csr_val_A,
                                                       const rocsparse_int*      csr_row_ptr_A,
                                                       const rocsparse_int*      csr_col_ind_A,
                                                       float                     percentage,
                                                       const rocsparse_mat_descr csr_descr_C,
                                                       const float*              csr_val_C,
                                                       const rocsparse_int*      csr_row_ptr_C,
                                                       const rocsparse_int*      csr_col_ind_C,
                                                       rocsparse_mat_info        info,
                                                       size_t*                   buffer_size)
{
    return rocsparse_prune_csr2csr_by_percentage_buffer_size_template(handle,
                                                                      m,
                                                                      n,
                                                                      nnz_A,
                                                                      csr_descr_A,
                                                                      csr_val_A,
                                                                      csr_row_ptr_A,
                                                                      csr_col_ind_A,
                                                                      percentage,
                                                                      csr_descr_C,
                                                                      csr_val_C,
                                                                      csr_row_ptr_C,
                                                                      csr_col_ind_C,
                                                                      info,
                                                                      buffer_size);
}

extern "C" rocsparse_status
    rocsparse_dprune_csr2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             nnz_A,
                                                       const rocsparse_mat_descr csr_descr_A,
                                                       const double*             csr_val_A,
                                                       const rocsparse_int*      csr_row_ptr_A,
                                                       const rocsparse_int*      csr_col_ind_A,
                                                       double                    percentage,
                                                       const rocsparse_mat_descr csr_descr_C,
                                                       const double*             csr_val_C,
                                                       const rocsparse_int*      csr_row_ptr_C,
                                                       const rocsparse_int*      csr_col_ind_C,
                                                       rocsparse_mat_info        info,
                                                       size_t*                   buffer_size)
{
    return rocsparse_prune_csr2csr_by_percentage_buffer_size_template(handle,
                                                                      m,
                                                                      n,
                                                                      nnz_A,
                                                                      csr_descr_A,
                                                                      csr_val_A,
                                                                      csr_row_ptr_A,
                                                                      csr_col_ind_A,
                                                                      percentage,
                                                                      csr_descr_C,
                                                                      csr_val_C,
                                                                      csr_row_ptr_C,
                                                                      csr_col_ind_C,
                                                                      info,
                                                                      buffer_size);
}

extern "C" rocsparse_status
    rocsparse_sprune_csr2csr_nnz_by_percentage(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               rocsparse_int             nnz_A,
                                               const rocsparse_mat_descr csr_descr_A,
                                               const float*              csr_val_A,
                                               const rocsparse_int*      csr_row_ptr_A,
                                               const rocsparse_int*      csr_col_ind_A,
                                               float                     percentage,
                                               const rocsparse_mat_descr csr_descr_C,
                                               rocsparse_int*            csr_row_ptr_C,
                                               rocsparse_int*            nnz_total_dev_host_ptr,
                                               rocsparse_mat_info        info,
                                               void*                     temp_buffer)
{
    return rocsparse_prune_csr2csr_nnz_by_percentage_template(handle,
                                                              m,
                                                              n,
                                                              nnz_A,
                                                              csr_descr_A,
                                                              csr_val_A,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              percentage,
                                                              csr_descr_C,
                                                              csr_row_ptr_C,
                                                              nnz_total_dev_host_ptr,
                                                              info,
                                                              temp_buffer);
}

extern "C" rocsparse_status
    rocsparse_dprune_csr2csr_nnz_by_percentage(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               rocsparse_int             nnz_A,
                                               const rocsparse_mat_descr csr_descr_A,
                                               const double*             csr_val_A,
                                               const rocsparse_int*      csr_row_ptr_A,
                                               const rocsparse_int*      csr_col_ind_A,
                                               double                    percentage,
                                               const rocsparse_mat_descr csr_descr_C,
                                               rocsparse_int*            csr_row_ptr_C,
                                               rocsparse_int*            nnz_total_dev_host_ptr,
                                               rocsparse_mat_info        info,
                                               void*                     temp_buffer)
{
    return rocsparse_prune_csr2csr_nnz_by_percentage_template(handle,
                                                              m,
                                                              n,
                                                              nnz_A,
                                                              csr_descr_A,
                                                              csr_val_A,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              percentage,
                                                              csr_descr_C,
                                                              csr_row_ptr_C,
                                                              nnz_total_dev_host_ptr,
                                                              info,
                                                              temp_buffer);
}

extern "C" rocsparse_status
    rocsparse_sprune_csr2csr_by_percentage(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           rocsparse_int             nnz_A,
                                           const rocsparse_mat_descr csr_descr_A,
                                           const float*              csr_val_A,
                                           const rocsparse_int*      csr_row_ptr_A,
                                           const rocsparse_int*      csr_col_ind_A,
                                           float                     percentage,
                                           const rocsparse_mat_descr csr_descr_C,
                                           float*                    csr_val_C,
                                           const rocsparse_int*      csr_row_ptr_C,
                                           rocsparse_int*            csr_col_ind_C,
                                           rocsparse_mat_info        info,
                                           void*                     temp_buffer)
{
    return rocsparse_prune_csr2csr_by_percentage_template(handle,
                                                          m,
                                                          n,
                                                          nnz_A,
                                                          csr_descr_A,
                                                          csr_val_A,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          percentage,
                                                          csr_descr_C,
                                                          csr_val_C,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          info,
                                                          temp_buffer);
}

extern "C" rocsparse_status
    rocsparse_dprune_csr2csr_by_percentage(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           rocsparse_int             nnz_A,
                                           const rocsparse_mat_descr csr_descr_A,
                                           const double*             csr_val_A,
                                           const rocsparse_int*      csr_row_ptr_A,
                                           const rocsparse_int*      csr_col_ind_A,
                                           double                    percentage,
                                           const rocsparse_mat_descr csr_descr_C,
                                           double*                   csr_val_C,
                                           const rocsparse_int*      csr_row_ptr_C,
                                           rocsparse_int*            csr_col_ind_C,
                                           rocsparse_mat_info        info,
                                           void*                     temp_buffer)
{
    return rocsparse_prune_csr2csr_by_percentage_template(handle,
                                                          m,
                                                          n,
                                                          nnz_A,
                                                          csr_descr_A,
                                                          csr_val_A,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          percentage,
                                                          csr_descr_C,
                                                          csr_val_C,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          info,
                                                          temp_buffer);
}
