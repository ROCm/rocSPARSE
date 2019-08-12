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

#include "rocsparse.hpp"

#include <rocsparse.h>

namespace rocsparse
{

    template <>
    rocsparse_status rocsparse_axpyi(rocsparse_handle     handle,
                                     rocsparse_int        nnz,
                                     const float*         alpha,
                                     const float*         x_val,
                                     const rocsparse_int* x_ind,
                                     float*               y,
                                     rocsparse_index_base idx_base)
    {
        return rocsparse_saxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
    }

    template <>
    rocsparse_status rocsparse_axpyi(rocsparse_handle     handle,
                                     rocsparse_int        nnz,
                                     const double*        alpha,
                                     const double*        x_val,
                                     const rocsparse_int* x_ind,
                                     double*              y,
                                     rocsparse_index_base idx_base)
    {
        return rocsparse_daxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
    }

    template <>
    rocsparse_status rocsparse_doti(rocsparse_handle     handle,
                                    rocsparse_int        nnz,
                                    const float*         x_val,
                                    const rocsparse_int* x_ind,
                                    const float*         y,
                                    float*               result,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_sdoti(handle, nnz, x_val, x_ind, y, result, idx_base);
    }

    template <>
    rocsparse_status rocsparse_doti(rocsparse_handle     handle,
                                    rocsparse_int        nnz,
                                    const double*        x_val,
                                    const rocsparse_int* x_ind,
                                    const double*        y,
                                    double*              result,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_ddoti(handle, nnz, x_val, x_ind, y, result, idx_base);
    }

    template <>
    rocsparse_status rocsparse_gthr(rocsparse_handle     handle,
                                    rocsparse_int        nnz,
                                    const float*         y,
                                    float*               x_val,
                                    const rocsparse_int* x_ind,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_sgthr(handle, nnz, y, x_val, x_ind, idx_base);
    }

    template <>
    rocsparse_status rocsparse_gthr(rocsparse_handle     handle,
                                    rocsparse_int        nnz,
                                    const double*        y,
                                    double*              x_val,
                                    const rocsparse_int* x_ind,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_dgthr(handle, nnz, y, x_val, x_ind, idx_base);
    }

    template <>
    rocsparse_status rocsparse_gthrz(rocsparse_handle     handle,
                                     rocsparse_int        nnz,
                                     float*               y,
                                     float*               x_val,
                                     const rocsparse_int* x_ind,
                                     rocsparse_index_base idx_base)
    {
        return rocsparse_sgthrz(handle, nnz, y, x_val, x_ind, idx_base);
    }

    template <>
    rocsparse_status rocsparse_gthrz(rocsparse_handle     handle,
                                     rocsparse_int        nnz,
                                     double*              y,
                                     double*              x_val,
                                     const rocsparse_int* x_ind,
                                     rocsparse_index_base idx_base)
    {
        return rocsparse_dgthrz(handle, nnz, y, x_val, x_ind, idx_base);
    }

    template <>
    rocsparse_status rocsparse_roti(rocsparse_handle     handle,
                                    rocsparse_int        nnz,
                                    float*               x_val,
                                    const rocsparse_int* x_ind,
                                    float*               y,
                                    const float*         c,
                                    const float*         s,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_sroti(handle, nnz, x_val, x_ind, y, c, s, idx_base);
    }

    template <>
    rocsparse_status rocsparse_roti(rocsparse_handle     handle,
                                    rocsparse_int        nnz,
                                    double*              x_val,
                                    const rocsparse_int* x_ind,
                                    double*              y,
                                    const double*        c,
                                    const double*        s,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_droti(handle, nnz, x_val, x_ind, y, c, s, idx_base);
    }

    template <>
    rocsparse_status rocsparse_sctr(rocsparse_handle     handle,
                                    rocsparse_int        nnz,
                                    const float*         x_val,
                                    const rocsparse_int* x_ind,
                                    float*               y,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_ssctr(handle, nnz, x_val, x_ind, y, idx_base);
    }

    template <>
    rocsparse_status rocsparse_sctr(rocsparse_handle     handle,
                                    rocsparse_int        nnz,
                                    const double*        x_val,
                                    const rocsparse_int* x_ind,
                                    double*              y,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_dsctr(handle, nnz, x_val, x_ind, y, idx_base);
    }

    template <>
    rocsparse_status rocsparse_coomv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     rocsparse_int             nnz,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              coo_val,
                                     const rocsparse_int*      coo_row_ind,
                                     const rocsparse_int*      coo_col_ind,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_scoomv(
            handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
    }

    template <>
    rocsparse_status rocsparse_coomv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     rocsparse_int             nnz,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             coo_val,
                                     const rocsparse_int*      coo_row_ind,
                                     const rocsparse_int*      coo_col_ind,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dcoomv(
            handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
    }

    template <>
    rocsparse_status rocsparse_csrmv_analysis(rocsparse_handle          handle,
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
        return rocsparse_scsrmv_analysis(
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
    }

    template <>
    rocsparse_status rocsparse_csrmv_analysis(rocsparse_handle          handle,
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
        return rocsparse_dcsrmv_analysis(
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
    }

    template <>
    rocsparse_status rocsparse_csrmv(rocsparse_handle          handle,
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
        return rocsparse_scsrmv(handle,
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

    template <>
    rocsparse_status rocsparse_csrmv(rocsparse_handle          handle,
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
        return rocsparse_dcsrmv(handle,
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

    template <>
    rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 rocsparse_int             m,
                                                 rocsparse_int             nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              csr_val,
                                                 const rocsparse_int*      csr_row_ptr,
                                                 const rocsparse_int*      csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_scsrsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 rocsparse_int             m,
                                                 rocsparse_int             nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             csr_val,
                                                 const rocsparse_int*      csr_row_ptr,
                                                 const rocsparse_int*      csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_dcsrsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_scsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_dcsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csrsv_solve(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse_int             m,
                                           rocsparse_int             nnz,
                                           const float*              alpha,
                                           const rocsparse_mat_descr descr,
                                           const float*              csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info,
                                           const float*              x,
                                           float*                    y,
                                           rocsparse_solve_policy    policy,
                                           void*                     temp_buffer)
    {
        return rocsparse_scsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      alpha,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csrsv_solve(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse_int             m,
                                           rocsparse_int             nnz,
                                           const double*             alpha,
                                           const rocsparse_mat_descr descr,
                                           const double*             csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info,
                                           const double*             x,
                                           double*                   y,
                                           rocsparse_solve_policy    policy,
                                           void*                     temp_buffer)
    {
        return rocsparse_dcsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      alpha,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_ellmv(rocsparse_handle          handle,
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
        return rocsparse_sellmv(
            handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
    }

    template <>
    rocsparse_status rocsparse_ellmv(rocsparse_handle          handle,
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
        return rocsparse_dellmv(
            handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
    }

    template <>
    rocsparse_status rocsparse_hybmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const rocsparse_hyb_mat   hyb,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_shybmv(handle, trans, alpha, descr, hyb, x, beta, y);
    }

    template <>
    rocsparse_status rocsparse_hybmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const rocsparse_hyb_mat   hyb,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dhybmv(handle, trans, alpha, descr, hyb, x, beta, y);
    }

    template <>
    rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
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
        return rocsparse_scsrmm(handle,
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

    template <>
    rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
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
        return rocsparse_dcsrmm(handle,
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

    template <>
    rocsparse_status rocsparse_csrgemm_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_int             m,
                                                   rocsparse_int             n,
                                                   rocsparse_int             k,
                                                   const float*              alpha,
                                                   const rocsparse_mat_descr descr_A,
                                                   rocsparse_int             nnz_A,
                                                   const rocsparse_int*      csr_row_ptr_A,
                                                   const rocsparse_int*      csr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   rocsparse_int             nnz_B,
                                                   const rocsparse_int*      csr_row_ptr_B,
                                                   const rocsparse_int*      csr_col_ind_B,
                                                   const float*              beta,
                                                   const rocsparse_mat_descr descr_D,
                                                   rocsparse_int             nnz_D,
                                                   const rocsparse_int*      csr_row_ptr_D,
                                                   const rocsparse_int*      csr_col_ind_D,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_scsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              beta,
                                              descr_D,
                                              nnz_D,
                                              csr_row_ptr_D,
                                              csr_col_ind_D,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparse_csrgemm_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_int             m,
                                                   rocsparse_int             n,
                                                   rocsparse_int             k,
                                                   const double*             alpha,
                                                   const rocsparse_mat_descr descr_A,
                                                   rocsparse_int             nnz_A,
                                                   const rocsparse_int*      csr_row_ptr_A,
                                                   const rocsparse_int*      csr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   rocsparse_int             nnz_B,
                                                   const rocsparse_int*      csr_row_ptr_B,
                                                   const rocsparse_int*      csr_col_ind_B,
                                                   const double*             beta,
                                                   const rocsparse_mat_descr descr_D,
                                                   rocsparse_int             nnz_D,
                                                   const rocsparse_int*      csr_row_ptr_D,
                                                   const rocsparse_int*      csr_col_ind_D,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dcsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              beta,
                                              descr_D,
                                              nnz_D,
                                              csr_row_ptr_D,
                                              csr_col_ind_D,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparse_csrgemm(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       rocsparse_int             k,
                                       const float*              alpha,
                                       const rocsparse_mat_descr descr_A,
                                       rocsparse_int             nnz_A,
                                       const float*              csr_val_A,
                                       const rocsparse_int*      csr_row_ptr_A,
                                       const rocsparse_int*      csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       rocsparse_int             nnz_B,
                                       const float*              csr_val_B,
                                       const rocsparse_int*      csr_row_ptr_B,
                                       const rocsparse_int*      csr_col_ind_B,
                                       const float*              beta,
                                       const rocsparse_mat_descr descr_D,
                                       rocsparse_int             nnz_D,
                                       const float*              csr_val_D,
                                       const rocsparse_int*      csr_row_ptr_D,
                                       const rocsparse_int*      csr_col_ind_D,
                                       const rocsparse_mat_descr descr_C,
                                       float*                    csr_val_C,
                                       const rocsparse_int*      csr_row_ptr_C,
                                       rocsparse_int*            csr_col_ind_C,
                                       const rocsparse_mat_info  info,
                                       void*                     temp_buffer)
    {
        return rocsparse_scsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descr_A,
                                  nnz_A,
                                  csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  beta,
                                  descr_D,
                                  nnz_D,
                                  csr_val_D,
                                  csr_row_ptr_D,
                                  csr_col_ind_D,
                                  descr_C,
                                  csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csrgemm(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       rocsparse_int             k,
                                       const double*             alpha,
                                       const rocsparse_mat_descr descr_A,
                                       rocsparse_int             nnz_A,
                                       const double*             csr_val_A,
                                       const rocsparse_int*      csr_row_ptr_A,
                                       const rocsparse_int*      csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       rocsparse_int             nnz_B,
                                       const double*             csr_val_B,
                                       const rocsparse_int*      csr_row_ptr_B,
                                       const rocsparse_int*      csr_col_ind_B,
                                       const double*             beta,
                                       const rocsparse_mat_descr descr_D,
                                       rocsparse_int             nnz_D,
                                       const double*             csr_val_D,
                                       const rocsparse_int*      csr_row_ptr_D,
                                       const rocsparse_int*      csr_col_ind_D,
                                       const rocsparse_mat_descr descr_C,
                                       double*                   csr_val_C,
                                       const rocsparse_int*      csr_row_ptr_C,
                                       rocsparse_int*            csr_col_ind_C,
                                       const rocsparse_mat_info  info,
                                       void*                     temp_buffer)
    {
        return rocsparse_dcsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descr_A,
                                  nnz_A,
                                  csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  beta,
                                  descr_D,
                                  nnz_D,
                                  csr_val_D,
                                  csr_row_ptr_D,
                                  csr_col_ind_D,
                                  descr_C,
                                  csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const float*              csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_scsrilu0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const double*             csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dcsrilu0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const float*              csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_scsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const double*             csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_dcsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csrilu0(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             nnz,
                                       const rocsparse_mat_descr descr,
                                       float*                    csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_scsrilu0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csrilu0(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             nnz,
                                       const rocsparse_mat_descr descr,
                                       double*                   csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_dcsrilu0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csr2csc(rocsparse_handle     handle,
                                       rocsparse_int        m,
                                       rocsparse_int        n,
                                       rocsparse_int        nnz,
                                       const float*         csr_val,
                                       const rocsparse_int* csr_row_ptr,
                                       const rocsparse_int* csr_col_ind,
                                       float*               csc_val,
                                       rocsparse_int*       csc_row_ind,
                                       rocsparse_int*       csc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer)
    {
        return rocsparse_scsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csr2csc(rocsparse_handle     handle,
                                       rocsparse_int        m,
                                       rocsparse_int        n,
                                       rocsparse_int        nnz,
                                       const double*        csr_val,
                                       const rocsparse_int* csr_row_ptr,
                                       const rocsparse_int* csr_col_ind,
                                       double*              csc_val,
                                       rocsparse_int*       csc_row_ind,
                                       rocsparse_int*       csc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer)
    {
        return rocsparse_dcsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparse_csr2ell(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       const rocsparse_mat_descr csr_descr,
                                       const float*              csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       const rocsparse_mat_descr ell_descr,
                                       rocsparse_int             ell_width,
                                       float*                    ell_val,
                                       rocsparse_int*            ell_col_ind)
    {
        return rocsparse_scsr2ell(handle,
                                  m,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind);
    }

    template <>
    rocsparse_status rocsparse_csr2ell(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       const rocsparse_mat_descr csr_descr,
                                       const double*             csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       const rocsparse_mat_descr ell_descr,
                                       rocsparse_int             ell_width,
                                       double*                   ell_val,
                                       rocsparse_int*            ell_col_ind)
    {
        return rocsparse_dcsr2ell(handle,
                                  m,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind);
    }

    template <>
    rocsparse_status rocsparse_csr2hyb(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr descr,
                                       const float*              csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_hyb_mat         hyb,
                                       rocsparse_int             user_ell_width,
                                       rocsparse_hyb_partition   partition_type)
    {
        return rocsparse_scsr2hyb(handle,
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

    template <>
    rocsparse_status rocsparse_csr2hyb(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr descr,
                                       const double*             csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_hyb_mat         hyb,
                                       rocsparse_int             user_ell_width,
                                       rocsparse_hyb_partition   partition_type)
    {
        return rocsparse_dcsr2hyb(handle,
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

    template <>
    rocsparse_status rocsparse_ell2csr(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr ell_descr,
                                       rocsparse_int             ell_width,
                                       const float*              ell_val,
                                       const rocsparse_int*      ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       float*                    csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       rocsparse_int*            csr_col_ind)
    {
        return rocsparse_sell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparse_ell2csr(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr ell_descr,
                                       rocsparse_int             ell_width,
                                       const double*             ell_val,
                                       const rocsparse_int*      ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       double*                   csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       rocsparse_int*            csr_col_ind)
    {
        return rocsparse_dell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

} // namespace rocsparse
