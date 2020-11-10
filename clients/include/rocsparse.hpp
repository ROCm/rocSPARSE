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

/*! \file
 *  \brief rocsparse.hpp exposes C++ templated Sparse Linear Algebra interface
 *  with only the precision templated.
 */

#pragma once
#ifndef ROCSPARSE_HPP
#define ROCSPARSE_HPP

#include <rocsparse.h>

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
// axpyi
template <typename T>
rocsparse_status rocsparse_axpyi(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const T*             alpha,
                                 const T*             x_vec,
                                 const rocsparse_int* x_ind,
                                 T*                   y,
                                 rocsparse_index_base idx_base);

// doti
template <typename T>
rocsparse_status rocsparse_doti(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const T*             x_val,
                                const rocsparse_int* x_ind,
                                const T*             y,
                                T*                   result,
                                rocsparse_index_base idx_base);

// dotci
template <typename T>
rocsparse_status rocsparse_dotci(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const T*             x_val,
                                 const rocsparse_int* x_ind,
                                 const T*             y,
                                 T*                   result,
                                 rocsparse_index_base idx_base);

// gthr
template <typename T>
rocsparse_status rocsparse_gthr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const T*             y,
                                T*                   x_val,
                                const rocsparse_int* x_ind,
                                rocsparse_index_base idx_base);

// gthrz
template <typename T>
rocsparse_status rocsparse_gthrz(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 T*                   y,
                                 T*                   x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);

// roti
template <typename T>
rocsparse_status (*rocsparse_roti)(rocsparse_handle     handle,
                                   rocsparse_int        nnz,
                                   T*                   x_val,
                                   const rocsparse_int* x_ind,
                                   T*                   y,
                                   const T*             c,
                                   const T*             s,
                                   rocsparse_index_base idx_base);

template <>
static constexpr auto rocsparse_roti<float> = rocsparse_sroti;

template <>
static constexpr auto rocsparse_roti<double> = rocsparse_droti;

// sctr
template <typename T>
rocsparse_status rocsparse_sctr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const T*             x_val,
                                const rocsparse_int* x_ind,
                                T*                   y,
                                rocsparse_index_base idx_base);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
// bsrmv
template <typename T>
rocsparse_status rocsparse_bsrmv(rocsparse_handle          handle,
                                 rocsparse_direction       dir,
                                 rocsparse_operation       trans,
                                 rocsparse_int             mb,
                                 rocsparse_int             nb,
                                 rocsparse_int             nnzb,
                                 const T*                  alpha,
                                 const rocsparse_mat_descr descr,
                                 const T*                  bsr_val,
                                 const rocsparse_int*      bsr_row_ptr,
                                 const rocsparse_int*      bsr_col_ind,
                                 rocsparse_int             bsr_dim,
                                 const T*                  x,
                                 const T*                  beta,
                                 T*                        y);

// bsrsv
template <typename T>
rocsparse_status rocsparse_bsrsv_buffer_size(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans,
                                             rocsparse_int             mb,
                                             rocsparse_int             nnzb,
                                             const rocsparse_mat_descr descr,
                                             const T*                  bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             bsr_dim,
                                             rocsparse_mat_info        info,
                                             size_t*                   buffer_size);

template <typename T>
rocsparse_status rocsparse_bsrsv_analysis(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans,
                                          rocsparse_int             mb,
                                          rocsparse_int             nnzb,
                                          const rocsparse_mat_descr descr,
                                          const T*                  bsr_val,
                                          const rocsparse_int*      bsr_row_ptr,
                                          const rocsparse_int*      bsr_col_ind,
                                          rocsparse_int             bsr_dim,
                                          rocsparse_mat_info        info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy    solve,
                                          void*                     temp_buffer);

template <typename T>
rocsparse_status rocsparse_bsrsv_solve(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_operation       trans,
                                       rocsparse_int             mb,
                                       rocsparse_int             nnzb,
                                       const T*                  alpha,
                                       const rocsparse_mat_descr descr,
                                       const T*                  bsr_val,
                                       const rocsparse_int*      bsr_row_ptr,
                                       const rocsparse_int*      bsr_col_ind,
                                       rocsparse_int             bsr_dim,
                                       rocsparse_mat_info        info,
                                       const T*                  x,
                                       T*                        y,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer);

// coomv
template <typename T>
rocsparse_status rocsparse_coomv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             nnz,
                                 const T*                  alpha,
                                 const rocsparse_mat_descr descr,
                                 const T*                  coo_val,
                                 const rocsparse_int*      coo_row_ind,
                                 const rocsparse_int*      coo_col_ind,
                                 const T*                  x,
                                 const T*                  beta,
                                 T*                        y);

// csrmv
template <typename T>
rocsparse_status rocsparse_csrmv_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          rocsparse_int             nnz,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          rocsparse_mat_info        info);

template <typename T>
rocsparse_status rocsparse_csrmv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             nnz,
                                 const T*                  alpha,
                                 const rocsparse_mat_descr descr,
                                 const T*                  csr_val,
                                 const rocsparse_int*      csr_row_ptr,
                                 const rocsparse_int*      csr_col_ind,
                                 rocsparse_mat_info        info,
                                 const T*                  x,
                                 const T*                  beta,
                                 T*                        y);

// csrsv
template <typename T>
rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             const T*                  csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             size_t*                   buffer_size);

template <typename T>
rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          rocsparse_int             m,
                                          rocsparse_int             nnz,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          rocsparse_mat_info        info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy    solve,
                                          void*                     temp_buffer);

template <typename T>
rocsparse_status rocsparse_csrsv_solve(rocsparse_handle          handle,
                                       rocsparse_operation       trans,
                                       rocsparse_int             m,
                                       rocsparse_int             nnz,
                                       const T*                  alpha,
                                       const rocsparse_mat_descr descr,
                                       const T*                  csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_mat_info        info,
                                       const T*                  x,
                                       T*                        y,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer);

// ellmv
template <typename T>
rocsparse_status rocsparse_ellmv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 const T*                  alpha,
                                 const rocsparse_mat_descr descr,
                                 const T*                  ell_val,
                                 const rocsparse_int*      ell_col_ind,
                                 rocsparse_int             ell_width,
                                 const T*                  x,
                                 const T*                  beta,
                                 T*                        y);

// hybmv
template <typename T>
rocsparse_status rocsparse_hybmv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 const T*                  alpha,
                                 const rocsparse_mat_descr descr,
                                 const rocsparse_hyb_mat   hyb,
                                 const T*                  x,
                                 const T*                  beta,
                                 T*                        y);

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
// bsrmm
template <typename T>
rocsparse_status rocsparse_bsrmm(rocsparse_handle          handle,
                                 rocsparse_direction       dir,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 rocsparse_int             mb,
                                 rocsparse_int             n,
                                 rocsparse_int             kb,
                                 rocsparse_int             nnzb,
                                 const T*                  alpha,
                                 const rocsparse_mat_descr descr,
                                 const T*                  bsr_val,
                                 const rocsparse_int*      bsr_row_ptr,
                                 const rocsparse_int*      bsr_col_ind,
                                 rocsparse_int             block_dim,
                                 const T*                  B,
                                 rocsparse_int             ldb,
                                 const T*                  beta,
                                 T*                        C,
                                 rocsparse_int             ldc);

// csrmm
template <typename T>
rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             k,
                                 rocsparse_int             nnz,
                                 const T*                  alpha,
                                 const rocsparse_mat_descr descr,
                                 const T*                  csr_val,
                                 const rocsparse_int*      csr_row_ptr,
                                 const rocsparse_int*      csr_col_ind,
                                 const T*                  B,
                                 rocsparse_int             ldb,
                                 const T*                  beta,
                                 T*                        C,
                                 rocsparse_int             ldc);

// csrsm
template <typename T>
rocsparse_status rocsparse_csrsm_buffer_size(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             rocsparse_int             m,
                                             rocsparse_int             nrhs,
                                             rocsparse_int             nnz,
                                             const T*                  alpha,
                                             const rocsparse_mat_descr descr,
                                             const T*                  csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             const T*                  B,
                                             rocsparse_int             ldb,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             size_t*                   buffer_size);

template <typename T>
rocsparse_status rocsparse_csrsm_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_int             m,
                                          rocsparse_int             nrhs,
                                          rocsparse_int             nnz,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          const T*                  B,
                                          rocsparse_int             ldb,
                                          rocsparse_mat_info        info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy    solve,
                                          void*                     temp_buffer);

template <typename T>
rocsparse_status rocsparse_csrsm_solve(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             m,
                                       rocsparse_int             nrhs,
                                       rocsparse_int             nnz,
                                       const T*                  alpha,
                                       const rocsparse_mat_descr descr,
                                       const T*                  csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       T*                        B,
                                       rocsparse_int             ldb,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer);

// gemmi
template <typename T>
rocsparse_status rocsparse_gemmi(rocsparse_handle          handle,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             k,
                                 rocsparse_int             nnz,
                                 const T*                  alpha,
                                 const T*                  A,
                                 rocsparse_int             lda,
                                 const rocsparse_mat_descr descr,
                                 const T*                  csr_val,
                                 const rocsparse_int*      csr_row_ptr,
                                 const rocsparse_int*      csr_col_ind,
                                 const T*                  beta,
                                 T*                        C,
                                 rocsparse_int             ldc);

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
// csrgeam
template <typename T>
rocsparse_status rocsparse_csrgeam(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const T*                  alpha,
                                   const rocsparse_mat_descr descr_A,
                                   rocsparse_int             nnz_A,
                                   const T*                  csr_val_A,
                                   const rocsparse_int*      csr_row_ptr_A,
                                   const rocsparse_int*      csr_col_ind_A,
                                   const T*                  beta,
                                   const rocsparse_mat_descr descr_B,
                                   rocsparse_int             nnz_B,
                                   const T*                  csr_val_B,
                                   const rocsparse_int*      csr_row_ptr_B,
                                   const rocsparse_int*      csr_col_ind_B,
                                   const rocsparse_mat_descr descr_C,
                                   T*                        csr_val_C,
                                   const rocsparse_int*      csr_row_ptr_C,
                                   rocsparse_int*            csr_col_ind_C);

// csrgemm
template <typename T>
rocsparse_status rocsparse_csrgemm_buffer_size(rocsparse_handle          handle,
                                               rocsparse_operation       trans_A,
                                               rocsparse_operation       trans_B,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               rocsparse_int             k,
                                               const T*                  alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int             nnz_A,
                                               const rocsparse_int*      csr_row_ptr_A,
                                               const rocsparse_int*      csr_col_ind_A,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int             nnz_B,
                                               const rocsparse_int*      csr_row_ptr_B,
                                               const rocsparse_int*      csr_col_ind_B,
                                               const T*                  beta,
                                               const rocsparse_mat_descr descr_D,
                                               rocsparse_int             nnz_D,
                                               const rocsparse_int*      csr_row_ptr_D,
                                               const rocsparse_int*      csr_col_ind_D,
                                               rocsparse_mat_info        info_C,
                                               size_t*                   buffer_size);

template <typename T>
rocsparse_status rocsparse_csrgemm(rocsparse_handle          handle,
                                   rocsparse_operation       trans_A,
                                   rocsparse_operation       trans_B,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   rocsparse_int             k,
                                   const T*                  alpha,
                                   const rocsparse_mat_descr descr_A,
                                   rocsparse_int             nnz_A,
                                   const T*                  csr_val_A,
                                   const rocsparse_int*      csr_row_ptr_A,
                                   const rocsparse_int*      csr_col_ind_A,
                                   const rocsparse_mat_descr descr_B,
                                   rocsparse_int             nnz_B,
                                   const T*                  csr_val_B,
                                   const rocsparse_int*      csr_row_ptr_B,
                                   const rocsparse_int*      csr_col_ind_B,
                                   const T*                  beta,
                                   const rocsparse_mat_descr descr_D,
                                   rocsparse_int             nnz_D,
                                   const T*                  csr_val_D,
                                   const rocsparse_int*      csr_row_ptr_D,
                                   const rocsparse_int*      csr_col_ind_D,
                                   const rocsparse_mat_descr descr_C,
                                   T*                        csr_val_C,
                                   const rocsparse_int*      csr_row_ptr_C,
                                   rocsparse_int*            csr_col_ind_C,
                                   const rocsparse_mat_info  info_C,
                                   void*                     temp_buffer);

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
// bsric0
template <typename T>
rocsparse_status rocsparse_bsric0_buffer_size(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_int             mb,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              const T*                  bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

template <typename T>
rocsparse_status rocsparse_bsric0_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_int             mb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           const T*                  bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             block_dim,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

template <typename T>
rocsparse_status rocsparse_bsric0(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_int             mb,
                                  rocsparse_int             nnzb,
                                  const rocsparse_mat_descr descr,
                                  T*                        bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer);

// bsrilu0
template <typename T>
rocsparse_status rocsparse_bsrilu0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nnzb,
                                               const rocsparse_mat_descr descr,
                                               const T*                  bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size);

template <typename T, typename U>
rocsparse_status rocsparse_bsrilu0_numeric_boost(rocsparse_handle   handle,
                                                 rocsparse_mat_info info,
                                                 int                enable_boost,
                                                 const U*           boost_tol,
                                                 const T*           boost_val);

template <typename T>
rocsparse_status rocsparse_bsrilu0_analysis(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_int             mb,
                                            rocsparse_int             nnzb,
                                            const rocsparse_mat_descr descr,
                                            const T*                  bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             block_dim,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer);

template <typename T>
rocsparse_status rocsparse_bsrilu0(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nnzb,
                                   const rocsparse_mat_descr descr,
                                   T*                        bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);

// csric0
template <typename T>
rocsparse_status rocsparse_csric0_buffer_size(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const T*                  csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

template <typename T>
rocsparse_status rocsparse_csric0_analysis(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const T*                  csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

template <typename T>
rocsparse_status rocsparse_csric0(rocsparse_handle          handle,
                                  rocsparse_int             m,
                                  rocsparse_int             nnz,
                                  const rocsparse_mat_descr descr,
                                  T*                        csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer);

// csrilu0
template <typename T>
rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               const T*                  csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size);

template <typename T, typename U>
rocsparse_status rocsparse_csrilu0_numeric_boost(rocsparse_handle   handle,
                                                 rocsparse_mat_info info,
                                                 int                enable_boost,
                                                 const U*           boost_tol,
                                                 const T*           boost_val);

template <typename T>
rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             nnz,
                                            const rocsparse_mat_descr descr,
                                            const T*                  csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer);

template <typename T>
rocsparse_status rocsparse_csrilu0(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   T*                        csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */

// nnz
template <typename T>
rocsparse_status rocsparse_nnz(rocsparse_handle          handle,
                               rocsparse_direction       dir,
                               rocsparse_int             m,
                               rocsparse_int             n,
                               const rocsparse_mat_descr descr,
                               const T*                  A,
                               rocsparse_int             lda,
                               rocsparse_int*            nnz_per_row_columns,
                               rocsparse_int*            nnz_total_dev_host_ptr);

// nnz_compress
template <typename T>
rocsparse_status rocsparse_nnz_compress(rocsparse_handle          handle,
                                        rocsparse_int             m,
                                        const rocsparse_mat_descr descr_A,
                                        const T*                  csr_val_A,
                                        const rocsparse_int*      csr_row_ptr_A,
                                        rocsparse_int*            nnz_per_row,
                                        rocsparse_int*            nnz_C,
                                        T                         tol);

// dense2csr
template <typename T>
rocsparse_status rocsparse_dense2csr(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const T*                  A,
                                     rocsparse_int             lda,
                                     const rocsparse_int*      nnz_per_rows,
                                     T*                        csr_val,
                                     rocsparse_int*            csr_row_ptr,
                                     rocsparse_int*            csr_col_ind);

// prune_dense2csr_buffer_size
template <typename T>
rocsparse_status rocsparse_prune_dense2csr_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       const T*                  A,
                                                       rocsparse_int             lda,
                                                       const T*                  threshold,
                                                       const rocsparse_mat_descr descr,
                                                       const T*                  csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       const rocsparse_int*      csr_col_ind,
                                                       size_t*                   buffer_size);

// prune_dense2csr_nnz
template <typename T>
rocsparse_status rocsparse_prune_dense2csr_nnz(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const T*                  A,
                                               rocsparse_int             lda,
                                               const T*                  threshold,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            nnz_total_dev_host_ptr,
                                               void*                     temp_buffer);

// prune_dense2csr
template <typename T>
rocsparse_status rocsparse_prune_dense2csr(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           const T*                  A,
                                           rocsparse_int             lda,
                                           const T*                  threshold,
                                           const rocsparse_mat_descr descr,
                                           T*                        csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           rocsparse_int*            csr_col_ind,
                                           void*                     temp_buffer);

// prune_dense2csr_by_percentage_buffer_size
template <typename T>
rocsparse_status
    rocsparse_prune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const T*                  A,
                                                        rocsparse_int             lda,
                                                        T                         percentage,
                                                        const rocsparse_mat_descr descr,
                                                        const T*                  csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        rocsparse_mat_info        info,
                                                        size_t*                   buffer_size);

// prune_dense2csr_nnz_by_percentage
template <typename T>
rocsparse_status rocsparse_prune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                             rocsparse_int             m,
                                                             rocsparse_int             n,
                                                             const T*                  A,
                                                             rocsparse_int             lda,
                                                             T                         percentage,
                                                             const rocsparse_mat_descr descr,
                                                             rocsparse_int*            csr_row_ptr,
                                                             rocsparse_int* nnz_total_dev_host_ptr,
                                                             rocsparse_mat_info info,
                                                             void*              temp_buffer);

// prune_dense2csr_by_percentage
template <typename T>
rocsparse_status rocsparse_prune_dense2csr_by_percentage(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const T*                  A,
                                                         rocsparse_int             lda,
                                                         T                         percentage,
                                                         const rocsparse_mat_descr descr,
                                                         T*                        csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         rocsparse_int*            csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         void*                     temp_buffer);

// dense2csc
template <typename T>
rocsparse_status rocsparse_dense2csc(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const T*                  A,
                                     rocsparse_int             lda,
                                     const rocsparse_int*      nnz_per_columns,
                                     T*                        csc_val,
                                     rocsparse_int*            csc_col_ptr,
                                     rocsparse_int*            csc_row_ind);

// csr2dense
template <typename T>
rocsparse_status rocsparse_csr2dense(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const T*                  csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     T*                        A,
                                     rocsparse_int             lda);

// csc2dense
template <typename T>
rocsparse_status rocsparse_csc2dense(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const T*                  csc_val,
                                     const rocsparse_int*      csc_col_ptr,
                                     const rocsparse_int*      csc_row_ind,
                                     T*                        A,
                                     rocsparse_int             lda);

// csr2csc
template <typename T>
rocsparse_status rocsparse_csr2csc(rocsparse_handle     handle,
                                   rocsparse_int        m,
                                   rocsparse_int        n,
                                   rocsparse_int        nnz,
                                   const T*             csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   const rocsparse_int* csr_col_ind,
                                   T*                   csc_val,
                                   rocsparse_int*       csc_row_ind,
                                   rocsparse_int*       csc_col_ptr,
                                   rocsparse_action     copy_values,
                                   rocsparse_index_base idx_base,
                                   void*                temp_buffer);
// gebsr2gebsc
template <typename T>
rocsparse_status rocsparse_gebsr2gebsc_buffer_size(rocsparse_handle     handle,
                                                   rocsparse_int        mb,
                                                   rocsparse_int        nb,
                                                   rocsparse_int        nnzb,
                                                   const T*             bsr_val,
                                                   const rocsparse_int* bsr_row_ptr,
                                                   const rocsparse_int* bsr_col_ind,
                                                   rocsparse_int        row_block_dim,
                                                   rocsparse_int        col_block_dim,
                                                   size_t*              p_buffer_size);

template <typename T>
rocsparse_status rocsparse_gebsr2gebsc(rocsparse_handle     handle,
                                       rocsparse_int        mb,
                                       rocsparse_int        nb,
                                       rocsparse_int        nnzb,
                                       const T*             bsr_val,
                                       const rocsparse_int* bsr_row_ptr,
                                       const rocsparse_int* bsr_col_ind,
                                       rocsparse_int        row_block_dim,
                                       rocsparse_int        col_block_dim,
                                       T*                   bsc_val,
                                       rocsparse_int*       bsc_row_ind,
                                       rocsparse_int*       bsc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer);

// csr2ell
template <typename T>
rocsparse_status rocsparse_csr2ell(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   const rocsparse_mat_descr csr_descr,
                                   const T*                  csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   const rocsparse_mat_descr ell_descr,
                                   rocsparse_int             ell_width,
                                   T*                        ell_val,
                                   rocsparse_int*            ell_col_ind);

// csr2hyb
template <typename T>
rocsparse_status rocsparse_csr2hyb(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr descr,
                                   const T*                  csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_hyb_mat         hyb,
                                   rocsparse_int             user_ell_width,
                                   rocsparse_hyb_partition   partition_type);

// csr2bsr
template <typename T>
rocsparse_status rocsparse_csr2bsr(rocsparse_handle          handle,
                                   rocsparse_direction       direction,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr csr_descr,
                                   const T*                  csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_int             block_dim,
                                   const rocsparse_mat_descr bsr_descr,
                                   T*                        bsr_val,
                                   rocsparse_int*            bsr_row_ptr,
                                   rocsparse_int*            bsr_col_ind);

// csr2gebsr_buffer_size
template <typename T>
rocsparse_status rocsparse_csr2gebsr_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_direction       direction,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const rocsparse_mat_descr csr_descr,
                                                 const T*                  csr_val,
                                                 const rocsparse_int*      csr_row_ptr,
                                                 const rocsparse_int*      csr_col_ind,
                                                 rocsparse_int             row_block_dim,
                                                 rocsparse_int             col_block_dim,
                                                 size_t*                   p_buffer_size);

// csr2gebsr
template <typename T>
rocsparse_status rocsparse_csr2gebsr(rocsparse_handle          handle,
                                     rocsparse_direction       direction,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr csr_descr,
                                     const T*                  csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     const rocsparse_mat_descr bsr_descr,
                                     T*                        bsr_val,
                                     rocsparse_int*            bsr_row_ptr,
                                     rocsparse_int*            bsr_col_ind,
                                     rocsparse_int             row_block_dim,
                                     rocsparse_int             col_block_dim,
                                     void*                     p_buffer);

// ell2csr
template <typename T>
rocsparse_status rocsparse_ell2csr(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr ell_descr,
                                   rocsparse_int             ell_width,
                                   const T*                  ell_val,
                                   const rocsparse_int*      ell_col_ind,
                                   const rocsparse_mat_descr csr_descr,
                                   T*                        csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   rocsparse_int*            csr_col_ind);

// hyb2csr
template <typename T>
rocsparse_status rocsparse_hyb2csr(rocsparse_handle          handle,
                                   const rocsparse_mat_descr descr,
                                   const rocsparse_hyb_mat   hyb,
                                   T*                        csr_val,
                                   rocsparse_int*            csr_row_ptr,
                                   rocsparse_int*            csr_col_ind,
                                   void*                     temp_buffer);

// bsr2csr
template <typename T>
rocsparse_status rocsparse_bsr2csr(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nb,
                                   const rocsparse_mat_descr bsr_descr,
                                   const T*                  bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   const rocsparse_mat_descr csr_descr,
                                   T*                        csr_val,
                                   rocsparse_int*            csr_row_ptr,
                                   rocsparse_int*            csr_col_ind);

// gebsr2csr
template <typename T>
rocsparse_status rocsparse_gebsr2csr(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_int             mb,
                                     rocsparse_int             nb,
                                     const rocsparse_mat_descr bsr_descr,
                                     const T*                  bsr_val,
                                     const rocsparse_int*      bsr_row_ptr,
                                     const rocsparse_int*      bsr_col_ind,
                                     rocsparse_int             row_block_dim,
                                     rocsparse_int             col_block_dim,
                                     const rocsparse_mat_descr csr_descr,
                                     T*                        csr_val,
                                     rocsparse_int*            csr_row_ptr,
                                     rocsparse_int*            csr_col_ind);

// gebsr2csr_buffer_size
template <typename T>
rocsparse_status rocsparse_gebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr_A,
                                                   const T*                  bsr_val_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   rocsparse_int             row_block_dim_A,
                                                   rocsparse_int             col_block_dim_A,
                                                   rocsparse_int             row_block_dim_C,
                                                   rocsparse_int             col_block_dim_C,
                                                   size_t*                   buffer_size);

// gebsr2gebsr
template <typename T>
rocsparse_status rocsparse_gebsr2gebsr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_int             mb,
                                       rocsparse_int             nb,
                                       rocsparse_int             nnzb,
                                       const rocsparse_mat_descr descr_A,
                                       const T*                  bsr_val_A,
                                       const rocsparse_int*      bsr_row_ptr_A,
                                       const rocsparse_int*      bsr_col_ind_A,
                                       rocsparse_int             row_block_dim_A,
                                       rocsparse_int             col_block_dim_A,
                                       const rocsparse_mat_descr descr_C,
                                       T*                        bsr_val_C,
                                       rocsparse_int*            bsr_row_ptr_C,
                                       rocsparse_int*            bsr_col_ind_C,
                                       rocsparse_int             row_block_dim_C,
                                       rocsparse_int             col_block_dim_C,
                                       void*                     temp_buffer);

// csr2csr_compress
template <typename T>
rocsparse_status rocsparse_csr2csr_compress(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const rocsparse_mat_descr descr_A,
                                            const T*                  csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            rocsparse_int             nnz_A,
                                            const rocsparse_int*      nnz_per_row,
                                            T*                        csr_val_C,
                                            rocsparse_int*            csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C,
                                            T                         tol);

// prune_csr2csr_buffer_size
template <typename T>
rocsparse_status rocsparse_prune_csr2csr_buffer_size(rocsparse_handle          handle,
                                                     rocsparse_int             m,
                                                     rocsparse_int             n,
                                                     rocsparse_int             nnz_A,
                                                     const rocsparse_mat_descr csr_descr_A,
                                                     const T*                  csr_val_A,
                                                     const rocsparse_int*      csr_row_ptr_A,
                                                     const rocsparse_int*      csr_col_ind_A,
                                                     const T*                  threshold,
                                                     const rocsparse_mat_descr csr_descr_C,
                                                     const T*                  csr_val_C,
                                                     const rocsparse_int*      csr_row_ptr_C,
                                                     const rocsparse_int*      csr_col_ind_C,
                                                     size_t*                   buffer_size);

// prune_csr2csr_nnz
template <typename T>
rocsparse_status rocsparse_prune_csr2csr_nnz(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             rocsparse_int             nnz_A,
                                             const rocsparse_mat_descr csr_descr_A,
                                             const T*                  csr_val_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             const T*                  threshold,
                                             const rocsparse_mat_descr csr_descr_C,
                                             rocsparse_int*            csr_row_ptr_C,
                                             rocsparse_int*            nnz_total_dev_host_ptr,
                                             void*                     buffer);

// prune_csr2csr
template <typename T>
rocsparse_status rocsparse_prune_csr2csr(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         rocsparse_int             nnz_A,
                                         const rocsparse_mat_descr csr_descr_A,
                                         const T*                  csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         const rocsparse_int*      csr_col_ind_A,
                                         const T*                  threshold,
                                         const rocsparse_mat_descr csr_descr_C,
                                         T*                        csr_val_C,
                                         const rocsparse_int*      csr_row_ptr_C,
                                         rocsparse_int*            csr_col_ind_C,
                                         void*                     buffer);

// prune_csr2csr_by_percentage_buffer_size
template <typename T>
rocsparse_status
    rocsparse_prune_csr2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                      rocsparse_int             m,
                                                      rocsparse_int             n,
                                                      rocsparse_int             nnz_A,
                                                      const rocsparse_mat_descr csr_descr_A,
                                                      const T*                  csr_val_A,
                                                      const rocsparse_int*      csr_row_ptr_A,
                                                      const rocsparse_int*      csr_col_ind_A,
                                                      T                         percentage,
                                                      const rocsparse_mat_descr csr_descr_C,
                                                      const T*                  csr_val_C,
                                                      const rocsparse_int*      csr_row_ptr_C,
                                                      const rocsparse_int*      csr_col_ind_C,
                                                      rocsparse_mat_info        info,
                                                      size_t*                   buffer_size);

// prune_csr2csr_nnz_by_percentage
template <typename T>
rocsparse_status rocsparse_prune_csr2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                           rocsparse_int             m,
                                                           rocsparse_int             n,
                                                           rocsparse_int             nnz_A,
                                                           const rocsparse_mat_descr csr_descr_A,
                                                           const T*                  csr_val_A,
                                                           const rocsparse_int*      csr_row_ptr_A,
                                                           const rocsparse_int*      csr_col_ind_A,
                                                           T                         percentage,
                                                           const rocsparse_mat_descr csr_descr_C,
                                                           rocsparse_int*            csr_row_ptr_C,
                                                           rocsparse_int* nnz_total_dev_host_ptr,
                                                           rocsparse_mat_info info,
                                                           void*              buffer);

// prune_csr2csr_by_percentage
template <typename T>
rocsparse_status rocsparse_prune_csr2csr_by_percentage(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             nnz_A,
                                                       const rocsparse_mat_descr csr_descr_A,
                                                       const T*                  csr_val_A,
                                                       const rocsparse_int*      csr_row_ptr_A,
                                                       const rocsparse_int*      csr_col_ind_A,
                                                       T                         percentage,
                                                       const rocsparse_mat_descr csr_descr_C,
                                                       T*                        csr_val_C,
                                                       const rocsparse_int*      csr_row_ptr_C,
                                                       rocsparse_int*            csr_col_ind_C,
                                                       rocsparse_mat_info        info,
                                                       void*                     buffer);

#endif // ROCSPARSE_HPP
