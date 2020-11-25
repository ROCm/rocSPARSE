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

#pragma once
#ifndef ROCSPARSE_HOST_HPP
#define ROCSPARSE_HOST_HPP

#include "rocsparse_test.hpp"

#include <hip/hip_runtime_api.h>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir) \
    ((dir == rocsparse_direction_row) ? BSR_IND_R(j, bi, bj) : BSR_IND_C(j, bi, bj))
#define BSR_IND_R(j, bi, bj) (bsr_dim * bsr_dim * (j) + (bi)*bsr_dim + (bj))
#define BSR_IND_C(j, bi, bj) (bsr_dim * bsr_dim * (j) + (bi) + (bj)*bsr_dim)

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template <typename T>
void host_axpyi(rocsparse_int        nnz,
                T                    alpha,
                const T*             x_val,
                const rocsparse_int* x_ind,
                T*                   y,
                rocsparse_index_base base);

template <typename T>
void host_doti(rocsparse_int        nnz,
               const T*             x_val,
               const rocsparse_int* x_ind,
               const T*             y,
               T*                   result,
               rocsparse_index_base base);
template <typename T>
void host_dotci(rocsparse_int        nnz,
                const T*             x_val,
                const rocsparse_int* x_ind,
                const T*             y,
                T*                   result,
                rocsparse_index_base base);
template <typename T>
void host_gthr(
    rocsparse_int nnz, const T* y, T* x_val, const rocsparse_int* x_ind, rocsparse_index_base base);
template <typename T>
void host_gthrz(
    rocsparse_int nnz, T* y, T* x_val, const rocsparse_int* x_ind, rocsparse_index_base base);
template <typename T>
void host_roti(rocsparse_int        nnz,
               T*                   x_val,
               const rocsparse_int* x_ind,
               T*                   y,
               const T*             c,
               const T*             s,
               rocsparse_index_base base);

template <typename T>
void host_sctr(
    rocsparse_int nnz, const T* x_val, const rocsparse_int* x_ind, T* y, rocsparse_index_base base);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T>
void host_bsrmv(rocsparse_direction  dir,
                rocsparse_operation  trans,
                rocsparse_int        mb,
                rocsparse_int        nb,
                rocsparse_int        nnzb,
                T                    alpha,
                const rocsparse_int* bsr_row_ptr,
                const rocsparse_int* bsr_col_ind,
                const T*             bsr_val,
                rocsparse_int        bsr_dim,
                const T*             x,
                T                    beta,
                T*                   y,
                rocsparse_index_base base);

template <typename T>
void host_bsrsv(rocsparse_operation               trans,
                rocsparse_direction               dir,
                rocsparse_int                     mb,
                rocsparse_int                     nnzb,
                T                                 alpha,
                const std::vector<rocsparse_int>& bsr_row_ptr,
                const std::vector<rocsparse_int>& bsr_col_ind,
                const std::vector<T>&             bsr_val,
                rocsparse_int                     bsr_dim,
                const std::vector<T>&             x,
                std::vector<T>&                   y,
                rocsparse_diag_type               diag_type,
                rocsparse_fill_mode               fill_mode,
                rocsparse_index_base              base,
                rocsparse_int*                    struct_pivot,
                rocsparse_int*                    numeric_pivot);

template <typename T>
void host_coomv(rocsparse_int        M,
                rocsparse_int        nnz,
                T                    alpha,
                const rocsparse_int* coo_row_ind,
                const rocsparse_int* coo_col_ind,
                const T*             coo_val,
                const T*             x,
                T                    beta,
                T*                   y,
                rocsparse_index_base base);

template <typename T>
void host_csrmv(rocsparse_int        M,
                rocsparse_int        nnz,
                T                    alpha,
                const rocsparse_int* csr_row_ptr,
                const rocsparse_int* csr_col_ind,
                const T*             csr_val,
                const T*             x,
                T                    beta,
                T*                   y,
                rocsparse_index_base base,
                int                  algo);

template <typename T>
void host_csrsv(rocsparse_operation               trans,
                rocsparse_int                     M,
                rocsparse_int                     nnz,
                T                                 alpha,
                const std::vector<rocsparse_int>& csr_row_ptr,
                const std::vector<rocsparse_int>& csr_col_ind,
                const std::vector<T>&             csr_val,
                const std::vector<T>&             x,
                std::vector<T>&                   y,
                rocsparse_diag_type               diag_type,
                rocsparse_fill_mode               fill_mode,
                rocsparse_index_base              base,
                rocsparse_int*                    struct_pivot,
                rocsparse_int*                    numeric_pivot);

template <typename T>
void host_ellmv(rocsparse_int        M,
                rocsparse_int        N,
                rocsparse_int        nnz,
                T                    alpha,
                const rocsparse_int* ell_col_ind,
                const T*             ell_val,
                rocsparse_int        ell_width,
                const T*             x,
                T                    beta,
                T*                   y,
                rocsparse_index_base base);

template <typename T>
void host_hybmv(rocsparse_int        M,
                rocsparse_int        N,
                T                    alpha,
                rocsparse_int        ell_nnz,
                const rocsparse_int* ell_col_ind,
                const T*             ell_val,
                rocsparse_int        ell_width,
                rocsparse_int        coo_nnz,
                const rocsparse_int* coo_row_ind,
                const rocsparse_int* coo_col_ind,
                const T*             coo_val,
                const T*             x,
                T                    beta,
                T*                   y,
                rocsparse_index_base base);

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T>
void host_bsrmm(rocsparse_int                     Mb,
                rocsparse_int                     N,
                rocsparse_int                     Kb,
                rocsparse_int                     block_dim,
                rocsparse_direction               dir,
                rocsparse_operation               transA,
                rocsparse_operation               transB,
                T                                 alpha,
                const std::vector<rocsparse_int>& bsr_row_ptr_A,
                const std::vector<rocsparse_int>& bsr_col_ind_A,
                const std::vector<T>&             bsr_val_A,
                const std::vector<T>&             B,
                rocsparse_int                     ldb,
                T                                 beta,
                std::vector<T>&                   C,
                rocsparse_int                     ldc,
                rocsparse_index_base              base);

template <typename T>
void host_csrmm(rocsparse_int                     M,
                rocsparse_int                     N,
                rocsparse_operation               transB,
                T                                 alpha,
                const std::vector<rocsparse_int>& csr_row_ptr_A,
                const std::vector<rocsparse_int>& csr_col_ind_A,
                const std::vector<T>&             csr_val_A,
                const std::vector<T>&             B,
                rocsparse_int                     ldb,
                T                                 beta,
                std::vector<T>&                   C,
                rocsparse_int                     ldc,
                rocsparse_index_base              base);

template <typename T>
void host_csrsm(rocsparse_int                     M,
                rocsparse_int                     nrhs,
                rocsparse_int                     nnz,
                rocsparse_operation               transA,
                rocsparse_operation               transB,
                T                                 alpha,
                const std::vector<rocsparse_int>& csr_row_ptr,
                const std::vector<rocsparse_int>& csr_col_ind,
                const std::vector<T>&             csr_val,
                std::vector<T>&                   B,
                rocsparse_int                     ldb,
                rocsparse_diag_type               diag_type,
                rocsparse_fill_mode               fill_mode,
                rocsparse_index_base              base,
                rocsparse_int*                    struct_pivot,
                rocsparse_int*                    numeric_pivot);
template <typename T>
void host_gemmi(rocsparse_int        M,
                rocsparse_int        N,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const T*             A,
                rocsparse_int        lda,
                const rocsparse_int* csr_row_ptr,
                const rocsparse_int* csr_col_ind,
                const T*             csr_val,
                T                    beta,
                T*                   C,
                rocsparse_int        ldc,
                rocsparse_index_base base);

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template <typename T>
void host_csrgeam_nnz(rocsparse_int                     M,
                      rocsparse_int                     N,
                      T                                 alpha,
                      const std::vector<rocsparse_int>& csr_row_ptr_A,
                      const std::vector<rocsparse_int>& csr_col_ind_A,
                      T                                 beta,
                      const std::vector<rocsparse_int>& csr_row_ptr_B,
                      const std::vector<rocsparse_int>& csr_col_ind_B,
                      std::vector<rocsparse_int>&       csr_row_ptr_C,
                      rocsparse_int*                    nnz_C,
                      rocsparse_index_base              base_A,
                      rocsparse_index_base              base_B,
                      rocsparse_index_base              base_C);

template <typename T>
void host_csrgeam(rocsparse_int                     M,
                  rocsparse_int                     N,
                  T                                 alpha,
                  const std::vector<rocsparse_int>& csr_row_ptr_A,
                  const std::vector<rocsparse_int>& csr_col_ind_A,
                  const std::vector<T>&             csr_val_A,
                  T                                 beta,
                  const std::vector<rocsparse_int>& csr_row_ptr_B,
                  const std::vector<rocsparse_int>& csr_col_ind_B,
                  const std::vector<T>&             csr_val_B,
                  const std::vector<rocsparse_int>& csr_row_ptr_C,
                  std::vector<rocsparse_int>&       csr_col_ind_C,
                  std::vector<T>&                   csr_val_C,
                  rocsparse_index_base              base_A,
                  rocsparse_index_base              base_B,
                  rocsparse_index_base              base_C);

template <typename T>
void host_csrgemm_nnz(rocsparse_int                     M,
                      rocsparse_int                     N,
                      rocsparse_int                     K,
                      const T*                          alpha,
                      const std::vector<rocsparse_int>& csr_row_ptr_A,
                      const std::vector<rocsparse_int>& csr_col_ind_A,
                      const std::vector<rocsparse_int>& csr_row_ptr_B,
                      const std::vector<rocsparse_int>& csr_col_ind_B,
                      const T*                          beta,
                      const std::vector<rocsparse_int>& csr_row_ptr_D,
                      const std::vector<rocsparse_int>& csr_col_ind_D,
                      std::vector<rocsparse_int>&       csr_row_ptr_C,
                      rocsparse_int*                    nnz_C,
                      rocsparse_index_base              base_A,
                      rocsparse_index_base              base_B,
                      rocsparse_index_base              base_C,
                      rocsparse_index_base              base_D);

template <typename T>
void host_csrgemm(rocsparse_int                     M,
                  rocsparse_int                     N,
                  rocsparse_int                     L,
                  const T*                          alpha,
                  const std::vector<rocsparse_int>& csr_row_ptr_A,
                  const std::vector<rocsparse_int>& csr_col_ind_A,
                  const std::vector<T>&             csr_val_A,
                  const std::vector<rocsparse_int>& csr_row_ptr_B,
                  const std::vector<rocsparse_int>& csr_col_ind_B,
                  const std::vector<T>&             csr_val_B,
                  const T*                          beta,
                  const std::vector<rocsparse_int>& csr_row_ptr_D,
                  const std::vector<rocsparse_int>& csr_col_ind_D,
                  const std::vector<T>&             csr_val_D,
                  const std::vector<rocsparse_int>& csr_row_ptr_C,
                  std::vector<rocsparse_int>&       csr_col_ind_C,
                  std::vector<T>&                   csr_val_C,
                  rocsparse_index_base              base_A,
                  rocsparse_index_base              base_B,
                  rocsparse_index_base              base_C,
                  rocsparse_index_base              base_D);

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template <typename T>
void host_bsric0(rocsparse_direction               direction,
                 rocsparse_int                     Mb,
                 rocsparse_int                     block_dim,
                 const std::vector<rocsparse_int>& bsr_row_ptr,
                 const std::vector<rocsparse_int>& bsr_col_ind,
                 std::vector<T>&                   bsr_val,
                 rocsparse_index_base              base,
                 rocsparse_int*                    struct_pivot,
                 rocsparse_int*                    numeric_pivot);

template <typename T, typename U>
void host_bsrilu0(rocsparse_direction               dir,
                  rocsparse_int                     mb,
                  const std::vector<rocsparse_int>& bsr_row_ptr,
                  const std::vector<rocsparse_int>& bsr_col_ind,
                  std::vector<T>&                   bsr_val,
                  rocsparse_int                     bsr_dim,
                  rocsparse_index_base              base,
                  rocsparse_int*                    struct_pivot,
                  rocsparse_int*                    numeric_pivot,
                  bool                              boost,
                  U                                 boost_tol,
                  T                                 boost_val);

template <typename T>
void host_csric0(rocsparse_int                     M,
                 const std::vector<rocsparse_int>& csr_row_ptr,
                 const std::vector<rocsparse_int>& csr_col_ind,
                 std::vector<T>&                   csr_val,
                 rocsparse_index_base              base,
                 rocsparse_int*                    struct_pivot,
                 rocsparse_int*                    numeric_pivot);

template <typename T, typename U>
void host_csrilu0(rocsparse_int                     M,
                  const std::vector<rocsparse_int>& csr_row_ptr,
                  const std::vector<rocsparse_int>& csr_col_ind,
                  std::vector<T>&                   csr_val,
                  rocsparse_index_base              base,
                  rocsparse_int*                    struct_pivot,
                  rocsparse_int*                    numeric_pivot,
                  bool                              boost,
                  U                                 boost_tol,
                  T                                 boost_val);

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template <typename T>
rocsparse_status host_nnz(rocsparse_direction       dirA,
                          rocsparse_int             m,
                          rocsparse_int             n,
                          const rocsparse_mat_descr descrA,
                          const T*                  A,
                          rocsparse_int             lda,
                          rocsparse_int*            nnz_per_row_columns,
                          rocsparse_int*            nnz_total_dev_host_ptr);

template <typename T>
void host_prune_dense2csr(rocsparse_int               m,
                          rocsparse_int               n,
                          const std::vector<T>&       A,
                          rocsparse_int               lda,
                          rocsparse_index_base        base,
                          T                           threshold,
                          rocsparse_int&              nnz,
                          std::vector<T>&             csr_val,
                          std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind);

template <typename T>
void host_prune_dense2csr_by_percentage(rocsparse_int               m,
                                        rocsparse_int               n,
                                        const std::vector<T>&       A,
                                        rocsparse_int               lda,
                                        rocsparse_index_base        base,
                                        T                           percentage,
                                        rocsparse_int&              nnz,
                                        std::vector<T>&             csr_val,
                                        std::vector<rocsparse_int>& csr_row_ptr,
                                        std::vector<rocsparse_int>& csr_col_ind);

template <rocsparse_direction DIRA, typename T>
rocsparse_status host_dense2csx(rocsparse_int        m,
                                rocsparse_int        n,
                                rocsparse_index_base base,
                                const T*             A,
                                rocsparse_int        ld,
                                const rocsparse_int* nnz_per_row_columns,
                                T*                   csx_val,
                                rocsparse_int*       csx_row_col_ptr,
                                rocsparse_int*       csx_col_row_ind);

template <rocsparse_direction DIRA, typename T>
rocsparse_status host_csx2dense(rocsparse_int        m,
                                rocsparse_int        n,
                                rocsparse_index_base base,
                                const T*             csx_val,
                                const rocsparse_int* csx_row_col_ptr,
                                const rocsparse_int* csx_col_row_ind,
                                T*                   A,
                                rocsparse_int        ld);

void host_csr_to_coo(rocsparse_int                     M,
                     rocsparse_int                     nnz,
                     const std::vector<rocsparse_int>& csr_row_ptr,
                     std::vector<rocsparse_int>&       coo_row_ind,
                     rocsparse_index_base              base);

template <typename T>
void host_csr_to_csc(rocsparse_int                     M,
                     rocsparse_int                     N,
                     rocsparse_int                     nnz,
                     const std::vector<rocsparse_int>& csr_row_ptr,
                     const std::vector<rocsparse_int>& csr_col_ind,
                     const std::vector<T>&             csr_val,
                     std::vector<rocsparse_int>&       csc_row_ind,
                     std::vector<rocsparse_int>&       csc_col_ptr,
                     std::vector<T>&                   csc_val,
                     rocsparse_action                  action,
                     rocsparse_index_base              base);

template <typename T>
void host_gebsr_to_gebsc(rocsparse_int                     Mb,
                         rocsparse_int                     Nb,
                         rocsparse_int                     nnzb,
                         const std::vector<rocsparse_int>& bsr_row_ptr,
                         const std::vector<rocsparse_int>& bsr_col_ind,
                         const std::vector<T>&             bsr_val,
                         rocsparse_int                     row_block_dim,
                         rocsparse_int                     col_block_dim,
                         std::vector<rocsparse_int>&       bsc_row_ind,
                         std::vector<rocsparse_int>&       bsc_col_ptr,
                         std::vector<T>&                   bsc_val,
                         rocsparse_action                  action,
                         rocsparse_index_base              base);

template <typename T>
void host_gebsr_to_csr(rocsparse_direction               direction,
                       rocsparse_int                     mb,
                       rocsparse_int                     nb,
                       rocsparse_int                     nnzb,
                       const std::vector<T>&             bsr_val,
                       const std::vector<rocsparse_int>& bsr_row_ptr,
                       const std::vector<rocsparse_int>& bsr_col_ind,
                       rocsparse_int                     row_block_dim,
                       rocsparse_int                     col_block_dim,
                       rocsparse_index_base              bsr_base,
                       std::vector<T>&                   csr_val,
                       std::vector<rocsparse_int>&       csr_row_ptr,
                       std::vector<rocsparse_int>&       csr_col_ind,
                       rocsparse_index_base              csr_base);

template <typename T>
void host_csr_to_gebsr(rocsparse_direction               direction,
                       rocsparse_int                     m,
                       rocsparse_int                     n,
                       rocsparse_int                     nnz,
                       const std::vector<T>&             csr_val,
                       const std::vector<rocsparse_int>& csr_row_ptr,
                       const std::vector<rocsparse_int>& csr_col_ind,
                       rocsparse_int                     row_block_dim,
                       rocsparse_int                     col_block_dim,
                       rocsparse_index_base              csr_base,
                       std::vector<T>&                   bsr_val,
                       std::vector<rocsparse_int>&       bsr_row_ptr,
                       std::vector<rocsparse_int>&       bsr_col_ind,
                       rocsparse_index_base              bsr_base);

template <typename T>
void host_gebsr_to_gebsr(rocsparse_direction               direction,
                         rocsparse_int                     mb,
                         rocsparse_int                     nb,
                         rocsparse_int                     nnzb,
                         const std::vector<T>&             bsr_val_A,
                         const std::vector<rocsparse_int>& bsr_row_ptr_A,
                         const std::vector<rocsparse_int>& bsr_col_ind_A,
                         rocsparse_int                     row_block_dim_A,
                         rocsparse_int                     col_block_dim_A,
                         rocsparse_index_base              base_A,
                         std::vector<T>&                   bsr_val_C,
                         std::vector<rocsparse_int>&       bsr_row_ptr_C,
                         std::vector<rocsparse_int>&       bsr_col_ind_C,
                         rocsparse_int                     row_block_dim_C,
                         rocsparse_int                     col_block_dim_C,
                         rocsparse_index_base              base_C);

template <typename T>
void host_bsr_to_bsc(rocsparse_int                     mb,
                     rocsparse_int                     nb,
                     rocsparse_int                     nnzb,
                     rocsparse_int                     bsr_dim,
                     const std::vector<rocsparse_int>& bsr_row_ptr,
                     const std::vector<rocsparse_int>& bsr_col_ind,
                     const std::vector<T>&             bsr_val,
                     std::vector<rocsparse_int>&       bsc_row_ind,
                     std::vector<rocsparse_int>&       bsc_col_ptr,
                     std::vector<T>&                   bsc_val,
                     rocsparse_index_base              bsr_base,
                     rocsparse_index_base              bsc_base);

template <typename T>
void host_csr_to_ell(rocsparse_int                     M,
                     const std::vector<rocsparse_int>& csr_row_ptr,
                     const std::vector<rocsparse_int>& csr_col_ind,
                     const std::vector<T>&             csr_val,
                     std::vector<rocsparse_int>&       ell_col_ind,
                     std::vector<T>&                   ell_val,
                     rocsparse_int&                    ell_width,
                     rocsparse_index_base              csr_base,
                     rocsparse_index_base              ell_base);

template <typename T>
void host_csr_to_hyb(rocsparse_int                     M,
                     rocsparse_int                     nnz,
                     const std::vector<rocsparse_int>& csr_row_ptr,
                     const std::vector<rocsparse_int>& csr_col_ind,
                     const std::vector<T>&             csr_val,
                     std::vector<rocsparse_int>&       ell_col_ind,
                     std::vector<T>&                   ell_val,
                     rocsparse_int&                    ell_width,
                     rocsparse_int&                    ell_nnz,
                     std::vector<rocsparse_int>&       coo_row_ind,
                     std::vector<rocsparse_int>&       coo_col_ind,
                     std::vector<T>&                   coo_val,
                     rocsparse_int&                    coo_nnz,
                     rocsparse_hyb_partition           part,
                     rocsparse_index_base              base);

template <typename T>
void host_csr_to_csr_compress(rocsparse_int                     M,
                              rocsparse_int                     N,
                              rocsparse_int                     nnz,
                              const std::vector<rocsparse_int>& csr_row_ptr_A,
                              const std::vector<rocsparse_int>& csr_col_ind_A,
                              const std::vector<T>&             csr_val_A,
                              std::vector<rocsparse_int>&       csr_row_ptr_C,
                              std::vector<rocsparse_int>&       csr_col_ind_C,
                              std::vector<T>&                   csr_val_C,
                              rocsparse_index_base              base,
                              T                                 tol);
template <typename T>
void host_prune_csr_to_csr(rocsparse_int                     M,
                           rocsparse_int                     N,
                           rocsparse_int                     nnz_A,
                           const std::vector<rocsparse_int>& csr_row_ptr_A,
                           const std::vector<rocsparse_int>& csr_col_ind_A,
                           const std::vector<T>&             csr_val_A,
                           rocsparse_int&                    nnz_C,
                           std::vector<rocsparse_int>&       csr_row_ptr_C,
                           std::vector<rocsparse_int>&       csr_col_ind_C,
                           std::vector<T>&                   csr_val_C,
                           rocsparse_index_base              csr_base_A,
                           rocsparse_index_base              csr_base_C,
                           T                                 threshold);

template <typename T>
void host_prune_csr_to_csr_by_percentage(rocsparse_int                     M,
                                         rocsparse_int                     N,
                                         rocsparse_int                     nnz_A,
                                         const std::vector<rocsparse_int>& csr_row_ptr_A,
                                         const std::vector<rocsparse_int>& csr_col_ind_A,
                                         const std::vector<T>&             csr_val_A,
                                         rocsparse_int&                    nnz_C,
                                         std::vector<rocsparse_int>&       csr_row_ptr_C,
                                         std::vector<rocsparse_int>&       csr_col_ind_C,
                                         std::vector<T>&                   csr_val_C,
                                         rocsparse_index_base              csr_base_A,
                                         rocsparse_index_base              csr_base_C,
                                         T                                 percentage);
void host_coo_to_csr(rocsparse_int                     M,
                     rocsparse_int                     nnz,
                     const std::vector<rocsparse_int>& coo_row_ind,
                     std::vector<rocsparse_int>&       csr_row_ptr,
                     rocsparse_index_base              base);

template <typename T>
void host_ell_to_csr(rocsparse_int                     M,
                     rocsparse_int                     N,
                     const std::vector<rocsparse_int>& ell_col_ind,
                     const std::vector<T>&             ell_val,
                     rocsparse_int                     ell_width,
                     std::vector<rocsparse_int>&       csr_row_ptr,
                     std::vector<rocsparse_int>&       csr_col_ind,
                     std::vector<T>&                   csr_val,
                     rocsparse_int&                    csr_nnz,
                     rocsparse_index_base              ell_base,
                     rocsparse_index_base              csr_base);

template <typename T>
void host_coosort_by_column(rocsparse_int               M,
                            rocsparse_int               nnz,
                            std::vector<rocsparse_int>& coo_row_ind,
                            std::vector<rocsparse_int>& coo_col_ind,
                            std::vector<T>&             coo_val);

#endif // ROCSPARSE_HOST_HPP
