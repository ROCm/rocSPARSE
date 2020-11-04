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
#ifndef ROCSPARSE_INIT_HPP
#define ROCSPARSE_INIT_HPP

#include "rocsparse_datatype2string.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_random.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <rocsparse.h>
#include <vector>

/* ==================================================================================== */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
void rocsparse_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1);

// Initializes sparse index vector with nnz entries ranging from start to end
void rocsparse_init_index(std::vector<rocsparse_int>& x, size_t nnz, size_t start, size_t end);

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
void rocsparse_init_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1);

/* ==================================================================================== */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
void rocsparse_init_nan(T* A, size_t N);

template <typename T>
void rocsparse_init_nan(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1);

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
void rocsparse_init_coo_matrix(std::vector<rocsparse_int>& row_ind,
                               std::vector<rocsparse_int>& col_ind,
                               std::vector<T>&             val,
                               size_t                      M,
                               size_t                      N,
                               size_t                      nnz,
                               rocsparse_index_base        base,
                               bool                        full_rank = false);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in CSR format */
template <typename T>
void rocsparse_init_csr_laplace2d(std::vector<rocsparse_int>& row_ptr,
                                  std::vector<rocsparse_int>& col_ind,
                                  std::vector<T>&             val,
                                  rocsparse_int               dim_x,
                                  rocsparse_int               dim_y,
                                  rocsparse_int&              M,
                                  rocsparse_int&              N,
                                  rocsparse_int&              nnz,
                                  rocsparse_index_base        base);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in COO format */
template <typename T>
void rocsparse_init_coo_laplace2d(std::vector<rocsparse_int>& row_ind,
                                  std::vector<rocsparse_int>& col_ind,
                                  std::vector<T>&             val,
                                  rocsparse_int               dim_x,
                                  rocsparse_int               dim_y,
                                  rocsparse_int&              M,
                                  rocsparse_int&              N,
                                  rocsparse_int&              nnz,
                                  rocsparse_index_base        base);

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in CSR format */
template <typename T>
void rocsparse_init_csr_laplace3d(std::vector<rocsparse_int>& row_ptr,
                                  std::vector<rocsparse_int>& col_ind,
                                  std::vector<T>&             val,
                                  rocsparse_int               dim_x,
                                  rocsparse_int               dim_y,
                                  rocsparse_int               dim_z,
                                  rocsparse_int&              M,
                                  rocsparse_int&              N,
                                  rocsparse_int&              nnz,
                                  rocsparse_index_base        base);

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in COO format */
template <typename T>
void rocsparse_init_coo_laplace3d(std::vector<rocsparse_int>& row_ind,
                                  std::vector<rocsparse_int>& col_ind,
                                  std::vector<T>&             val,
                                  rocsparse_int               dim_x,
                                  rocsparse_int               dim_y,
                                  rocsparse_int               dim_z,
                                  rocsparse_int&              M,
                                  rocsparse_int&              N,
                                  rocsparse_int&              nnz,
                                  rocsparse_index_base        base);

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
static void
    read_mtx_value(std::istringstream& is, rocsparse_int& row, rocsparse_int& col, float& val);

static void
    read_mtx_value(std::istringstream& is, rocsparse_int& row, rocsparse_int& col, double& val);

static void read_mtx_value(std::istringstream&      is,
                           rocsparse_int&           row,
                           rocsparse_int&           col,
                           rocsparse_float_complex& val);

static void read_mtx_value(std::istringstream&       is,
                           rocsparse_int&            row,
                           rocsparse_int&            col,
                           rocsparse_double_complex& val);

template <typename T>
void rocsparse_init_coo_mtx(const char*                 filename,
                            std::vector<rocsparse_int>& coo_row_ind,
                            std::vector<rocsparse_int>& coo_col_ind,
                            std::vector<T>&             coo_val,
                            rocsparse_int&              M,
                            rocsparse_int&              N,
                            rocsparse_int&              nnz,
                            rocsparse_index_base        base);

template <typename T>
void rocsparse_init_csr_mtx(const char*                 filename,
                            std::vector<rocsparse_int>& csr_row_ptr,
                            std::vector<rocsparse_int>& csr_col_ind,
                            std::vector<T>&             csr_val,
                            rocsparse_int&              M,
                            rocsparse_int&              N,
                            rocsparse_int&              nnz,
                            rocsparse_index_base        base);

template <typename T>
void rocsparse_init_bsr_mtx(const char*                 filename,
                            std::vector<rocsparse_int>& bsr_row_ptr,
                            std::vector<rocsparse_int>& bsr_col_ind,
                            std::vector<T>&             bsr_val,
                            rocsparse_direction         direction,
                            rocsparse_int&              Mb,
                            rocsparse_int&              Nb,
                            rocsparse_int               block_dim,
                            rocsparse_int&              nnzb,
                            rocsparse_index_base        base);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
static void read_csr_values(std::ifstream& in, rocsparse_int nnz, float* csr_val, bool mod);

static void read_csr_values(std::ifstream& in, rocsparse_int nnz, double* csr_val, bool mod);

static void read_csr_values(std::ifstream&           in,
                            rocsparse_int            nnz,
                            rocsparse_float_complex* csr_val,
                            bool                     mod);

static void read_csr_values(std::ifstream&            in,
                            rocsparse_int             nnz,
                            rocsparse_double_complex* csr_val,
                            bool                      mod);

template <typename T>
void rocsparse_init_csr_rocalution(const char*                 filename,
                                   std::vector<rocsparse_int>& row_ptr,
                                   std::vector<rocsparse_int>& col_ind,
                                   std::vector<T>&             val,
                                   rocsparse_int&              M,
                                   rocsparse_int&              N,
                                   rocsparse_int&              nnz,
                                   rocsparse_index_base        base,
                                   bool                        toint);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename T>
void rocsparse_init_coo_rocalution(const char*                 filename,
                                   std::vector<rocsparse_int>& row_ind,
                                   std::vector<rocsparse_int>& col_ind,
                                   std::vector<T>&             val,
                                   rocsparse_int&              M,
                                   rocsparse_int&              N,
                                   rocsparse_int&              nnz,
                                   rocsparse_index_base        base,
                                   bool                        toint);

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in CSR format */
template <typename T>
void rocsparse_init_csr_random(std::vector<rocsparse_int>& row_ptr,
                               std::vector<rocsparse_int>& col_ind,
                               std::vector<T>&             val,
                               rocsparse_int               M,
                               rocsparse_int               N,
                               rocsparse_int&              nnz,
                               rocsparse_index_base        base,
                               bool                        full_rank = false);

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
void rocsparse_init_coo_random(std::vector<rocsparse_int>& row_ind,
                               std::vector<rocsparse_int>& col_ind,
                               std::vector<T>&             val,
                               rocsparse_int               M,
                               rocsparse_int               N,
                               rocsparse_int&              nnz,
                               rocsparse_index_base        base,
                               bool                        full_rank = false);

/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in CSR format */
template <typename T>
void rocsparse_init_csr_matrix(std::vector<rocsparse_int>& csr_row_ptr,
                               std::vector<rocsparse_int>& csr_col_ind,
                               std::vector<T>&             csr_val,
                               rocsparse_int&              M,
                               rocsparse_int&              N,
                               rocsparse_int&              K,
                               rocsparse_int               dim_x,
                               rocsparse_int               dim_y,
                               rocsparse_int               dim_z,
                               rocsparse_int&              nnz,
                               rocsparse_index_base        base,
                               rocsparse_matrix_init       matrix,
                               const char*                 filename,
                               bool                        toint     = false,
                               bool                        full_rank = false);

/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in COO format */

template <typename T>
void rocsparse_init_coo_matrix(std::vector<rocsparse_int>& coo_row_ind,
                               std::vector<rocsparse_int>& coo_col_ind,
                               std::vector<T>&             coo_val,
                               rocsparse_int&              M,
                               rocsparse_int&              N,
                               rocsparse_int&              K,
                               rocsparse_int               dim_x,
                               rocsparse_int               dim_y,
                               rocsparse_int               dim_z,
                               rocsparse_int&              nnz,
                               rocsparse_index_base        base,
                               rocsparse_matrix_init       matrix,
                               const char*                 filename,
                               bool                        toint     = false,
                               bool                        full_rank = false);

#endif // ROCSPARSE_INIT_HPP
