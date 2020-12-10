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
#ifndef ROCSPARSE_INIT_HPP
#define ROCSPARSE_INIT_HPP

#include "rocsparse_host.hpp"
#include "rocsparse_random.hpp"

#include <fstream>

/* ==================================================================================== */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
void rocsparse_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1);

// Initializes sparse index vector with nnz entries ranging from start to end
template <typename I>
void rocsparse_init_index(std::vector<I>& x, size_t nnz, size_t start, size_t end);

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
template <typename I, typename T>
void rocsparse_init_coo_matrix(std::vector<I>&      row_ind,
                               std::vector<I>&      col_ind,
                               std::vector<T>&      val,
                               I                    M,
                               I                    N,
                               I                    nnz,
                               rocsparse_index_base base,
                               bool                 full_rank = false);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_laplace2d(std::vector<I>&      row_ptr,
                                  std::vector<J>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  J&                   M,
                                  J&                   N,
                                  I&                   nnz,
                                  rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in COO format */
template <typename I, typename T>
void rocsparse_init_coo_laplace2d(std::vector<I>&      row_ind,
                                  std::vector<I>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  I&                   M,
                                  I&                   N,
                                  I&                   nnz,
                                  rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in ELL format */
template <typename I, typename T>
void rocsparse_init_ell_laplace2d(std::vector<I>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  I&                   M,
                                  I&                   N,
                                  I&                   width,
                                  rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_laplace3d(std::vector<I>&      row_ptr,
                                  std::vector<J>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  int32_t              dim_z,
                                  J&                   M,
                                  J&                   N,
                                  I&                   nnz,
                                  rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in COO format */
template <typename I, typename T>
void rocsparse_init_coo_laplace3d(std::vector<I>&      row_ind,
                                  std::vector<I>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  int32_t              dim_z,
                                  I&                   M,
                                  I&                   N,
                                  I&                   nnz,
                                  rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
template <typename I, typename T>
void rocsparse_init_coo_mtx(const char*          filename,
                            std::vector<I>&      coo_row_ind,
                            std::vector<I>&      coo_col_ind,
                            std::vector<T>&      coo_val,
                            I&                   M,
                            I&                   N,
                            I&                   nnz,
                            rocsparse_index_base base);

template <typename I, typename J, typename T>
void rocsparse_init_csr_mtx(const char*          filename,
                            std::vector<I>&      csr_row_ptr,
                            std::vector<J>&      csr_col_ind,
                            std::vector<T>&      csr_val,
                            J&                   M,
                            J&                   N,
                            I&                   nnz,
                            rocsparse_index_base base);

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
template <typename I, typename J, typename T>
void rocsparse_init_csr_rocalution(const char*          filename,
                                   std::vector<I>&      row_ptr,
                                   std::vector<J>&      col_ind,
                                   std::vector<T>&      val,
                                   J&                   M,
                                   J&                   N,
                                   I&                   nnz,
                                   rocsparse_index_base base,
                                   bool                 toint);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename I, typename T>
void rocsparse_init_coo_rocalution(const char*          filename,
                                   std::vector<I>&      row_ind,
                                   std::vector<I>&      col_ind,
                                   std::vector<T>&      val,
                                   I&                   M,
                                   I&                   N,
                                   I&                   nnz,
                                   rocsparse_index_base base,
                                   bool                 toint);

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_random(std::vector<I>&      row_ptr,
                               std::vector<J>&      col_ind,
                               std::vector<T>&      val,
                               J                    M,
                               J                    N,
                               I&                   nnz,
                               rocsparse_index_base base,
                               bool                 full_rank = false);

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename I, typename T>
void rocsparse_init_coo_random(std::vector<I>&      row_ind,
                               std::vector<I>&      col_ind,
                               std::vector<T>&      val,
                               I                    M,
                               I                    N,
                               I&                   nnz,
                               rocsparse_index_base base,
                               bool                 full_rank = false);

/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_matrix(std::vector<I>&       csr_row_ptr,
                               std::vector<J>&       csr_col_ind,
                               std::vector<T>&       csr_val,
                               J&                    M,
                               J&                    N,
                               J&                    K,
                               int32_t               dim_x,
                               int32_t               dim_y,
                               int32_t               dim_z,
                               I&                    nnz,
                               rocsparse_index_base  base,
                               rocsparse_matrix_init matrix,
                               const char*           filename,
                               bool                  toint     = false,
                               bool                  full_rank = false);

/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in ELL format */
template <typename I, typename T>
void rocsparse_init_ell_matrix(std::vector<I>&       ell_col_ind,
                               std::vector<T>&       ell_val,
                               I&                    M,
                               I&                    N,
                               I&                    K,
                               int32_t               dim_x,
                               int32_t               dim_y,
                               int32_t               dim_z,
                               I&                    ell_width,
                               rocsparse_index_base  base,
                               rocsparse_matrix_init matrix,
                               const char*           filename,
                               bool                  toint     = false,
                               bool                  full_rank = false);

/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in COO format */

template <typename I, typename T>
void rocsparse_init_coo_matrix(std::vector<I>&       coo_row_ind,
                               std::vector<I>&       coo_col_ind,
                               std::vector<T>&       coo_val,
                               I&                    M,
                               I&                    N,
                               I&                    K,
                               int32_t               dim_x,
                               int32_t               dim_y,
                               int32_t               dim_z,
                               I&                    nnz,
                               rocsparse_index_base  base,
                               rocsparse_matrix_init matrix,
                               const char*           filename,
                               bool                  toint     = false,
                               bool                  full_rank = false);

template <typename T>
struct rocsparse_initializer_base
{
    virtual ~rocsparse_initializer_base(){};
    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
        = 0;

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
        = 0;
};

template <typename T>
struct rocsparse_initializer_random : public rocsparse_initializer_base<T>
{
private:
    bool m_fullrank;

public:
    rocsparse_initializer_random(bool fullrank)
        : m_fullrank(fullrank){};

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        // Compute non-zero entries of the matrix
        nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;

        // Sample random matrix
        std::vector<rocsparse_int> row_ind(nnz);

        // Sample COO matrix
        rocsparse_init_coo_matrix(row_ind, csr_col_ind, csr_val, M, N, nnz, base, this->m_fullrank);

        // Convert to CSR
        host_coo_to_csr(M, row_ind, csr_row_ptr, base);
    };

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        // Compute non-zero entries of the matrix
        nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;

        // Sample random matrix
        rocsparse_init_coo_matrix(
            coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, this->m_fullrank);
    }
};

template <typename T>
struct rocsparse_initializer_rocalution : public rocsparse_initializer_base<T>
{
private:
    const char* m_filename;

public:
    rocsparse_initializer_rocalution(const char* filename)
        : m_filename(filename){};

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_csr_rocalution(
            this->m_filename, csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base, 0);
    };

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_coo_rocalution(
            this->m_filename, coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, 0);
    }
};

template <typename T>
struct rocsparse_initializer_mtx : public rocsparse_initializer_base<T>
{
private:
    const char* m_filename;

public:
    rocsparse_initializer_mtx(const char* filename)
        : m_filename(filename){};

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_csr_mtx(
            this->m_filename, csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base);
    };

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_coo_mtx(
            this->m_filename, coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base);
    }
};

template <typename T>
struct rocsparse_initializer_laplace2d : public rocsparse_initializer_base<T>
{
private:
    rocsparse_int m_dimx, m_dimy;

public:
    rocsparse_initializer_laplace2d(rocsparse_int dimx, rocsparse_int dimy)
        : m_dimx(dimx)
        , m_dimy(dimy){};

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_csr_laplace2d(
            csr_row_ptr, csr_col_ind, csr_val, this->m_dimx, this->m_dimy, M, N, nnz, base);
    };

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_coo_laplace2d(
            coo_row_ind, coo_col_ind, coo_val, this->m_dimx, this->m_dimy, M, N, nnz, base);
    }
};

template <typename T>
struct rocsparse_initializer_laplace3d : public rocsparse_initializer_base<T>
{
private:
    rocsparse_int m_dimx, m_dimy, m_dimz;

public:
    rocsparse_initializer_laplace3d(rocsparse_int dimx, rocsparse_int dimy, rocsparse_int dimz)
        : m_dimx(dimx)
        , m_dimy(dimy)
        , m_dimz(dimz){};

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_csr_laplace3d(csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     this->m_dimx,
                                     this->m_dimy,
                                     this->m_dimz,
                                     M,
                                     N,
                                     nnz,
                                     base);
    };

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_coo_laplace3d(coo_row_ind,
                                     coo_col_ind,
                                     coo_val,
                                     this->m_dimx,
                                     this->m_dimy,
                                     this->m_dimz,
                                     M,
                                     N,
                                     nnz,
                                     base);
    }
};

std::string rocsparse_exepath();

template <typename T>
struct rocsparse_initializer : public rocsparse_initializer_base<T>
{
private:
    rocsparse_initializer_base<T>* m_instance;

public:
    virtual ~rocsparse_initializer()
    {
        if(this->m_instance)
        {
            delete this->m_instance;
            this->m_instance = nullptr;
        }
    }

    rocsparse_initializer(const Arguments& arg, bool to_int = false, bool full_rank = false)
    {
        switch(arg.matrix)
        {
        case rocsparse_matrix_random:
        {
            this->m_instance = new rocsparse_initializer_random<T>(full_rank);
            break;
        }

        case rocsparse_matrix_laplace_2d:
        {
            this->m_instance = new rocsparse_initializer_laplace2d<T>(arg.dimx, arg.dimy);
            break;
        }

        case rocsparse_matrix_laplace_3d:
        {
            this->m_instance = new rocsparse_initializer_laplace3d<T>(arg.dimx, arg.dimy, arg.dimz);
            break;
        }

        case rocsparse_matrix_file_rocalution:
        {
            std::string filename
                = arg.timing ? arg.filename
                             : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";
            this->m_instance = new rocsparse_initializer_rocalution<T>(filename.c_str());
            break;
        }

        case rocsparse_matrix_file_mtx:
        {
            std::string filename
                = arg.timing ? arg.filename
                             : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";
            this->m_instance = new rocsparse_initializer_mtx<T>(filename.c_str());
            break;
        }

        default:
        {
            this->m_instance = nullptr;
            break;
        }
        }
        assert(this->m_instance != nullptr);
    };

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        this->m_instance->init_csr(csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base);
    };

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        this->m_instance->init_coo(coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base);
    }
};

#endif // ROCSPARSE_INIT_HPP
