/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <iostream>
#include <rocsparse.h>
#include <vector>

/* ==================================================================================== */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
inline void rocsparse_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator<T>();
}

// Initializes sparse index vector with nnz entries ranging from start to end
inline void
    rocsparse_init_index(std::vector<rocsparse_int>& x, size_t nnz, size_t start, size_t end)
{
    std::vector<bool> check(end - start, false);

    rocsparse_int num = 0;

    while(num < nnz)
    {
        rocsparse_int val = random_generator<rocsparse_int>(start, end - 1);
        if(!check[val - start])
        {
            x[num++]           = val;
            check[val - start] = true;
        }
    }

    std::sort(x.begin(), x.end());
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
inline void rocsparse_init_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value                        = random_generator<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : -value;
            }
}

/* ==================================================================================== */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
inline void rocsparse_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocsparse_nan_rng());
}

template <typename T>
inline void rocsparse_init_nan(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocsparse_nan_rng());
}

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
inline void rocsparse_init_coo_matrix(std::vector<rocsparse_int>& row_ind,
                                      std::vector<rocsparse_int>& col_ind,
                                      std::vector<T>&             val,
                                      size_t                      M,
                                      size_t                      N,
                                      size_t                      nnz,
                                      rocsparse_index_base        base,
                                      bool                        full_rank = false)
{
    // If M > N, full rank is not possible
    if(full_rank && M > N)
    {
        std::cerr << "ERROR: M > N, cannot generate matrix with full rank" << std::endl;
        full_rank = false;
    }

    // If nnz < M, full rank is not possible
    if(full_rank && nnz < M)
    {
        std::cerr << "ERROR: nnz < M, cannot generate matrix with full rank" << std::endl;
        full_rank = false;
    }

    if(row_ind.size() != nnz)
    {
        row_ind.resize(nnz);
    }
    if(col_ind.size() != nnz)
    {
        col_ind.resize(nnz);
    }
    if(val.size() != nnz)
    {
        val.resize(nnz);
    }

    // Add diagonal entry, if full rank is flagged
    size_t i = 0;

    if(full_rank)
    {
        for(; i < M; ++i)
        {
            row_ind[i] = i;
        }
    }

    // Uniform distributed row indices
    for(; i < nnz; ++i)
    {
        row_ind[i] = random_generator<rocsparse_int>(0, M - 1);
    }

    // Sort row indices
    std::sort(row_ind.begin(), row_ind.end());

    // Sample column indices
    std::vector<bool> check(nnz, false);

    i = 0;
    while(i < nnz)
    {
        size_t begin = i;
        while(row_ind[i] == row_ind[begin])
        {
            ++i;
            if(i >= nnz)
            {
                break;
            }
        }

        // Sample i disjunct column indices
        size_t idx = begin;

        if(full_rank)
        {
            check[row_ind[idx]] = true;
            col_ind[idx++]      = row_ind[begin];
        }

        while(idx < i)
        {
            // Normal distribution around the diagonal
            rocsparse_int rng = (i - begin) * random_generator_normal<T>();

            if(M <= N)
            {
                rng += row_ind[begin];
            }

            // Repeat if running out of bounds
            if(rng < 0 || rng > N - 1)
            {
                continue;
            }

            // Check for disjunct column index in current row
            if(!check[rng])
            {
                check[rng]     = true;
                col_ind[idx++] = rng;
            }
        }

        // Reset disjunct check array
        for(size_t j = begin; j < i; ++j)
        {
            check[col_ind[j]] = false;
        }

        // Partially sort column indices
        std::sort(&col_ind[begin], &col_ind[i]);
    }

    // Correct index base accordingly
    if(base == rocsparse_index_base_one)
    {
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++row_ind[i];
            ++col_ind[i];
        }
    }

    // Sample random off-diagonal values
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        if(row_ind[i] == col_ind[i])
        {
            // Sample diagonal values
            val[i] = random_generator<T>();
        }
        else
        {
            // Samples off-diagonal values
            val[i] = random_generator<T>();
        }
    }
}

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in CSR format */
template <typename T>
inline void rocsparse_init_csr_laplace2d(std::vector<rocsparse_int>& row_ptr,
                                         std::vector<rocsparse_int>& col_ind,
                                         std::vector<T>&             val,
                                         rocsparse_int               dim_x,
                                         rocsparse_int               dim_y,
                                         rocsparse_int&              M,
                                         rocsparse_int&              N,
                                         rocsparse_int&              nnz,
                                         rocsparse_index_base        idx_base)
{
    // Do nothing
    if(dim_x == 0 || dim_y == 0)
    {
        return;
    }

    M = dim_x * dim_y;
    N = dim_x * dim_y;

    // Approximate 9pt stencil
    rocsparse_int nnz_mat = 9 * M;

    row_ptr.resize(M + 1);
    col_ind.resize(nnz_mat);
    val.resize(nnz_mat);

    nnz        = idx_base;
    row_ptr[0] = idx_base;

    // Fill local arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int iy = 0; iy < dim_y; ++iy)
    {
        for(rocsparse_int ix = 0; ix < dim_x; ++ix)
        {
            rocsparse_int row = iy * dim_x + ix;

            for(int sy = -1; sy <= 1; ++sy)
            {
                if(iy + sy > -1 && iy + sy < dim_y)
                {
                    for(int sx = -1; sx <= 1; ++sx)
                    {
                        if(ix + sx > -1 && ix + sx < dim_x)
                        {
                            rocsparse_int col = row + sy * dim_x + sx;

                            col_ind[nnz - idx_base] = col + idx_base;
                            val[nnz - idx_base]     = (col == row) ? 8.0 : -1.0;

                            ++nnz;
                        }
                    }
                }
            }

            row_ptr[row + 1] = nnz;
        }
    }

    // Adjust nnz by index base
    nnz -= idx_base;
}

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in COO format */
template <typename T>
inline void rocsparse_init_coo_laplace2d(std::vector<rocsparse_int>& row_ind,
                                         std::vector<rocsparse_int>& col_ind,
                                         std::vector<T>&             val,
                                         rocsparse_int               dim_x,
                                         rocsparse_int               dim_y,
                                         rocsparse_int&              M,
                                         rocsparse_int&              N,
                                         rocsparse_int&              nnz,
                                         rocsparse_index_base        base)
{
    std::vector<rocsparse_int> row_ptr(M + 1);

    // Sample CSR matrix
    rocsparse_init_csr_laplace2d(row_ptr, col_ind, val, dim_x, dim_y, M, N, nnz, base);

    // Convert to COO
    host_csr_to_coo(M, nnz, row_ptr, row_ind, base);
}

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in CSR format */
template <typename T>
inline void rocsparse_init_csr_laplace3d(std::vector<rocsparse_int>& row_ptr,
                                         std::vector<rocsparse_int>& col_ind,
                                         std::vector<T>&             val,
                                         rocsparse_int               dim_x,
                                         rocsparse_int               dim_y,
                                         rocsparse_int               dim_z,
                                         rocsparse_int&              M,
                                         rocsparse_int&              N,
                                         rocsparse_int&              nnz,
                                         rocsparse_index_base        idx_base)
{
    // Do nothing
    if(dim_x == 0 || dim_y == 0 || dim_z == 0)
    {
        return;
    }

    M = dim_x * dim_y * dim_z;
    N = dim_x * dim_y * dim_z;

    // Approximate 27pt stencil
    rocsparse_int nnz_mat = 27 * M;

    row_ptr.resize(M + 1);
    col_ind.resize(nnz_mat);
    val.resize(nnz_mat);

    nnz        = idx_base;
    row_ptr[0] = idx_base;

    // Fill local arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int iz = 0; iz < dim_z; ++iz)
    {
        for(rocsparse_int iy = 0; iy < dim_y; ++iy)
        {
            for(rocsparse_int ix = 0; ix < dim_x; ++ix)
            {
                rocsparse_int row = iz * dim_x * dim_y + iy * dim_x + ix;

                for(int sz = -1; sz <= 1; ++sz)
                {
                    if(iz + sz > -1 && iz + sz < dim_z)
                    {
                        for(int sy = -1; sy <= 1; ++sy)
                        {
                            if(iy + sy > -1 && iy + sy < dim_y)
                            {
                                for(int sx = -1; sx <= 1; ++sx)
                                {
                                    if(ix + sx > -1 && ix + sx < dim_x)
                                    {
                                        rocsparse_int col
                                            = row + sz * dim_x * dim_y + sy * dim_x + sx;

                                        col_ind[nnz - idx_base] = col + idx_base;
                                        val[nnz - idx_base]     = (col == row) ? 26.0 : -1.0;

                                        ++nnz;
                                    }
                                }
                            }
                        }
                    }
                }

                row_ptr[row + 1] = nnz;
            }
        }
    }

    // Adjust nnz by index base
    nnz -= idx_base;
}

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in COO format */
template <typename T>
inline void rocsparse_init_coo_laplace3d(std::vector<rocsparse_int>& row_ind,
                                         std::vector<rocsparse_int>& col_ind,
                                         std::vector<T>&             val,
                                         rocsparse_int               dim_x,
                                         rocsparse_int               dim_y,
                                         rocsparse_int               dim_z,
                                         rocsparse_int&              M,
                                         rocsparse_int&              N,
                                         rocsparse_int&              nnz,
                                         rocsparse_index_base        base)
{
    std::vector<rocsparse_int> row_ptr(M + 1);

    // Sample CSR matrix
    rocsparse_init_csr_laplace3d(row_ptr, col_ind, val, dim_x, dim_y, dim_z, M, N, nnz, base);

    // Convert to COO
    host_csr_to_coo(M, nnz, row_ptr, row_ind, base);
}

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename T>
inline void rocsparse_init_csr_rocalution(const char*                 filename,
                                          std::vector<rocsparse_int>& row_ptr,
                                          std::vector<rocsparse_int>& col_ind,
                                          std::vector<T>&             val,
                                          rocsparse_int&              M,
                                          rocsparse_int&              N,
                                          rocsparse_int&              nnz,
                                          rocsparse_index_base        base,
                                          bool                        toint)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Reading matrix " << filename << " ... ";
    }

    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if(!in.is_open())
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    std::string header;
    std::getline(in, header);

    if(header != "#rocALUTION binary csr file")
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    int version;
    in.read((char*)&version, sizeof(int));

    int iM;
    int iN;
    int innz;

    in.read((char*)&iM, sizeof(int));
    in.read((char*)&iN, sizeof(int));
    in.read((char*)&innz, sizeof(int));

    M   = iM;
    N   = iN;
    nnz = innz;

    // Allocate memory
    row_ptr.resize(M + 1);
    col_ind.resize(nnz);
    val.resize(nnz);

    std::vector<int>    iptr(M + 1);
    std::vector<int>    icol(nnz);
    std::vector<double> dval(nnz);

    in.read((char*)iptr.data(), sizeof(int) * (M + 1));
    in.read((char*)icol.data(), sizeof(int) * nnz);
    in.read((char*)dval.data(), sizeof(double) * nnz);

    in.close();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M + 1; ++i)
    {
        row_ptr[i] = static_cast<rocsparse_int>(iptr[i]);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        col_ind[i] = static_cast<rocsparse_int>(icol[i]);

        if(toint)
        {
            // Transform all values to integers to avoid rounding errors when testing but
            // preserving the sparsity pattern
            // val[i] = (dval[i] > static_cast<T>(0)) ? static_cast<T>(ceil(dval[i])) : static_cast<T>(floor(dval[i]));
            val[i] = std::abs(dval[i]);
        }
        else
        {
            val[i] = static_cast<T>(dval[i]);
        }
    }

    if(base == rocsparse_index_base_one)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int i = 0; i < M + 1; ++i)
        {
            ++row_ptr[i];
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++col_ind[i];
        }
    }

    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "done." << std::endl;
    }
}

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename T>
inline void rocsparse_init_coo_rocalution(const char*                 filename,
                                          std::vector<rocsparse_int>& row_ind,
                                          std::vector<rocsparse_int>& col_ind,
                                          std::vector<T>&             val,
                                          rocsparse_int&              M,
                                          rocsparse_int&              N,
                                          rocsparse_int&              nnz,
                                          rocsparse_index_base        base,
                                          bool                        toint)
{
    std::vector<rocsparse_int> row_ptr(M + 1);

    // Sample CSR matrix
    rocsparse_init_csr_rocalution(filename, row_ptr, col_ind, val, M, N, nnz, base, toint);

    // Convert to COO
    host_csr_to_coo(M, nnz, row_ptr, row_ind, base);
}

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in CSR format */
template <typename T>
inline void rocsparse_init_csr_random(std::vector<rocsparse_int>& row_ptr,
                                      std::vector<rocsparse_int>& col_ind,
                                      std::vector<T>&             val,
                                      rocsparse_int               M,
                                      rocsparse_int               N,
                                      rocsparse_int&              nnz,
                                      rocsparse_index_base        base,
                                      bool                        full_rank = false)
{
    // Compute non-zero entries of the matrix
    nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;

    // Sample random matrix
    std::vector<rocsparse_int> row_ind(nnz);

    // Sample COO matrix
    rocsparse_init_coo_matrix(row_ind, col_ind, val, M, N, nnz, base, full_rank);

    // Convert to CSR
    host_coo_to_csr(M, nnz, row_ind, row_ptr, base);
}

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
inline void rocsparse_init_coo_random(std::vector<rocsparse_int>& row_ind,
                                      std::vector<rocsparse_int>& col_ind,
                                      std::vector<T>&             val,
                                      rocsparse_int               M,
                                      rocsparse_int               N,
                                      rocsparse_int&              nnz,
                                      rocsparse_index_base        base,
                                      bool                        full_rank = false)
{
    // Compute non-zero entries of the matrix
    nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;

    // Sample random matrix
    rocsparse_init_coo_matrix(row_ind, col_ind, val, M, N, nnz, base, full_rank);
}

/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in CSR format */
template <typename T>
inline void rocsparse_init_csr_matrix(std::vector<rocsparse_int>& csr_row_ptr,
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
                                      bool                        full_rank = false)
{
    // Differentiate the different matrix generators
    if(matrix == rocsparse_matrix_random)
    {
        rocsparse_init_csr_random(csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base, full_rank);
    }
    else if(matrix == rocsparse_matrix_laplace_2d)
    {
        rocsparse_init_csr_laplace2d(
            csr_row_ptr, csr_col_ind, csr_val, dim_x, dim_y, M, N, nnz, base);
    }
    else if(matrix == rocsparse_matrix_laplace_3d)
    {
        rocsparse_init_csr_laplace3d(
            csr_row_ptr, csr_col_ind, csr_val, dim_x, dim_y, dim_z, M, N, nnz, base);
    }
    else if(matrix == rocsparse_matrix_file_rocalution)
    {
        rocsparse_init_csr_rocalution(
            filename, csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base, toint);
    }
}

/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in COO format */
template <typename T>
inline void rocsparse_init_coo_matrix(std::vector<rocsparse_int>& coo_row_ind,
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
                                      bool                        full_rank = false)
{
    // Differentiate the different matrix generators
    if(matrix == rocsparse_matrix_random)
    {
        rocsparse_init_coo_random(coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, full_rank);
    }
    else if(matrix == rocsparse_matrix_laplace_2d)
    {
        rocsparse_init_coo_laplace2d(
            coo_row_ind, coo_col_ind, coo_val, dim_x, dim_y, M, N, nnz, base);
    }
    else if(matrix == rocsparse_matrix_laplace_3d)
    {
        rocsparse_init_coo_laplace3d(
            coo_row_ind, coo_col_ind, coo_val, dim_x, dim_y, dim_z, M, N, nnz, base);
    }
    else if(matrix == rocsparse_matrix_file_rocalution)
    {
        rocsparse_init_coo_rocalution(
            filename, coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, toint);
    }
}

#endif // ROCSPARSE_INIT_HPP
