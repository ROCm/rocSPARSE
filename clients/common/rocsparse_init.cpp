/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#include "rocsparse_init.hpp"
#include "rocsparse_import.hpp"
#include "rocsparse_matrix.hpp"

template <typename I, typename J>
void host_coo_to_csr(
    J M, I nnz, const J* coo_row_ind, std::vector<I>& csr_row_ptr, rocsparse_index_base base)
{
    // Resize and initialize csr_row_ptr with zeros
    csr_row_ptr.resize(M + 1, 0);
    for(size_t i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr[coo_row_ind[i] + 1 - base];
    }

    csr_row_ptr[0] = base;
    for(J i = 0; i < M; ++i)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }
}

template <typename I, typename J>
void host_csr_to_coo(J                     M,
                     I                     nnz,
                     const std::vector<I>& csr_row_ptr,
                     std::vector<J>&       coo_row_ind,
                     rocsparse_index_base  base)
{
    // Resize coo_row_ind
    coo_row_ind.resize(nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr[i] - base;
        I row_end   = csr_row_ptr[i + 1] - base;

        for(I j = row_begin; j < row_end; ++j)
        {
            coo_row_ind[j] = i + base;
        }
    }
}

template <typename I, typename J>
void host_csr_to_coo_aos(J                     M,
                         I                     nnz,
                         const std::vector<I>& csr_row_ptr,
                         const std::vector<J>& csr_col_ind,
                         std::vector<I>&       coo_ind,
                         rocsparse_index_base  base)
{
    // Resize coo_ind
    coo_ind.resize(2 * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr[i] - base;
        I row_end   = csr_row_ptr[i + 1] - base;

        for(I j = row_begin; j < row_end; ++j)
        {
            coo_ind[2 * j]     = i + base;
            coo_ind[2 * j + 1] = static_cast<I>(csr_col_ind[j]);
        }
    }
}

template <typename I, typename J, typename T>
void host_csr_to_ell(J                     M,
                     const std::vector<I>& csr_row_ptr,
                     const std::vector<J>& csr_col_ind,
                     const std::vector<T>& csr_val,
                     std::vector<J>&       ell_col_ind,
                     std::vector<T>&       ell_val,
                     J&                    ell_width,
                     rocsparse_index_base  csr_base,
                     rocsparse_index_base  ell_base)
{
    // Determine ELL width
    ell_width = 0;

    for(J i = 0; i < M; ++i)
    {
        J row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        ell_width = std::max(row_nnz, ell_width);
    }

    // Compute ELL non-zeros
    I ell_nnz = ell_width * M;

    ell_col_ind.resize(ell_nnz);
    ell_val.resize(ell_nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        J p = 0;

        I row_begin = csr_row_ptr[i] - csr_base;
        I row_end   = csr_row_ptr[i + 1] - csr_base;
        J row_nnz   = row_end - row_begin;

        // Fill ELL matrix with data
        for(I j = row_begin; j < row_end; ++j)
        {
            I idx = p * M + i;

            ell_col_ind[idx] = csr_col_ind[j] - csr_base + ell_base;
            ell_val[idx]     = csr_val[j];

            ++p;
        }

        // Add padding to ELL structures
        for(J j = row_nnz; j < ell_width; ++j)
        {
            I idx = p * M + i;

            ell_col_ind[idx] = -1;
            ell_val[idx]     = static_cast<T>(0);

            ++p;
        }
    }
}

/* ==================================================================================== */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values

template <typename T>
void rocsparse_init_exact(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, int a, int b)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t j = 0; j < N; ++j)
            for(size_t i = 0; i < M; ++i)
            {
                A[i + j * lda + i_batch * stride] = random_generator_exact<T>(a, b);
            }
}

template <typename T>
void rocsparse_init(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, T a, T b)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t j = 0; j < N; ++j)
            for(size_t i = 0; i < M; ++i)
            {
                A[i + j * lda + i_batch * stride] = random_generator<T>(a, b);
            }
}

template <typename T>
void rocsparse_init_exact(std::vector<T>& A,
                          size_t          M,
                          size_t          N,
                          size_t          lda,
                          size_t          stride,
                          size_t          batch_count,
                          int             a,
                          int             b)
{
    rocsparse_init_exact(A.data(), M, N, lda, stride, batch_count, a, b);
}

template <typename T>
void rocsparse_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, T a, T b)
{
    rocsparse_init(A.data(), M, N, lda, stride, batch_count, a, b);
}

// Initializes sparse index vector with nnz entries ranging from start to end
template <typename I>
void rocsparse_init_index(std::vector<I>& x, size_t nnz, size_t start, size_t end)
{
    std::vector<bool> check(end - start, false);

    I num = 0;

    while(num < nnz)
    {
        I val = random_generator<I>(start, end - 1);
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
void rocsparse_init_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value                        = random_generator_exact<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : -value;
            }
}

/* ==================================================================================== */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
void rocsparse_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocsparse_nan_rng());
}

template void rocsparse_init_nan<int32_t>(int32_t* A, size_t N);
template void rocsparse_init_nan<int64_t>(int64_t* A, size_t N);
template void rocsparse_init_nan<size_t>(size_t* A, size_t N);

template <typename T>
void rocsparse_init_nan(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocsparse_nan_rng());
}

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
                               bool                 full_rank,
                               bool                 to_int)
{
    if(nnz == 0)
    {
        row_ind.resize(nnz);
        col_ind.resize(nnz);
        val.resize(nnz);
        return;
    }
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

    //
    // Compute row indices.
    //
    std::vector<int> count(M, 0);
    I                start = 0;
    if(full_rank)
    {
        for(I k = 0; k < std::min(M, nnz); ++k)
        {
            count[k] += 1;
            row_ind[k] = k;
        }
        start = std::min(M, nnz);
    }

    for(I k = start; k < nnz; ++k)
    {
        I   i       = random_generator<I>(0, M - 1);
        int maxiter = 0;
        while(count[i] >= N && maxiter++ < 10)
        {
            i = random_generator<I>(0, M - 1);
        }
        if(maxiter >= 10)
        {
            for(i = 0; i < M; ++i)
            {
                if(count[i] < N)
                {
                    break;
                }
            }
            if(i == M)
            {
                std::cerr << "rocsparse_init_coo_matrix error" << std::endl;
                exit(1);
            }
        }
        count[i] += 1;
        row_ind[k] = i;
    }

    std::sort(row_ind.begin(), row_ind.end());

    std::vector<bool> marker(N, false);
    std::vector<I>    select(N, 0);
    I                 mx = count[0];
    for(I i = 1; i < M; ++i)
    {
        if(mx < count[i])
            mx = count[i];
    }
    I sec = std::min(2 * mx, N);
    I at  = 0;
    for(I i = 0; i < M; ++i)
    {
        I select_n = 0;
        I begin    = at;
        for(I k = 0; k < count[i]; ++k)
        {
            I j;
            if(full_rank && k == 0)
            {
                j = i;
            }
            else
            {
                I maxiter = 0;
                do
                {
                    //
                    // Generate coefficients close to the diagonal
                    //
                    I bmax = std::min(i + sec, N - 1);
                    I bmin = std::max(bmax - 2 * sec, ((I)0));

                    j = random_generator<I>(bmin, bmax);
                    ++maxiter;
                    if(maxiter == 10)
                    {
                        break;
                    }
                } while(j < 0 || j >= N || marker[j]);
                if(maxiter == 10)
                {
                    for(j = 0; j < N; ++j)
                    {
                        if(!marker[j])
                        {
                            break;
                        }
                    }
                    if(j == N)
                    {
                        std::cerr << "rocsparse_init_coo_matrix error" << std::endl;
                        exit(1);
                    }
                }
            }

            select[select_n++] = j;
            marker[j]          = true;
            col_ind[at++]      = j;
        }

        for(I k = 0; k < select_n; ++k)
        {
            marker[select[k]] = false;
        }

        if(count[i] > 0)
        {
            std::sort(&col_ind[begin], &col_ind[begin + count[i]]);
        }
    }

    // Correct index base accordingly
    if(base == rocsparse_index_base_one)
    {
        for(I i = 0; i < nnz; ++i)
        {
            ++row_ind[i];
            ++col_ind[i];
        }
    }

    if(to_int)
    {

        // Sample random off-diagonal values
        for(I i = 0; i < nnz; ++i)
        {
            if(row_ind[i] == col_ind[i])
            {
                // Sample diagonal values
                val[i] = random_generator_exact<T>();
            }
            else
            {
                // Samples off-diagonal values
                val[i] = random_generator_exact<T>();
            }
        }
    }
    else
    {
        if(full_rank)
        {
            // Sample random off-diagonal values
            for(I i = 0; i < nnz; ++i)
            {
                if(row_ind[i] == col_ind[i])
                {
                    // Sample diagonal values
                    val[i] = random_generator<T>(static_cast<T>(4.0), static_cast<T>(8.0));
                    val[i]
                        += val[i]
                           * random_generator<T>(static_cast<T>(-1.0e-2), static_cast<T>(1.0e-2));
                }
                else
                {
                    // Samples off-diagonal values
                    val[i] = random_generator<T>(static_cast<T>(-0.5), static_cast<T>(0.5));
                }
            }
        }
        else
        {
            // Sample random off-diagonal values
            for(I i = 0; i < nnz; ++i)
            {
                val[i] = random_generator<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
            }
        }
    }
}

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
                                  rocsparse_index_base base)
{
    // Do nothing
    if(dim_x == 0 || dim_y == 0)
    {
        return;
    }

    M = dim_x * dim_y;
    N = dim_x * dim_y;

    // Approximate 9pt stencil
    I nnz_mat = 9 * M;

    row_ptr.resize(M + 1);
    col_ind.resize(nnz_mat);
    val.resize(nnz_mat);

    nnz        = base;
    row_ptr[0] = base;

    // Fill local arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int32_t iy = 0; iy < dim_y; ++iy)
    {
        for(int32_t ix = 0; ix < dim_x; ++ix)
        {
            J row = iy * dim_x + ix;

            for(int32_t sy = -1; sy <= 1; ++sy)
            {
                if(iy + sy > -1 && iy + sy < dim_y)
                {
                    for(int32_t sx = -1; sx <= 1; ++sx)
                    {
                        if(ix + sx > -1 && ix + sx < dim_x)
                        {
                            J col = row + sy * dim_x + sx;

                            col_ind[nnz - base] = col + base;
                            val[nnz - base]     = (col == row) ? 8.0 : -1.0;

                            ++nnz;
                        }
                    }
                }
            }

            row_ptr[row + 1] = nnz;
        }
    }

    // Adjust nnz by index base
    nnz -= base;
}

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
                                  rocsparse_index_base base)
{
    std::vector<I> row_ptr;

    // Sample CSR matrix
    rocsparse_init_csr_laplace2d(row_ptr, col_ind, val, dim_x, dim_y, M, N, nnz, base);

    // Convert to COO
    host_csr_to_coo(M, nnz, row_ptr, row_ind, base);
}

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_laplace2d(std::vector<I>&      row_ptr,
                                    std::vector<J>&      col_ind,
                                    std::vector<T>&      val,
                                    int32_t              dim_x,
                                    int32_t              dim_y,
                                    J&                   Mb,
                                    J&                   Nb,
                                    I&                   nnzb,
                                    J                    row_block_dim,
                                    J                    col_block_dim,
                                    rocsparse_index_base base)
{
    rocsparse_init_csr_laplace2d(row_ptr, col_ind, val, dim_x, dim_y, Mb, Nb, nnzb, base);

    I nvalues = nnzb * row_block_dim * col_block_dim;
    val.resize(nvalues);
    for(I i = 0; i < nvalues; ++i)
    {
        val[i] = random_generator<T>();
    }
}

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
                                  rocsparse_index_base base)
{
    I csr_nnz;

    std::vector<I> csr_row_ptr;
    std::vector<I> csr_col_ind;
    std::vector<T> csr_val;

    // Sample CSR matrix
    rocsparse_init_csr_laplace2d(
        csr_row_ptr, csr_col_ind, csr_val, dim_x, dim_y, M, N, csr_nnz, base);

    // Convert to ELL
    host_csr_to_ell(M, csr_row_ptr, csr_col_ind, csr_val, col_ind, val, width, base, base);
}

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
                                  rocsparse_index_base base)
{
    // Do nothing
    if(dim_x == 0 || dim_y == 0 || dim_z == 0)
    {
        return;
    }

    M = dim_x * dim_y * dim_z;
    N = dim_x * dim_y * dim_z;

    // Approximate 27pt stencil
    I nnz_mat = 27 * M;

    row_ptr.resize(M + 1);
    col_ind.resize(nnz_mat);
    val.resize(nnz_mat);

    nnz        = base;
    row_ptr[0] = base;

    // Fill local arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int32_t iz = 0; iz < dim_z; ++iz)
    {
        for(int32_t iy = 0; iy < dim_y; ++iy)
        {
            for(int32_t ix = 0; ix < dim_x; ++ix)
            {
                J row = iz * dim_x * dim_y + iy * dim_x + ix;

                for(int32_t sz = -1; sz <= 1; ++sz)
                {
                    if(iz + sz > -1 && iz + sz < dim_z)
                    {
                        for(int32_t sy = -1; sy <= 1; ++sy)
                        {
                            if(iy + sy > -1 && iy + sy < dim_y)
                            {
                                for(int32_t sx = -1; sx <= 1; ++sx)
                                {
                                    if(ix + sx > -1 && ix + sx < dim_x)
                                    {
                                        J col = row + sz * dim_x * dim_y + sy * dim_x + sx;

                                        col_ind[nnz - base] = col + base;
                                        val[nnz - base]     = (col == row) ? 26.0 : -1.0;

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
    nnz -= base;
}

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
                                  rocsparse_index_base base)
{
    std::vector<I> row_ptr(M + 1);

    // Sample CSR matrix
    rocsparse_init_csr_laplace3d(row_ptr, col_ind, val, dim_x, dim_y, dim_z, M, N, nnz, base);

    // Convert to COO
    host_csr_to_coo(M, nnz, row_ptr, row_ind, base);
}

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_laplace3d(std::vector<I>&      row_ptr,
                                    std::vector<J>&      col_ind,
                                    std::vector<T>&      val,
                                    int32_t              dim_x,
                                    int32_t              dim_y,
                                    int32_t              dim_z,
                                    J&                   Mb,
                                    J&                   Nb,
                                    I&                   nnzb,
                                    J                    row_block_dim,
                                    J                    col_block_dim,
                                    rocsparse_index_base base)
{
    rocsparse_init_csr_laplace3d(row_ptr, col_ind, val, dim_x, dim_y, dim_z, Mb, Nb, nnzb, base);

    I nvalues = nnzb * row_block_dim * col_block_dim;
    val.resize(nvalues);
    for(I i = 0; i < nvalues; ++i)
    {
        val[i] = random_generator<T>();
    }
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */

/* ==================================================================================== */
/*! \brief  Read matrix from mtx file in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_mtx(const char*          filename,
                            std::vector<I>&      csr_row_ptr,
                            std::vector<J>&      csr_col_ind,
                            std::vector<T>&      csr_val,
                            J&                   M,
                            J&                   N,
                            I&                   nnz,
                            rocsparse_index_base base)
{
    I coo_M, coo_N;

    std::vector<I> coo_row_ind;
    std::vector<I> coo_col_ind;

    // Read COO matrix
    rocsparse_init_coo_mtx(filename, coo_row_ind, coo_col_ind, csr_val, coo_M, coo_N, nnz, base);

    // Convert to CSR
    M = (J)coo_M;
    N = (J)coo_N;

    csr_row_ptr.resize(M + 1);
    csr_col_ind.resize(nnz);

    host_coo_to_csr(coo_M, nnz, coo_row_ind.data(), csr_row_ptr, base);

    for(I i = 0; i < nnz; ++i)
    {
        csr_col_ind[i] = (J)coo_col_ind[i];
    }
}

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
                            rocsparse_index_base base)
{
    rocsparse_importer_matrixmarket importer(filename);
    rocsparse_status                status
        = rocsparse_import_sparse_coo(importer, coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base);
    CHECK_ROCSPARSE_ERROR(status);
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_mtx(const char*          filename,
                              std::vector<I>&      bsr_row_ptr,
                              std::vector<J>&      bsr_col_ind,
                              std::vector<T>&      bsr_val,
                              J&                   Mb,
                              J&                   Nb,
                              I&                   nnzb,
                              J                    row_block_dim,
                              J                    col_block_dim,
                              rocsparse_index_base base)
{
    //this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);
    rocsparse_init_csr_mtx(filename, bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

    I nvalues = nnzb * row_block_dim * col_block_dim;
    bsr_val.resize(nvalues);
    for(I i = 0; i < nvalues; ++i)
    {
        bsr_val[i] = random_generator<T>();
    }
}

template <typename I, typename J, typename T>
void rocsparse_init_csr_rocalution(const char*          filename,
                                   std::vector<I>&      row_ptr,
                                   std::vector<J>&      col_ind,
                                   std::vector<T>&      val,
                                   J&                   M,
                                   J&                   N,
                                   I&                   nnz,
                                   rocsparse_index_base base,
                                   bool                 toint)
{
    rocsparse_importer_rocalution importer(filename);
    rocsparse_status              status
        = rocsparse_import_sparse_csr(importer, row_ptr, col_ind, val, M, N, nnz, base);

    if(toint)
    {
        //
        // On GPU ?
        //
        size_t bound = static_cast<size_t>(nnz);
        for(size_t i = 0; i < bound; ++i)
            val[i] = std::abs(val[i]);
    }
    CHECK_ROCSPARSE_ERROR(status);
}

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
                                   bool                 toint)
{
    std::vector<I> row_ptr(M + 1);

    // Sample CSR matrix
    rocsparse_init_csr_rocalution(filename, row_ptr, col_ind, val, M, N, nnz, base, toint);

    // Convert to COO
    host_csr_to_coo(M, nnz, row_ptr, row_ind, base);
}

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_rocalution(const char*          filename,
                                     std::vector<I>&      row_ptr,
                                     std::vector<J>&      col_ind,
                                     std::vector<T>&      val,
                                     J&                   Mb,
                                     J&                   Nb,
                                     I&                   nnzb,
                                     J                    row_block_dim,
                                     J                    col_block_dim,
                                     rocsparse_index_base base,
                                     bool                 toint)
{
    // Temporarily the file contains a CSR matrix.
    rocsparse_init_csr_rocalution(filename, row_ptr, col_ind, val, Mb, Nb, nnzb, base, toint);

    // Then temporarily skip the values.
    I nvalues = nnzb * row_block_dim * col_block_dim;
    val.resize(nvalues);
    for(I i = 0; i < nvalues; ++i)
    {
        val[i] = random_generator<T>();
    }
}

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_random(std::vector<I>&            csr_row_ptr,
                               std::vector<J>&            csr_col_ind,
                               std::vector<T>&            csr_val,
                               J                          M,
                               J                          N,
                               I&                         nnz,
                               rocsparse_index_base       base,
                               rocsparse_matrix_init_kind init_kind,
                               bool                       full_rank,
                               bool                       to_int)
{
    switch(init_kind)
    {
    case rocsparse_matrix_init_kind_tunedavg:
    {
        rocsparse_int alpha = static_cast<rocsparse_int>(0);
        if(N >= 16384)
        {
            alpha = static_cast<rocsparse_int>(4);
        }
        else if(N >= 8192)
        {
            alpha = static_cast<rocsparse_int>(8);
        }
        else if(N >= 4096)
        {
            alpha = static_cast<rocsparse_int>(16);
        }
        else if(N >= 1024)
        {
            alpha = static_cast<rocsparse_int>(32);
        }
        else
        {
            alpha = static_cast<rocsparse_int>(64);
        }

        nnz = M * alpha;
        nnz = std::min(nnz, static_cast<I>(M) * static_cast<I>(N));

        // Sample random matrix
        std::vector<J> row_ind(nnz);
        // Sample COO matrix
        rocsparse_init_coo_matrix<J>(
            row_ind, csr_col_ind, csr_val, M, N, nnz, base, full_rank, to_int);

        // Convert to CSR
        host_coo_to_csr(M, nnz, row_ind.data(), csr_row_ptr, base);
        break;
    }

    case rocsparse_matrix_init_kind_default:
    {
        if(M < 32 && N < 32)
        {
            nnz = (M * N) / 4;
            if(full_rank)
            {
                nnz = std::max(nnz, static_cast<I>(M));
            }
            nnz = std::max(nnz, static_cast<I>(M));
            nnz = std::min(nnz, static_cast<I>(M) * static_cast<I>(N));
        }
        else
        {
            nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;
        }

        // Sample random matrix
        std::vector<J> row_ind(nnz);
        // Sample COO matrix
        rocsparse_init_coo_matrix<J>(
            row_ind, csr_col_ind, csr_val, M, N, nnz, base, full_rank, to_int);

        // Convert to CSR
        host_coo_to_csr(M, nnz, row_ind.data(), csr_row_ptr, base);
        break;
    }
    }
}

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
                               bool                 full_rank,
                               bool                 to_int)
{
    // Compute non-zero entries of the matrix
    if(M < 32 && N < 32)
    {
        nnz = (M * N) / 4;
    }
    else
    {
        nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;
    }
    // Sample random matrix

    rocsparse_init_coo_matrix(row_ind, col_ind, val, M, N, nnz, base, full_rank, to_int);
}

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_random(std::vector<I>&            row_ptr,
                                 std::vector<J>&            col_ind,
                                 std::vector<T>&            val,
                                 J                          Mb,
                                 J                          Nb,
                                 I&                         nnzb,
                                 J                          row_block_dim,
                                 J                          col_block_dim,
                                 rocsparse_index_base       base,
                                 rocsparse_matrix_init_kind init_kind,
                                 bool                       full_rank,
                                 bool                       to_int)
{
    rocsparse_init_csr_random(
        row_ptr, col_ind, val, Mb, Nb, nnzb, base, init_kind, full_rank, to_int);

    I nvalues = nnzb * row_block_dim * col_block_dim;
    val.resize(nvalues);
    if(to_int)
    {
        for(I i = 0; i < nvalues; ++i)
        {
            val[i] = random_generator_exact<T>();
        }
    }
    else
    {
        for(I i = 0; i < nvalues; ++i)
        {
            val[i] = random_generator<T>();
        }
    }
}

#define INSTANTIATEI(TYPE)                    \
    template void rocsparse_init_index<TYPE>( \
        std::vector<TYPE> & x, size_t nnz, size_t start, size_t end);

#define INSTANTIATE(TYPE)                                                                          \
    template void rocsparse_init<TYPE>(TYPE * A,                                                   \
                                       size_t M,                                                   \
                                       size_t N,                                                   \
                                       size_t lda,                                                 \
                                       size_t stride,                                              \
                                       size_t batch_count = 1,                                     \
                                       TYPE   a           = static_cast<TYPE>(0),                  \
                                       TYPE   b           = static_cast<TYPE>(1));                             \
    template void rocsparse_init_exact<TYPE>(TYPE * A,                                             \
                                             size_t M,                                             \
                                             size_t N,                                             \
                                             size_t lda,                                           \
                                             size_t stride,                                        \
                                             size_t batch_count,                                   \
                                             int    a = 1,                                         \
                                             int    b = 10);                                          \
    template void rocsparse_init<TYPE>(std::vector<TYPE> & A,                                      \
                                       size_t M,                                                   \
                                       size_t N,                                                   \
                                       size_t lda,                                                 \
                                       size_t stride,                                              \
                                       size_t batch_count = 1,                                     \
                                       TYPE   a           = static_cast<TYPE>(0),                  \
                                       TYPE   b           = static_cast<TYPE>(1));                             \
    template void rocsparse_init_exact<TYPE>(std::vector<TYPE> & A,                                \
                                             size_t M,                                             \
                                             size_t N,                                             \
                                             size_t lda,                                           \
                                             size_t stride,                                        \
                                             size_t batch_count,                                   \
                                             int    a = 1,                                         \
                                             int    b = 10);                                          \
    template void rocsparse_init_alternating_sign<TYPE>(                                           \
        std::vector<TYPE> & A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count); \
    template void rocsparse_init_nan<TYPE>(TYPE * A, size_t N);                                    \
    template void rocsparse_init_nan<TYPE>(std::vector<TYPE> & A,                                  \
                                           size_t M,                                               \
                                           size_t N,                                               \
                                           size_t lda,                                             \
                                           size_t stride = 0,                                      \
                                           size_t batch_count);

#define INSTANTIATE1(ITYPE, JTYPE)                                                         \
    template void host_csr_to_coo<ITYPE, JTYPE>(JTYPE                     M,               \
                                                ITYPE                     nnz,             \
                                                const std::vector<ITYPE>& csr_row_ptr,     \
                                                std::vector<JTYPE>&       coo_row_ind,     \
                                                rocsparse_index_base      base);                \
    template void host_coo_to_csr<ITYPE, JTYPE>(JTYPE                M,                    \
                                                ITYPE                NNZ,                  \
                                                const JTYPE*         coo_row_ind,          \
                                                std::vector<ITYPE>&  csr_row_ptr,          \
                                                rocsparse_index_base base);                \
    template void host_csr_to_coo_aos<ITYPE, JTYPE>(JTYPE                     M,           \
                                                    ITYPE                     nnz,         \
                                                    const std::vector<ITYPE>& csr_row_ptr, \
                                                    const std::vector<JTYPE>& csr_col_ind, \
                                                    std::vector<ITYPE>&       coo_ind,     \
                                                    rocsparse_index_base      base)

#define INSTANTIATE2(ITYPE, TTYPE)                                                           \
    template void rocsparse_init_coo_laplace2d<ITYPE, TTYPE>(std::vector<ITYPE> & row_ind,   \
                                                             std::vector<ITYPE> & col_ind,   \
                                                             std::vector<TTYPE> & val,       \
                                                             int32_t dim_x,                  \
                                                             int32_t dim_y,                  \
                                                             ITYPE & M,                      \
                                                             ITYPE & N,                      \
                                                             ITYPE & nnz,                    \
                                                             rocsparse_index_base base);     \
    template void rocsparse_init_ell_laplace2d<ITYPE, TTYPE>(std::vector<ITYPE> & col_ind,   \
                                                             std::vector<TTYPE> & val,       \
                                                             int32_t dim_x,                  \
                                                             int32_t dim_y,                  \
                                                             ITYPE & M,                      \
                                                             ITYPE & N,                      \
                                                             ITYPE & width,                  \
                                                             rocsparse_index_base base);     \
    template void rocsparse_init_coo_matrix<ITYPE, TTYPE>(std::vector<ITYPE> & row_ind,      \
                                                          std::vector<ITYPE> & col_ind,      \
                                                          std::vector<TTYPE> & val,          \
                                                          ITYPE                M,            \
                                                          ITYPE                N,            \
                                                          ITYPE                nnz,          \
                                                          rocsparse_index_base base,         \
                                                          bool                 full_rank,    \
                                                          bool                 to_int);                      \
    template void rocsparse_init_coo_laplace3d<ITYPE, TTYPE>(std::vector<ITYPE> & row_ind,   \
                                                             std::vector<ITYPE> & col_ind,   \
                                                             std::vector<TTYPE> & val,       \
                                                             int32_t dim_x,                  \
                                                             int32_t dim_y,                  \
                                                             int32_t dim_z,                  \
                                                             ITYPE & M,                      \
                                                             ITYPE & N,                      \
                                                             ITYPE & nnz,                    \
                                                             rocsparse_index_base base);     \
    template void rocsparse_init_coo_mtx<ITYPE, TTYPE>(const char*          filename,        \
                                                       std::vector<ITYPE>&  coo_row_ind,     \
                                                       std::vector<ITYPE>&  coo_col_ind,     \
                                                       std::vector<TTYPE>&  coo_val,         \
                                                       ITYPE&               M,               \
                                                       ITYPE&               N,               \
                                                       ITYPE&               nnz,             \
                                                       rocsparse_index_base base);           \
    template void rocsparse_init_coo_rocalution<ITYPE, TTYPE>(const char*          filename, \
                                                              std::vector<ITYPE>&  row_ind,  \
                                                              std::vector<ITYPE>&  col_ind,  \
                                                              std::vector<TTYPE>&  val,      \
                                                              ITYPE&               M,        \
                                                              ITYPE&               N,        \
                                                              ITYPE&               nnz,      \
                                                              rocsparse_index_base base,     \
                                                              bool                 toint);                   \
    template void rocsparse_init_coo_random<ITYPE, TTYPE>(std::vector<ITYPE> & row_ind,      \
                                                          std::vector<ITYPE> & col_ind,      \
                                                          std::vector<TTYPE> & val,          \
                                                          ITYPE M,                           \
                                                          ITYPE N,                           \
                                                          ITYPE & nnz,                       \
                                                          rocsparse_index_base base,         \
                                                          bool                 full_rank,    \
                                                          bool                 to_int);

#define INSTANTIATE3(ITYPE, JTYPE, TTYPE)                                                           \
    template void rocsparse_init_csr_laplace2d<ITYPE, JTYPE, TTYPE>(std::vector<ITYPE> & row_ptr,   \
                                                                    std::vector<JTYPE> & col_ind,   \
                                                                    std::vector<TTYPE> & val,       \
                                                                    int32_t dim_x,                  \
                                                                    int32_t dim_y,                  \
                                                                    JTYPE & M,                      \
                                                                    JTYPE & N,                      \
                                                                    ITYPE & nnz,                    \
                                                                    rocsparse_index_base base);     \
    template void rocsparse_init_csr_laplace3d<ITYPE, JTYPE, TTYPE>(std::vector<ITYPE> & row_ptr,   \
                                                                    std::vector<JTYPE> & col_ind,   \
                                                                    std::vector<TTYPE> & val,       \
                                                                    int32_t dim_x,                  \
                                                                    int32_t dim_y,                  \
                                                                    int32_t dim_z,                  \
                                                                    JTYPE & M,                      \
                                                                    JTYPE & N,                      \
                                                                    ITYPE & nnz,                    \
                                                                    rocsparse_index_base base);     \
    template void rocsparse_init_csr_mtx<ITYPE, JTYPE, TTYPE>(const char*          filename,        \
                                                              std::vector<ITYPE>&  csr_row_ptr,     \
                                                              std::vector<JTYPE>&  csr_col_ind,     \
                                                              std::vector<TTYPE>&  csr_val,         \
                                                              JTYPE&               M,               \
                                                              JTYPE&               N,               \
                                                              ITYPE&               nnz,             \
                                                              rocsparse_index_base base);           \
    template void rocsparse_init_csr_rocalution<ITYPE, JTYPE, TTYPE>(const char*          filename, \
                                                                     std::vector<ITYPE>&  row_ptr,  \
                                                                     std::vector<JTYPE>&  col_ind,  \
                                                                     std::vector<TTYPE>&  val,      \
                                                                     JTYPE&               M,        \
                                                                     JTYPE&               N,        \
                                                                     ITYPE&               nnz,      \
                                                                     rocsparse_index_base base,     \
                                                                     bool                 toint);                   \
    template void rocsparse_init_csr_random<ITYPE, JTYPE, TTYPE>(                                   \
        std::vector<ITYPE> & row_ptr,                                                               \
        std::vector<JTYPE> & col_ind,                                                               \
        std::vector<TTYPE> & val,                                                                   \
        JTYPE M,                                                                                    \
        JTYPE N,                                                                                    \
        ITYPE & nnz,                                                                                \
        rocsparse_index_base       base,                                                            \
        rocsparse_matrix_init_kind init_kind,                                                       \
        bool                       full_rank,                                                       \
        bool                       to_int);                                                                               \
    template void rocsparse_init_gebsr_laplace2d<ITYPE, JTYPE, TTYPE>(                              \
        std::vector<ITYPE> & row_ptr,                                                               \
        std::vector<JTYPE> & col_ind,                                                               \
        std::vector<TTYPE> & val,                                                                   \
        int32_t dim_x,                                                                              \
        int32_t dim_y,                                                                              \
        JTYPE & Mb,                                                                                 \
        JTYPE & Nb,                                                                                 \
        ITYPE & nnzb,                                                                               \
        JTYPE                row_block_dim,                                                         \
        JTYPE                col_block_dim,                                                         \
        rocsparse_index_base base);                                                                 \
    template void rocsparse_init_gebsr_laplace3d<ITYPE, JTYPE, TTYPE>(                              \
        std::vector<ITYPE> & row_ptr,                                                               \
        std::vector<JTYPE> & col_ind,                                                               \
        std::vector<TTYPE> & val,                                                                   \
        int32_t dim_x,                                                                              \
        int32_t dim_y,                                                                              \
        int32_t dim_z,                                                                              \
        JTYPE & Mb,                                                                                 \
        JTYPE & Nb,                                                                                 \
        ITYPE & nnzb,                                                                               \
        JTYPE                row_block_dim,                                                         \
        JTYPE                col_block_dim,                                                         \
        rocsparse_index_base base);                                                                 \
    template void rocsparse_init_gebsr_mtx<ITYPE, JTYPE, TTYPE>(const char*          filename,      \
                                                                std::vector<ITYPE>&  bsr_row_ptr,   \
                                                                std::vector<JTYPE>&  bsr_col_ind,   \
                                                                std::vector<TTYPE>&  bsr_val,       \
                                                                JTYPE&               Mb,            \
                                                                JTYPE&               Nb,            \
                                                                ITYPE&               nnzb,          \
                                                                JTYPE                row_block_dim, \
                                                                JTYPE                col_block_dim, \
                                                                rocsparse_index_base base);         \
    template void rocsparse_init_gebsr_rocalution<ITYPE, JTYPE, TTYPE>(                             \
        const char*          filename,                                                              \
        std::vector<ITYPE>&  row_ptr,                                                               \
        std::vector<JTYPE>&  col_ind,                                                               \
        std::vector<TTYPE>&  val,                                                                   \
        JTYPE&               Mb,                                                                    \
        JTYPE&               Nb,                                                                    \
        ITYPE&               nnzb,                                                                  \
        JTYPE                row_block_dim,                                                         \
        JTYPE                col_block_dim,                                                         \
        rocsparse_index_base base,                                                                  \
        bool                 toint);                                                                                \
    template void rocsparse_init_gebsr_random<ITYPE, JTYPE, TTYPE>(                                 \
        std::vector<ITYPE> & row_ptr,                                                               \
        std::vector<JTYPE> & col_ind,                                                               \
        std::vector<TTYPE> & val,                                                                   \
        JTYPE Mb,                                                                                   \
        JTYPE Nb,                                                                                   \
        ITYPE & nnzb,                                                                               \
        JTYPE                      row_block_dim,                                                   \
        JTYPE                      col_block_dim,                                                   \
        rocsparse_index_base       base,                                                            \
        rocsparse_matrix_init_kind init_kind,                                                       \
        bool                       full_rank,                                                       \
        bool                       to_int);                                                                               \
    template void host_csr_to_ell<ITYPE, JTYPE, TTYPE>(JTYPE                     M,                 \
                                                       const std::vector<ITYPE>& csr_row_ptr,       \
                                                       const std::vector<JTYPE>& csr_col_ind,       \
                                                       const std::vector<TTYPE>& csr_val,           \
                                                       std::vector<JTYPE>&       ell_col_ind,       \
                                                       std::vector<TTYPE>&       ell_val,           \
                                                       JTYPE&                    ell_width,         \
                                                       rocsparse_index_base      csr_base,          \
                                                       rocsparse_index_base      ell_base);

INSTANTIATEI(int32_t);
INSTANTIATEI(int64_t);

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

INSTANTIATE1(int32_t, int32_t);
INSTANTIATE1(int64_t, int32_t);
INSTANTIATE1(int64_t, int64_t);

INSTANTIATE2(int32_t, float);
INSTANTIATE2(int32_t, double);
INSTANTIATE2(int32_t, rocsparse_float_complex);
INSTANTIATE2(int32_t, rocsparse_double_complex);
INSTANTIATE2(int64_t, float);
INSTANTIATE2(int64_t, double);
INSTANTIATE2(int64_t, rocsparse_float_complex);
INSTANTIATE2(int64_t, rocsparse_double_complex);

INSTANTIATE3(int32_t, int32_t, float);
INSTANTIATE3(int64_t, int32_t, float);
INSTANTIATE3(int64_t, int64_t, float);
INSTANTIATE3(int32_t, int32_t, double);
INSTANTIATE3(int64_t, int32_t, double);
INSTANTIATE3(int64_t, int64_t, double);
INSTANTIATE3(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE3(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE3(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE3(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE3(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE3(int64_t, int64_t, rocsparse_double_complex);
