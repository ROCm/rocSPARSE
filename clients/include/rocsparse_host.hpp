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
#ifndef ROCSPARSE_HOST_HPP
#define ROCSPARSE_HOST_HPP

#include "rocsparse_math.hpp"
#include "rocsparse_test.hpp"

#include <algorithm>
#include <cmath>
#include <hip/hip_runtime_api.h>
#include <rocsparse.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template <typename T>
inline void host_axpyi(rocsparse_int        nnz,
                       T                    alpha,
                       const T*             x_val,
                       const rocsparse_int* x_ind,
                       T*                   y,
                       rocsparse_index_base base)
{
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        rocsparse_int idx = x_ind[i] - base;
        y[idx]            = math_fma(alpha, x_val[i], y[idx]);
    }
}

template <typename T>
inline void host_doti(rocsparse_int        nnz,
                      const T*             x_val,
                      const rocsparse_int* x_ind,
                      const T*             y,
                      T*                   result,
                      rocsparse_index_base base)
{
    *result = static_cast<T>(0);

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        *result = math_fma(y[x_ind[i] - base], x_val[i], *result);
    }
}

template <typename T>
inline void host_dotci(rocsparse_int        nnz,
                       const T*             x_val,
                       const rocsparse_int* x_ind,
                       const T*             y,
                       T*                   result,
                       rocsparse_index_base base)
{
    *result = static_cast<T>(0);

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        *result = math_fma(std::conj(x_val[i]), y[x_ind[i] - base], *result);
    }
}

template <typename T>
inline void host_gthr(
    rocsparse_int nnz, const T* y, T* x_val, const rocsparse_int* x_ind, rocsparse_index_base base)
{
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        x_val[i] = y[x_ind[i] - base];
    }
}

template <typename T>
inline void host_gthrz(
    rocsparse_int nnz, T* y, T* x_val, const rocsparse_int* x_ind, rocsparse_index_base base)
{
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        x_val[i]           = y[x_ind[i] - base];
        y[x_ind[i] - base] = static_cast<T>(0);
    }
}

template <typename T>
inline void host_roti(rocsparse_int        nnz,
                      T*                   x_val,
                      const rocsparse_int* x_ind,
                      T*                   y,
                      const T*             c,
                      const T*             s,
                      rocsparse_index_base base)
{
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        rocsparse_int idx = x_ind[i] - base;

        T xs = x_val[i];
        T ys = y[idx];

        x_val[i] = *c * xs + *s * ys;
        y[idx]   = *c * ys - *s * xs;
    }
}

template <typename T>
inline void host_sctr(
    rocsparse_int nnz, const T* x_val, const rocsparse_int* x_ind, T* y, rocsparse_index_base base)
{
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        y[x_ind[i] - base] = x_val[i];
    }
}

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T>
inline void host_coomv(rocsparse_int        M,
                       rocsparse_int        nnz,
                       T                    alpha,
                       const rocsparse_int* coo_row_ind,
                       const rocsparse_int* coo_col_ind,
                       const T*             coo_val,
                       const T*             x,
                       T                    beta,
                       T*                   y,
                       rocsparse_index_base base)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        y[i] *= beta;
    }

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        y[coo_row_ind[i] - base]
            = math_fma(alpha * coo_val[i], x[coo_col_ind[i] - base], y[coo_row_ind[i] - base]);
    }
}

template <typename T>
inline void host_csrmv(rocsparse_int        M,
                       rocsparse_int        nnz,
                       T                    alpha,
                       const rocsparse_int* csr_row_ptr,
                       const rocsparse_int* csr_col_ind,
                       const T*             csr_val,
                       const T*             x,
                       T                    beta,
                       T*                   y,
                       rocsparse_index_base base,
                       int                  algo)
{
    if(algo == 0)
    {
        // Get device properties
        int             dev;
        hipDeviceProp_t prop;

        hipGetDevice(&dev);
        hipGetDeviceProperties(&prop, dev);

        rocsparse_int WF_SIZE;
        rocsparse_int nnz_per_row = nnz / M;

        if(prop.warpSize == 32)
        {
            if(nnz_per_row < 4)
                WF_SIZE = 2;
            else if(nnz_per_row < 8)
                WF_SIZE = 4;
            else if(nnz_per_row < 16)
                WF_SIZE = 8;
            else if(nnz_per_row < 32)
                WF_SIZE = 16;
            else
                WF_SIZE = 32;
        }
        else if(prop.warpSize == 64)
        {
            if(nnz_per_row < 4)
                WF_SIZE = 2;
            else if(nnz_per_row < 8)
                WF_SIZE = 4;
            else if(nnz_per_row < 16)
                WF_SIZE = 8;
            else if(nnz_per_row < 32)
                WF_SIZE = 16;
            else if(nnz_per_row < 64)
                WF_SIZE = 32;
            else
                WF_SIZE = 64;
        }
        else
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_int row_begin = csr_row_ptr[i] - base;
            rocsparse_int row_end   = csr_row_ptr[i + 1] - base;

            std::vector<T> sum(WF_SIZE, static_cast<T>(0));

            for(rocsparse_int j = row_begin; j < row_end; j += WF_SIZE)
            {
                for(rocsparse_int k = 0; k < WF_SIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        sum[k] = math_fma(
                            alpha * csr_val[j + k], x[csr_col_ind[j + k] - base], sum[k]);
                    }
                }
            }

            for(rocsparse_int j = 1; j < WF_SIZE; j <<= 1)
            {
                for(rocsparse_int k = 0; k < WF_SIZE - j; ++k)
                {
                    sum[k] += sum[k + j];
                }
            }

            if(beta == static_cast<T>(0))
            {
                y[i] = sum[0];
            }
            else
            {
                y[i] = math_fma(beta, y[i], sum[0]);
            }
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int i = 0; i < M; ++i)
        {
            T sum = static_cast<T>(0);
            T err = static_cast<T>(0);

            rocsparse_int row_begin = csr_row_ptr[i] - base;
            rocsparse_int row_end   = csr_row_ptr[i + 1] - base;

            for(rocsparse_int j = row_begin; j < row_end; ++j)
            {
                T old  = sum;
                T prod = alpha * csr_val[j] * x[csr_col_ind[j] - base];

                sum = sum + prod;
                err = (old - (sum - (sum - old))) + (prod - (sum - old)) + err;
            }

            if(beta != static_cast<T>(0))
            {
                y[i] = math_fma(beta, y[i], sum + err);
            }
            else
            {
                y[i] = sum + err;
            }
        }
    }
}

template <typename T>
inline void host_csrsv(rocsparse_int        M,
                       T                    alpha,
                       const rocsparse_int* csr_row_ptr,
                       const rocsparse_int* csr_col_ind,
                       const T*             csr_val,
                       const T*             x,
                       T*                   y,
                       rocsparse_diag_type  diag_type,
                       rocsparse_fill_mode  fill_mode,
                       rocsparse_index_base base,
                       rocsparse_int*       struct_pivot,
                       rocsparse_int*       numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = M + 1;
    *numeric_pivot = M + 1;

    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    hipGetDevice(&dev);
    hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

    if(fill_mode == rocsparse_fill_mode_lower)
    {
        // Process lower triangular part
        for(rocsparse_int row = 0; row < M; ++row)
        {
            temp.assign(prop.warpSize, static_cast<T>(0));
            temp[0] = alpha * x[row];

            rocsparse_int diag      = -1;
            rocsparse_int row_begin = csr_row_ptr[row] - base;
            rocsparse_int row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(rocsparse_int l = row_begin; l < row_end; l += prop.warpSize)
            {
                for(unsigned int k = 0; k < prop.warpSize; ++k)
                {
                    rocsparse_int j = l + k;

                    // Do not run out of bounds
                    if(j >= row_end)
                    {
                        break;
                    }

                    rocsparse_int local_col = csr_col_ind[j] - base;
                    T             local_val = csr_val[j];

                    if(local_val == static_cast<T>(0) && local_col == row
                       && diag_type == rocsparse_diag_type_non_unit)
                    {
                        // Numerical zero pivot found, avoid division by 0
                        // and store index for later use.
                        *numeric_pivot = std::min(*numeric_pivot, row + base);
                        local_val      = static_cast<T>(1);
                    }

                    // Ignore all entries that are above the diagonal
                    if(local_col > row)
                    {
                        break;
                    }

                    // Diagonal entry
                    if(local_col == row)
                    {
                        // If diagonal type is non unit, do division by diagonal entry
                        // This is not required for unit diagonal for obvious reasons
                        if(diag_type == rocsparse_diag_type_non_unit)
                        {
                            diag     = j;
                            diag_val = static_cast<T>(1) / local_val;
                        }

                        break;
                    }

                    // Lower triangular part
                    temp[k] = math_fma(-local_val, y[local_col], temp[k]);
                }
            }

            for(unsigned int j = 1; j < prop.warpSize; j <<= 1)
            {
                for(unsigned int k = 0; k < prop.warpSize - j; ++k)
                {
                    temp[k] += temp[k + j];
                }
            }

            if(diag_type == rocsparse_diag_type_non_unit)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                y[row] = temp[0] * diag_val;
            }
            else
            {
                y[row] = temp[0];
            }
        }
    }
    else
    {
        // Process upper triangular part
        for(rocsparse_int row = M - 1; row >= 0; --row)
        {
            temp.assign(prop.warpSize, static_cast<T>(0));
            temp[0] = alpha * x[row];

            rocsparse_int diag      = -1;
            rocsparse_int row_begin = csr_row_ptr[row] - base;
            rocsparse_int row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(rocsparse_int l = row_end - 1; l >= row_begin; l -= prop.warpSize)
            {
                for(unsigned int k = 0; k < prop.warpSize; ++k)
                {
                    rocsparse_int j = l - k;

                    // Do not run out of bounds
                    if(j < row_begin)
                    {
                        break;
                    }

                    rocsparse_int local_col = csr_col_ind[j] - base;
                    T             local_val = csr_val[j];

                    // Ignore all entries that are below the diagonal
                    if(local_col < row)
                    {
                        continue;
                    }

                    // Diagonal entry
                    if(local_col == row)
                    {
                        if(diag_type == rocsparse_diag_type_non_unit)
                        {
                            // Check for numerical zero
                            if(local_val == static_cast<T>(0))
                            {
                                *numeric_pivot = std::min(*numeric_pivot, row + base);
                                local_val      = static_cast<T>(1);
                            }

                            diag     = j;
                            diag_val = static_cast<T>(1) / local_val;
                        }

                        continue;
                    }

                    // Upper triangular part
                    temp[k] = math_fma(-local_val, y[local_col], temp[k]);
                }
            }

            for(unsigned int j = 1; j < prop.warpSize; j <<= 1)
            {
                for(unsigned int k = 0; k < prop.warpSize - j; ++k)
                {
                    temp[k] += temp[k + j];
                }
            }

            if(diag_type == rocsparse_diag_type_non_unit)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                y[row] = temp[0] * diag_val;
            }
            else
            {
                y[row] = temp[0];
            }
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
}

template <typename T>
inline void host_ellmv(rocsparse_int        M,
                       rocsparse_int        N,
                       rocsparse_int        nnz,
                       T                    alpha,
                       const rocsparse_int* ell_col_ind,
                       const T*             ell_val,
                       rocsparse_int        ell_width,
                       const T*             x,
                       T                    beta,
                       T*                   y,
                       rocsparse_index_base base)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        T sum = static_cast<T>(0);
        for(rocsparse_int p = 0; p < ell_width; ++p)
        {
            rocsparse_int idx = p * M + i;
            rocsparse_int col = ell_col_ind[idx] - base;

            if(col >= 0 && col < N)
            {
                sum = math_fma(ell_val[idx], x[col], sum);
            }
            else
            {
                break;
            }
        }

        if(beta != static_cast<T>(0))
        {
            y[i] = math_fma(beta, y[i], alpha * sum);
        }
        else
        {
            y[i] = alpha * sum;
        }
    }
}

template <typename T>
inline void host_hybmv(rocsparse_int        M,
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
                       rocsparse_index_base base)
{
    T coo_beta = beta;

    // ELL part
    if(ell_nnz > 0)
    {
        host_ellmv<T>(M, N, ell_nnz, alpha, ell_col_ind, ell_val, ell_width, x, beta, y, base);
        coo_beta = static_cast<T>(1);
    }

    // COO part
    if(coo_nnz > 0)
    {
        host_coomv<T>(M, coo_nnz, alpha, coo_row_ind, coo_col_ind, coo_val, x, coo_beta, y, base);
    }
}

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T>
inline void host_csrmm(rocsparse_int                     M,
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
                       rocsparse_index_base              base)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        for(rocsparse_int j = 0; j < N; ++j)
        {
            rocsparse_int row_begin = csr_row_ptr_A[i] - base;
            rocsparse_int row_end   = csr_row_ptr_A[i + 1] - base;
            rocsparse_int idx_C     = i + j * ldc;

            T sum = static_cast<T>(0);

            for(rocsparse_int k = row_begin; k < row_end; ++k)
            {
                rocsparse_int idx_B = (transB == rocsparse_operation_none)
                                          ? (csr_col_ind_A[k] - base + j * ldb)
                                          : (j + (csr_col_ind_A[k] - base) * ldb);

                sum = math_fma(alpha * csr_val_A[k], B[idx_B], sum);
            }

            if(beta == static_cast<T>(0))
            {
                C[idx_C] = sum;
            }
            else
            {
                C[idx_C] = math_fma(beta, C[idx_C], sum);
            }
        }
    }
}

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template <typename T>
inline void host_csrgemm_nnz(rocsparse_int                     M,
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
                             rocsparse_index_base              base_D)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<rocsparse_int> nnz(N, -1);

#ifdef _OPENMP
        rocsparse_int nthreads = omp_get_num_threads();
        rocsparse_int tid      = omp_get_thread_num();
#else
        rocsparse_int nthreads = 1;
        rocsparse_int tid      = 0;
#endif

        rocsparse_int rows_per_thread = (M + nthreads - 1) / nthreads;
        rocsparse_int chunk_begin     = rows_per_thread * tid;
        rocsparse_int chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Index base
        csr_row_ptr_C[0] = base_C;

        // Loop over rows of A
        for(rocsparse_int i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            if(alpha)
            {
                rocsparse_int row_begin_A = csr_row_ptr_A[i] - base_A;
                rocsparse_int row_end_A   = csr_row_ptr_A[i + 1] - base_A;

                // Loop over columns of A
                for(rocsparse_int j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    rocsparse_int col_A = csr_col_ind_A[j] - base_A;

                    rocsparse_int row_begin_B = csr_row_ptr_B[col_A] - base_B;
                    rocsparse_int row_end_B   = csr_row_ptr_B[col_A + 1] - base_B;

                    // Loop over columns of B in row col_A
                    for(rocsparse_int k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        rocsparse_int col_B = csr_col_ind_B[k] - base_B;

                        // Check if a new nnz is generated
                        if(nnz[col_B] != i)
                        {
                            nnz[col_B] = i;
                            ++csr_row_ptr_C[i + 1];
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                rocsparse_int row_begin_D = csr_row_ptr_D[i] - base_D;
                rocsparse_int row_end_D   = csr_row_ptr_D[i + 1] - base_D;

                // Loop over columns of D
                for(rocsparse_int j = row_begin_D; j < row_end_D; ++j)
                {
                    rocsparse_int col_D = csr_col_ind_D[j] - base_D;

                    // Check if a new nnz is generated
                    if(nnz[col_D] != i)
                    {
                        nnz[col_D] = i;
                        ++csr_row_ptr_C[i + 1];
                    }
                }
            }
        }
    }

    // Scan to obtain row offsets
    for(rocsparse_int i = 0; i < M; ++i)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    *nnz_C = csr_row_ptr_C[M] - base_C;
}

template <typename T>
inline void host_csrgemm(rocsparse_int                     M,
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
                         rocsparse_index_base              base_D)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<rocsparse_int> nnz(N, -1);

#ifdef _OPENMP
        rocsparse_int nthreads = omp_get_num_threads();
        rocsparse_int tid      = omp_get_thread_num();
#else
        rocsparse_int nthreads = 1;
        rocsparse_int tid      = 0;
#endif

        rocsparse_int rows_per_thread = (M + nthreads - 1) / nthreads;
        rocsparse_int chunk_begin     = rows_per_thread * tid;
        rocsparse_int chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Loop over rows of A
        for(rocsparse_int i = chunk_begin; i < chunk_end; ++i)
        {
            rocsparse_int row_begin_C = csr_row_ptr_C[i] - base_C;
            rocsparse_int row_end_C   = row_begin_C;

            if(alpha)
            {
                rocsparse_int row_begin_A = csr_row_ptr_A[i] - base_A;
                rocsparse_int row_end_A   = csr_row_ptr_A[i + 1] - base_A;

                // Loop over columns of A
                for(rocsparse_int j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    rocsparse_int col_A = csr_col_ind_A[j] - base_A;
                    // Current value of A
                    T val_A = *alpha * csr_val_A[j];

                    rocsparse_int row_begin_B = csr_row_ptr_B[col_A] - base_B;
                    rocsparse_int row_end_B   = csr_row_ptr_B[col_A + 1] - base_B;

                    // Loop over columns of B in row col_A
                    for(rocsparse_int k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        rocsparse_int col_B = csr_col_ind_B[k] - base_B;
                        // Current value of B
                        T val_B = csr_val_B[k];

                        // Check if a new nnz is generated or if the product is appended
                        if(nnz[col_B] < row_begin_C)
                        {
                            nnz[col_B]               = row_end_C;
                            csr_col_ind_C[row_end_C] = col_B + base_C;
                            csr_val_C[row_end_C]     = val_A * val_B;
                            ++row_end_C;
                        }
                        else
                        {
                            csr_val_C[nnz[col_B]] += val_A * val_B;
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                rocsparse_int row_begin_D = csr_row_ptr_D[i] - base_D;
                rocsparse_int row_end_D   = csr_row_ptr_D[i + 1] - base_D;

                // Loop over columns of D
                for(rocsparse_int j = row_begin_D; j < row_end_D; ++j)
                {
                    // Current column of D
                    rocsparse_int col_D = csr_col_ind_D[j] - base_D;
                    // Current value of D
                    T val_D = *beta * csr_val_D[j];

                    // Check if a new nnz is generated or if the value is added
                    if(nnz[col_D] < row_begin_C)
                    {
                        nnz[col_D] = row_end_C;

                        csr_col_ind_C[row_end_C] = col_D + base_C;
                        csr_val_C[row_end_C]     = val_D;
                        ++row_end_C;
                    }
                    else
                    {
                        csr_val_C[nnz[col_D]] += val_D;
                    }
                }
            }
        }
    }

    rocsparse_int nnz = csr_row_ptr_C[M] - base_C;

    std::vector<rocsparse_int> col(nnz);
    std::vector<T>             val(nnz);

    col = csr_col_ind_C;
    val = csr_val_C;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int row_begin = csr_row_ptr_C[i] - base_C;
        rocsparse_int row_end   = csr_row_ptr_C[i + 1] - base_C;
        rocsparse_int row_nnz   = row_end - row_begin;

        std::vector<rocsparse_int> perm(row_nnz);
        for(rocsparse_int j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        rocsparse_int* col_entry = &col[row_begin];
        T*             val_entry = &val[row_begin];

        std::sort(perm.begin(), perm.end(), [&](const rocsparse_int& a, const rocsparse_int& b) {
            return col_entry[a] <= col_entry[b];
        });

        for(rocsparse_int j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template <typename T>
inline void host_csrilu0(rocsparse_int                     M,
                         const std::vector<rocsparse_int>& csr_row_ptr,
                         const std::vector<rocsparse_int>& csr_col_ind,
                         std::vector<T>&                   csr_val,
                         rocsparse_index_base              base,
                         rocsparse_int*                    struct_pivot,
                         rocsparse_int*                    numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = -1;
    *numeric_pivot = -1;

    // pointer of upper part of each row
    std::vector<rocsparse_int> diag_offset(M);
    std::vector<rocsparse_int> nnz_entries(M, 0);

    // ai = 0 to N loop over all rows
    for(rocsparse_int ai = 0; ai < M; ++ai)
    {
        // ai-th row entries
        rocsparse_int row_begin = csr_row_ptr[ai] - base;
        rocsparse_int row_end   = csr_row_ptr[ai + 1] - base;
        rocsparse_int j;

        // nnz position of ai-th row in val array
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = j;
        }

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            // if nnz entry is in lower matrix
            if(csr_col_ind[j] - base < ai)
            {

                rocsparse_int col_j  = csr_col_ind[j] - base;
                rocsparse_int diag_j = diag_offset[col_j];

                if(csr_val[diag_j] != static_cast<T>(0))
                {
                    // multiplication factor
                    csr_val[j] = csr_val[j] / csr_val[diag_j];

                    // loop over upper offset pointer and do linear combination for nnz entry
                    for(rocsparse_int k = diag_j + 1; k < csr_row_ptr[col_j + 1] - base; ++k)
                    {
                        // if nnz at this position do linear combination
                        if(nnz_entries[csr_col_ind[k] - base] != 0)
                        {
                            rocsparse_int idx = nnz_entries[csr_col_ind[k] - base];
                            csr_val[idx]      = math_fma(-csr_val[j], csr_val[k], csr_val[idx]);
                        }
                    }
                }
                else
                {
                    // Numerical zero diagonal
                    *numeric_pivot = col_j + base;
                    return;
                }
            }
            else if(csr_col_ind[j] - base == ai)
            {
                has_diag = true;
                break;
            }
            else
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural (and numerical) zero diagonal
            *struct_pivot  = ai + base;
            *numeric_pivot = ai + base;
            return;
        }

        // set diagonal pointer to diagonal element
        diag_offset[ai] = j;

        // clear nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = 0;
        }
    }
}

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
inline void host_csr_to_coo(rocsparse_int                     M,
                            rocsparse_int                     nnz,
                            const std::vector<rocsparse_int>& csr_row_ptr,
                            std::vector<rocsparse_int>&       coo_row_ind,
                            rocsparse_index_base              base)
{
    // Resize coo_row_ind
    coo_row_ind.resize(nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int row_begin = csr_row_ptr[i] - base;
        rocsparse_int row_end   = csr_row_ptr[i + 1] - base;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            coo_row_ind[j] = i + base;
        }
    }
}

template <typename T>
inline void host_csr_to_csc(rocsparse_int                     M,
                            rocsparse_int                     N,
                            rocsparse_int                     nnz,
                            const std::vector<rocsparse_int>& csr_row_ptr,
                            const std::vector<rocsparse_int>& csr_col_ind,
                            const std::vector<T>&             csr_val,
                            std::vector<rocsparse_int>&       csc_row_ind,
                            std::vector<rocsparse_int>&       csc_col_ptr,
                            std::vector<T>&                   csc_val,
                            rocsparse_action                  action,
                            rocsparse_index_base              base)
{
    csc_row_ind.resize(nnz);
    csc_col_ptr.resize(N + 1, 0);
    csc_val.resize(nnz);

    // Determine nnz per column
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        ++csc_col_ptr[csr_col_ind[i] + 1 - base];
    }

    // Scan
    for(rocsparse_int i = 0; i < N; ++i)
    {
        csc_col_ptr[i + 1] += csc_col_ptr[i];
    }

    // Fill row indices and values
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int row_begin = csr_row_ptr[i] - base;
        rocsparse_int row_end   = csr_row_ptr[i + 1] - base;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col = csr_col_ind[j] - base;
            rocsparse_int idx = csc_col_ptr[col];

            csc_row_ind[idx] = i + base;
            csc_val[idx]     = csr_val[j];

            ++csc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(rocsparse_int i = N; i > 0; --i)
    {
        csc_col_ptr[i] = csc_col_ptr[i - 1] + base;
    }

    csc_col_ptr[0] = base;
}

template <typename T>
inline void host_csr_to_ell(rocsparse_int                     M,
                            const std::vector<rocsparse_int>& csr_row_ptr,
                            const std::vector<rocsparse_int>& csr_col_ind,
                            const std::vector<T>&             csr_val,
                            std::vector<rocsparse_int>&       ell_col_ind,
                            std::vector<T>&                   ell_val,
                            rocsparse_int&                    ell_width,
                            rocsparse_index_base              csr_base,
                            rocsparse_index_base              ell_base)
{
    // Determine ELL width
    ell_width = 0;

    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        ell_width             = std::max(row_nnz, ell_width);
    }

    // Compute ELL non-zeros
    rocsparse_int ell_nnz = ell_width * M;

    ell_col_ind.resize(ell_nnz);
    ell_val.resize(ell_nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int p = 0;

        rocsparse_int row_begin = csr_row_ptr[i] - csr_base;
        rocsparse_int row_end   = csr_row_ptr[i + 1] - csr_base;
        rocsparse_int row_nnz   = row_end - row_begin;

        // Fill ELL matrix with data
        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            rocsparse_int idx = p * M + i;

            ell_col_ind[idx] = csr_col_ind[j] - csr_base + ell_base;
            ell_val[idx]     = csr_val[j];

            ++p;
        }

        // Add padding to ELL structures
        for(rocsparse_int j = row_nnz; j < ell_width; ++j)
        {
            rocsparse_int idx = p * M + i;

            ell_col_ind[idx] = -1;
            ell_val[idx]     = static_cast<T>(0);

            ++p;
        }
    }
}

template <typename T>
inline void host_csr_to_hyb(rocsparse_int                     M,
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
                            rocsparse_index_base              base)
{
    ell_nnz = 0;
    coo_nnz = 0;

    // Auto and user width
    if(part == rocsparse_hyb_partition_auto || part == rocsparse_hyb_partition_user)
    {
        // Determine ELL width
        ell_width = (part == rocsparse_hyb_partition_auto) ? (nnz - 1) / M + 1 : ell_width;

        // Determine COO nnz
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];

            if(row_nnz > ell_width)
            {
                coo_nnz += row_nnz - ell_width;
            }
        }
    }
    else if(part == rocsparse_hyb_partition_max)
    {
        ell_width = 0;

        // Determine max nnz per row
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
            ell_width             = std::max(ell_width, row_nnz);
        }
    }

    // ELL nnz
    ell_nnz = ell_width * M;

    // Allocate memory for HYB matrix
    if(ell_nnz > 0)
    {
        ell_col_ind.resize(ell_nnz);
        ell_val.resize(ell_nnz);
    }

    if(coo_nnz > 0)
    {
        coo_row_ind.resize(coo_nnz);
        coo_col_ind.resize(coo_nnz);
        coo_val.resize(coo_nnz);
    }

    // Fill HYB
    rocsparse_int coo_idx = 0;
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int p         = 0;
        rocsparse_int row_begin = csr_row_ptr[i] - base;
        rocsparse_int row_end   = csr_row_ptr[i + 1] - base;
        rocsparse_int row_nnz   = row_end - row_begin;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            if(p < ell_width)
            {
                rocsparse_int idx = p++ * M + i;

                ell_col_ind[idx] = csr_col_ind[j];
                ell_val[idx]     = csr_val[j];
            }
            else
            {
                coo_row_ind[coo_idx] = i + base;
                coo_col_ind[coo_idx] = csr_col_ind[j];
                coo_val[coo_idx++]   = csr_val[j];
            }
        }

        for(rocsparse_int j = row_nnz; j < ell_width; ++j)
        {
            rocsparse_int idx = p++ * M + i;

            ell_col_ind[idx] = -1;
            ell_val[idx]     = static_cast<T>(0);
        }
    }
}

inline void host_coo_to_csr(rocsparse_int                     M,
                            rocsparse_int                     nnz,
                            const std::vector<rocsparse_int>& coo_row_ind,
                            std::vector<rocsparse_int>&       csr_row_ptr,
                            rocsparse_index_base              base)
{
    // Resize and initialize csr_row_ptr with zeros
    csr_row_ptr.resize(M + 1, 0);

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr[coo_row_ind[i] + 1 - base];
    }

    csr_row_ptr[0] = base;
    for(rocsparse_int i = 0; i < M; ++i)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }
}

template <typename T>
inline void host_ell_to_csr(rocsparse_int                     M,
                            rocsparse_int                     N,
                            const std::vector<rocsparse_int>& ell_col_ind,
                            const std::vector<T>&             ell_val,
                            rocsparse_int                     ell_width,
                            std::vector<rocsparse_int>&       csr_row_ptr,
                            std::vector<rocsparse_int>&       csr_col_ind,
                            std::vector<T>&                   csr_val,
                            rocsparse_int&                    csr_nnz,
                            rocsparse_index_base              ell_base,
                            rocsparse_index_base              csr_base)
{
    csr_row_ptr.resize(M + 1, 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        for(rocsparse_int p = 0; p < ell_width; ++p)
        {
            rocsparse_int idx = p * M + i;
            rocsparse_int col = ell_col_ind[idx] - ell_base;

            if(col >= 0 && col < N)
            {
                ++csr_row_ptr[i];
            }
        }
    }

    // Determine row pointers
    csr_nnz = csr_base;
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int tmp = csr_row_ptr[i];
        csr_row_ptr[i]    = csr_nnz;
        csr_nnz += tmp;
    }

    csr_row_ptr[M] = csr_nnz;
    csr_nnz -= csr_base;

    // Allocate memory for columns and values
    csr_col_ind.resize(csr_nnz);
    csr_val.resize(csr_nnz);

    // Fill CSR structure
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int csr_idx = csr_row_ptr[i] - csr_base;

        for(rocsparse_int p = 0; p < ell_width; ++p)
        {
            rocsparse_int idx = p * M + i;
            rocsparse_int col = ell_col_ind[idx] - ell_base;

            if(col >= 0 && col < N)
            {
                csr_col_ind[csr_idx] = col + csr_base;
                csr_val[csr_idx]     = ell_val[idx];

                ++csr_idx;
            }
        }
    }
}

template <typename T>
inline void host_coosort_by_column(rocsparse_int               M,
                                   rocsparse_int               nnz,
                                   std::vector<rocsparse_int>& coo_row_ind,
                                   std::vector<rocsparse_int>& coo_col_ind,
                                   std::vector<T>&             coo_val)
{
    // Permutation vector
    std::vector<rocsparse_int> perm(nnz);

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    std::vector<rocsparse_int> tmp_row(nnz);
    std::vector<rocsparse_int> tmp_col(nnz);
    std::vector<T>             tmp_val(nnz);

    tmp_row = coo_row_ind;
    tmp_col = coo_col_ind;
    tmp_val = coo_val;

    // Sort
    std::sort(perm.begin(), perm.end(), [&](const rocsparse_int& a, const rocsparse_int& b) {
        if(tmp_col[a] < tmp_col[b])
        {
            return true;
        }
        else if(tmp_col[a] == tmp_col[b])
        {
            return (tmp_row[a] < tmp_row[b]);
        }
        else
        {
            return false;
        }
    });

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        coo_row_ind[i] = tmp_row[perm[i]];
        coo_col_ind[i] = tmp_col[perm[i]];
        coo_val[i]     = tmp_val[perm[i]];
    }
}

#endif // ROCSPARSE_HOST_HPP
