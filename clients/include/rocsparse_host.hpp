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

#include <algorithm>
#include <cmath>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <rocsparse.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declaration
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
                            rocsparse_index_base              base);
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
        y[idx]            = std::fma(alpha, x_val[i], y[idx]);
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
        *result = std::fma(y[x_ind[i] - base], x_val[i], *result);
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
        *result = std::fma(std::conj(x_val[i]), y[x_ind[i] - base], *result);
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
inline void host_bsrmv(rocsparse_direction  dir,
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
                       rocsparse_index_base base)
{
    // Quick return
    if(alpha == static_cast<T>(0))
    {
        if(beta != static_cast<T>(1))
        {
            for(rocsparse_int i = 0; i < mb * bsr_dim; ++i)
            {
                y[i] *= beta;
            }
        }

        return;
    }

    rocsparse_int WFSIZE;

    if(bsr_dim == 2)
    {
        rocsparse_int blocks_per_row = nnzb / mb;

        if(blocks_per_row < 8)
        {
            WFSIZE = 4;
        }
        else if(blocks_per_row < 16)
        {
            WFSIZE = 8;
        }
        else if(blocks_per_row < 32)
        {
            WFSIZE = 16;
        }
        else if(blocks_per_row < 64)
        {
            WFSIZE = 32;
        }
        else
        {
            WFSIZE = 64;
        }
    }
    else if(bsr_dim <= 8)
    {
        WFSIZE = 8;
    }
    else if(bsr_dim <= 16)
    {
        WFSIZE = 16;
    }
    else
    {
        WFSIZE = 32;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int row = 0; row < mb; ++row)
    {
        rocsparse_int row_begin = bsr_row_ptr[row] - base;
        rocsparse_int row_end   = bsr_row_ptr[row + 1] - base;

        if(bsr_dim == 2)
        {
            std::vector<T> sum0(WFSIZE, static_cast<T>(0));
            std::vector<T> sum1(WFSIZE, static_cast<T>(0));

            for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
            {
                for(rocsparse_int k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        rocsparse_int col = bsr_col_ind[j + k] - base;

                        if(dir == rocsparse_direction_column)
                        {
                            sum0[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 0],
                                               x[col * bsr_dim + 0],
                                               sum0[k]);
                            sum1[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 1],
                                               x[col * bsr_dim + 0],
                                               sum1[k]);
                            sum0[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 2],
                                               x[col * bsr_dim + 1],
                                               sum0[k]);
                            sum1[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 3],
                                               x[col * bsr_dim + 1],
                                               sum1[k]);
                        }
                        else
                        {
                            sum0[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 0],
                                               x[col * bsr_dim + 0],
                                               sum0[k]);
                            sum0[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 1],
                                               x[col * bsr_dim + 1],
                                               sum0[k]);
                            sum1[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 2],
                                               x[col * bsr_dim + 0],
                                               sum1[k]);
                            sum1[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 3],
                                               x[col * bsr_dim + 1],
                                               sum1[k]);
                        }
                    }
                }
            }

            for(unsigned int j = 1; j < WFSIZE; j <<= 1)
            {
                for(unsigned int k = 0; k < WFSIZE - j; ++k)
                {
                    sum0[k] += sum0[k + j];
                    sum1[k] += sum1[k + j];
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[row * bsr_dim + 0] = std::fma(beta, y[row * bsr_dim + 0], alpha * sum0[0]);
                y[row * bsr_dim + 1] = std::fma(beta, y[row * bsr_dim + 1], alpha * sum1[0]);
            }
            else
            {
                y[row * bsr_dim + 0] = alpha * sum0[0];
                y[row * bsr_dim + 1] = alpha * sum1[0];
            }
        }
        else
        {
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                std::vector<T> sum(WFSIZE, static_cast<T>(0));

                for(rocsparse_int j = row_begin; j < row_end; ++j)
                {
                    rocsparse_int col = bsr_col_ind[j] - base;

                    for(rocsparse_int bj = 0; bj < bsr_dim; bj += WFSIZE)
                    {
                        for(unsigned int k = 0; k < WFSIZE; ++k)
                        {
                            if(bj + k < bsr_dim)
                            {
                                if(dir == rocsparse_direction_column)
                                {
                                    sum[k] = std::fma(
                                        bsr_val[bsr_dim * bsr_dim * j + bsr_dim * (bj + k) + bi],
                                        x[bsr_dim * col + (bj + k)],
                                        sum[k]);
                                }
                                else
                                {
                                    sum[k] = std::fma(
                                        bsr_val[bsr_dim * bsr_dim * j + bsr_dim * bi + (bj + k)],
                                        x[bsr_dim * col + (bj + k)],
                                        sum[k]);
                                }
                            }
                        }
                    }
                }

                for(unsigned int j = 1; j < WFSIZE; j <<= 1)
                {
                    for(unsigned int k = 0; k < WFSIZE - j; ++k)
                    {
                        sum[k] += sum[k + j];
                    }
                }

                if(beta != static_cast<T>(0))
                {
                    y[row * bsr_dim + bi] = std::fma(beta, y[row * bsr_dim + bi], alpha * sum[0]);
                }
                else
                {
                    y[row * bsr_dim + bi] = alpha * sum[0];
                }
            }
        }
    }
}

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
            = std::fma(alpha * coo_val[i], x[coo_col_ind[i] - base], y[coo_row_ind[i] - base]);
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
                        sum[k] = std::fma(
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
                y[i] = std::fma(beta, y[i], sum[0]);
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
                y[i] = std::fma(beta, y[i], sum + err);
            }
            else
            {
                y[i] = sum + err;
            }
        }
    }
}

template <typename T>
static inline void host_lsolve(rocsparse_int                     M,
                               T                                 alpha,
                               const std::vector<rocsparse_int>& csr_row_ptr,
                               const std::vector<rocsparse_int>& csr_col_ind,
                               const std::vector<T>&             csr_val,
                               const std::vector<T>&             x,
                               std::vector<T>&                   y,
                               rocsparse_diag_type               diag_type,
                               rocsparse_index_base              base,
                               rocsparse_int*                    struct_pivot,
                               rocsparse_int*                    numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    hipGetDevice(&dev);
    hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

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
                temp[k] = std::fma(-local_val, y[local_col], temp[k]);
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

template <typename T>
static inline void host_usolve(rocsparse_int                     M,
                               T                                 alpha,
                               const std::vector<rocsparse_int>& csr_row_ptr,
                               const std::vector<rocsparse_int>& csr_col_ind,
                               const std::vector<T>&             csr_val,
                               const std::vector<T>&             x,
                               std::vector<T>&                   y,
                               rocsparse_diag_type               diag_type,
                               rocsparse_index_base              base,
                               rocsparse_int*                    struct_pivot,
                               rocsparse_int*                    numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    hipGetDevice(&dev);
    hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

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
                temp[k] = std::fma(-local_val, y[local_col], temp[k]);
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

template <typename T>
inline void host_csrsv(rocsparse_operation               trans,
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
                       rocsparse_int*                    numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = M + 1;
    *numeric_pivot = M + 1;

    if(trans == rocsparse_operation_none)
    {
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_lsolve(M,
                        alpha,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        x,
                        y,
                        diag_type,
                        base,
                        struct_pivot,
                        numeric_pivot);
        }
        else
        {
            host_usolve(M,
                        alpha,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        x,
                        y,
                        diag_type,
                        base,
                        struct_pivot,
                        numeric_pivot);
        }
    }
    else if(trans == rocsparse_operation_transpose)
    {
        // Transpose matrix
        std::vector<rocsparse_int> csrt_row_ptr(M + 1);
        std::vector<rocsparse_int> csrt_col_ind(nnz);
        std::vector<T>             csrt_val(nnz);

        host_csr_to_csc(M,
                        M,
                        nnz,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        csrt_col_ind,
                        csrt_row_ptr,
                        csrt_val,
                        rocsparse_action_numeric,
                        base);

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_usolve(M,
                        alpha,
                        csrt_row_ptr,
                        csrt_col_ind,
                        csrt_val,
                        x,
                        y,
                        diag_type,
                        base,
                        struct_pivot,
                        numeric_pivot);
        }
        else
        {
            host_lsolve(M,
                        alpha,
                        csrt_row_ptr,
                        csrt_col_ind,
                        csrt_val,
                        x,
                        y,
                        diag_type,
                        base,
                        struct_pivot,
                        numeric_pivot);
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
                sum = std::fma(ell_val[idx], x[col], sum);
            }
            else
            {
                break;
            }
        }

        if(beta != static_cast<T>(0))
        {
            y[i] = std::fma(beta, y[i], alpha * sum);
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

                sum = std::fma(alpha * csr_val_A[k], B[idx_B], sum);
            }

            if(beta == static_cast<T>(0))
            {
                C[idx_C] = sum;
            }
            else
            {
                C[idx_C] = std::fma(beta, C[idx_C], sum);
            }
        }
    }
}

template <typename T>
static inline void host_lssolve(rocsparse_int                     M,
                                rocsparse_int                     nrhs,
                                rocsparse_operation               transB,
                                T                                 alpha,
                                const std::vector<rocsparse_int>& csr_row_ptr,
                                const std::vector<rocsparse_int>& csr_col_ind,
                                const std::vector<T>&             csr_val,
                                std::vector<T>&                   B,
                                rocsparse_int                     ldb,
                                rocsparse_diag_type               diag_type,
                                rocsparse_index_base              base,
                                rocsparse_int*                    struct_pivot,
                                rocsparse_int*                    numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(rocsparse_int i = 0; i < nrhs; ++i)
    {
        // Process lower triangular part
        for(rocsparse_int row = 0; row < M; ++row)
        {
            rocsparse_int idx_B
                = (transB == rocsparse_operation_none) ? i * ldb + row : row * ldb + i;

            T sum = alpha * B[idx_B];

            rocsparse_int diag      = -1;
            rocsparse_int row_begin = csr_row_ptr[row] - base;
            rocsparse_int row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(rocsparse_int j = row_begin; j < row_end; ++j)
            {
                rocsparse_int local_col = csr_col_ind[j] - base;
                T             local_val = csr_val[j];

                if(local_val == static_cast<T>(0) && local_col == row
                   && diag_type == rocsparse_diag_type_non_unit)
                {
                    // Numerical zero pivot found, avoid division by 0 and store
                    // index for later use
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
                rocsparse_int idx = (transB == rocsparse_operation_none) ? i * ldb + local_col
                                                                         : local_col * ldb + i;
                sum = std::fma(-local_val, B[idx], sum);
            }

            if(diag_type == rocsparse_diag_type_non_unit)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                B[idx_B] = sum * diag_val;
            }
            else
            {
                B[idx_B] = sum;
            }
        }
    }
}

template <typename T>
static inline void host_ussolve(rocsparse_int                     M,
                                rocsparse_int                     nrhs,
                                rocsparse_operation               transB,
                                T                                 alpha,
                                const std::vector<rocsparse_int>& csr_row_ptr,
                                const std::vector<rocsparse_int>& csr_col_ind,
                                const std::vector<T>&             csr_val,
                                std::vector<T>&                   B,
                                rocsparse_int                     ldb,
                                rocsparse_diag_type               diag_type,
                                rocsparse_index_base              base,
                                rocsparse_int*                    struct_pivot,
                                rocsparse_int*                    numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(rocsparse_int i = 0; i < nrhs; ++i)
    {
        // Process upper triangular part
        for(rocsparse_int row = M - 1; row >= 0; --row)
        {
            rocsparse_int idx_B
                = (transB == rocsparse_operation_none) ? i * ldb + row : row * ldb + i;

            T sum = alpha * B[idx_B];

            rocsparse_int diag      = -1;
            rocsparse_int row_begin = csr_row_ptr[row] - base;
            rocsparse_int row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(rocsparse_int j = row_end - 1; j >= row_begin; --j)
            {
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
                rocsparse_int idx = (transB == rocsparse_operation_none) ? i * ldb + local_col
                                                                         : local_col * ldb + i;
                sum = std::fma(-local_val, B[idx], sum);
            }

            if(diag_type == rocsparse_diag_type_non_unit)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                B[idx_B] = sum * diag_val;
            }
            else
            {
                B[idx_B] = sum;
            }
        }
    }
}

template <typename T>
inline void host_csrsm(rocsparse_int                     M,
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
                       rocsparse_int*                    numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = M + 1;
    *numeric_pivot = M + 1;

    if(transA == rocsparse_operation_none)
    {
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_lssolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csr_row_ptr,
                         csr_col_ind,
                         csr_val,
                         B,
                         ldb,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
        else
        {
            host_ussolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csr_row_ptr,
                         csr_col_ind,
                         csr_val,
                         B,
                         ldb,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
    }
    else if(transA == rocsparse_operation_transpose)
    {
        // Transpose matrix
        std::vector<rocsparse_int> csrt_row_ptr(M + 1);
        std::vector<rocsparse_int> csrt_col_ind(nnz);
        std::vector<T>             csrt_val(nnz);

        host_csr_to_csc(M,
                        M,
                        nnz,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        csrt_col_ind,
                        csrt_row_ptr,
                        csrt_val,
                        rocsparse_action_numeric,
                        base);

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_ussolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csrt_row_ptr,
                         csrt_col_ind,
                         csrt_val,
                         B,
                         ldb,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
        else
        {
            host_lssolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csrt_row_ptr,
                         csrt_col_ind,
                         csrt_val,
                         B,
                         ldb,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
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
inline void host_csric0(rocsparse_int                     M,
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

        T sum = static_cast<T>(0);

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col_j = csr_col_ind[j] - base;
            T             val_j = csr_val[j];

            // Mark diagonal and skip row
            if(col_j == ai)
            {
                has_diag = true;
                break;
            }

            // Skip upper triangular
            if(col_j > ai)
            {
                break;
            }

            rocsparse_int row_begin_j = csr_row_ptr[col_j] - base;
            rocsparse_int row_diag_j  = diag_offset[col_j];

            T local_sum = static_cast<T>(0);
            T inv_diag  = csr_val[row_diag_j];

            // Check for numeric zero
            if(inv_diag == static_cast<T>(0))
            {
                // Numerical zero diagonal
                *numeric_pivot = col_j + base;
                return;
            }

            inv_diag = static_cast<T>(1) / inv_diag;

            // loop over upper offset pointer and do linear combination for nnz entry
            for(rocsparse_int k = row_begin_j; k < row_diag_j; ++k)
            {
                rocsparse_int col_k = csr_col_ind[k] - base;

                // if nnz at this position do linear combination
                if(nnz_entries[col_k] != 0)
                {
                    rocsparse_int idx = nnz_entries[col_k];
                    local_sum         = std::fma(csr_val[k], csr_val[idx], local_sum);
                }
            }

            val_j = (val_j - local_sum) * inv_diag;
            sum   = std::fma(val_j, val_j, sum);

            csr_val[j] = val_j;
        }

        if(!has_diag)
        {
            // Structural (and numerical) zero diagonal
            *struct_pivot  = ai + base;
            *numeric_pivot = ai + base;
            return;
        }

        // Process diagonal entry
        T diag_entry = std::sqrt(std::abs(csr_val[j] - sum));
        csr_val[j]   = diag_entry;

        // Store diagonal offset
        diag_offset[ai] = j;

        // clear nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = 0;
        }
    }
}

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
                            csr_val[idx]      = std::fma(-csr_val[j], csr_val[k], csr_val[idx]);
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
template <typename T>
rocsparse_status host_nnz(rocsparse_direction       dirA,
                          rocsparse_int             m,
                          rocsparse_int             n,
                          const rocsparse_mat_descr descrA,
                          const T*                  A,
                          rocsparse_int             lda,
                          rocsparse_int*            nnz_per_row_columns,
                          rocsparse_int*            nnz_total_dev_host_ptr)
{

    rocsparse_int mn = (dirA == rocsparse_direction_row) ? m : n;
    for(rocsparse_int j = 0; j < mn; ++j)
    {
        nnz_per_row_columns[j] = 0;
    }

    for(rocsparse_int j = 0; j < n; ++j)
    {
        for(rocsparse_int i = 0; i < m; ++i)
        {
            if(A[j * lda + i] != 0)
            {
                if(dirA == rocsparse_direction_row)
                {
                    nnz_per_row_columns[i] += 1;
                }
                else
                {
                    nnz_per_row_columns[j] += 1;
                }
            }
        }
    }

    nnz_total_dev_host_ptr[0] = 0;
    for(rocsparse_int j = 0; j < mn; ++j)
    {
        nnz_total_dev_host_ptr[0] += nnz_per_row_columns[j];
    }

    return rocsparse_status_success;
}

template <rocsparse_direction DIRA, typename T>
rocsparse_status host_dense2csx(rocsparse_int        m,
                                rocsparse_int        n,
                                rocsparse_index_base base,
                                const T*             A,
                                rocsparse_int        ld,
                                const rocsparse_int* nnz_per_row_columns,
                                T*                   csx_val,
                                rocsparse_int*       csx_row_col_ptr,
                                rocsparse_int*       csx_col_row_ind)
{
    static constexpr T s_zero = {};
    rocsparse_int      len    = (rocsparse_direction_row == DIRA) ? m : n;
    *csx_row_col_ptr          = base;
    for(rocsparse_int i = 0; i < len; ++i)
    {
        csx_row_col_ptr[i + 1] = nnz_per_row_columns[i] + csx_row_col_ptr[i];
    }

    switch(DIRA)
    {
    case rocsparse_direction_column:
    {
        for(rocsparse_int j = 0; j < n; ++j)
        {
            for(rocsparse_int i = 0; i < m; ++i)
            {
                if(A[j * ld + i] != s_zero)
                {
                    *csx_val++         = A[j * ld + i];
                    *csx_col_row_ind++ = i + base;
                }
            }
        }
        return rocsparse_status_success;
    }

    case rocsparse_direction_row:
    {
        //
        // Does not matter having an orthogonal traversal ... testing only.
        // Otherwise, we would use csxRowPtrA to store the shifts.
        // and once the job is done a simple memory move would reinitialize the csxRowPtrA to its initial state)
        //
        for(rocsparse_int i = 0; i < m; ++i)
        {
            for(rocsparse_int j = 0; j < n; ++j)
            {
                if(A[j * ld + i] != s_zero)
                {
                    *csx_val++         = A[j * ld + i];
                    *csx_col_row_ind++ = j + base;
                }
            }
        }
        return rocsparse_status_success;
    }
    }

    return rocsparse_status_invalid_value;
}

template <rocsparse_direction DIRA, typename T>
rocsparse_status host_csx2dense(rocsparse_int        m,
                                rocsparse_int        n,
                                rocsparse_index_base base,
                                const T*             csx_val,
                                const rocsparse_int* csx_row_col_ptr,
                                const rocsparse_int* csx_col_row_ind,
                                T*                   A,
                                rocsparse_int        ld)
{
    static constexpr T s_zero = {};
    switch(DIRA)
    {
    case rocsparse_direction_column:
    {
        static constexpr T s_zero = {};
        for(rocsparse_int col = 0; col < n; ++col)
        {
            for(rocsparse_int row = 0; row < m; ++row)
            {
                A[row + ld * col] = s_zero;
            }
            const rocsparse_int bound = csx_row_col_ptr[col + 1] - base;
            for(rocsparse_int at = csx_row_col_ptr[col] - base; at < bound; ++at)
            {
                A[(csx_col_row_ind[at] - base) + ld * col] = csx_val[at];
            }
        }
        return rocsparse_status_success;
    }

    case rocsparse_direction_row:
    {
        static constexpr T s_zero = {};
        for(rocsparse_int row = 0; row < m; ++row)
        {
            for(rocsparse_int col = 0; col < n; ++col)
            {
                A[col * ld + row] = s_zero;
            }

            const rocsparse_int bound = csx_row_col_ptr[row + 1] - base;
            for(rocsparse_int at = csx_row_col_ptr[row] - base; at < bound; ++at)
            {
                A[(csx_col_row_ind[at] - base) * ld + row] = csx_val[at];
            }
        }
        return rocsparse_status_success;
    }
    }

    return rocsparse_status_invalid_value;
}

inline void host_csr_to_bsr_nnz(rocsparse_direction               dir,
                                rocsparse_int                     M,
                                rocsparse_int                     N,
                                rocsparse_index_base              csr_base,
                                const std::vector<rocsparse_int>& csr_row_ptr,
                                const std::vector<rocsparse_int>& csr_col_ind,
                                rocsparse_int                     block_dim,
                                rocsparse_index_base              bsr_base,
                                std::vector<rocsparse_int>&       bsr_row_ptr,
                                rocsparse_int*                    nnzTotalDevHostPtr)
{
    rocsparse_int Mb  = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb  = (N + block_dim - 1) / block_dim;
    rocsparse_int nnz = csr_row_ptr[M] - csr_row_ptr[0];

    // determine number of non-zero block columns for each block row of the bsr matrix
    bsr_row_ptr.resize(Mb + 1, 0);
    bsr_row_ptr[0] = bsr_base;

    std::vector<rocsparse_int> temp(nnz, -1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < nnz; i++)
    {
        temp[i] = (csr_col_ind[i] - csr_base) / block_dim;
    }

    // perform segmented scan on temp array to count unique values
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < Mb; i++)
    {
        rocsparse_int start = csr_row_ptr[block_dim * i] - csr_base;
        rocsparse_int end   = csr_row_ptr[min(block_dim * i + block_dim, M)] - csr_base;

        // replace duplicates with -1 marker
        for(rocsparse_int j = start; j < end; j++)
        {
            for(rocsparse_int k = j + 1; k < end; k++)
            {
                if(temp[k] == temp[j])
                {
                    temp[k] = -1;
                }
            }
        }

        // add up the number of unique elements in the segment
        rocsparse_int uniqueCount = 0;
        for(rocsparse_int j = start; j < end; j++)
        {
            if(temp[j] == -1)
            {
                continue;
            }

            uniqueCount++;
        }

        bsr_row_ptr[i + 1] = uniqueCount;
    }

    // update row pointer array
    for(int i = 0; i < Mb; i++)
    {
        bsr_row_ptr[i + 1] = bsr_row_ptr[i + 1] + bsr_row_ptr[i];
    }

    *nnzTotalDevHostPtr = bsr_row_ptr[Mb] - bsr_row_ptr[0];
}

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

template <typename T>
inline void host_csr_to_csr_compress(rocsparse_int                     M,
                                     rocsparse_int                     N,
                                     rocsparse_int                     nnz,
                                     const std::vector<rocsparse_int>& csr_row_ptr_A,
                                     const std::vector<rocsparse_int>& csr_col_ind_A,
                                     const std::vector<T>&             csr_val_A,
                                     std::vector<rocsparse_int>&       csr_row_ptr_C,
                                     std::vector<rocsparse_int>&       csr_col_ind_C,
                                     std::vector<T>&                   csr_val_C,
                                     rocsparse_index_base              base,
                                     T                                 tol)
{
    if(M <= 0 || N <= 0)
    {
        return;
    }

    // find how many entries will be in each compressed CSR matrix row
    std::vector<rocsparse_int> nnz_per_row(M);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; i++)
    {
        rocsparse_int start = csr_row_ptr_A[i] - base;
        rocsparse_int end   = csr_row_ptr_A[i + 1] - base;
        rocsparse_int count = 0;

        for(rocsparse_int j = start; j < end; j++)
        {
            if(std::abs(csr_val_A[j]) > std::real(tol)
               && std::abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                count++;
            }
        }

        nnz_per_row[i] = count;
    }

    // add up total number of entries
    rocsparse_int nnz_C = 0;
    for(rocsparse_int i = 0; i < M; i++)
    {
        nnz_C += nnz_per_row[i];
    }

    //column indices and value arrays for compressed CSR matrix
    csr_col_ind_C.resize(nnz_C);
    csr_val_C.resize(nnz_C);

    // fill in row pointer array for compressed CSR matrix
    csr_row_ptr_C.resize(M + 1);

    csr_row_ptr_C[0] = base;
    for(rocsparse_int i = 0; i < M; i++)
    {
        csr_row_ptr_C[i + 1] = csr_row_ptr_C[i] + nnz_per_row[i];
    }

    // fill in column indices and value arrays for compressed CSR matrix
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; i++)
    {
        rocsparse_int start = csr_row_ptr_A[i] - base;
        rocsparse_int end   = csr_row_ptr_A[i + 1] - base;
        rocsparse_int index = csr_row_ptr_C[i] - base;

        for(rocsparse_int j = start; j < end; j++)
        {
            if(std::abs(csr_val_A[j]) > std::real(tol)
               && std::abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                csr_col_ind_C[index] = csr_col_ind_A[j];
                csr_val_C[index]     = csr_val_A[j];
                index++;
            }
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
inline void host_bsr_to_csr(rocsparse_direction               direction,
                            rocsparse_int                     Mb,
                            rocsparse_int                     Nb,
                            rocsparse_int                     block_dim,
                            rocsparse_index_base              bsr_base,
                            const std::vector<rocsparse_int>& bsr_row_ptr,
                            const std::vector<rocsparse_int>& bsr_col_ind,
                            const std::vector<T>&             bsr_val,
                            rocsparse_index_base              csr_base,
                            std::vector<rocsparse_int>&       csr_row_ptr,
                            std::vector<rocsparse_int>&       csr_col_ind,
                            std::vector<T>&                   csr_val)
{
    rocsparse_int m    = Mb * block_dim;
    rocsparse_int n    = Nb * block_dim;
    rocsparse_int nnzb = bsr_row_ptr[Mb] - bsr_row_ptr[0];

    csr_row_ptr.resize(m + 1, 0);
    csr_col_ind.resize(nnzb * block_dim * block_dim, 0);
    csr_val.resize(nnzb * block_dim * block_dim, 0);

    // quick return if block_dim == 1
    if(block_dim == 1)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < bsr_row_ptr.size(); i++)
        {
            csr_row_ptr[i] = (bsr_row_ptr[i] - bsr_base) + csr_base;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < bsr_col_ind.size(); i++)
        {
            csr_col_ind[i] = (bsr_col_ind[i] - bsr_base) + csr_base;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < bsr_val.size(); i++)
        {
            csr_val[i] = bsr_val[i];
        }

        return;
    }

    csr_row_ptr[0] = csr_base;

    // find csr row ptr array
    for(rocsparse_int i = 0; i < Mb; i++)
    {
        rocsparse_int entries_in_row = block_dim * (bsr_row_ptr[i + 1] - bsr_row_ptr[i]);

        for(rocsparse_int j = 0; j < block_dim; j++)
        {
            csr_row_ptr[i * block_dim + j + 1] = csr_row_ptr[i * block_dim + j] + entries_in_row;
        }
    }

    rocsparse_int entries_in_block = block_dim * block_dim;

    // find csr col indices and values arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < Mb; i++)
    {
        rocsparse_int entries_in_Row     = (bsr_row_ptr[i + 1] - bsr_row_ptr[i]) * block_dim;
        rocsparse_int entries_in_row_sum = (bsr_row_ptr[i] - bsr_base) * entries_in_block;

        for(rocsparse_int j = bsr_row_ptr[i] - bsr_base; j < bsr_row_ptr[i + 1] - bsr_base; j++)
        {
            rocsparse_int col = bsr_col_ind[j] - bsr_base;
            rocsparse_int offset
                = entries_in_row_sum + block_dim * (j - (bsr_row_ptr[i] - bsr_base));

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                for(rocsparse_int l = 0; l < block_dim; l++)
                {
                    csr_col_ind[offset + k * entries_in_Row + l] = block_dim * col + l + csr_base;
                    if(direction == rocsparse_direction_row)
                    {
                        csr_val[offset + k * entries_in_Row + l]
                            = bsr_val[j * entries_in_block + k * block_dim + l];
                    }
                    else
                    {
                        csr_val[offset + k * entries_in_Row + l]
                            = bsr_val[j * entries_in_block + k + block_dim * l];
                    }
                }
            }
        }
    }
}

template <typename T>
inline void host_csr_to_bsr(rocsparse_direction               direction,
                            rocsparse_int                     M,
                            rocsparse_int                     N,
                            rocsparse_int                     block_dim,
                            rocsparse_int&                    nnzb,
                            rocsparse_index_base              csr_base,
                            const std::vector<rocsparse_int>& csr_row_ptr,
                            const std::vector<rocsparse_int>& csr_col_ind,
                            const std::vector<T>&             csr_val,
                            rocsparse_index_base              bsr_base,
                            std::vector<rocsparse_int>&       bsr_row_ptr,
                            std::vector<rocsparse_int>&       bsr_col_ind,
                            std::vector<T>&                   bsr_val)
{
    rocsparse_int mb = (M + block_dim - 1) / block_dim;
    rocsparse_int nb = (N + block_dim - 1) / block_dim;

    // quick return if block_dim == 1
    if(block_dim == 1)
    {
        bsr_row_ptr.resize(mb + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < csr_row_ptr.size(); i++)
        {
            bsr_row_ptr[i] = (csr_row_ptr[i] - csr_base) + bsr_base;
        }

        nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];

        bsr_col_ind.resize(nnzb, 0);
        bsr_val.resize(nnzb * block_dim * block_dim, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < csr_col_ind.size(); i++)
        {
            bsr_col_ind[i] = (csr_col_ind[i] - csr_base) + bsr_base;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < csr_val.size(); i++)
        {
            bsr_val[i] = csr_val[i];
        }

        return;
    }

    // determine number of non-zero block columns for each block row of the bsr matrix
    bsr_row_ptr.resize(mb + 1, 0);

    bsr_row_ptr[0] = bsr_base;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < mb; i++)
    {
        rocsparse_int start = csr_row_ptr[i * block_dim] - csr_base;
        rocsparse_int end   = csr_row_ptr[std::min(M, block_dim * i + block_dim)] - csr_base;

        std::vector<rocsparse_int> temp(nb, 0);
        for(rocsparse_int j = start; j < end; j++)
        {
            rocsparse_int blockCol = (csr_col_ind[j] - csr_base) / block_dim;
            temp[blockCol]         = 1;
        }

        rocsparse_int sum = 0;
        for(rocsparse_int j = 0; j < temp.size(); j++)
        {
            sum += temp[j];
        }

        bsr_row_ptr[i + 1] = sum;
    }

    for(rocsparse_int i = 0; i < mb; i++)
    {
        bsr_row_ptr[i + 1] += bsr_row_ptr[i];
    }

    nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];

    // find bsr col indices array
    bsr_col_ind.resize(nnzb, 0);
    bsr_val.resize(nnzb * block_dim * block_dim, 0);

    rocsparse_int colIndex = 0;

    for(rocsparse_int i = 0; i < mb; i++)
    {
        rocsparse_int start = csr_row_ptr[i * block_dim] - csr_base;
        rocsparse_int end   = csr_row_ptr[std::min(M, block_dim * i + block_dim)] - csr_base;

        std::vector<rocsparse_int> temp(nb, 0);

        for(rocsparse_int j = start; j < end; j++)
        {
            rocsparse_int blockCol = (csr_col_ind[j] - csr_base) / block_dim;
            temp[blockCol]         = 1;
        }

        for(rocsparse_int j = 0; j < nb; j++)
        {
            if(temp[j] == 1)
            {
                bsr_col_ind[colIndex] = j + bsr_base;
                colIndex++;
            }
        }
    }

    // find bsr values array
    for(rocsparse_int i = 0; i < M; i++)
    {
        rocsparse_int blockRow = i / block_dim;

        rocsparse_int start = csr_row_ptr[i] - csr_base;
        rocsparse_int end   = csr_row_ptr[i + 1] - csr_base;

        for(rocsparse_int j = start; j < end; j++)
        {
            rocsparse_int blockCol = (csr_col_ind[j] - csr_base) / block_dim;

            colIndex = -1;
            for(rocsparse_int k = bsr_row_ptr[blockRow] - bsr_base;
                k < bsr_row_ptr[blockRow + 1] - bsr_base;
                k++)
            {
                if(bsr_col_ind[k] - bsr_base == blockCol)
                {
                    colIndex = k - (bsr_row_ptr[blockRow] - bsr_base);
                    break;
                }
            }

            assert(colIndex != -1);

            rocsparse_int blockIndex = 0;
            if(direction == rocsparse_direction_row)
            {
                blockIndex = (csr_col_ind[j] - csr_base) % block_dim + (i % block_dim) * block_dim;
            }
            else
            {
                blockIndex
                    = ((csr_col_ind[j] - csr_base) % block_dim) * block_dim + (i % block_dim);
            }

            rocsparse_int index = (bsr_row_ptr[blockRow] - bsr_base) * block_dim * block_dim
                                  + colIndex * block_dim * block_dim + blockIndex;

            bsr_val[index] = csr_val[j];
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
