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
#include "utility.hpp"

#include <limits>

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
template <typename I, typename T>
void host_axpby(
    I size, I nnz, T alpha, const T* x_val, const I* x_ind, T beta, T* y, rocsparse_index_base base)
{
    for(I i = 0; i < size; ++i)
    {
        y[i] *= beta;
    }

    for(I i = 0; i < nnz; ++i)
    {
        y[x_ind[i] - base] = std::fma(alpha, x_val[i], y[x_ind[i] - base]);
    }
}

template <typename I, typename T>
void host_doti(
    I nnz, const T* x_val, const I* x_ind, const T* y, T* result, rocsparse_index_base base)
{
    *result = static_cast<T>(0);

    for(I i = 0; i < nnz; ++i)
    {
        *result = std::fma(y[x_ind[i] - base], x_val[i], *result);
    }
}

template <typename I, typename T>
void host_dotci(
    I nnz, const T* x_val, const I* x_ind, const T* y, T* result, rocsparse_index_base base)
{
    *result = static_cast<T>(0);

    for(I i = 0; i < nnz; ++i)
    {
        *result = std::fma(rocsparse_conj(x_val[i]), y[x_ind[i] - base], *result);
    }
}

template <typename I, typename T>
void host_gthr(I nnz, const T* y, T* x_val, const I* x_ind, rocsparse_index_base base)
{
    for(I i = 0; i < nnz; ++i)
    {
        x_val[i] = y[x_ind[i] - base];
    }
}

template <typename T>
void host_gthrz(
    rocsparse_int nnz, T* y, T* x_val, const rocsparse_int* x_ind, rocsparse_index_base base)
{
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        x_val[i]           = y[x_ind[i] - base];
        y[x_ind[i] - base] = static_cast<T>(0);
    }
}

template <typename I, typename T>
void host_roti(
    I nnz, T* x_val, const I* x_ind, T* y, const T* c, const T* s, rocsparse_index_base base)
{
    for(I i = 0; i < nnz; ++i)
    {
        I idx = x_ind[i] - base;

        T xs = x_val[i];
        T ys = y[idx];

        x_val[i] = *c * xs + *s * ys;
        y[idx]   = *c * ys - *s * xs;
    }
}

template <typename I, typename T>
void host_sctr(I nnz, const T* x_val, const I* x_ind, T* y, rocsparse_index_base base)
{
    for(I i = 0; i < nnz; ++i)
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
void host_bsrxmv(rocsparse_direction  dir,
                 rocsparse_operation  trans,
                 rocsparse_int        size_of_mask,
                 rocsparse_int        mb,
                 rocsparse_int        nb,
                 rocsparse_int        nnzb,
                 T                    alpha,
                 const rocsparse_int* bsr_mask_ptr,
                 const rocsparse_int* bsr_row_ptr,
                 const rocsparse_int* bsr_end_ptr,
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
            for(rocsparse_int i = 0; i < size_of_mask; ++i)
            {
                rocsparse_int shift = (bsr_mask_ptr[i] - base) * bsr_dim;
                for(rocsparse_int j = 0; j < bsr_dim; ++j)
                {
                    y[shift + j] *= beta;
                }
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
    for(rocsparse_int mask_idx = 0; mask_idx < size_of_mask; ++mask_idx)
    {
        rocsparse_int row       = bsr_mask_ptr[mask_idx] - base;
        rocsparse_int row_begin = bsr_row_ptr[row] - base;
        rocsparse_int row_end   = bsr_end_ptr[row] - base;

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
void host_gebsrmv(rocsparse_direction  dir,
                  rocsparse_operation  trans,
                  rocsparse_int        mb,
                  rocsparse_int        nb,
                  rocsparse_int        nnzb,
                  T                    alpha,
                  const rocsparse_int* bsr_row_ptr,
                  const rocsparse_int* bsr_col_ind,
                  const T*             bsr_val,
                  rocsparse_int        row_block_dim,
                  rocsparse_int        col_block_dim,
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
            for(rocsparse_int i = 0; i < mb * row_block_dim; ++i)
            {
                y[i] *= beta;
            }
        }

        return;
    }

    if(row_block_dim == col_block_dim)
    {
        host_bsrmv(dir,
                   trans,
                   mb,
                   nb,
                   nnzb,
                   alpha,
                   bsr_row_ptr,
                   bsr_col_ind,
                   bsr_val,
                   row_block_dim,
                   x,
                   beta,
                   y,
                   base);

        return;
    }

    rocsparse_int WFSIZE;

    if(row_block_dim == 2 || row_block_dim == 3 || row_block_dim == 4)
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
    else if(row_block_dim <= 8)
    {
        WFSIZE = 8;
    }
    else if(row_block_dim <= 16)
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

        if(row_block_dim == 2)
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

                        for(rocsparse_int l = 0; l < col_block_dim; l++)
                        {
                            if(dir == rocsparse_direction_column)
                            {
                                sum0[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l],
                                                   x[col * col_block_dim + l],
                                                   sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 1],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                            }
                            else
                            {
                                sum0[k]
                                    = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k) + l],
                                               x[col * col_block_dim + l],
                                               sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                            }
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
                y[row * row_block_dim + 0]
                    = std::fma(beta, y[row * row_block_dim + 0], alpha * sum0[0]);
                y[row * row_block_dim + 1]
                    = std::fma(beta, y[row * row_block_dim + 1], alpha * sum1[0]);
            }
            else
            {
                y[row * row_block_dim + 0] = alpha * sum0[0];
                y[row * row_block_dim + 1] = alpha * sum1[0];
            }
        }
        else if(row_block_dim == 3)
        {
            std::vector<T> sum0(WFSIZE, static_cast<T>(0));
            std::vector<T> sum1(WFSIZE, static_cast<T>(0));
            std::vector<T> sum2(WFSIZE, static_cast<T>(0));

            for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
            {
                for(rocsparse_int k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        rocsparse_int col = bsr_col_ind[j + k] - base;

                        for(rocsparse_int l = 0; l < col_block_dim; l++)
                        {
                            if(dir == rocsparse_direction_column)
                            {
                                sum0[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l],
                                                   x[col * col_block_dim + l],
                                                   sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 1],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                                sum2[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 2],
                                                   x[col * col_block_dim + l],
                                                   sum2[k]);
                            }
                            else
                            {
                                sum0[k]
                                    = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k) + l],
                                               x[col * col_block_dim + l],
                                               sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                                sum2[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + 2 * col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum2[k]);
                            }
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
                    sum2[k] += sum2[k + j];
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[row * row_block_dim + 0]
                    = std::fma(beta, y[row * row_block_dim + 0], alpha * sum0[0]);
                y[row * row_block_dim + 1]
                    = std::fma(beta, y[row * row_block_dim + 1], alpha * sum1[0]);
                y[row * row_block_dim + 2]
                    = std::fma(beta, y[row * row_block_dim + 2], alpha * sum2[0]);
            }
            else
            {
                y[row * row_block_dim + 0] = alpha * sum0[0];
                y[row * row_block_dim + 1] = alpha * sum1[0];
                y[row * row_block_dim + 2] = alpha * sum2[0];
            }
        }
        else if(row_block_dim == 4)
        {
            std::vector<T> sum0(WFSIZE, static_cast<T>(0));
            std::vector<T> sum1(WFSIZE, static_cast<T>(0));
            std::vector<T> sum2(WFSIZE, static_cast<T>(0));
            std::vector<T> sum3(WFSIZE, static_cast<T>(0));

            for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
            {
                for(rocsparse_int k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        rocsparse_int col = bsr_col_ind[j + k] - base;

                        for(rocsparse_int l = 0; l < col_block_dim; l++)
                        {
                            if(dir == rocsparse_direction_column)
                            {
                                sum0[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l],
                                                   x[col * col_block_dim + l],
                                                   sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 1],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                                sum2[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 2],
                                                   x[col * col_block_dim + l],
                                                   sum2[k]);
                                sum3[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 3],
                                                   x[col * col_block_dim + l],
                                                   sum3[k]);
                            }
                            else
                            {
                                sum0[k]
                                    = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k) + l],
                                               x[col * col_block_dim + l],
                                               sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                                sum2[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + 2 * col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum2[k]);
                                sum3[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + 3 * col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum3[k]);
                            }
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
                    sum2[k] += sum2[k + j];
                    sum3[k] += sum3[k + j];
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[row * row_block_dim + 0]
                    = std::fma(beta, y[row * row_block_dim + 0], alpha * sum0[0]);
                y[row * row_block_dim + 1]
                    = std::fma(beta, y[row * row_block_dim + 1], alpha * sum1[0]);
                y[row * row_block_dim + 2]
                    = std::fma(beta, y[row * row_block_dim + 2], alpha * sum2[0]);
                y[row * row_block_dim + 3]
                    = std::fma(beta, y[row * row_block_dim + 3], alpha * sum3[0]);
            }
            else
            {
                y[row * row_block_dim + 0] = alpha * sum0[0];
                y[row * row_block_dim + 1] = alpha * sum1[0];
                y[row * row_block_dim + 2] = alpha * sum2[0];
                y[row * row_block_dim + 3] = alpha * sum3[0];
            }
        }
        else
        {
            for(rocsparse_int bi = 0; bi < row_block_dim; ++bi)
            {
                std::vector<T> sum(WFSIZE, static_cast<T>(0));

                for(rocsparse_int j = row_begin; j < row_end; ++j)
                {
                    rocsparse_int col = bsr_col_ind[j] - base;

                    for(rocsparse_int bj = 0; bj < col_block_dim; bj += WFSIZE)
                    {
                        for(unsigned int k = 0; k < WFSIZE; ++k)
                        {
                            if(bj + k < col_block_dim)
                            {
                                if(dir == rocsparse_direction_column)
                                {
                                    sum[k] = std::fma(bsr_val[row_block_dim * col_block_dim * j
                                                              + row_block_dim * (bj + k) + bi],
                                                      x[col_block_dim * col + (bj + k)],
                                                      sum[k]);
                                }
                                else
                                {
                                    sum[k] = std::fma(bsr_val[row_block_dim * col_block_dim * j
                                                              + col_block_dim * bi + (bj + k)],
                                                      x[col_block_dim * col + (bj + k)],
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
                    y[row * row_block_dim + bi]
                        = std::fma(beta, y[row * row_block_dim + bi], alpha * sum[0]);
                }
                else
                {
                    y[row * row_block_dim + bi] = alpha * sum[0];
                }
            }
        }
    }
}

template <typename T>
static inline void host_bsr_lsolve(rocsparse_direction  dir,
                                   rocsparse_operation  trans_X,
                                   rocsparse_int        mb,
                                   rocsparse_int        nrhs,
                                   T                    alpha,
                                   const rocsparse_int* bsr_row_ptr,
                                   const rocsparse_int* bsr_col_ind,
                                   const T*             bsr_val,
                                   rocsparse_int        bsr_dim,
                                   const T*             B,
                                   rocsparse_int        ldb,
                                   T*                   X,
                                   rocsparse_int        ldx,
                                   rocsparse_diag_type  diag_type,
                                   rocsparse_index_base base,
                                   rocsparse_int*       struct_pivot,
                                   rocsparse_int*       numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(rocsparse_int i = 0; i < nrhs; ++i)
    {
        // Process lower triangular part
        for(rocsparse_int bsr_row = 0; bsr_row < mb; ++bsr_row)
        {
            rocsparse_int bsr_row_begin = bsr_row_ptr[bsr_row] - base;
            rocsparse_int bsr_row_end   = bsr_row_ptr[bsr_row + 1] - base;

            // Loop over blocks rows
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                rocsparse_int diag      = -1;
                rocsparse_int local_row = bsr_row * bsr_dim + bi;

                rocsparse_int idx_B = (trans_X == rocsparse_operation_none) ? i * ldb + local_row
                                                                            : local_row * ldb + i;
                rocsparse_int idx_X = (trans_X == rocsparse_operation_none) ? i * ldx + local_row
                                                                            : local_row * ldx + i;

                T sum      = alpha * B[idx_B];
                T diag_val = static_cast<T>(0);

                // Loop over BSR columns
                for(rocsparse_int j = bsr_row_begin; j < bsr_row_end; ++j)
                {
                    rocsparse_int bsr_col = bsr_col_ind[j] - base;

                    // Loop over blocks columns
                    for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
                    {
                        rocsparse_int local_col = bsr_col * bsr_dim + bj;
                        T             local_val = (dir == rocsparse_direction_row)
                                                      ? bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj]
                                                      : bsr_val[bsr_dim * bsr_dim * j + bi + bj * bsr_dim];

                        if(local_val == static_cast<T>(0) && local_col == local_row
                           && diag_type == rocsparse_diag_type_non_unit)
                        {
                            // Numerical zero pivot found, avoid division by 0
                            // and store index for later use.
                            *numeric_pivot = std::min(*numeric_pivot, bsr_row + base);
                            local_val      = static_cast<T>(1);
                        }

                        // Ignore all entries that are above the diagonal
                        if(local_col > local_row)
                        {
                            break;
                        }

                        // Diagonal
                        if(local_col == local_row)
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
                        rocsparse_int idx = (trans_X == rocsparse_operation_none)
                                                ? i * ldx + local_col
                                                : local_col * ldx + i;
                        sum               = std::fma(-local_val, X[idx], sum);
                    }
                }

                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    if(diag == -1)
                    {
                        *struct_pivot = std::min(*struct_pivot, bsr_row + base);
                    }

                    X[idx_X] = sum * diag_val;
                }
                else
                {
                    X[idx_X] = sum;
                }
            }
        }
    }
}

template <typename T>
static inline void host_bsr_usolve(rocsparse_direction  dir,
                                   rocsparse_operation  trans_X,
                                   rocsparse_int        mb,
                                   rocsparse_int        nrhs,
                                   T                    alpha,
                                   const rocsparse_int* bsr_row_ptr,
                                   const rocsparse_int* bsr_col_ind,
                                   const T*             bsr_val,
                                   rocsparse_int        bsr_dim,
                                   const T*             B,
                                   rocsparse_int        ldb,
                                   T*                   X,
                                   rocsparse_int        ldx,
                                   rocsparse_diag_type  diag_type,
                                   rocsparse_index_base base,
                                   rocsparse_int*       struct_pivot,
                                   rocsparse_int*       numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < nrhs; ++i)
    {
        // Process upper triangular part
        for(rocsparse_int bsr_row = mb - 1; bsr_row >= 0; --bsr_row)
        {
            rocsparse_int bsr_row_begin = bsr_row_ptr[bsr_row] - base;
            rocsparse_int bsr_row_end   = bsr_row_ptr[bsr_row + 1] - base;

            for(rocsparse_int bi = bsr_dim - 1; bi >= 0; --bi)
            {
                rocsparse_int local_row = bsr_row * bsr_dim + bi;

                rocsparse_int idx_B = (trans_X == rocsparse_operation_none) ? i * ldb + local_row
                                                                            : local_row * ldb + i;
                rocsparse_int idx_X = (trans_X == rocsparse_operation_none) ? i * ldx + local_row
                                                                            : local_row * ldx + i;
                T             sum   = alpha * B[idx_B];

                rocsparse_int diag     = -1;
                T             diag_val = static_cast<T>(0);

                for(rocsparse_int j = bsr_row_end - 1; j >= bsr_row_begin; --j)
                {
                    rocsparse_int bsr_col = bsr_col_ind[j] - base;

                    for(rocsparse_int bj = bsr_dim - 1; bj >= 0; --bj)
                    {
                        rocsparse_int local_col = bsr_col * bsr_dim + bj;
                        T             local_val = dir == rocsparse_direction_row
                                                      ? bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj]
                                                      : bsr_val[bsr_dim * bsr_dim * j + bi + bj * bsr_dim];

                        // Ignore all entries that are below the diagonal
                        if(local_col < local_row)
                        {
                            continue;
                        }

                        // Diagonal
                        if(local_col == local_row)
                        {
                            if(diag_type == rocsparse_diag_type_non_unit)
                            {
                                // Check for numerical zero
                                if(local_val == static_cast<T>(0))
                                {
                                    *numeric_pivot = std::min(*numeric_pivot, bsr_row + base);
                                    local_val      = static_cast<T>(1);
                                }

                                diag     = j;
                                diag_val = static_cast<T>(1) / local_val;
                            }

                            continue;
                        }

                        // Upper triangular part
                        rocsparse_int idx = (trans_X == rocsparse_operation_none)
                                                ? i * ldx + local_col
                                                : local_col * ldx + i;
                        sum               = std::fma(-local_val, X[idx], sum);
                    }
                }

                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    if(diag == -1)
                    {
                        *struct_pivot = std::min(*struct_pivot, bsr_row + base);
                    }

                    X[idx_X] = sum * diag_val;
                }
                else
                {
                    X[idx_X] = sum;
                }
            }
        }
    }
}

template <typename T>
void host_bsrsv(rocsparse_operation  trans,
                rocsparse_direction  dir,
                rocsparse_int        mb,
                rocsparse_int        nnzb,
                T                    alpha,
                const rocsparse_int* bsr_row_ptr,
                const rocsparse_int* bsr_col_ind,
                const T*             bsr_val,
                rocsparse_int        bsr_dim,
                const T*             x,
                T*                   y,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                rocsparse_int*       struct_pivot,
                rocsparse_int*       numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    if(trans == rocsparse_operation_none)
    {
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_bsr_lsolve(dir,
                            rocsparse_operation_none,
                            mb,
                            1,
                            alpha,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_val,
                            bsr_dim,
                            x,
                            mb * bsr_dim,
                            y,
                            mb * bsr_dim,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_bsr_usolve(dir,
                            rocsparse_operation_none,
                            mb,
                            1,
                            alpha,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_val,
                            bsr_dim,
                            x,
                            mb * bsr_dim,
                            y,
                            mb * bsr_dim,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }
    else if(trans == rocsparse_operation_transpose)
    {
        // Transpose matrix
        std::vector<rocsparse_int> bsrt_row_ptr;
        std::vector<rocsparse_int> bsrt_col_ind;
        std::vector<T>             bsrt_val;

        host_bsr_to_bsc(mb,
                        mb,
                        nnzb,
                        bsr_dim,
                        bsr_row_ptr,
                        bsr_col_ind,
                        bsr_val,
                        bsrt_col_ind,
                        bsrt_row_ptr,
                        bsrt_val,
                        base,
                        base);

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_bsr_usolve(dir,
                            rocsparse_operation_none,
                            mb,
                            1,
                            alpha,
                            bsrt_row_ptr.data(),
                            bsrt_col_ind.data(),
                            bsrt_val.data(),
                            bsr_dim,
                            x,
                            mb * bsr_dim,
                            y,
                            mb * bsr_dim,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_bsr_lsolve(dir,
                            rocsparse_operation_none,
                            mb,
                            1,
                            alpha,
                            bsrt_row_ptr.data(),
                            bsrt_col_ind.data(),
                            bsrt_val.data(),
                            bsr_dim,
                            x,
                            mb * bsr_dim,
                            y,
                            mb * bsr_dim,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;
}

template <typename I, typename T>
void host_coomv(rocsparse_operation  trans,
                I                    M,
                I                    N,
                I                    nnz,
                T                    alpha,
                const I*             coo_row_ind,
                const I*             coo_col_ind,
                const T*             coo_val,
                const T*             x,
                T                    beta,
                T*                   y,
                rocsparse_index_base base)
{
    if(trans == rocsparse_operation_none)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            y[i] *= beta;
        }

        for(I i = 0; i < nnz; ++i)
        {
            y[coo_row_ind[i] - base]
                = std::fma(alpha * coo_val[i], x[coo_col_ind[i] - base], y[coo_row_ind[i] - base]);
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        for(I i = 0; i < nnz; ++i)
        {
            I row = coo_row_ind[i] - base;
            I col = coo_col_ind[i] - base;
            T val = (trans == rocsparse_operation_transpose) ? coo_val[i]
                                                             : rocsparse_conj(coo_val[i]);

            y[col] = std::fma(alpha * val, x[row], y[col]);
        }
    }
}

template <typename I, typename T>
void host_coomv_aos(rocsparse_operation  trans,
                    I                    M,
                    I                    N,
                    I                    nnz,
                    T                    alpha,
                    const I*             coo_ind,
                    const T*             coo_val,
                    const T*             x,
                    T                    beta,
                    T*                   y,
                    rocsparse_index_base base)
{
    switch(trans)
    {
    case rocsparse_operation_none:
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            y[i] *= beta;
        }

        for(I i = 0; i < nnz; ++i)
        {
            y[coo_ind[2 * i] - base] = std::fma(
                alpha * coo_val[i], x[coo_ind[2 * i + 1] - base], y[coo_ind[2 * i] - base]);
        }

        break;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        for(I i = 0; i < nnz; ++i)
        {
            I row = coo_ind[2 * i] - base;
            I col = coo_ind[2 * i + 1] - base;
            T val = (trans == rocsparse_operation_transpose) ? coo_val[i]
                                                             : rocsparse_conj(coo_val[i]);

            y[col] = std::fma(alpha * val, x[row], y[col]);
        }

        break;
    }
    }
}

template <typename I, typename J, typename T>
static void host_csrmv_general(rocsparse_operation  trans,
                               J                    M,
                               J                    N,
                               I                    nnz,
                               T                    alpha,
                               const I*             csr_row_ptr,
                               const J*             csr_col_ind,
                               const T*             csr_val,
                               const T*             x,
                               T                    beta,
                               T*                   y,
                               rocsparse_index_base base,
                               rocsparse_spmv_alg   algo)
{
    if(trans == rocsparse_operation_none)
    {
        if(algo == rocsparse_spmv_alg_csr_stream)
        {
            // Get device properties
            int             dev;
            hipDeviceProp_t prop;

            hipGetDevice(&dev);
            hipGetDeviceProperties(&prop, dev);

            int WF_SIZE;
            J   nnz_per_row = nnz / M;

            if(nnz_per_row < 4)
                WF_SIZE = 2;
            else if(nnz_per_row < 8)
                WF_SIZE = 4;
            else if(nnz_per_row < 16)
                WF_SIZE = 8;
            else if(nnz_per_row < 32)
                WF_SIZE = 16;
            else if(nnz_per_row < 64 || prop.warpSize == 32)
                WF_SIZE = 32;
            else
                WF_SIZE = 64;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(J i = 0; i < M; ++i)
            {
                I row_begin = csr_row_ptr[i] - base;
                I row_end   = csr_row_ptr[i + 1] - base;

                std::vector<T> sum(WF_SIZE, static_cast<T>(0));

                for(I j = row_begin; j < row_end; j += WF_SIZE)
                {
                    for(int k = 0; k < WF_SIZE; ++k)
                    {
                        if(j + k < row_end)
                        {
                            sum[k] = std::fma(
                                alpha * csr_val[j + k], x[csr_col_ind[j + k] - base], sum[k]);
                        }
                    }
                }

                for(int j = 1; j < WF_SIZE; j <<= 1)
                {
                    for(int k = 0; k < WF_SIZE - j; ++k)
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
            for(J i = 0; i < M; ++i)
            {
                T sum = static_cast<T>(0);
                T err = static_cast<T>(0);

                I row_begin = csr_row_ptr[i] - base;
                I row_end   = csr_row_ptr[i + 1] - base;

                for(I j = row_begin; j < row_end; ++j)
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
    else
    {
        // Scale y with beta
        for(J i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        // Transposed SpMV
        for(J i = 0; i < M; ++i)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;
            T row_val   = alpha * x[i];

            for(I j = row_begin; j < row_end; ++j)
            {
                J col  = csr_col_ind[j] - base;
                T val  = (trans == rocsparse_operation_conjugate_transpose)
                             ? rocsparse_conj(csr_val[j])
                             : csr_val[j];
                y[col] = std::fma(val, row_val, y[col]);
            }
        }
    }
}

template <typename I, typename J, typename T>
static void host_csrmv_symmetric(rocsparse_operation  trans,
                                 J                    M,
                                 J                    N,
                                 I                    nnz,
                                 T                    alpha,
                                 const I*             csr_row_ptr,
                                 const J*             csr_col_ind,
                                 const T*             csr_val,
                                 const T*             x,
                                 T                    beta,
                                 T*                   y,
                                 rocsparse_index_base base,
                                 rocsparse_spmv_alg   algo)
{
    if(algo == rocsparse_spmv_alg_csr_stream || trans != rocsparse_operation_none)
    {
        // Get device properties
        int             dev;
        hipDeviceProp_t prop;

        hipGetDevice(&dev);
        hipGetDeviceProperties(&prop, dev);

        int WF_SIZE;
        J   nnz_per_row = nnz / M;

        if(nnz_per_row < 4)
            WF_SIZE = 2;
        else if(nnz_per_row < 8)
            WF_SIZE = 4;
        else if(nnz_per_row < 16)
            WF_SIZE = 8;
        else if(nnz_per_row < 32)
            WF_SIZE = 16;
        else if(nnz_per_row < 64 || prop.warpSize == 32)
            WF_SIZE = 32;
        else
            WF_SIZE = 64;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; ++i)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            std::vector<T> sum(WF_SIZE, static_cast<T>(0));

            for(I j = row_begin; j < row_end; j += WF_SIZE)
            {
                for(int k = 0; k < WF_SIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        T val  = (trans == rocsparse_operation_conjugate_transpose)
                                     ? rocsparse_conj(csr_val[j + k])
                                     : csr_val[j + k];
                        sum[k] = std::fma(alpha * val, x[csr_col_ind[j + k] - base], sum[k]);
                    }
                }
            }

            for(int j = 1; j < WF_SIZE; j <<= 1)
            {
                for(int k = 0; k < WF_SIZE - j; ++k)
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

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            T x_val = alpha * x[i];
            for(I j = row_begin; j < row_end; ++j)
            {
                if((csr_col_ind[j] - base) != i)
                {
                    y[csr_col_ind[j] - base] += (trans == rocsparse_operation_conjugate_transpose)
                                                    ? rocsparse_conj(csr_val[j]) * x_val
                                                    : csr_val[j] * x_val;
                }
            }
        }
    }
    else
    {
        // Scale y with beta
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; ++i)
        {
            y[i] *= beta;
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; i++)
        {
            T sum = static_cast<T>(0);
            T err = static_cast<T>(0);

            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            for(I j = row_begin; j < row_end; ++j)
            {
                T old  = sum;
                T prod = alpha * csr_val[j] * x[csr_col_ind[j] - base];

                sum = sum + prod;
                err = (old - (sum - (sum - old))) + (prod - (sum - old)) + err;
            }

            y[i] += sum + err;
        }

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            T x_val = alpha * x[i];
            for(I j = row_begin; j < row_end; ++j)
            {
                if((csr_col_ind[j] - base) != i)
                {
                    y[csr_col_ind[j] - base] += csr_val[j] * x_val;
                }
            }
        }
    }
}

template <typename I, typename J, typename T>
void host_csrmv(rocsparse_operation   trans,
                J                     M,
                J                     N,
                I                     nnz,
                T                     alpha,
                const I*              csr_row_ptr,
                const J*              csr_col_ind,
                const T*              csr_val,
                const T*              x,
                T                     beta,
                T*                    y,
                rocsparse_index_base  base,
                rocsparse_matrix_type matrix_type,
                rocsparse_spmv_alg    algo)
{
    switch(matrix_type)
    {
    case rocsparse_matrix_type_symmetric:
    {
        host_csrmv_symmetric(
            trans, M, N, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, base, algo);
        break;
    }
    default:
    {
        host_csrmv_general(
            trans, M, N, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, base, algo);
        break;
    }
    }
}

template <typename I, typename J, typename T>
static void host_csr_lsolve(J                    M,
                            T                    alpha,
                            const I*             csr_row_ptr,
                            const J*             csr_col_ind,
                            const T*             csr_val,
                            const T*             x,
                            T*                   y,
                            rocsparse_diag_type  diag_type,
                            rocsparse_index_base base,
                            J*                   struct_pivot,
                            J*                   numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    hipGetDevice(&dev);
    hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

    // Process lower triangular part
    for(J row = 0; row < M; ++row)
    {
        temp.assign(prop.warpSize, static_cast<T>(0));
        temp[0] = alpha * x[row];

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = static_cast<T>(0);

        for(I l = row_begin; l < row_end; l += prop.warpSize)
        {
            for(unsigned int k = 0; k < prop.warpSize; ++k)
            {
                I j = l + k;

                // Do not run out of bounds
                if(j >= row_end)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

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

template <typename I, typename J, typename T>
static void host_csr_usolve(J                    M,
                            T                    alpha,
                            const I*             csr_row_ptr,
                            const J*             csr_col_ind,
                            const T*             csr_val,
                            const T*             x,
                            T*                   y,
                            rocsparse_diag_type  diag_type,
                            rocsparse_index_base base,
                            J*                   struct_pivot,
                            J*                   numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    hipGetDevice(&dev);
    hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

    // Process upper triangular part
    for(J row = M - 1; row >= 0; --row)
    {
        temp.assign(prop.warpSize, static_cast<T>(0));
        temp[0] = alpha * x[row];

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = static_cast<T>(0);

        for(I l = row_end - 1; l >= row_begin; l -= prop.warpSize)
        {
            for(unsigned int k = 0; k < prop.warpSize; ++k)
            {
                I j = l - k;

                // Do not run out of bounds
                if(j < row_begin)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

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

template <typename I, typename J, typename T>
void host_csrsv(rocsparse_operation  trans,
                J                    M,
                I                    nnz,
                T                    alpha,
                const I*             csr_row_ptr,
                const J*             csr_col_ind,
                const T*             csr_val,
                const T*             x,
                T*                   y,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                J*                   struct_pivot,
                J*                   numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = M + 1;
    *numeric_pivot = M + 1;

    if(trans == rocsparse_operation_none)
    {
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_csr_lsolve(M,
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
            host_csr_usolve(M,
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
    else if(trans == rocsparse_operation_transpose
            || trans == rocsparse_operation_conjugate_transpose)
    {
        // Transpose matrix
        std::vector<I> csrt_row_ptr(M + 1);
        std::vector<J> csrt_col_ind(nnz);
        std::vector<T> csrt_val(nnz);

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

        if(trans == rocsparse_operation_conjugate_transpose)
        {
            for(size_t i = 0; i < csrt_val.size(); i++)
            {
                csrt_val[i] = rocsparse_conj(csrt_val[i]);
            }
        }

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_csr_usolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
                            x,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_csr_lsolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
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

template <typename I, typename T>
void host_coosv(rocsparse_operation   trans,
                I                     M,
                I                     nnz,
                T                     alpha,
                const std::vector<I>& coo_row_ind,
                const std::vector<I>& coo_col_ind,
                const std::vector<T>& coo_val,
                const std::vector<T>& x,
                std::vector<T>&       y,
                rocsparse_diag_type   diag_type,
                rocsparse_fill_mode   fill_mode,
                rocsparse_index_base  base,
                I*                    struct_pivot,
                I*                    numeric_pivot)
{
    std::vector<I> csr_row_ptr(M + 1);

    host_coo_to_csr(M, nnz, coo_row_ind.data(), csr_row_ptr, base);

    host_csrsv(trans,
               M,
               nnz,
               alpha,
               csr_row_ptr.data(),
               coo_col_ind.data(),
               coo_val.data(),
               x.data(),
               y.data(),
               diag_type,
               fill_mode,
               base,
               struct_pivot,
               numeric_pivot);
}

template <typename I, typename T>
void host_ellmv(rocsparse_operation  trans,
                I                    M,
                I                    N,
                T                    alpha,
                const I*             ell_col_ind,
                const T*             ell_val,
                I                    ell_width,
                const T*             x,
                T                    beta,
                T*                   y,
                rocsparse_index_base base)
{
    if(trans == rocsparse_operation_none)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            T sum = static_cast<T>(0);
            for(I p = 0; p < ell_width; ++p)
            {
                I idx = p * M + i;
                I col = ell_col_ind[idx] - base;

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
    else
    {
        // Scale y with beta
        for(I i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        // Transposed SpMV
        for(I i = 0; i < M; ++i)
        {
            T row_val = alpha * x[i];

            for(I p = 0; p < ell_width; ++p)
            {
                I idx = p * M + i;
                I col = ell_col_ind[idx] - base;

                if(col >= 0 && col < N)
                {
                    T val = (trans == rocsparse_operation_conjugate_transpose)
                                ? rocsparse_conj(ell_val[idx])
                                : ell_val[idx];

                    y[col] = std::fma(val, row_val, y[col]);
                }
                else
                {
                    break;
                }
            }
        }
    }
}

template <typename T>
void host_hybmv(rocsparse_operation  trans,
                rocsparse_int        M,
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
        host_ellmv(trans, M, N, alpha, ell_col_ind, ell_val, ell_width, x, beta, y, base);
        coo_beta = static_cast<T>(1);
    }

    // COO part
    if(coo_nnz > 0)
    {
        host_coomv(
            trans, M, N, coo_nnz, alpha, coo_row_ind, coo_col_ind, coo_val, x, coo_beta, y, base);
    }
}

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T>
void host_bsrmm(rocsparse_handle          handle,
                rocsparse_direction       dir,
                rocsparse_operation       transA,
                rocsparse_operation       transB,
                rocsparse_int             Mb,
                rocsparse_int             N,
                rocsparse_int             Kb,
                rocsparse_int             nnzb,
                const T*                  alpha,
                const rocsparse_mat_descr descr,
                const T*                  bsr_val_A,
                const rocsparse_int*      bsr_row_ptr_A,
                const rocsparse_int*      bsr_col_ind_A,
                rocsparse_int             block_dim,
                const T*                  B,
                rocsparse_int             ldb,
                const T*                  beta,
                T*                        C,
                rocsparse_int             ldc)
{
    rocsparse_index_base base = rocsparse_get_mat_index_base(descr);

    if(transA != rocsparse_operation_none)
    {
        return;
    }

    if(transB != rocsparse_operation_none && transB != rocsparse_operation_transpose)
    {
        return;
    }

    rocsparse_int M = Mb * block_dim;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; i++)
    {
        rocsparse_int local_row = i % block_dim;

        rocsparse_int row_begin = bsr_row_ptr_A[i / block_dim] - base;
        rocsparse_int row_end   = bsr_row_ptr_A[i / block_dim + 1] - base;

        for(rocsparse_int j = 0; j < N; j++)
        {
            rocsparse_int idx_C = i + j * ldc;

            T sum = static_cast<T>(0);

            for(rocsparse_int s = row_begin; s < row_end; s++)
            {
                for(rocsparse_int t = 0; t < block_dim; t++)
                {
                    rocsparse_int idx_A
                        = (dir == rocsparse_direction_row)
                              ? block_dim * block_dim * s + block_dim * local_row + t
                              : block_dim * block_dim * s + block_dim * t + local_row;
                    rocsparse_int idx_B
                        = (transB == rocsparse_operation_none)
                              ? j * ldb + block_dim * (bsr_col_ind_A[s] - base) + t
                              : (block_dim * (bsr_col_ind_A[s] - base) + t) * ldb + j;

                    sum = std::fma(bsr_val_A[idx_A], B[idx_B], sum);
                }
            }

            if(*beta == static_cast<T>(0))
            {
                C[idx_C] = *alpha * sum;
            }
            else
            {
                C[idx_C] = std::fma(*beta, C[idx_C], *alpha * sum);
            }
        }
    }
}

template <typename T>
void host_gebsrmm(rocsparse_handle          handle,
                  rocsparse_direction       dir,
                  rocsparse_operation       transA,
                  rocsparse_operation       transB,
                  rocsparse_int             Mb,
                  rocsparse_int             N,
                  rocsparse_int             Kb,
                  rocsparse_int             nnzb,
                  const T*                  alpha,
                  const rocsparse_mat_descr descr,
                  const T*                  bsr_val_A,
                  const rocsparse_int*      bsr_row_ptr_A,
                  const rocsparse_int*      bsr_col_ind_A,
                  rocsparse_int             row_block_dim,
                  rocsparse_int             col_block_dim,
                  const T*                  B,
                  rocsparse_int             ldb,
                  const T*                  beta,
                  T*                        C,
                  rocsparse_int             ldc)
{
    if(transA != rocsparse_operation_none)
    {
        return;
    }

    if(transB != rocsparse_operation_none && transB != rocsparse_operation_transpose)
    {
        return;
    }
    rocsparse_index_base base = rocsparse_get_mat_index_base(descr);

    rocsparse_int M = Mb * row_block_dim;

    const rocsparse_int rowXcol_block_dim = row_block_dim * col_block_dim;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int row_idx = 0; row_idx < M; ++row_idx)
    {
        const rocsparse_int row_block_idx = row_idx / row_block_dim,
                            row_local_idx = row_idx % row_block_dim;

        const rocsparse_int start = bsr_row_ptr_A[row_block_idx] - base,
                            bound = bsr_row_ptr_A[row_block_idx + 1] - base;

        for(rocsparse_int col_idx = 0; col_idx < N; ++col_idx)
        {
            const rocsparse_int idx_C = ldc * col_idx + row_idx;

            T sum = static_cast<T>(0);

            for(rocsparse_int at = start; at < bound; ++at)
            {
                for(rocsparse_int col_local_idx = 0; col_local_idx < col_block_dim; ++col_local_idx)
                {
                    const rocsparse_int idx_A
                        = (dir == rocsparse_direction_row)
                              ? rowXcol_block_dim * at + col_block_dim * row_local_idx
                                    + col_local_idx
                              : rowXcol_block_dim * at + row_block_dim * col_local_idx
                                    + row_local_idx;

                    const rocsparse_int idx_B
                        = (transB == rocsparse_operation_none)
                              ? col_idx * ldb + col_block_dim * (bsr_col_ind_A[at] - base)
                                    + col_local_idx
                              : (col_block_dim * (bsr_col_ind_A[at] - base) + col_local_idx) * ldb
                                    + col_idx;

                    sum = std::fma(bsr_val_A[idx_A], B[idx_B], sum);
                }
            }

            if(*beta == static_cast<T>(0))
            {
                C[idx_C] = *alpha * sum;
            }
            else
            {
                C[idx_C] = std::fma(*beta, C[idx_C], *alpha * sum);
            }
        }
    }
}

template <typename T, typename I, typename J>
void host_csrmm(J                    M,
                J                    N,
                J                    K,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             csr_row_ptr_A,
                const J*             csr_col_ind_A,
                const T*             csr_val_A,
                const T*             B,
                J                    ldb,
                T                    beta,
                T*                   C,
                J                    ldc,
                rocsparse_order      order,
                rocsparse_index_base base)
{
    if(transA == rocsparse_operation_none)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr_A[i] - base;
            I row_end   = csr_row_ptr_A[i + 1] - base;

            for(J j = 0; j < N; ++j)
            {
                T sum = static_cast<T>(0);

                for(I k = row_begin; k < row_end; ++k)
                {
                    J idx_B = 0;
                    if((transB == rocsparse_operation_none && order == rocsparse_order_column)
                       || (transB == rocsparse_operation_transpose && order == rocsparse_order_row)
                       || (transB == rocsparse_operation_conjugate_transpose
                           && order == rocsparse_order_row))
                    {
                        idx_B = (csr_col_ind_A[k] - base + j * ldb);
                    }
                    else
                    {
                        idx_B = (j + (csr_col_ind_A[k] - base) * ldb);
                    }

                    if(transB == rocsparse_operation_conjugate_transpose)
                    {
                        sum = std::fma(csr_val_A[k], rocsparse_conj(B[idx_B]), sum);
                    }
                    else
                    {
                        sum = std::fma(csr_val_A[k], B[idx_B], sum);
                    }
                }

                J idx_C = (order == rocsparse_order_column) ? i + j * ldc : i * ldc + j;

                if(beta == static_cast<T>(0))
                {
                    C[idx_C] = alpha * sum;
                }
                else
                {
                    C[idx_C] = std::fma(beta, C[idx_C], alpha * sum);
                }
            }
        }
    }
    else
    {
        // scale C by beta
        for(J i = 0; i < K; i++)
        {
            for(J j = 0; j < N; ++j)
            {
                J idx_C  = (order == rocsparse_order_column) ? i + j * ldc : i * ldc + j;
                C[idx_C] = beta * C[idx_C];
            }
        }

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr_A[i] - base;
            I row_end   = csr_row_ptr_A[i + 1] - base;

            for(J j = 0; j < N; ++j)
            {
                for(I k = row_begin; k < row_end; ++k)
                {
                    J col = csr_col_ind_A[k] - base;
                    T val = csr_val_A[k];
                    if(transA == rocsparse_operation_conjugate_transpose)
                    {
                        val = rocsparse_conj(val);
                    }

                    J idx_B = 0;

                    if((transB == rocsparse_operation_none && order == rocsparse_order_column)
                       || (transB == rocsparse_operation_transpose && order == rocsparse_order_row)
                       || (transB == rocsparse_operation_conjugate_transpose
                           && order == rocsparse_order_row))
                    {
                        idx_B = (i + j * ldb);
                    }
                    else
                    {
                        idx_B = (j + i * ldb);
                    }

                    J idx_C = (order == rocsparse_order_column) ? col + j * ldc : col * ldc + j;
                    if(transB == rocsparse_operation_conjugate_transpose)
                    {
                        C[idx_C] += alpha * val * rocsparse_conj(B[idx_B]);
                    }
                    else
                    {
                        C[idx_C] += alpha * val * B[idx_B];
                    }
                }
            }
        }
    }
}

template <typename T, typename I>
void host_coomm(rocsparse_spmm_alg   alg,
                I                    M,
                I                    N,
                I                    nnz,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             coo_row_ind_A,
                const I*             coo_col_ind_A,
                const T*             coo_val_A,
                const T*             B,
                I                    ldb,
                T                    beta,
                T*                   C,
                I                    ldc,
                rocsparse_order      order,
                rocsparse_index_base base)
{
    for(I j = 0; j < N; j++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            I idx_C = order == rocsparse_order_column ? i + j * ldc : i * ldc + j;
            C[idx_C] *= beta;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I j = 0; j < N; j++)
    {
        for(I i = 0; i < nnz; ++i)
        {
            I row = coo_row_ind_A[i] - base;
            I col = coo_col_ind_A[i] - base;
            T val = alpha * coo_val_A[i];

            I idx_C = order == rocsparse_order_column ? row + j * ldc : row * ldc + j;

            I idx_B = 0;
            if((transB == rocsparse_operation_none && order == rocsparse_order_column)
               || (transB != rocsparse_operation_none && order != rocsparse_order_column))
            {
                idx_B = (col + j * ldb);
            }
            else
            {
                idx_B = (j + col * ldb);
            }

            if(transB == rocsparse_operation_conjugate_transpose)
            {
                C[idx_C] = std::fma(val, rocsparse_conj(B[idx_B]), C[idx_C]);
            }
            else
            {
                C[idx_C] = std::fma(val, B[idx_B], C[idx_C]);
            }
        }
    }
}

template <typename I, typename J, typename T>
static inline void host_lssolve(J                    M,
                                J                    nrhs,
                                rocsparse_operation  transB,
                                T                    alpha,
                                const I*             csr_row_ptr,
                                const J*             csr_col_ind,
                                const T*             csr_val,
                                T*                   B,
                                J                    ldb,
                                rocsparse_diag_type  diag_type,
                                rocsparse_index_base base,
                                J*                   struct_pivot,
                                J*                   numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(J i = 0; i < nrhs; ++i)
    {
        // Process lower triangular part
        for(J row = 0; row < M; ++row)
        {
            J idx_B = (transB == rocsparse_operation_none) ? i * ldb + row : row * ldb + i;

            T sum = static_cast<T>(0);
            if(transB == rocsparse_operation_conjugate_transpose)
            {
                sum = alpha * rocsparse_conj(B[idx_B]);
            }
            else
            {
                sum = alpha * B[idx_B];
            }

            I diag      = -1;
            I row_begin = csr_row_ptr[row] - base;
            I row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(I j = row_begin; j < row_end; ++j)
            {
                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

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
                J idx = (transB == rocsparse_operation_none) ? i * ldb + local_col
                                                             : local_col * ldb + i;

                if(transB == rocsparse_operation_conjugate_transpose)
                {
                    sum = std::fma(-local_val, rocsparse_conj(B[idx]), sum);
                }
                else
                {
                    sum = std::fma(-local_val, B[idx], sum);
                }
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

template <typename I, typename J, typename T>
static inline void host_ussolve(J                    M,
                                J                    nrhs,
                                rocsparse_operation  transB,
                                T                    alpha,
                                const I*             csr_row_ptr,
                                const J*             csr_col_ind,
                                const T*             csr_val,
                                T*                   B,
                                J                    ldb,
                                rocsparse_diag_type  diag_type,
                                rocsparse_index_base base,
                                J*                   struct_pivot,
                                J*                   numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(J i = 0; i < nrhs; ++i)
    {
        // Process upper triangular part
        for(J row = M - 1; row >= 0; --row)
        {
            J idx_B = (transB == rocsparse_operation_none) ? i * ldb + row : row * ldb + i;

            T sum = static_cast<T>(0);
            if(transB == rocsparse_operation_conjugate_transpose)
            {
                sum = alpha * rocsparse_conj(B[idx_B]);
            }
            else
            {
                sum = alpha * B[idx_B];
            }

            I diag      = -1;
            I row_begin = csr_row_ptr[row] - base;
            I row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(I j = row_end - 1; j >= row_begin; --j)
            {
                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

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
                J idx = (transB == rocsparse_operation_none) ? i * ldb + local_col
                                                             : local_col * ldb + i;

                if(transB == rocsparse_operation_conjugate_transpose)
                {
                    sum = std::fma(-local_val, rocsparse_conj(B[idx]), sum);
                }
                else
                {
                    sum = std::fma(-local_val, B[idx], sum);
                }
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

template <typename I, typename J, typename T>
void host_csrsm(J                    M,
                J                    nrhs,
                I                    nnz,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             csr_row_ptr,
                const J*             csr_col_ind,
                const T*             csr_val,
                T*                   B,
                J                    ldb,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                J*                   struct_pivot,
                J*                   numeric_pivot)
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
    else if(transA == rocsparse_operation_transpose
            || transA == rocsparse_operation_conjugate_transpose)
    {
        // Transpose matrix
        std::vector<I> csrt_row_ptr(M + 1);
        std::vector<J> csrt_col_ind(nnz);
        std::vector<T> csrt_val(nnz);

        host_csr_to_csc<I, J, T>(M,
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

        if(transA == rocsparse_operation_conjugate_transpose)
        {
            for(size_t i = 0; i < csrt_val.size(); i++)
            {
                csrt_val[i] = rocsparse_conj(csrt_val[i]);
            }
        }

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_ussolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csrt_row_ptr.data(),
                         csrt_col_ind.data(),
                         csrt_val.data(),
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
                         csrt_row_ptr.data(),
                         csrt_col_ind.data(),
                         csrt_val.data(),
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

template <typename I, typename T>
void host_coosm(I                    M,
                I                    nrhs,
                I                    nnz,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             coo_row_ind,
                const I*             coo_col_ind,
                const T*             coo_val,
                T*                   B,
                I                    ldb,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                I*                   struct_pivot,
                I*                   numeric_pivot)
{
    std::vector<I> csr_row_ptr(M + 1);

    host_coo_to_csr(M, nnz, coo_row_ind, csr_row_ptr, base);

    host_csrsm(M,
               nrhs,
               nnz,
               transA,
               transB,
               alpha,
               csr_row_ptr.data(),
               coo_col_ind,
               coo_val,
               B,
               ldb,
               diag_type,
               fill_mode,
               base,
               struct_pivot,
               numeric_pivot);
}

template <typename T>
void host_bsrsm(rocsparse_int       mb,
                rocsparse_int       nrhs,
                rocsparse_int       nnzb,
                rocsparse_direction dir,
                rocsparse_operation transA,
                rocsparse_operation transX,
                T                   alpha,
                const rocsparse_int* __restrict bsr_row_ptr,
                const rocsparse_int* __restrict bsr_col_ind,
                const T* __restrict bsr_val,
                rocsparse_int        bsr_dim,
                const T*             B,
                rocsparse_int        ldb,
                T*                   X,
                rocsparse_int        ldx,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                rocsparse_int*       struct_pivot,
                rocsparse_int*       numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    if(transA == rocsparse_operation_none)
    {
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_bsr_lsolve(dir,
                            transX,
                            mb,
                            nrhs,
                            alpha,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_val,
                            bsr_dim,
                            B,
                            ldb,
                            X,
                            ldx,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_bsr_usolve(dir,
                            transX,
                            mb,
                            nrhs,
                            alpha,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_val,
                            bsr_dim,
                            B,
                            ldb,
                            X,
                            ldx,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }
    else if(transA == rocsparse_operation_transpose)
    {
        // Transpose matrix
        std::vector<rocsparse_int> bsrt_row_ptr(mb + 1);
        std::vector<rocsparse_int> bsrt_col_ind(nnzb);
        std::vector<T>             bsrt_val(nnzb * bsr_dim * bsr_dim);

        host_bsr_to_bsc(mb,
                        mb,
                        nnzb,
                        bsr_dim,
                        bsr_row_ptr,
                        bsr_col_ind,
                        bsr_val,
                        bsrt_col_ind,
                        bsrt_row_ptr,
                        bsrt_val,
                        base,
                        base);

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_bsr_usolve(dir,
                            transX,
                            mb,
                            nrhs,
                            alpha,
                            bsrt_row_ptr.data(),
                            bsrt_col_ind.data(),
                            bsrt_val.data(),
                            bsr_dim,
                            B,
                            ldb,
                            X,
                            ldx,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_bsr_lsolve(dir,
                            transX,
                            mb,
                            nrhs,
                            alpha,
                            bsrt_row_ptr.data(),
                            bsrt_col_ind.data(),
                            bsrt_val.data(),
                            bsr_dim,
                            B,
                            ldb,
                            X,
                            ldx,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;
}

template <typename I, typename T>
void host_gemvi(I                    M,
                I                    N,
                T                    alpha,
                const T*             A,
                I                    lda,
                I                    nnz,
                const T*             x_val,
                const I*             x_ind,
                T                    beta,
                T*                   y,
                rocsparse_index_base base)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I i = 0; i < M; ++i)
    {
        T sum = static_cast<T>(0);

        for(I j = 0; j < nnz; ++j)
        {
            sum = std::fma(x_val[j], A[x_ind[j] * lda + i], sum);
        }

        y[i] = std::fma(alpha, sum, beta * y[i]);
    }
}

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
                rocsparse_index_base base)
{
    if(transB == rocsparse_operation_transpose)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            for(rocsparse_int j = 0; j < N; ++j)
            {
                T sum = static_cast<T>(0);

                rocsparse_int row_begin = csr_row_ptr[j] - base;
                rocsparse_int row_end   = csr_row_ptr[j + 1] - base;

                for(rocsparse_int k = row_begin; k < row_end; ++k)
                {
                    rocsparse_int col_B = csr_col_ind[k] - base;
                    T             val_B = csr_val[k];
                    T             val_A = A[col_B * lda + i];

                    sum = std::fma(val_A, val_B, sum);
                }

                C[j * ldc + i] = std::fma(beta, C[j * ldc + i], alpha * sum);
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
                      rocsparse_index_base              base_C)
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

        // Loop over rows
        for(rocsparse_int i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            rocsparse_int row_begin_A = csr_row_ptr_A[i] - base_A;
            rocsparse_int row_end_A   = csr_row_ptr_A[i + 1] - base_A;

            // Loop over columns of A
            for(rocsparse_int j = row_begin_A; j < row_end_A; ++j)
            {
                rocsparse_int col_A = csr_col_ind_A[j] - base_A;

                nnz[col_A] = i;
                ++csr_row_ptr_C[i + 1];
            }

            rocsparse_int row_begin_B = csr_row_ptr_B[i] - base_B;
            rocsparse_int row_end_B   = csr_row_ptr_B[i + 1] - base_B;

            // Loop over columns of B
            for(rocsparse_int j = row_begin_B; j < row_end_B; ++j)
            {
                rocsparse_int col_B = csr_col_ind_B[j] - base_B;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    ++csr_row_ptr_C[i + 1];
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
                  rocsparse_index_base              base_C)
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

        // Loop over rows
        for(rocsparse_int i = chunk_begin; i < chunk_end; ++i)
        {
            rocsparse_int row_begin_C = csr_row_ptr_C[i] - base_C;
            rocsparse_int row_end_C   = row_begin_C;

            rocsparse_int row_begin_A = csr_row_ptr_A[i] - base_A;
            rocsparse_int row_end_A   = csr_row_ptr_A[i + 1] - base_A;

            // Copy A into C
            for(rocsparse_int j = row_begin_A; j < row_end_A; ++j)
            {
                // Current column of A
                rocsparse_int col_A = csr_col_ind_A[j] - base_A;

                // Current value of A
                T val_A = alpha * csr_val_A[j];

                nnz[col_A] = row_end_C;

                csr_col_ind_C[row_end_C] = col_A + base_C;
                csr_val_C[row_end_C]     = val_A;
                ++row_end_C;
            }

            rocsparse_int row_begin_B = csr_row_ptr_B[i] - base_B;
            rocsparse_int row_end_B   = csr_row_ptr_B[i + 1] - base_B;

            // Loop over columns of B
            for(rocsparse_int j = row_begin_B; j < row_end_B; ++j)
            {
                // Current column of B
                rocsparse_int col_B = csr_col_ind_B[j] - base_B;

                // Current value of B
                T val_B = beta * csr_val_B[j];

                // Check if a new nnz is generated or if the value is added
                if(nnz[col_B] < row_begin_C)
                {
                    nnz[col_B] = row_end_C;

                    csr_col_ind_C[row_end_C] = col_B + base_C;
                    csr_val_C[row_end_C]     = val_B;
                    ++row_end_C;
                }
                else
                {
                    csr_val_C[nnz[col_B]] += val_B;
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

template <typename T, typename I, typename J>
void host_csrgemm_nnz(J                    M,
                      J                    N,
                      J                    K,
                      const T*             alpha,
                      const I*             csr_row_ptr_A,
                      const J*             csr_col_ind_A,
                      const I*             csr_row_ptr_B,
                      const J*             csr_col_ind_B,
                      const T*             beta,
                      const I*             csr_row_ptr_D,
                      const J*             csr_col_ind_D,
                      I*                   csr_row_ptr_C,
                      I*                   nnz_C,
                      rocsparse_index_base base_A,
                      rocsparse_index_base base_B,
                      rocsparse_index_base base_C,
                      rocsparse_index_base base_D)
{
    if(M == 0 || N == 0)
    {
        *nnz_C = 0;
        if(M > 0)
        {
            for(J i = 0; i <= M; ++i)
            {
                csr_row_ptr_C[i] = base_C;
            }
        }
        return;
    }
    else if(alpha && !beta && (K == 0))
    {
        *nnz_C = 0;
        if(M > 0)
        {
            for(J i = 0; i <= M; ++i)
            {
                csr_row_ptr_C[i] = base_C;
            }
        }
        return;
    }
    else if(!alpha && !beta)
    {
        *nnz_C = 0;
        if(M > 0)
        {
            for(J i = 0; i <= M; ++i)
            {
                csr_row_ptr_C[i] = base_C;
            }
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<J> nnz(N, -1);

        int nthreads = 1;
        int tid      = 0;

#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        tid      = omp_get_thread_num();
#endif

        J rows_per_thread = (M + nthreads - 1) / nthreads;
        J chunk_begin     = rows_per_thread * tid;
        J chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Index base
        csr_row_ptr_C[0] = base_C;

        // Loop over rows of A
        for(J i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            if(alpha)
            {
                I row_begin_A = csr_row_ptr_A[i] - base_A;
                I row_end_A   = csr_row_ptr_A[i + 1] - base_A;

                // Loop over columns of A
                for(I j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    J col_A = csr_col_ind_A[j] - base_A;

                    I row_begin_B = csr_row_ptr_B[col_A] - base_B;
                    I row_end_B   = csr_row_ptr_B[col_A + 1] - base_B;

                    // Loop over columns of B in row col_A
                    for(I k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        J col_B = csr_col_ind_B[k] - base_B;

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
                I row_begin_D = csr_row_ptr_D[i] - base_D;
                I row_end_D   = csr_row_ptr_D[i + 1] - base_D;

                // Loop over columns of D
                for(I j = row_begin_D; j < row_end_D; ++j)
                {
                    J col_D = csr_col_ind_D[j] - base_D;

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
    for(J i = 0; i < M; ++i)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    *nnz_C = csr_row_ptr_C[M] - base_C;
}

template <typename T, typename I, typename J>
void host_csrgemm(J                    M,
                  J                    N,
                  J                    L,
                  const T*             alpha,
                  const I*             csr_row_ptr_A,
                  const J*             csr_col_ind_A,
                  const T*             csr_val_A,
                  const I*             csr_row_ptr_B,
                  const J*             csr_col_ind_B,
                  const T*             csr_val_B,
                  const T*             beta,
                  const I*             csr_row_ptr_D,
                  const J*             csr_col_ind_D,
                  const T*             csr_val_D,
                  const I*             csr_row_ptr_C,
                  J*                   csr_col_ind_C,
                  T*                   csr_val_C,
                  rocsparse_index_base base_A,
                  rocsparse_index_base base_B,
                  rocsparse_index_base base_C,
                  rocsparse_index_base base_D)
{
    if(M == 0 || N == 0)
    {
        return;
    }
    else if(alpha && !beta && (L == 0))
    {
        return;
    }
    else if(!alpha && !beta)
    {
        return;
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<I> nnz(N, -1);

        int nthreads = 1;
        int tid      = 0;

#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        tid      = omp_get_thread_num();
#endif

        J rows_per_thread = (M + nthreads - 1) / nthreads;
        J chunk_begin     = rows_per_thread * tid;
        J chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Loop over rows of A
        for(J i = chunk_begin; i < chunk_end; ++i)
        {
            I row_begin_C = csr_row_ptr_C[i] - base_C;
            I row_end_C   = row_begin_C;

            if(alpha)
            {
                I row_begin_A = csr_row_ptr_A[i] - base_A;
                I row_end_A   = csr_row_ptr_A[i + 1] - base_A;

                // Loop over columns of A
                for(I j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    J col_A = csr_col_ind_A[j] - base_A;
                    // Current value of A
                    T val_A = *alpha * csr_val_A[j];

                    I row_begin_B = csr_row_ptr_B[col_A] - base_B;
                    I row_end_B   = csr_row_ptr_B[col_A + 1] - base_B;

                    // Loop over columns of B in row col_A
                    for(I k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        J col_B = csr_col_ind_B[k] - base_B;
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
                I row_begin_D = csr_row_ptr_D[i] - base_D;
                I row_end_D   = csr_row_ptr_D[i + 1] - base_D;

                // Loop over columns of D
                for(I j = row_begin_D; j < row_end_D; ++j)
                {
                    // Current column of D
                    J col_D = csr_col_ind_D[j] - base_D;
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

    I nnz = csr_row_ptr_C[M] - base_C;

    std::vector<J> col(nnz);
    std::vector<T> val(nnz);

    memcpy(col.data(), csr_col_ind_C, sizeof(J) * nnz);
    memcpy(val.data(), csr_val_C, sizeof(T) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr_C[i] - base_C;
        I row_end   = csr_row_ptr_C[i + 1] - base_C;
        J row_nnz   = row_end - row_begin;

        std::vector<J> perm(row_nnz);
        for(J j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        J* col_entry = &col[row_begin];
        T* val_entry = &val[row_begin];

        std::sort(perm.begin(), perm.end(), [&](const J& a, const J& b) {
            return col_entry[a] <= col_entry[b];
        });

        for(J j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

template <typename T, typename I, typename J>
void rocsparse_host<T, I, J>::cooddmm(rocsparse_operation  transA,
                                      rocsparse_operation  transB,
                                      rocsparse_order      orderA,
                                      rocsparse_order      orderB,
                                      J                    M,
                                      J                    N,
                                      J                    K,
                                      I                    nnz,
                                      const T*             alpha,
                                      const T*             A,
                                      J                    lda,
                                      const T*             B,
                                      J                    ldb,
                                      const T*             beta,
                                      const I*             coo_row_ind_C,
                                      const I*             coo_col_ind_C,
                                      T*                   coo_val_C,
                                      rocsparse_index_base base_C)
{

    const T a = *alpha;
    const T b = *beta;

    const J incx = (orderA == rocsparse_order_column)
                       ? ((transA == rocsparse_operation_none) ? lda : 1)
                       : ((transA == rocsparse_operation_none) ? 1 : lda);
    const J incy = (orderB == rocsparse_order_column)
                       ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                       : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I s = 0; s < nnz; ++s)
    {
        const I i = coo_row_ind_C[s] - base_C;
        const I j = coo_col_ind_C[s] - base_C;

        const T* x = (orderA == rocsparse_order_column)
                         ? ((transA == rocsparse_operation_none) ? (A + i) : (A + lda * i))
                         : ((transA == rocsparse_operation_none) ? (A + lda * i) : (A + i));

        const T* y = (orderB == rocsparse_order_column)
                         ? ((transB == rocsparse_operation_none) ? (B + ldb * j) : (B + j))
                         : ((transB == rocsparse_operation_none) ? (B + j) : (B + ldb * j));

        T sum = static_cast<T>(0);
        for(J k = 0; k < K; ++k)
        {
            sum += x[incx * k] * y[incy * k];
        }
        coo_val_C[s] = coo_val_C[s] * b + a * sum;
    }
}

template <typename T, typename I, typename J>
void rocsparse_host<T, I, J>::cooaosddmm(rocsparse_operation  transA,
                                         rocsparse_operation  transB,
                                         rocsparse_order      orderA,
                                         rocsparse_order      orderB,
                                         J                    M,
                                         J                    N,
                                         J                    K,
                                         I                    nnz,
                                         const T*             alpha,
                                         const T*             A,
                                         J                    lda,
                                         const T*             B,
                                         J                    ldb,
                                         const T*             beta,
                                         const I*             coo_row_ind_C,
                                         const I*             coo_col_ind_C,
                                         T*                   coo_val_C,
                                         rocsparse_index_base base_C)
{

    const T a = *alpha;
    const T b = *beta;

    const J incx = (orderA == rocsparse_order_column)
                       ? ((transA == rocsparse_operation_none) ? lda : 1)
                       : ((transA == rocsparse_operation_none) ? 1 : lda);
    const J incy = (orderB == rocsparse_order_column)
                       ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                       : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I s = 0; s < nnz; ++s)
    {
        const I i = coo_row_ind_C[2 * s] - base_C;
        const I j = coo_col_ind_C[2 * s] - base_C;

        const T* x = (orderA == rocsparse_order_column)
                         ? ((transA == rocsparse_operation_none) ? (A + i) : (A + lda * i))
                         : ((transA == rocsparse_operation_none) ? (A + lda * i) : (A + i));

        const T* y = (orderB == rocsparse_order_column)
                         ? ((transB == rocsparse_operation_none) ? (B + ldb * j) : (B + j))
                         : ((transB == rocsparse_operation_none) ? (B + j) : (B + ldb * j));

        T sum = static_cast<T>(0);
        for(J k = 0; k < K; ++k)
        {
            sum += x[incx * k] * y[incy * k];
        }
        coo_val_C[s] = coo_val_C[s] * b + a * sum;
    }
}

template <typename T, typename I, typename J>
void rocsparse_host<T, I, J>::csrddmm(rocsparse_operation  transA,
                                      rocsparse_operation  transB,
                                      rocsparse_order      orderA,
                                      rocsparse_order      orderB,
                                      J                    M,
                                      J                    N,
                                      J                    K,
                                      I                    nnz,
                                      const T*             alpha,
                                      const T*             A,
                                      J                    lda,
                                      const T*             B,
                                      J                    ldb,
                                      const T*             beta,
                                      const I*             csr_row_ptr_C,
                                      const J*             csr_col_ind_C,
                                      T*                   csr_val_C,
                                      rocsparse_index_base base_C)
{
    const T a = *alpha;
    const T b = *beta;

    const J incx = (orderA == rocsparse_order_column)
                       ? ((transA == rocsparse_operation_none) ? lda : 1)
                       : ((transA == rocsparse_operation_none) ? 1 : lda);

    const J incy = (orderB == rocsparse_order_column)
                       ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                       : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        for(I at = csr_row_ptr_C[i] - base_C; at < csr_row_ptr_C[i + 1] - base_C; ++at)
        {
            J j = csr_col_ind_C[at] - base_C;

            const T* x = (orderA == rocsparse_order_column)
                             ? ((transA == rocsparse_operation_none) ? (A + i) : (A + lda * i))
                             : ((transA == rocsparse_operation_none) ? (A + lda * i) : (A + i));
            const T* y = (orderB == rocsparse_order_column)
                             ? ((transB == rocsparse_operation_none) ? (B + ldb * j) : (B + j))
                             : ((transB == rocsparse_operation_none) ? (B + j) : (B + ldb * j));

            T sum = static_cast<T>(0);
            for(J k = 0; k < K; ++k)
            {
                sum += x[incx * k] * y[incy * k];
            }
            csr_val_C[at] = csr_val_C[at] * b + a * sum;
        }
    }
}

template <typename T, typename I, typename J>
void rocsparse_host<T, I, J>::ellddmm(rocsparse_operation  transA,
                                      rocsparse_operation  transB,
                                      rocsparse_order      orderA,
                                      rocsparse_order      orderB,
                                      J                    M,
                                      J                    N,
                                      J                    K,
                                      I                    nnz,
                                      const T*             alpha,
                                      const T*             A,
                                      J                    lda,
                                      const T*             B,
                                      J                    ldb,
                                      const T*             beta,
                                      const J              ell_width,
                                      const I*             ell_ind_C,
                                      T*                   ell_val_C,
                                      rocsparse_index_base ell_base)
{
    const T a = *alpha;
    const T b = *beta;

    const J incx = (orderA == rocsparse_order_column)
                       ? ((transA == rocsparse_operation_none) ? lda : 1)
                       : ((transA == rocsparse_operation_none) ? 1 : lda);

    const J incy = (orderB == rocsparse_order_column)
                       ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                       : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        for(J p = 0; p < ell_width; ++p)
        {
            I at = p * M + i;
            J j  = ell_ind_C[at] - ell_base;
            if(j >= 0 && j < N)
            {
                const T* x = (orderA == rocsparse_order_column)
                                 ? ((transA == rocsparse_operation_none) ? (A + i) : (A + lda * i))
                                 : ((transA == rocsparse_operation_none) ? (A + lda * i) : (A + i));
                const T* y = (orderB == rocsparse_order_column)
                                 ? ((transB == rocsparse_operation_none) ? (B + ldb * j) : (B + j))
                                 : ((transB == rocsparse_operation_none) ? (B + j) : (B + ldb * j));

                T sum = static_cast<T>(0);
                for(J k = 0; k < K; ++k)
                {
                    sum += x[incx * k] * y[incy * k];
                }
                ell_val_C[at] = ell_val_C[at] * b + a * sum;
            }
        }
    }
}

template <typename T, typename I, typename J>
void rocsparse_host<T, I, J>::cscddmm(rocsparse_operation  transA,
                                      rocsparse_operation  transB,
                                      rocsparse_order      orderA,
                                      rocsparse_order      orderB,
                                      J                    M,
                                      J                    N,
                                      J                    K,
                                      I                    nnz,
                                      const T*             alpha,
                                      const T*             A,
                                      J                    lda,
                                      const T*             B,
                                      J                    ldb,
                                      const T*             beta,
                                      const I*             csr_ptr_C,
                                      const J*             csr_ind_C,
                                      T*                   csr_val_C,
                                      rocsparse_index_base base_C)
{
    const T a = *alpha;
    const T b = *beta;

    const J incx = (orderA == rocsparse_order_column)
                       ? ((transA == rocsparse_operation_none) ? lda : 1)
                       : ((transA == rocsparse_operation_none) ? 1 : lda);

    const J incy = (orderB == rocsparse_order_column)
                       ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                       : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif

    for(J j = 0; j < N; ++j)
    {
        for(I at = csr_ptr_C[j] - base_C; at < csr_ptr_C[j + 1] - base_C; ++at)
        {
            J        i = csr_ind_C[at] - base_C;
            const T* x = (orderA == rocsparse_order_column)
                             ? ((transA == rocsparse_operation_none) ? (A + i) : (A + lda * i))
                             : ((transA == rocsparse_operation_none) ? (A + lda * i) : (A + i));
            const T* y = (orderB == rocsparse_order_column)
                             ? ((transB == rocsparse_operation_none) ? (B + ldb * j) : (B + j))
                             : ((transB == rocsparse_operation_none) ? (B + j) : (B + ldb * j));

            T sum = static_cast<T>(0);
            for(J k = 0; k < K; ++k)
            {
                sum += x[incx * k] * y[incy * k];
            }
            csr_val_C[at] = csr_val_C[at] * b + a * sum;
        }
    }
}

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
                 rocsparse_int*                    numeric_pivot)

{
    rocsparse_int M = Mb * block_dim;

    // Initialize pivot
    *struct_pivot  = -1;
    *numeric_pivot = -1;

    // pointer of upper part of each row
    std::vector<rocsparse_int> diag_block_offset(Mb);
    std::vector<rocsparse_int> diag_offset(M, -1);
    std::vector<rocsparse_int> nnz_entries(M, -1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < Mb; i++)
    {
        rocsparse_int row_begin = bsr_row_ptr[i] - base;
        rocsparse_int row_end   = bsr_row_ptr[i + 1] - base;

        for(rocsparse_int j = row_begin; j < row_end; j++)
        {
            if(bsr_col_ind[j] - base == i)
            {
                diag_block_offset[i] = j;
                break;
            }
        }
    }

    for(rocsparse_int i = 0; i < M; i++)
    {
        rocsparse_int local_row = i % block_dim;

        rocsparse_int row_begin = bsr_row_ptr[i / block_dim] - base;
        rocsparse_int row_end   = bsr_row_ptr[i / block_dim + 1] - base;

        for(rocsparse_int j = row_begin; j < row_end; j++)
        {
            rocsparse_int block_col_j = bsr_col_ind[j] - base;

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                if(direction == rocsparse_direction_row)
                {
                    nnz_entries[block_dim * block_col_j + k]
                        = block_dim * block_dim * j + block_dim * local_row + k;
                }
                else
                {
                    nnz_entries[block_dim * block_col_j + k]
                        = block_dim * block_dim * j + block_dim * k + local_row;
                }
            }
        }

        T             sum            = static_cast<T>(0);
        rocsparse_int diag_val_index = -1;

        bool has_diag         = false;
        bool break_outer_loop = false;

        for(rocsparse_int j = row_begin; j < row_end; j++)
        {
            rocsparse_int block_col_j = bsr_col_ind[j] - base;

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                rocsparse_int col_j = block_dim * block_col_j + k;

                // Mark diagonal and skip row
                if(col_j == i)
                {
                    diag_val_index = block_dim * block_dim * j + block_dim * k + k;

                    has_diag         = true;
                    break_outer_loop = true;
                    break;
                }

                // Skip upper triangular
                if(col_j > i)
                {
                    break_outer_loop = true;
                    break;
                }

                T val_j = static_cast<T>(0);
                if(direction == rocsparse_direction_row)
                {
                    val_j = bsr_val[block_dim * block_dim * j + block_dim * local_row + k];
                }
                else
                {
                    val_j = bsr_val[block_dim * block_dim * j + block_dim * k + local_row];
                }

                rocsparse_int local_row_j = col_j % block_dim;

                rocsparse_int row_begin_j = bsr_row_ptr[col_j / block_dim] - base;
                rocsparse_int row_end_j   = diag_block_offset[col_j / block_dim];
                rocsparse_int row_diag_j  = diag_offset[col_j];

                T local_sum = static_cast<T>(0);
                T inv_diag  = row_diag_j != -1 ? bsr_val[row_diag_j] : static_cast<T>(0);

                // Check for numeric zero
                if(inv_diag == static_cast<T>(0))
                {
                    // Numerical non-invertible block diagonal
                    if(*numeric_pivot == -1)
                    {
                        *numeric_pivot = block_col_j + base;
                    }

                    *numeric_pivot = std::min(*numeric_pivot, block_col_j + base);

                    inv_diag = static_cast<T>(1);
                }

                inv_diag = static_cast<T>(1) / inv_diag;

                // loop over upper offset pointer and do linear combination for nnz entry
                for(rocsparse_int l = row_begin_j; l < row_end_j + 1; l++)
                {
                    rocsparse_int block_col_l = bsr_col_ind[l] - base;

                    for(rocsparse_int m = 0; m < block_dim; m++)
                    {
                        rocsparse_int idx = nnz_entries[block_dim * block_col_l + m];

                        if(idx != -1 && block_dim * block_col_l + m < col_j)
                        {
                            if(direction == rocsparse_direction_row)
                            {
                                local_sum = std::fma(bsr_val[block_dim * block_dim * l
                                                             + block_dim * local_row_j + m],
                                                     rocsparse_conj(bsr_val[idx]),
                                                     local_sum);
                            }
                            else
                            {
                                local_sum = std::fma(bsr_val[block_dim * block_dim * l
                                                             + block_dim * m + local_row_j],
                                                     rocsparse_conj(bsr_val[idx]),
                                                     local_sum);
                            }
                        }
                    }
                }

                val_j = (val_j - local_sum) * inv_diag;
                sum   = std::fma(val_j, rocsparse_conj(val_j), sum);

                if(direction == rocsparse_direction_row)
                {
                    bsr_val[block_dim * block_dim * j + block_dim * local_row + k] = val_j;
                }
                else
                {
                    bsr_val[block_dim * block_dim * j + block_dim * k + local_row] = val_j;
                }
            }

            if(break_outer_loop)
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural missing block diagonal
            if(*struct_pivot == -1)
            {
                *struct_pivot = i / block_dim + base;
            }
        }

        // Process diagonal entry
        if(has_diag)
        {
            T diag_entry            = std::sqrt(std::abs(bsr_val[diag_val_index] - sum));
            bsr_val[diag_val_index] = diag_entry;

            if(diag_entry == static_cast<T>(0))
            {
                // Numerical non-invertible block diagonal
                if(*numeric_pivot == -1)
                {
                    *numeric_pivot = i / block_dim + base;
                }

                *numeric_pivot = std::min(*numeric_pivot, i / block_dim + base);
            }

            // Store diagonal offset
            diag_offset[i] = diag_val_index;
        }

        for(rocsparse_int j = row_begin; j < row_end; j++)
        {
            rocsparse_int block_col_j = bsr_col_ind[j] - base;

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                if(direction == rocsparse_direction_row)
                {
                    nnz_entries[block_dim * block_col_j + k] = -1;
                }
                else
                {
                    nnz_entries[block_dim * block_col_j + k] = -1;
                }
            }
        }
    }
}

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
                  T                                 boost_val)

{
    // Initialize pivots
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    // Temporary vector to hold diagonal offset to access diagonal BSR block
    std::vector<rocsparse_int> diag_offset(mb);
    std::vector<rocsparse_int> nnz_entries(mb, -1);

    // First diagonal block is index 0
    diag_offset[0] = 0;

    // Loop over all BSR rows
    for(rocsparse_int i = 0; i < mb; ++i)
    {
        // Flag whether we have a diagonal block or not
        bool has_diag = false;

        // BSR column entry and exit point
        rocsparse_int row_begin = bsr_row_ptr[i] - base;
        rocsparse_int row_end   = bsr_row_ptr[i + 1] - base;

        rocsparse_int j;

        // Set up entry points for linear combination
        for(j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col_j = bsr_col_ind[j] - base;
            nnz_entries[col_j]  = j;
        }

        // Process lower diagonal BSR blocks (diagonal BSR block is excluded)
        for(j = row_begin; j < row_end; ++j)
        {
            // Column index of current BSR block
            rocsparse_int bsr_col = bsr_col_ind[j] - base;

            // If this is a diagonal block, set diagonal flag to true and skip
            // all upcoming blocks as we exceed the lower matrix part
            if(bsr_col == i)
            {
                has_diag = true;
                break;
            }

            // Skip all upper matrix blocks
            if(bsr_col > i)
            {
                break;
            }

            // Process all lower matrix BSR blocks

            // Obtain corresponding row entry and exit point that corresponds with the
            // current BSR column. Actually, we skip all lower matrix column indices,
            // therefore starting with the diagonal entry.
            rocsparse_int diag_j    = diag_offset[bsr_col];
            rocsparse_int row_end_j = bsr_row_ptr[bsr_col + 1] - base;

            // Loop through all rows within the BSR block
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                T diag = bsr_val[BSR_IND(diag_j, bi, bi, dir)];

                // Process all rows within the BSR block
                for(rocsparse_int bk = 0; bk < bsr_dim; ++bk)
                {
                    T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                    // Multiplication factor
                    bsr_val[BSR_IND(j, bk, bi, dir)] = val /= diag;

                    // Loop through columns of bk-th row and do linear combination
                    for(rocsparse_int bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = std::fma(-val,
                                       bsr_val[BSR_IND(diag_j, bi, bj, dir)],
                                       bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }

            // Loop over upper offset pointer and do linear combination for nnz entry
            for(rocsparse_int k = diag_j + 1; k < row_end_j; ++k)
            {
                rocsparse_int bsr_col_k = bsr_col_ind[k] - base;

                if(nnz_entries[bsr_col_k] != -1)
                {
                    rocsparse_int m = nnz_entries[bsr_col_k];

                    // Loop through all rows within the BSR block
                    for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
                    {
                        // Loop through columns of bi-th row and do linear combination
                        for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
                        {
                            T sum = static_cast<T>(0);

                            for(rocsparse_int bk = 0; bk < bsr_dim; ++bk)
                            {
                                sum = std::fma(bsr_val[BSR_IND(j, bi, bk, dir)],
                                               bsr_val[BSR_IND(k, bk, bj, dir)],
                                               sum);
                            }

                            bsr_val[BSR_IND(m, bi, bj, dir)] -= sum;
                        }
                    }
                }
            }
        }

        // Check for structural pivot
        if(!has_diag)
        {
            *struct_pivot = std::min(*struct_pivot, i + base);
            break;
        }

        // Process diagonal
        if(bsr_col_ind[j] - base == i)
        {
            // Loop through all rows within the BSR block
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                T diag = bsr_val[BSR_IND(j, bi, bi, dir)];

                if(boost)
                {
                    diag = (boost_tol >= std::abs(diag)) ? boost_val : diag;
                    bsr_val[BSR_IND(j, bi, bi, dir)] = diag;
                }
                else
                {
                    // Check for numeric pivot
                    if(diag == static_cast<T>(0))
                    {
                        *numeric_pivot = std::min(*numeric_pivot, bsr_col_ind[j]);
                        continue;
                    }
                }

                // Process all rows within the BSR block after bi-th row
                for(rocsparse_int bk = bi + 1; bk < bsr_dim; ++bk)
                {
                    T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                    // Multiplication factor
                    bsr_val[BSR_IND(j, bk, bi, dir)] = val /= diag;

                    // Loop through remaining columns of bk-th row and do linear combination
                    for(rocsparse_int bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = std::fma(-val,
                                       bsr_val[BSR_IND(j, bi, bj, dir)],
                                       bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }
        }

        // Store diagonal BSR block entry point
        rocsparse_int row_diag = diag_offset[i] = j;

        // Process upper diagonal BSR blocks
        for(j = row_diag + 1; j < row_end; ++j)
        {
            // Loop through all rows within the BSR block
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                // Process all rows within the BSR block after bi-th row
                for(rocsparse_int bk = bi + 1; bk < bsr_dim; ++bk)
                {
                    // Loop through columns of bk-th row and do linear combination
                    for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = std::fma(-bsr_val[BSR_IND(row_diag, bk, bi, dir)],
                                       bsr_val[BSR_IND(j, bi, bj, dir)],
                                       bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }
        }

        // Reset entry points
        for(j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col_j = bsr_col_ind[j] - base;
            nnz_entries[col_j]  = -1;
        }
    }

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;
}

template <typename T>
void host_csric0(rocsparse_int                     M,
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
                    local_sum = std::fma(csr_val[k], rocsparse_conj(csr_val[idx]), local_sum);
                }
            }

            val_j = (val_j - local_sum) * inv_diag;
            sum   = std::fma(val_j, rocsparse_conj(val_j), sum);

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
                  T                                 boost_val)
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

                T diag_val = csr_val[diag_j];

                if(boost)
                {
                    diag_val        = (boost_tol >= std::abs(diag_val)) ? boost_val : diag_val;
                    csr_val[diag_j] = diag_val;
                }
                else
                {
                    // Check for numeric pivot
                    if(diag_val == static_cast<T>(0))
                    {
                        *numeric_pivot = col_j + base;
                        return;
                    }
                }

                // multiplication factor
                csr_val[j] = csr_val[j] / diag_val;

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

// Parallel Cyclic reduction based on paper "Fast Tridiagonal Solvers on the GPU" by Yao Zhang
template <typename T>
void host_gtsv_no_pivot(rocsparse_int         m,
                        rocsparse_int         n,
                        const std::vector<T>& dl,
                        const std::vector<T>& d,
                        const std::vector<T>& du,
                        std::vector<T>&       B,
                        rocsparse_int         ldb)
{
    rocsparse_int BLOCKSIZE = 0;
    if((m & (m - 1)) == 0)
    {
        BLOCKSIZE = m;
    }
    else
    {
        BLOCKSIZE = pow(2, static_cast<rocsparse_int>(log2(m)) + 1);
    }

    for(rocsparse_int col = 0; col < n; col++)
    {
        rocsparse_int iter   = static_cast<rocsparse_int>(log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        std::vector<T> sa(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> sb(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> sc(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> srhs(BLOCKSIZE, static_cast<T>(0));

        std::vector<T> a(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> b(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> c(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> rhs(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> x(BLOCKSIZE, static_cast<T>(0));

        for(rocsparse_int i = 0; i < m; i++)
        {
            a[i]   = dl[i];
            b[i]   = d[i];
            c[i]   = du[i];
            rhs[i] = B[ldb * col + i];
        }

        for(rocsparse_int j = 0; j < iter; j++)
        {
            for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
            {
                rocsparse_int right = tid + stride;
                if(right >= m)
                    right = m - 1;

                rocsparse_int left = tid - stride;
                if(left < 0)
                    left = 0;

                T k1 = a[tid] / b[left];
                T k2 = c[tid] / b[right];

                T tb   = b[tid] - c[left] * k1 - a[right] * k2;
                T trhs = rhs[tid] - rhs[left] * k1 - rhs[right] * k2;
                T ta   = -a[left] * k1;
                T tc   = -c[right] * k2;

                sb[tid]   = tb;
                srhs[tid] = trhs;
                sa[tid]   = ta;
                sc[tid]   = tc;
            }

            for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
            {
                a[tid]   = sa[tid];
                b[tid]   = sb[tid];
                c[tid]   = sc[tid];
                rhs[tid] = srhs[tid];
            }

            stride *= 2;
        }

        for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
        {
            if(tid < BLOCKSIZE / 2)
            {
                rocsparse_int i = tid;
                rocsparse_int j = tid + stride;

                if(j < m)
                {
                    // Solve 2x2 systems
                    T det = b[j] * b[i] - c[i] * a[j];
                    x[i]  = (b[j] * rhs[i] - c[i] * rhs[j]) / det;
                    x[j]  = (rhs[j] * b[i] - rhs[i] * a[j]) / det;
                }
                else
                {
                    // Solve 1x1 systems
                    x[i] = rhs[i] / b[i];
                }
            }
        }

        for(rocsparse_int i = 0; i < m; i++)
        {
            B[ldb * col + i] = x[i];
        }
    }
}

// Parallel Cyclic reduction based on paper "Fast Tridiagonal Solvers on the GPU" by Yao Zhang
template <typename T>
void host_gtsv_no_pivot_strided_batch(rocsparse_int         m,
                                      const std::vector<T>& dl,
                                      const std::vector<T>& d,
                                      const std::vector<T>& du,
                                      std::vector<T>&       x,
                                      rocsparse_int         batch_count,
                                      rocsparse_int         batch_stride)
{
    rocsparse_int BLOCKSIZE = 0;
    if((m & (m - 1)) == 0)
    {
        BLOCKSIZE = m;
    }
    else
    {
        BLOCKSIZE = pow(2, static_cast<rocsparse_int>(log2(m)) + 1);
    }

    for(rocsparse_int col = 0; col < batch_count; col++)
    {
        rocsparse_int iter   = static_cast<rocsparse_int>(log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        std::vector<T> sa(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> sb(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> sc(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> srhs(BLOCKSIZE, static_cast<T>(0));

        std::vector<T> a(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> b(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> c(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> rhs(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> y(BLOCKSIZE, static_cast<T>(0));

        for(rocsparse_int i = 0; i < m; i++)
        {
            a[i]   = dl[batch_stride * col + i];
            b[i]   = d[batch_stride * col + i];
            c[i]   = du[batch_stride * col + i];
            rhs[i] = x[batch_stride * col + i];
        }

        for(rocsparse_int j = 0; j < iter; j++)
        {
            for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
            {
                rocsparse_int right = tid + stride;
                if(right >= m)
                    right = m - 1;

                rocsparse_int left = tid - stride;
                if(left < 0)
                    left = 0;

                T k1 = a[tid] / b[left];
                T k2 = c[tid] / b[right];

                T tb   = b[tid] - c[left] * k1 - a[right] * k2;
                T trhs = rhs[tid] - rhs[left] * k1 - rhs[right] * k2;
                T ta   = -a[left] * k1;
                T tc   = -c[right] * k2;

                sb[tid]   = tb;
                srhs[tid] = trhs;
                sa[tid]   = ta;
                sc[tid]   = tc;
            }

            for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
            {
                a[tid]   = sa[tid];
                b[tid]   = sb[tid];
                c[tid]   = sc[tid];
                rhs[tid] = srhs[tid];
            }

            stride *= 2;
        }

        for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
        {
            if(tid < BLOCKSIZE / 2)
            {
                rocsparse_int i = tid;
                rocsparse_int j = tid + stride;

                if(j < m)
                {
                    // Solve 2x2 systems
                    T det = b[j] * b[i] - c[i] * a[j];
                    y[i]  = (b[j] * rhs[i] - c[i] * rhs[j]) / det;
                    y[j]  = (rhs[j] * b[i] - rhs[i] * a[j]) / det;
                }
                else
                {
                    // Solve 1x1 systems
                    y[i] = rhs[i] / b[i];
                }
            }
        }

        for(rocsparse_int i = 0; i < m; i++)
        {
            x[batch_stride * col + i] = y[i];
        }
    }
}

template <typename T>
void host_gtsv_interleaved_batch_thomas(rocsparse_int m,
                                        const T*      dl,
                                        const T*      d,
                                        const T*      du,
                                        T*            x,
                                        rocsparse_int batch_count,
                                        rocsparse_int batch_stride)
{
    std::vector<T> c1(m * batch_count, 0);
    std::vector<T> x1(m * batch_count, 0);

    // Forward elimination
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        c1[j] = du[j] / d[j];
        x1[j] = x[j] / d[j];
    }

    for(rocsparse_int i = 1; i < m; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            rocsparse_int index = batch_count * i + j;
            rocsparse_int minus = batch_count * (i - 1) + j;

            T tdu = du[batch_stride * i + j];
            T td  = d[batch_stride * i + j];
            T tdl = dl[batch_stride * i + j];
            T tx  = x[batch_stride * i + j];

            c1[index] = tdu / (td - c1[minus] * tdl);
            x1[index] = (tx - x1[minus] * tdl) / (td - c1[minus] * tdl);
        }
    }

    // backward substitution
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        x[batch_stride * (m - 1) + j] = x1[batch_count * (m - 1) + j];
    }

    for(rocsparse_int i = m - 2; i >= 0; i--)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            rocsparse_int index = batch_count * i + j;

            x[batch_stride * i + j] = x1[index] - c1[index] * x[batch_stride * (i + 1) + j];
        }
    }
}

template <typename T>
void host_gtsv_interleaved_batch_lu(rocsparse_int m,
                                    const T*      dl,
                                    const T*      d,
                                    const T*      du,
                                    T*            x,
                                    rocsparse_int batch_count,
                                    rocsparse_int batch_stride)
{
    std::vector<T>             l(m * batch_count, 0);
    std::vector<T>             u0(m * batch_count, 0);
    std::vector<T>             u1(m * batch_count, 0);
    std::vector<T>             u2(m * batch_count, 0);
    std::vector<rocsparse_int> p(m * batch_count, 0);

    for(rocsparse_int i = 0; i < m; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            l[batch_count * i + j]  = dl[batch_stride * i + j];
            u0[batch_count * i + j] = d[batch_stride * i + j];
            u1[batch_count * i + j] = du[batch_stride * i + j];
        }
    }

    // LU decomposition
    for(rocsparse_int i = 0; i < m - 1; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            T ak_1 = l[batch_count * (i + 1) + j];
            T bk   = u0[batch_count * i + j];

            if(std::abs(bk) < std::abs(ak_1))
            {
                T bk_1 = u0[batch_count * (i + 1) + j];
                T ck   = u1[batch_count * i + j];
                T ck_1 = u1[batch_count * (i + 1) + j];
                T dk   = u2[batch_count * i + j];

                u0[batch_count * i + j] = ak_1;
                u1[batch_count * i + j] = bk_1;
                u2[batch_count * i + j] = ck_1;

                u0[batch_count * (i + 1) + j] = ck;
                u1[batch_count * (i + 1) + j] = dk;

                rocsparse_int pk             = p[batch_count * i + j];
                p[batch_count * i + j]       = i + 1;
                p[batch_count * (i + 1) + j] = pk;

                T xk                          = x[batch_stride * i + j];
                x[batch_stride * i + j]       = x[batch_stride * (i + 1) + j];
                x[batch_stride * (i + 1) + j] = xk;

                T lk_1                       = bk / ak_1;
                l[batch_count * (i + 1) + j] = lk_1;

                u0[batch_count * (i + 1) + j]
                    = u0[batch_count * (i + 1) + j] - lk_1 * u1[batch_count * i + j];
                u1[batch_count * (i + 1) + j]
                    = u1[batch_count * (i + 1) + j] - lk_1 * u2[batch_count * i + j];
            }
            else
            {
                p[batch_count * (i + 1) + j] = i + 1;

                T lk_1                       = ak_1 / bk;
                l[batch_count * (i + 1) + j] = lk_1;

                u0[batch_count * (i + 1) + j]
                    = u0[batch_count * (i + 1) + j] - lk_1 * u1[batch_count * i + j];
                u1[batch_count * (i + 1) + j]
                    = u1[batch_count * (i + 1) + j] - lk_1 * u2[batch_count * i + j];
            }
        }
    }

    // Forward elimination (L * x_new = x_old)
    std::vector<rocsparse_int> start(batch_count, 0);
    for(rocsparse_int i = 1; i < m; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            if(p[batch_count * i + j] <= i) // no pivoting occured, sum up result
            {
                T temp = static_cast<T>(0);
                for(rocsparse_int s = start[j]; s < i; s++)
                {
                    temp = temp - l[batch_count * (s + 1) + j] * x[batch_stride * s + j];
                }
                x[batch_stride * i + j] = x[batch_stride * i + j] + temp;
                start[j] += i - start[j];
            }
        }
    }

    // backward substitution (U * x_newest = x_new)
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        x[batch_stride * (m - 1) + j]
            = x[batch_stride * (m - 1) + j] / u0[batch_count * (m - 1) + j];
        x[batch_stride * (m - 2) + j]
            = (x[batch_stride * (m - 2) + j]
               - u1[batch_count * (m - 2) + j] * x[batch_stride * (m - 1) + j])
              / u0[batch_count * (m - 2) + j];
    }

    for(rocsparse_int i = m - 3; i >= 0; i--)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            x[batch_stride * i + j]
                = (x[batch_stride * i + j] - u1[batch_count * i + j] * x[batch_stride * (i + 1) + j]
                   - u2[batch_count * i + j] * x[batch_stride * (i + 2) + j])
                  / u0[batch_count * i + j];
        }
    }
}

template <typename T>
void host_gtsv_interleaved_batch_qr(rocsparse_int m,
                                    const T*      dl,
                                    const T*      d,
                                    const T*      du,
                                    T*            x,
                                    rocsparse_int batch_count,
                                    rocsparse_int batch_stride)
{
    std::vector<T> r0(m * batch_count, 0);
    std::vector<T> r1(m * batch_count, 0);
    std::vector<T> r2(m * batch_count, 0);

    for(rocsparse_int i = 0; i < m; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            r0[batch_count * i + j] = d[batch_stride * i + j];
            r1[batch_count * i + j] = du[batch_stride * i + j];
        }
    }

    // Reduce A = Q*R where Q is orthonormal and R is upper triangular
    // This means when solving A * x          = b
    //                      => Q * R * x      = b
    //                      => Q' * Q * R * x = Q' * b
    //                      => R * x          = Q' * b
    // Because A is tri-diagonal, we use Givens rotations
    // Note on notation used here. I consider the A matrix to have form:
    // A = b0 c0 0  0  0
    //     a1 b1 c1 0  0
    //     0  a2 b2 c2 0
    //     0  0  a3 b3 c3
    //     0  0  0  a4 b4
    for(rocsparse_int i = 0; i < m - 1; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            T ak_1 = dl[batch_stride * (i + 1) + j];
            T bk   = r0[batch_count * i + j];
            T bk_1 = r0[batch_count * (i + 1) + j];
            T ck   = r1[batch_count * i + j];
            T ck_1 = r1[batch_count * (i + 1) + j];

            T radius = std::sqrt(std::abs(bk * rocsparse_conj(bk) + ak_1 * rocsparse_conj(ak_1)));

            // Apply Givens rotation
            // | cos  sin | |bk    ck   0   |
            // |-sin  cos | |ak_1  bk_1 ck_1|
            T cos_theta = rocsparse_conj(bk) / radius;
            T sin_theta = rocsparse_conj(ak_1) / radius;

            r0[batch_count * i + j] = std::fma(bk, cos_theta, ak_1 * sin_theta);
            r0[batch_count * (i + 1) + j]
                = std::fma(-ck, rocsparse_conj(sin_theta), bk_1 * rocsparse_conj(cos_theta));
            r1[batch_count * i + j]       = std::fma(ck, cos_theta, bk_1 * sin_theta);
            r1[batch_count * (i + 1) + j] = ck_1 * rocsparse_conj(cos_theta);
            r2[batch_count * i + j]       = ck_1 * sin_theta;

            // Apply Givens rotation to rhs vector
            // | cos  sin | |xk  |
            // |-sin  cos | |xk_1|
            T xk                    = x[batch_stride * i + j];
            T xk_1                  = x[batch_stride * (i + 1) + j];
            x[batch_stride * i + j] = std::fma(xk, cos_theta, xk_1 * sin_theta);
            x[batch_stride * (i + 1) + j]
                = std::fma(-xk, rocsparse_conj(sin_theta), xk_1 * rocsparse_conj(cos_theta));
        }
    }

    // Backward substitution on upper triangular R * x = x
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        x[batch_stride * (m - 1) + j]
            = x[batch_stride * (m - 1) + j] / r0[batch_count * (m - 1) + j];
        x[batch_stride * (m - 2) + j]
            = (x[batch_stride * (m - 2) + j]
               - r1[batch_count * (m - 2) + j] * x[batch_stride * (m - 1) + j])
              / r0[batch_count * (m - 2) + j];
    }

    for(rocsparse_int i = m - 3; i >= 0; i--)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            x[batch_stride * i + j]
                = (x[batch_stride * i + j] - r1[batch_count * i + j] * x[batch_stride * (i + 1) + j]
                   - r2[batch_count * i + j] * x[batch_stride * (i + 2) + j])
                  / r0[batch_count * i + j];
        }
    }
}

template <typename T>
void host_gtsv_interleaved_batch(rocsparse_gtsv_interleaved_alg algo,
                                 rocsparse_int                  m,
                                 const T*                       dl,
                                 const T*                       d,
                                 const T*                       du,
                                 T*                             x,
                                 rocsparse_int                  batch_count,
                                 rocsparse_int                  batch_stride)
{
    switch(algo)
    {
    case rocsparse_gtsv_interleaved_alg_thomas:
    {
        host_gtsv_interleaved_batch_thomas(m, dl, d, du, x, batch_count, batch_stride);
        break;
    }
    case rocsparse_gtsv_interleaved_alg_lu:
    {
        host_gtsv_interleaved_batch_lu(m, dl, d, du, x, batch_count, batch_stride);
        break;
    }
    case rocsparse_gtsv_interleaved_alg_default:
    case rocsparse_gtsv_interleaved_alg_qr:
    {
        host_gtsv_interleaved_batch_qr(m, dl, d, du, x, batch_count, batch_stride);
        break;
    }
    }
}

template <typename T>
void host_gpsv_interleaved_batch_qr(rocsparse_int m,
                                    T*            ds,
                                    T*            dl,
                                    T*            d,
                                    T*            du,
                                    T*            dw,
                                    T*            x,
                                    rocsparse_int batch_count,
                                    rocsparse_int batch_stride)
{
    std::vector<T> r3(m * batch_count, 0);
    std::vector<T> r4(m * batch_count, 0);

    // Reduce A = Q*R where Q is orthonormal and R is upper triangular
    // This means when solving A * x          = b
    //                      => Q * R * x      = b
    //                      => Q' * Q * R * x = Q' * b
    //                      => R * x          = Q' * b
    // Because A is penta-diagonal, we use Givens rotations
    // Note on notation used here. I consider the A matrix to have form:
    // A = d0 u0 w0  0  0
    //     l1 d1 u1 w1  0
    //     s2  l2 d2 u2 w2
    //     0  s3  l3 d3 u3
    //     0  0  s4  l4 d4
    for(rocsparse_int i = 0; i < m - 2; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            // For penta diagonal matrices, need to apply two givens rotations to remove lower and lower - 1 entries
            T radius    = static_cast<T>(0);
            T cos_theta = static_cast<T>(0);
            T sin_theta = static_cast<T>(0);

            // Apply first Givens rotation
            // | cos  sin | |lk_1 dk_1 uk_1 wk_1 0   |
            // |-sin  cos | |sk_2 lk_2 dk_2 uk_2 wk_2|
            T sk_2 = ds[batch_stride * (i + 2) + j];
            T lk_1 = dl[batch_stride * (i + 1) + j];
            T lk_2 = dl[batch_stride * (i + 2) + j];
            T dk_1 = d[batch_stride * (i + 1) + j];
            T dk_2 = d[batch_stride * (i + 2) + j];
            T uk_1 = du[batch_stride * (i + 1) + j];
            T uk_2 = du[batch_stride * (i + 2) + j];
            T wk_1 = dw[batch_stride * (i + 1) + j];
            T wk_2 = dw[batch_stride * (i + 2) + j];

            radius = std::sqrt(
                std::abs(std::fma(lk_1, rocsparse_conj(lk_1), sk_2 * rocsparse_conj(sk_2))));
            cos_theta = rocsparse_conj(lk_1) / radius;
            sin_theta = rocsparse_conj(sk_2) / radius;

            T dlk_1_new = std::fma(lk_1, cos_theta, sk_2 * sin_theta);
            T dk_1_new  = std::fma(dk_1, cos_theta, lk_2 * sin_theta);
            T duk_1_new = std::fma(uk_1, cos_theta, dk_2 * sin_theta);
            T dwk_1_new = std::fma(wk_1, cos_theta, uk_2 * sin_theta);

            dl[batch_stride * (i + 1) + j] = dlk_1_new;
            dl[batch_stride * (i + 2) + j]
                = std::fma(-dk_1, rocsparse_conj(sin_theta), lk_2 * rocsparse_conj(cos_theta));
            d[batch_stride * (i + 1) + j] = dk_1_new;
            d[batch_stride * (i + 2) + j]
                = std::fma(-uk_1, rocsparse_conj(sin_theta), dk_2 * rocsparse_conj(cos_theta));
            du[batch_stride * (i + 1) + j] = duk_1_new;
            du[batch_stride * (i + 2) + j]
                = std::fma(-wk_1, rocsparse_conj(sin_theta), uk_2 * rocsparse_conj(cos_theta));
            dw[batch_stride * (i + 1) + j] = dwk_1_new;
            dw[batch_stride * (i + 2) + j] = wk_2 * rocsparse_conj(cos_theta);
            r3[batch_count * (i + 1) + j]  = wk_2 * sin_theta;

            // Apply first Givens rotation to rhs vector
            // | cos  sin | |xk_1|
            // |-sin  cos | |xk_2|
            T xk_1                        = x[batch_stride * (i + 1) + j];
            T xk_2                        = x[batch_stride * (i + 2) + j];
            x[batch_stride * (i + 1) + j] = std::fma(xk_1, cos_theta, xk_2 * sin_theta);
            x[batch_stride * (i + 2) + j]
                = std::fma(-xk_1, rocsparse_conj(sin_theta), xk_2 * rocsparse_conj(cos_theta));

            // Apply second Givens rotation
            // | cos  sin | |dk   uk   wk   rk   0   |
            // |-sin  cos | |lk_1 dk_1 uk_1 wk_1 rk_1|
            lk_1   = dlk_1_new;
            T dk   = d[batch_stride * i + j];
            dk_1   = dk_1_new;
            T uk   = du[batch_stride * i + j];
            uk_1   = duk_1_new;
            T wk   = dw[batch_stride * i + j];
            wk_1   = dwk_1_new;
            T rk   = r3[batch_count * i + j];
            T rk_1 = r3[batch_count * (i + 1) + j];

            radius = std::sqrt(
                std::abs(std::fma(dk, rocsparse_conj(dk), lk_1 * rocsparse_conj(lk_1))));
            cos_theta = rocsparse_conj(dk) / radius;
            sin_theta = rocsparse_conj(lk_1) / radius;

            d[batch_stride * i + j] = std::fma(dk, cos_theta, lk_1 * sin_theta);
            d[batch_stride * (i + 1) + j]
                = std::fma(-uk, rocsparse_conj(sin_theta), dk_1 * rocsparse_conj(cos_theta));
            du[batch_stride * i + j] = std::fma(uk, cos_theta, dk_1 * sin_theta);
            du[batch_stride * (i + 1) + j]
                = std::fma(-wk, rocsparse_conj(sin_theta), uk_1 * rocsparse_conj(cos_theta));
            dw[batch_stride * i + j] = std::fma(wk, cos_theta, uk_1 * sin_theta);
            dw[batch_stride * (i + 1) + j]
                = std::fma(-rk, rocsparse_conj(sin_theta), wk_1 * rocsparse_conj(cos_theta));
            r3[batch_count * i + j]       = std::fma(rk, cos_theta, wk_1 * sin_theta);
            r3[batch_count * (i + 1) + j] = rk_1 * rocsparse_conj(cos_theta);
            r4[batch_count * i + j]       = rk_1 * sin_theta;

            // Apply second Givens rotation to rhs vector
            // | cos  sin | |xk  |
            // |-sin  cos | |xk_1|
            T xk                    = x[batch_stride * i + j];
            xk_1                    = x[batch_stride * (i + 1) + j];
            x[batch_stride * i + j] = std::fma(xk, cos_theta, xk_1 * sin_theta);
            x[batch_stride * (i + 1) + j]
                = std::fma(-xk, rocsparse_conj(sin_theta), xk_1 * rocsparse_conj(cos_theta));
        }
    }

    // Apply last givens rotation
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        // Apply last Givens rotation
        // | cos  sin | |dk   uk   wk   rk   0   |
        // |-sin  cos | |lk_1 dk_1 uk_1 wk_1 rk_1|
        T lk_1 = dl[batch_stride * (m - 1) + j];
        T dk   = d[batch_stride * (m - 2) + j];
        T dk_1 = d[batch_stride * (m - 1) + j];
        T uk   = du[batch_stride * (m - 2) + j];
        T uk_1 = du[batch_stride * (m - 1) + j];
        T wk   = dw[batch_stride * (m - 2) + j];
        T wk_1 = dw[batch_stride * (m - 1) + j];
        T rk   = r3[batch_count * (m - 2) + j];
        T rk_1 = r3[batch_count * (m - 1) + j];

        T radius
            = std::sqrt(std::abs(std::fma(dk, rocsparse_conj(dk), lk_1 * rocsparse_conj(lk_1))));
        T cos_theta = rocsparse_conj(dk) / radius;
        T sin_theta = rocsparse_conj(lk_1) / radius;

        d[batch_stride * (m - 2) + j] = std::fma(dk, cos_theta, lk_1 * sin_theta);
        d[batch_stride * (m - 1) + j]
            = std::fma(-uk, rocsparse_conj(sin_theta), dk_1 * rocsparse_conj(cos_theta));
        du[batch_stride * (m - 2) + j] = std::fma(uk, cos_theta, dk_1 * sin_theta);
        du[batch_stride * (m - 1) + j]
            = std::fma(-wk, rocsparse_conj(sin_theta), uk_1 * rocsparse_conj(cos_theta));
        dw[batch_stride * (m - 2) + j] = std::fma(wk, cos_theta, uk_1 * sin_theta);
        dw[batch_stride * (m - 1) + j]
            = std::fma(-rk, rocsparse_conj(sin_theta), wk_1 * rocsparse_conj(cos_theta));
        r3[batch_count * (m - 2) + j] = std::fma(rk, cos_theta, wk_1 * sin_theta);
        r3[batch_count * (m - 1) + j] = rk_1 * rocsparse_conj(cos_theta);
        r4[batch_count * (m - 2) + j] = rk_1 * sin_theta;

        // Apply last Givens rotation to rhs vector
        // | cos  sin | |xk  |
        // |-sin  cos | |xk_1|
        T xk                          = x[batch_stride * (m - 2) + j];
        T xk_1                        = x[batch_stride * (m - 1) + j];
        x[batch_stride * (m - 2) + j] = std::fma(xk, cos_theta, xk_1 * sin_theta);
        x[batch_stride * (m - 1) + j]
            = std::fma(-xk, rocsparse_conj(sin_theta), xk_1 * rocsparse_conj(cos_theta));
    }

    // Backward substitution on upper triangular R * x = x
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        x[batch_stride * (m - 1) + j]
            = x[batch_stride * (m - 1) + j] / d[batch_stride * (m - 1) + j];
        x[batch_stride * (m - 2) + j]
            = (x[batch_stride * (m - 2) + j]
               - du[batch_stride * (m - 2) + j] * x[batch_stride * (m - 1) + j])
              / d[batch_stride * (m - 2) + j];

        x[batch_stride * (m - 3) + j]
            = (x[batch_stride * (m - 3) + j]
               - du[batch_stride * (m - 3) + j] * x[batch_stride * (m - 2) + j]
               - dw[batch_stride * (m - 3) + j] * x[batch_stride * (m - 1) + j])
              / d[batch_stride * (m - 3) + j];

        x[batch_stride * (m - 4) + j]
            = (x[batch_stride * (m - 4) + j]
               - du[batch_stride * (m - 4) + j] * x[batch_stride * (m - 3) + j]
               - dw[batch_stride * (m - 4) + j] * x[batch_stride * (m - 2) + j]
               - r3[batch_count * (m - 4) + j] * x[batch_stride * (m - 1) + j])
              / d[batch_stride * (m - 4) + j];
    }

    for(rocsparse_int i = m - 5; i >= 0; i--)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            x[batch_stride * i + j] = (x[batch_stride * i + j]
                                       - du[batch_stride * i + j] * x[batch_stride * (i + 1) + j]
                                       - dw[batch_stride * i + j] * x[batch_stride * (i + 2) + j]
                                       - r3[batch_count * i + j] * x[batch_stride * (i + 3) + j]
                                       - r4[batch_count * i + j] * x[batch_stride * (i + 4) + j])
                                      / d[batch_stride * i + j];
        }
    }
}

template <typename T>
void host_gpsv_interleaved_batch(rocsparse_gpsv_interleaved_alg algo,
                                 rocsparse_int                  m,
                                 T*                             ds,
                                 T*                             dl,
                                 T*                             d,
                                 T*                             du,
                                 T*                             dw,
                                 T*                             x,
                                 rocsparse_int                  batch_count,
                                 rocsparse_int                  batch_stride)
{
    switch(algo)
    {
    case rocsparse_gpsv_interleaved_alg_default:
    case rocsparse_gpsv_interleaved_alg_qr:
    {
        host_gpsv_interleaved_batch_qr(m, ds, dl, d, du, dw, x, batch_count, batch_stride);
        break;
    }
    }
}

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template <typename T>
rocsparse_status host_nnz(rocsparse_direction dirA,
                          rocsparse_int       m,
                          rocsparse_int       n,
                          const T*            A,
                          rocsparse_int       lda,
                          rocsparse_int*      nnz_per_row_columns,
                          rocsparse_int*      nnz_total_dev_host_ptr)
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
                          std::vector<rocsparse_int>& csr_col_ind)
{
    csr_row_ptr.resize(m + 1, 0);
    csr_row_ptr[0] = base;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < m; i++)
    {
        for(rocsparse_int j = 0; j < n; j++)
        {
            if(std::abs(A[lda * j + i]) > threshold)
            {
                csr_row_ptr[i + 1]++;
            }
        }
    }

    for(rocsparse_int i = 1; i <= m; i++)
    {
        csr_row_ptr[i] += csr_row_ptr[i - 1];
    }

    nnz = csr_row_ptr[m] - csr_row_ptr[0];

    csr_col_ind.resize(nnz);
    csr_val.resize(nnz);

    rocsparse_int index = 0;
    for(rocsparse_int i = 0; i < m; i++)
    {
        for(rocsparse_int j = 0; j < n; j++)
        {
            if(std::abs(A[lda * j + i]) > threshold)
            {
                csr_val[index]     = A[lda * j + i];
                csr_col_ind[index] = j + base;

                index++;
            }
        }
    }
}

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
                                        std::vector<rocsparse_int>& csr_col_ind)
{
    rocsparse_int nnz_A = m * n;
    rocsparse_int pos   = std::ceil(nnz_A * (percentage / 100)) - 1;
    pos                 = std::min(pos, nnz_A - 1);
    pos                 = std::max(pos, 0);

    std::vector<T> sorted_A(m * n);
    for(rocsparse_int i = 0; i < n; i++)
    {
        for(rocsparse_int j = 0; j < m; j++)
        {
            sorted_A[m * i + j] = std::abs(A[lda * i + j]);
        }
    }

    std::sort(sorted_A.begin(), sorted_A.end());

    T threshold = sorted_A[pos];
    host_prune_dense2csr<T>(m, n, A, lda, base, threshold, nnz, csr_val, csr_row_ptr, csr_col_ind);
}

template <rocsparse_direction DIRA, typename T, typename I, typename J>
void host_dense2csx(J                    m,
                    J                    n,
                    rocsparse_index_base base,
                    const T*             A,
                    I                    ld,
                    rocsparse_order      order,
                    const I*             nnz_per_row_columns,
                    T*                   csx_val,
                    I*                   csx_row_col_ptr,
                    J*                   csx_col_row_ind)
{
    static constexpr T s_zero = {};
    J                  len    = (rocsparse_direction_row == DIRA) ? m : n;
    *csx_row_col_ptr          = base;
    for(J i = 0; i < len; ++i)
    {
        csx_row_col_ptr[i + 1] = nnz_per_row_columns[i] + csx_row_col_ptr[i];
    }

    switch(DIRA)
    {
    case rocsparse_direction_column:
    {
        for(J j = 0; j < n; ++j)
        {
            for(J i = 0; i < m; ++i)
            {
                if(order == rocsparse_order_column)
                {
                    if(A[j * ld + i] != s_zero)
                    {
                        *csx_val++         = A[j * ld + i];
                        *csx_col_row_ind++ = i + base;
                    }
                }
                else
                {
                    if(A[i * ld + j] != s_zero)
                    {
                        *csx_val++         = A[i * ld + j];
                        *csx_col_row_ind++ = i + base;
                    }
                }
            }
        }

        break;
    }

    case rocsparse_direction_row:
    {
        //
        // Does not matter having an orthogonal traversal ... testing only.
        // Otherwise, we would use csx_row_ptr_A to store the shifts.
        // and once the job is done a simple memory move would reinitialize the csx_row_ptr_A to its initial state)
        //
        for(J i = 0; i < m; ++i)
        {
            for(J j = 0; j < n; ++j)
            {
                if(order == rocsparse_order_column)
                {
                    if(A[j * ld + i] != s_zero)
                    {
                        *csx_val++         = A[j * ld + i];
                        *csx_col_row_ind++ = j + base;
                    }
                }
                else
                {
                    if(A[i * ld + j] != s_zero)
                    {
                        *csx_val++         = A[i * ld + j];
                        *csx_col_row_ind++ = j + base;
                    }
                }
            }
        }

        break;
    }
    }
}

template <rocsparse_direction DIRA, typename T, typename I, typename J>
void host_csx2dense(J                    m,
                    J                    n,
                    rocsparse_index_base base,
                    rocsparse_order      order,
                    const T*             csx_val,
                    const I*             csx_row_col_ptr,
                    const J*             csx_col_row_ind,
                    T*                   A,
                    I                    ld)
{
    if(order == rocsparse_order_column)
    {
        for(J col = 0; col < n; ++col)
        {
            for(J row = 0; row < m; ++row)
            {
                A[row + ld * col] = static_cast<T>(0);
            }
        }
    }
    else
    {
        for(J row = 0; row < m; ++row)
        {
            for(J col = 0; col < n; ++col)
            {
                A[col + ld * row] = static_cast<T>(0);
            }
        }
    }

    if(DIRA == rocsparse_direction_column)
    {
        for(J col = 0; col < n; ++col)
        {
            I start = csx_row_col_ptr[col] - base;
            I end   = csx_row_col_ptr[col + 1] - base;

            if(order == rocsparse_order_column)
            {
                for(I at = start; at < end; ++at)
                {
                    A[(csx_col_row_ind[at] - base) + ld * col] = csx_val[at];
                }
            }
            else
            {
                for(I at = start; at < end; ++at)
                {
                    A[col + ld * (csx_col_row_ind[at] - base)] = csx_val[at];
                }
            }
        }
    }
    else
    {
        for(J row = 0; row < m; ++row)
        {
            I start = csx_row_col_ptr[row] - base;
            I end   = csx_row_col_ptr[row + 1] - base;

            if(order == rocsparse_order_column)
            {
                for(I at = start; at < end; ++at)
                {
                    A[(csx_col_row_ind[at] - base) * ld + row] = csx_val[at];
                }
            }
            else
            {
                for(I at = start; at < end; ++at)
                {

                    A[row * ld + (csx_col_row_ind[at] - base)] = csx_val[at];
                }
            }
        }
    }
}

template <typename I, typename T>
void host_dense_to_coo(I                     m,
                       I                     n,
                       rocsparse_index_base  base,
                       const std::vector<T>& A,
                       I                     ld,
                       rocsparse_order       order,
                       const std::vector<I>& nnz_per_row,
                       std::vector<T>&       coo_val,
                       std::vector<I>&       coo_row_ind,
                       std::vector<I>&       coo_col_ind)
{
    // Find number of non-zeros in dense matrix
    int nnz = 0;
    for(I i = 0; i < m; ++i)
    {
        nnz += nnz_per_row[i];
    }

    coo_val.resize(nnz, static_cast<T>(0));
    coo_row_ind.resize(nnz, 0);
    coo_col_ind.resize(nnz, 0);

    // Fill COO matrix
    int index = 0;
    for(I i = 0; i < m; i++)
    {
        for(I j = 0; j < n; j++)
        {
            if(order == rocsparse_order_column)
            {
                if(A[ld * j + i] != static_cast<T>(0))
                {
                    coo_val[index]     = A[ld * j + i];
                    coo_row_ind[index] = i + base;
                    coo_col_ind[index] = j + base;

                    index++;
                }
            }
            else
            {
                if(A[ld * i + j] != static_cast<T>(0))
                {
                    coo_val[index]     = A[ld * i + j];
                    coo_row_ind[index] = i + base;
                    coo_col_ind[index] = j + base;

                    index++;
                }
            }
        }
    }
}

template <typename I, typename T>
void host_coo_to_dense(I                     m,
                       I                     n,
                       I                     nnz,
                       rocsparse_index_base  base,
                       const std::vector<T>& coo_val,
                       const std::vector<I>& coo_row_ind,
                       const std::vector<I>& coo_col_ind,
                       std::vector<T>&       A,
                       I                     ld,
                       rocsparse_order       order)
{
    I nm = order == rocsparse_order_column ? n : m;

    A.resize(ld * nm);

    for(I i = 0; i < nnz; i++)
    {
        I row = coo_row_ind[i] - base;
        I col = coo_col_ind[i] - base;
        T val = coo_val[i];

        if(order == rocsparse_order_column)
        {
            A[ld * col + row] = val;
        }
        else
        {
            A[ld * row + col] = val;
        }
    }
}

template <typename I, typename J, typename T>
void host_csr_to_csc(J                    M,
                     J                    N,
                     I                    nnz,
                     const I*             csr_row_ptr,
                     const J*             csr_col_ind,
                     const T*             csr_val,
                     std::vector<J>&      csc_row_ind,
                     std::vector<I>&      csc_col_ptr,
                     std::vector<T>&      csc_val,
                     rocsparse_action     action,
                     rocsparse_index_base base)
{
    csc_row_ind.resize(nnz);
    csc_col_ptr.resize(N + 1, 0);
    csc_val.resize(nnz);

    // Determine nnz per column
    for(I i = 0; i < nnz; ++i)
    {
        ++csc_col_ptr[csr_col_ind[i] + 1 - base];
    }

    // Scan
    for(J i = 0; i < N; ++i)
    {
        csc_col_ptr[i + 1] += csc_col_ptr[i];
    }

    // Fill row indices and values
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr[i] - base;
        I row_end   = csr_row_ptr[i + 1] - base;

        for(I j = row_begin; j < row_end; ++j)
        {
            J col = csr_col_ind[j] - base;
            I idx = csc_col_ptr[col];

            csc_row_ind[idx] = i + base;
            csc_val[idx]     = csr_val[j];

            ++csc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(J i = N; i > 0; --i)
    {
        csc_col_ptr[i] = csc_col_ptr[i - 1] + base;
    }

    csc_col_ptr[0] = base;
}

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
                       rocsparse_index_base              bsr_base)
{
    rocsparse_int mb = (m + row_block_dim - 1) / row_block_dim;

    bsr_row_ptr.resize(mb + 1, 0);

    std::vector<rocsparse_int> temp(nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < nnz; i++)
    {
        temp[i] = (csr_col_ind[i] - csr_base) / col_block_dim;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < mb; i++)
    {
        rocsparse_int frow = row_block_dim * i;
        rocsparse_int lrow = row_block_dim * (i + 1);

        if(lrow > m)
        {
            lrow = m;
        }

        rocsparse_int start = csr_row_ptr[frow] - csr_base;
        rocsparse_int end   = csr_row_ptr[lrow] - csr_base;

        std::sort(temp.begin() + start, temp.begin() + end);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < mb; i++)
    {
        rocsparse_int frow = row_block_dim * i;
        rocsparse_int lrow = row_block_dim * (i + 1);

        if(lrow > m)
        {
            lrow = m;
        }

        rocsparse_int start = csr_row_ptr[frow] - csr_base;
        rocsparse_int end   = csr_row_ptr[lrow] - csr_base;

        rocsparse_int col   = -1;
        rocsparse_int count = 0;
        for(rocsparse_int j = start; j < end; j++)
        {
            if(temp[j] > col)
            {
                col                 = temp[j];
                temp[j]             = -1;
                temp[start + count] = col;
                count++;
            }
            else
            {
                temp[j] = -1;
            }
        }

        bsr_row_ptr[i + 1] = count;
    }

    // fill GEBSR row pointer array
    bsr_row_ptr[0] = bsr_base;
    for(rocsparse_int i = 0; i < mb; i++)
    {
        bsr_row_ptr[i + 1] += bsr_row_ptr[i];
    }

    rocsparse_int nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];
    bsr_col_ind.resize(nnzb);
    bsr_val.resize(nnzb * row_block_dim * col_block_dim, 0);

    // fill GEBSR col indices array
    {
        rocsparse_int index = 0;
        for(rocsparse_int i = 0; i < nnz; i++)
        {
            if(temp[i] != -1)
            {
                bsr_col_ind[index] = temp[i] + bsr_base;
                index++;
            }
        }
    }

    // fill GEBSR values array
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < m; i++)
    {
        rocsparse_int start = csr_row_ptr[i] - csr_base;
        rocsparse_int end   = csr_row_ptr[i + 1] - csr_base;

        rocsparse_int bstart = bsr_row_ptr[i / row_block_dim] - bsr_base;
        rocsparse_int bend   = bsr_row_ptr[i / row_block_dim + 1] - bsr_base;

        rocsparse_int local_row = i % row_block_dim;

        for(rocsparse_int j = start; j < end; j++)
        {
            rocsparse_int col = csr_col_ind[j] - csr_base;

            rocsparse_int local_col = col % col_block_dim;

            rocsparse_int index = 0;
            for(rocsparse_int k = bstart; k < bend; k++)
            {
                if(bsr_col_ind[k] - bsr_base == col / col_block_dim)
                {
                    index  = k;
                    bstart = k;
                    break;
                }
            }

            if(direction == rocsparse_direction_row)
            {
                bsr_val[row_block_dim * col_block_dim * index + col_block_dim * local_row
                        + local_col]
                    = csr_val[j];
            }
            else
            {
                bsr_val[row_block_dim * col_block_dim * index + row_block_dim * local_col
                        + local_row]
                    = csr_val[j];
            }
        }
    }
}

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
                         rocsparse_index_base              base)
{
    bsc_row_ind.resize(nnzb);
    bsc_col_ptr.resize(Nb + 1, 0);
    bsc_val.resize(nnzb * row_block_dim * col_block_dim);

    const rocsparse_int block_shift = row_block_dim * col_block_dim;

    //
    // Determine nnz per column
    //
    for(rocsparse_int i = 0; i < nnzb; ++i)
    {
        ++bsc_col_ptr[bsr_col_ind[i] + 1 - base];
    }

    // Scan
    for(rocsparse_int i = 0; i < Nb; ++i)
    {
        bsc_col_ptr[i + 1] += bsc_col_ptr[i];
    }

    // Fill row indices and values
    for(rocsparse_int i = 0; i < Mb; ++i)
    {
        const rocsparse_int row_begin = bsr_row_ptr[i] - base;
        const rocsparse_int row_end   = bsr_row_ptr[i + 1] - base;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            const rocsparse_int col = bsr_col_ind[j] - base;
            const rocsparse_int idx = bsc_col_ptr[col];

            bsc_row_ind[idx] = i + base;
            for(rocsparse_int k = 0; k < block_shift; ++k)
            {
                bsc_val[idx * block_shift + k] = bsr_val[j * block_shift + k];
            }

            ++bsc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(rocsparse_int i = Nb; i > 0; --i)
    {
        bsc_col_ptr[i] = bsc_col_ptr[i - 1] + base;
    }

    bsc_col_ptr[0] = base;
}

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
                       rocsparse_index_base              csr_base)
{

    csr_col_ind.resize(nnzb * row_block_dim * col_block_dim);
    csr_row_ptr.resize(mb * row_block_dim + 1);
    csr_val.resize(nnzb * row_block_dim * col_block_dim);
    rocsparse_int at = 0;
    csr_row_ptr[0]   = csr_base;
    for(rocsparse_int i = 0; i < mb; ++i)
    {
        for(rocsparse_int r = 0; r < row_block_dim; ++r)
        {
            rocsparse_int row = i * row_block_dim + r;
            for(rocsparse_int k = bsr_row_ptr[i] - bsr_base; k < bsr_row_ptr[i + 1] - bsr_base; ++k)
            {
                rocsparse_int j = bsr_col_ind[k] - bsr_base;
                for(rocsparse_int c = 0; c < col_block_dim; ++c)
                {
                    rocsparse_int col = col_block_dim * j + c;
                    csr_col_ind[at]   = col + csr_base;
                    if(direction == rocsparse_direction_row)
                    {
                        csr_val[at]
                            = bsr_val[k * row_block_dim * col_block_dim + col_block_dim * r + c];
                    }
                    else
                    {
                        csr_val[at]
                            = bsr_val[k * row_block_dim * col_block_dim + row_block_dim * c + r];
                    }
                    ++at;
                }
            }

            csr_row_ptr[row + 1]
                = csr_row_ptr[row] + (bsr_row_ptr[i + 1] - bsr_row_ptr[i]) * col_block_dim;
        }
    }
}

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
                         rocsparse_index_base              base_C)
{
    rocsparse_int m = mb * row_block_dim_A;
    rocsparse_int n = nb * col_block_dim_A;

    // convert GEBSR to CSR format
    std::vector<rocsparse_int> csr_row_ptr;
    std::vector<rocsparse_int> csr_col_ind;
    std::vector<T>             csr_val;

    host_gebsr_to_csr(direction,
                      mb,
                      nb,
                      nnzb,
                      bsr_val_A,
                      bsr_row_ptr_A,
                      bsr_col_ind_A,
                      row_block_dim_A,
                      col_block_dim_A,
                      base_A,
                      csr_val,
                      csr_row_ptr,
                      csr_col_ind,
                      rocsparse_index_base_zero);

    rocsparse_int nnz = csr_row_ptr[m] - csr_row_ptr[0];

    // convert CSR to GEBSR format
    host_csr_to_gebsr(direction,
                      m,
                      n,
                      nnz,
                      csr_val,
                      csr_row_ptr,
                      csr_col_ind,
                      row_block_dim_C,
                      col_block_dim_C,
                      rocsparse_index_base_zero,
                      bsr_val_C,
                      bsr_row_ptr_C,
                      bsr_col_ind_C,
                      base_C);
}

template <typename T>
void host_bsr_to_bsc(rocsparse_int               mb,
                     rocsparse_int               nb,
                     rocsparse_int               nnzb,
                     rocsparse_int               bsr_dim,
                     const rocsparse_int*        bsr_row_ptr,
                     const rocsparse_int*        bsr_col_ind,
                     const T*                    bsr_val,
                     std::vector<rocsparse_int>& bsc_row_ind,
                     std::vector<rocsparse_int>& bsc_col_ptr,
                     std::vector<T>&             bsc_val,
                     rocsparse_index_base        bsr_base,
                     rocsparse_index_base        bsc_base)
{
    bsc_row_ind.resize(nnzb);
    bsc_col_ptr.resize(nb + 1, 0);
    bsc_val.resize(nnzb * bsr_dim * bsr_dim);

    // Determine nnz per column
    for(rocsparse_int i = 0; i < nnzb; ++i)
    {
        ++bsc_col_ptr[bsr_col_ind[i] + 1 - bsr_base];
    }

    // Scan
    for(rocsparse_int i = 0; i < nb; ++i)
    {
        bsc_col_ptr[i + 1] += bsc_col_ptr[i];
    }

    // Fill row indices and values
    for(rocsparse_int i = 0; i < mb; ++i)
    {
        rocsparse_int row_begin = bsr_row_ptr[i] - bsr_base;
        rocsparse_int row_end   = bsr_row_ptr[i + 1] - bsr_base;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col = bsr_col_ind[j] - bsr_base;
            rocsparse_int idx = bsc_col_ptr[col];

            bsc_row_ind[idx] = i + bsc_base;

            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
                {
                    bsc_val[bsr_dim * bsr_dim * idx + bi + bj * bsr_dim]
                        = bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj];
                }
            }

            ++bsc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(rocsparse_int i = nb; i > 0; --i)
    {
        bsc_col_ptr[i] = bsc_col_ptr[i - 1] + bsc_base;
    }

    bsc_col_ptr[0] = bsc_base;
}

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
                           T                                 threshold)
{
    csr_row_ptr_C.resize(M + 1, 0);
    csr_row_ptr_C[0] = csr_base_C;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; i++)
    {
        for(rocsparse_int j = csr_row_ptr_A[i] - csr_base_A; j < csr_row_ptr_A[i + 1] - csr_base_A;
            j++)
        {
            if(std::abs(csr_val_A[j]) > threshold
               && std::abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                csr_row_ptr_C[i + 1]++;
            }
        }
    }

    for(rocsparse_int i = 1; i <= M; i++)
    {
        csr_row_ptr_C[i] += csr_row_ptr_C[i - 1];
    }

    nnz_C = csr_row_ptr_C[M] - csr_row_ptr_C[0];

    csr_col_ind_C.resize(nnz_C);
    csr_val_C.resize(nnz_C);

    rocsparse_int index = 0;
    for(rocsparse_int i = 0; i < M; i++)
    {
        for(rocsparse_int j = csr_row_ptr_A[i] - csr_base_A; j < csr_row_ptr_A[i + 1] - csr_base_A;
            j++)
        {
            if(std::abs(csr_val_A[j]) > threshold
               && std::abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                csr_col_ind_C[index] = (csr_col_ind_A[j] - csr_base_A) + csr_base_C;
                csr_val_C[index]     = csr_val_A[j];

                index++;
            }
        }
    }
}

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
                                         T                                 percentage)
{
    rocsparse_int pos = std::ceil(nnz_A * (percentage / 100)) - 1;
    pos               = std::min(pos, nnz_A - 1);
    pos               = std::max(pos, 0);

    std::vector<T> sorted_A(nnz_A);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < nnz_A; i++)
    {
        sorted_A[i] = std::abs(csr_val_A[i]);
    }

    std::sort(sorted_A.begin(), sorted_A.end());

    T threshold = nnz_A != 0 ? sorted_A[pos] : static_cast<T>(0);

    host_prune_csr_to_csr<T>(M,
                             N,
                             nnz_A,
                             csr_row_ptr_A,
                             csr_col_ind_A,
                             csr_val_A,
                             nnz_C,
                             csr_row_ptr_C,
                             csr_col_ind_C,
                             csr_val_C,
                             csr_base_A,
                             csr_base_C,
                             threshold);
}

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
void host_coosort_by_column(rocsparse_int               M,
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

// INSTANTIATE

template struct rocsparse_host<float, int32_t, int32_t>;
template struct rocsparse_host<double, int32_t, int32_t>;
template struct rocsparse_host<rocsparse_float_complex, int32_t, int32_t>;
template struct rocsparse_host<rocsparse_double_complex, int32_t, int32_t>;

template struct rocsparse_host<float, int64_t, int32_t>;
template struct rocsparse_host<double, int64_t, int32_t>;
template struct rocsparse_host<rocsparse_float_complex, int64_t, int32_t>;
template struct rocsparse_host<rocsparse_double_complex, int64_t, int32_t>;

template struct rocsparse_host<float, int64_t, int64_t>;
template struct rocsparse_host<double, int64_t, int64_t>;
template struct rocsparse_host<rocsparse_float_complex, int64_t, int64_t>;
template struct rocsparse_host<rocsparse_double_complex, int64_t, int64_t>;

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template void host_gthrz(rocsparse_int        nnz,
                         float*               y,
                         float*               x_val,
                         const rocsparse_int* x_ind,
                         rocsparse_index_base base);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template void host_bsrmv(rocsparse_direction  dir,
                         rocsparse_operation  trans,
                         rocsparse_int        mb,
                         rocsparse_int        nb,
                         rocsparse_int        nnzb,
                         float                alpha,
                         const rocsparse_int* bsr_row_ptr,
                         const rocsparse_int* bsr_col_ind,
                         const float*         bsr_val,
                         rocsparse_int        bsr_dim,
                         const float*         x,
                         float                beta,
                         float*               y,
                         rocsparse_index_base base);

template void host_bsrxmv(rocsparse_direction  dir,
                          rocsparse_operation  trans,
                          rocsparse_int        size_of_mask,
                          rocsparse_int        mb,
                          rocsparse_int        nb,
                          rocsparse_int        nnzb,
                          float                alpha,
                          const rocsparse_int* bsr_mask_ptr,
                          const rocsparse_int* bsr_row_ptr,
                          const rocsparse_int* bsr_end_ptr,
                          const rocsparse_int* bsr_col_ind,
                          const float*         bsr_val,
                          rocsparse_int        bsr_dim,
                          const float*         x,
                          float                beta,
                          float*               y,
                          rocsparse_index_base base);

template void host_bsrsv(rocsparse_operation  trans,
                         rocsparse_direction  dir,
                         rocsparse_int        mb,
                         rocsparse_int        nnzb,
                         float                alpha,
                         const rocsparse_int* bsr_row_ptr,
                         const rocsparse_int* bsr_col_ind,
                         const float*         bsr_val,
                         rocsparse_int        bsr_dim,
                         const float*         x,
                         float*               y,
                         rocsparse_diag_type  diag_type,
                         rocsparse_fill_mode  fill_mode,
                         rocsparse_index_base base,
                         rocsparse_int*       struct_pivot,
                         rocsparse_int*       numeric_pivot);

template void host_hybmv(rocsparse_operation  trans,
                         rocsparse_int        M,
                         rocsparse_int        N,
                         float                alpha,
                         rocsparse_int        ell_nnz,
                         const rocsparse_int* ell_col_ind,
                         const float*         ell_val,
                         rocsparse_int        ell_width,
                         rocsparse_int        coo_nnz,
                         const rocsparse_int* coo_row_ind,
                         const rocsparse_int* coo_col_ind,
                         const float*         coo_val,
                         const float*         x,
                         float                beta,
                         float*               y,
                         rocsparse_index_base base);

template void host_gebsrmv(rocsparse_direction  dir,
                           rocsparse_operation  trans,
                           rocsparse_int        mb,
                           rocsparse_int        nb,
                           rocsparse_int        nnzb,
                           float                alpha,
                           const rocsparse_int* bsr_row_ptr,
                           const rocsparse_int* bsr_col_ind,
                           const float*         bsr_val,
                           rocsparse_int        row_block_dim,
                           rocsparse_int        col_block_dim,
                           const float*         x,
                           float                beta,
                           float*               y,
                           rocsparse_index_base base);

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template void host_bsrmm(rocsparse_handle          handle,
                         rocsparse_direction       dir,
                         rocsparse_operation       transA,
                         rocsparse_operation       transB,
                         rocsparse_int             Mb,
                         rocsparse_int             N,
                         rocsparse_int             Kb,
                         rocsparse_int             nnzb,
                         const float*              alpha,
                         const rocsparse_mat_descr descr,
                         const float*              bsr_val_A,
                         const rocsparse_int*      bsr_row_ptr_A,
                         const rocsparse_int*      bsr_col_ind_A,
                         rocsparse_int             block_dim,
                         const float*              B,
                         rocsparse_int             ldb,
                         const float*              beta,
                         float*                    C,
                         rocsparse_int             ldc);

template void host_gebsrmm(rocsparse_handle          handle,
                           rocsparse_direction       dir,
                           rocsparse_operation       trans_A,
                           rocsparse_operation       trans_B,
                           rocsparse_int             mb,
                           rocsparse_int             n,
                           rocsparse_int             kb,
                           rocsparse_int             nnzb,
                           const float*              alpha,
                           const rocsparse_mat_descr descr,
                           const float*              bsr_val,
                           const rocsparse_int*      bsr_row_ptr,
                           const rocsparse_int*      bsr_col_ind,
                           rocsparse_int             row_block_dim,
                           rocsparse_int             col_block_dim,
                           const float*              B,
                           rocsparse_int             ldb,
                           const float*              beta,
                           float*                    C,
                           rocsparse_int             ldc);

template void host_gebsrmm(rocsparse_handle          handle,
                           rocsparse_direction       dir,
                           rocsparse_operation       trans_A,
                           rocsparse_operation       trans_B,
                           rocsparse_int             mb,
                           rocsparse_int             n,
                           rocsparse_int             kb,
                           rocsparse_int             nnzb,
                           const double*             alpha,
                           const rocsparse_mat_descr descr,
                           const double*             bsr_val,
                           const rocsparse_int*      bsr_row_ptr,
                           const rocsparse_int*      bsr_col_ind,
                           rocsparse_int             row_block_dim,
                           rocsparse_int             col_block_dim,
                           const double*             B,
                           rocsparse_int             ldb,
                           const double*             beta,
                           double*                   C,
                           rocsparse_int             ldc);

template void host_gebsrmm(rocsparse_handle               handle,
                           rocsparse_direction            dir,
                           rocsparse_operation            trans_A,
                           rocsparse_operation            trans_B,
                           rocsparse_int                  mb,
                           rocsparse_int                  n,
                           rocsparse_int                  kb,
                           rocsparse_int                  nnzb,
                           const rocsparse_float_complex* alpha,
                           const rocsparse_mat_descr      descr,
                           const rocsparse_float_complex* bsr_val,
                           const rocsparse_int*           bsr_row_ptr,
                           const rocsparse_int*           bsr_col_ind,
                           rocsparse_int                  row_block_dim,
                           rocsparse_int                  col_block_dim,
                           const rocsparse_float_complex* B,
                           rocsparse_int                  ldb,
                           const rocsparse_float_complex* beta,
                           rocsparse_float_complex*       C,
                           rocsparse_int                  ldc);

template void host_gebsrmm(rocsparse_handle                handle,
                           rocsparse_direction             dir,
                           rocsparse_operation             trans_A,
                           rocsparse_operation             trans_B,
                           rocsparse_int                   mb,
                           rocsparse_int                   n,
                           rocsparse_int                   kb,
                           rocsparse_int                   nnzb,
                           const rocsparse_double_complex* alpha,
                           const rocsparse_mat_descr       descr,
                           const rocsparse_double_complex* bsr_val,
                           const rocsparse_int*            bsr_row_ptr,
                           const rocsparse_int*            bsr_col_ind,
                           rocsparse_int                   row_block_dim,
                           rocsparse_int                   col_block_dim,
                           const rocsparse_double_complex* B,
                           rocsparse_int                   ldb,
                           const rocsparse_double_complex* beta,
                           rocsparse_double_complex*       C,
                           rocsparse_int                   ldc);

template void host_bsrsm(rocsparse_int        mb,
                         rocsparse_int        nrhs,
                         rocsparse_int        nnzb,
                         rocsparse_direction  dir,
                         rocsparse_operation  transA,
                         rocsparse_operation  transX,
                         float                alpha,
                         const rocsparse_int* bsr_row_ptr,
                         const rocsparse_int* bsr_col_ind,
                         const float*         bsr_val,
                         rocsparse_int        bsr_dim,
                         const float*         B,
                         rocsparse_int        ldb,
                         float*               X,
                         rocsparse_int        ldx,
                         rocsparse_diag_type  diag_type,
                         rocsparse_fill_mode  fill_mode,
                         rocsparse_index_base base,
                         rocsparse_int*       struct_pivot,
                         rocsparse_int*       numeric_pivot);
template void host_gemmi(rocsparse_int        M,
                         rocsparse_int        N,
                         rocsparse_operation  transA,
                         rocsparse_operation  transB,
                         float                alpha,
                         const float*         A,
                         rocsparse_int        lda,
                         const rocsparse_int* csr_row_ptr,
                         const rocsparse_int* csr_col_ind,
                         const float*         csr_val,
                         float                beta,
                         float*               C,
                         rocsparse_int        ldc,
                         rocsparse_index_base base);

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template void host_csrgeam_nnz(rocsparse_int                     M,
                               rocsparse_int                     N,
                               float                             alpha,
                               const std::vector<rocsparse_int>& csr_row_ptr_A,
                               const std::vector<rocsparse_int>& csr_col_ind_A,
                               float                             beta,
                               const std::vector<rocsparse_int>& csr_row_ptr_B,
                               const std::vector<rocsparse_int>& csr_col_ind_B,
                               std::vector<rocsparse_int>&       csr_row_ptr_C,
                               rocsparse_int*                    nnz_C,
                               rocsparse_index_base              base_A,
                               rocsparse_index_base              base_B,
                               rocsparse_index_base              base_C);

template void host_csrgeam(rocsparse_int                     M,
                           rocsparse_int                     N,
                           float                             alpha,
                           const std::vector<rocsparse_int>& csr_row_ptr_A,
                           const std::vector<rocsparse_int>& csr_col_ind_A,
                           const std::vector<float>&         csr_val_A,
                           float                             beta,
                           const std::vector<rocsparse_int>& csr_row_ptr_B,
                           const std::vector<rocsparse_int>& csr_col_ind_B,
                           const std::vector<float>&         csr_val_B,
                           const std::vector<rocsparse_int>& csr_row_ptr_C,
                           std::vector<rocsparse_int>&       csr_col_ind_C,
                           std::vector<float>&               csr_val_C,
                           rocsparse_index_base              base_A,
                           rocsparse_index_base              base_B,
                           rocsparse_index_base              base_C);

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template void host_bsric0(rocsparse_direction               direction,
                          rocsparse_int                     Mb,
                          rocsparse_int                     block_dim,
                          const std::vector<rocsparse_int>& bsr_row_ptr,
                          const std::vector<rocsparse_int>& bsr_col_ind,
                          std::vector<float>&               bsr_val,
                          rocsparse_index_base              base,
                          rocsparse_int*                    struct_pivot,
                          rocsparse_int*                    numeric_pivot);

template void host_bsrilu0(rocsparse_direction               dir,
                           rocsparse_int                     mb,
                           const std::vector<rocsparse_int>& bsr_row_ptr,
                           const std::vector<rocsparse_int>& bsr_col_ind,
                           std::vector<float>&               bsr_val,
                           rocsparse_int                     bsr_dim,
                           rocsparse_index_base              base,
                           rocsparse_int*                    struct_pivot,
                           rocsparse_int*                    numeric_pivot,
                           bool                              boost,
                           float                             boost_tol,
                           float                             boost_val);

template void host_csric0(rocsparse_int                     M,
                          const std::vector<rocsparse_int>& csr_row_ptr,
                          const std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<float>&               csr_val,
                          rocsparse_index_base              base,
                          rocsparse_int*                    struct_pivot,
                          rocsparse_int*                    numeric_pivot);

template void host_csrilu0(rocsparse_int                     M,
                           const std::vector<rocsparse_int>& csr_row_ptr,
                           const std::vector<rocsparse_int>& csr_col_ind,
                           std::vector<float>&               csr_val,
                           rocsparse_index_base              base,
                           rocsparse_int*                    struct_pivot,
                           rocsparse_int*                    numeric_pivot,
                           bool                              boost,
                           float                             boost_tol,
                           float                             boost_val);

template void host_gtsv_no_pivot(rocsparse_int             m,
                                 rocsparse_int             n,
                                 const std::vector<float>& dl,
                                 const std::vector<float>& d,
                                 const std::vector<float>& du,
                                 std::vector<float>&       B,
                                 rocsparse_int             ldb);

template void host_gtsv_no_pivot_strided_batch(rocsparse_int             m,
                                               const std::vector<float>& dl,
                                               const std::vector<float>& d,
                                               const std::vector<float>& du,
                                               std::vector<float>&       x,
                                               rocsparse_int             batch_count,
                                               rocsparse_int             batch_stride);

template void host_gtsv_interleaved_batch(rocsparse_gtsv_interleaved_alg algo,
                                          rocsparse_int                  m,
                                          const float*                   dl,
                                          const float*                   d,
                                          const float*                   du,
                                          float*                         x,
                                          rocsparse_int                  batch_count,
                                          rocsparse_int                  batch_stride);

template void host_gpsv_interleaved_batch(rocsparse_gpsv_interleaved_alg algo,
                                          rocsparse_int                  m,
                                          float*                         ds,
                                          float*                         dl,
                                          float*                         d,
                                          float*                         du,
                                          float*                         dw,
                                          float*                         x,
                                          rocsparse_int                  batch_count,
                                          rocsparse_int                  batch_stride);

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template rocsparse_status host_nnz(rocsparse_direction dirA,
                                   rocsparse_int       m,
                                   rocsparse_int       n,
                                   const float*        A,
                                   rocsparse_int       lda,
                                   rocsparse_int*      nnz_per_row_columns,
                                   rocsparse_int*      nnz_total_dev_host_ptr);

template void host_prune_dense2csr(rocsparse_int               m,
                                   rocsparse_int               n,
                                   const std::vector<float>&   A,
                                   rocsparse_int               lda,
                                   rocsparse_index_base        base,
                                   float                       threshold,
                                   rocsparse_int&              nnz,
                                   std::vector<float>&         csr_val,
                                   std::vector<rocsparse_int>& csr_row_ptr,
                                   std::vector<rocsparse_int>& csr_col_ind);

template void host_prune_dense2csr_by_percentage(rocsparse_int               m,
                                                 rocsparse_int               n,
                                                 const std::vector<float>&   A,
                                                 rocsparse_int               lda,
                                                 rocsparse_index_base        base,
                                                 float                       percentage,
                                                 rocsparse_int&              nnz,
                                                 std::vector<float>&         csr_val,
                                                 std::vector<rocsparse_int>& csr_row_ptr,
                                                 std::vector<rocsparse_int>& csr_col_ind);

template void host_csr_to_gebsr(rocsparse_direction               direction,
                                rocsparse_int                     m,
                                rocsparse_int                     n,
                                rocsparse_int                     nnz,
                                const std::vector<float>&         csr_val,
                                const std::vector<rocsparse_int>& csr_row_ptr,
                                const std::vector<rocsparse_int>& csr_col_ind,
                                rocsparse_int                     row_block_dim,
                                rocsparse_int                     col_block_dim,
                                rocsparse_index_base              csr_base,
                                std::vector<float>&               bsr_val,
                                std::vector<rocsparse_int>&       bsr_row_ptr,
                                std::vector<rocsparse_int>&       bsr_col_ind,
                                rocsparse_index_base              bsr_base);

template void host_gebsr_to_gebsc(rocsparse_int                     Mb,
                                  rocsparse_int                     Nb,
                                  rocsparse_int                     nnzb,
                                  const std::vector<rocsparse_int>& bsr_row_ptr,
                                  const std::vector<rocsparse_int>& bsr_col_ind,
                                  const std::vector<float>&         bsr_val,
                                  rocsparse_int                     row_block_dim,
                                  rocsparse_int                     col_block_dim,
                                  std::vector<rocsparse_int>&       bsc_row_ind,
                                  std::vector<rocsparse_int>&       bsc_col_ptr,
                                  std::vector<float>&               bsc_val,
                                  rocsparse_action                  action,
                                  rocsparse_index_base              base);

template void host_gebsr_to_csr(rocsparse_direction               direction,
                                rocsparse_int                     mb,
                                rocsparse_int                     nb,
                                rocsparse_int                     nnzb,
                                const std::vector<float>&         bsr_val,
                                const std::vector<rocsparse_int>& bsr_row_ptr,
                                const std::vector<rocsparse_int>& bsr_col_ind,
                                rocsparse_int                     row_block_dim,
                                rocsparse_int                     col_block_dim,
                                rocsparse_index_base              bsr_base,
                                std::vector<float>&               csr_val,
                                std::vector<rocsparse_int>&       csr_row_ptr,
                                std::vector<rocsparse_int>&       csr_col_ind,
                                rocsparse_index_base              csr_base);

template void host_gebsr_to_gebsr(rocsparse_direction               direction,
                                  rocsparse_int                     mb,
                                  rocsparse_int                     nb,
                                  rocsparse_int                     nnzb,
                                  const std::vector<float>&         bsr_val_A,
                                  const std::vector<rocsparse_int>& bsr_row_ptr_A,
                                  const std::vector<rocsparse_int>& bsr_col_ind_A,
                                  rocsparse_int                     row_block_dim_A,
                                  rocsparse_int                     col_block_dim_A,
                                  rocsparse_index_base              base_A,
                                  std::vector<float>&               bsr_val_C,
                                  std::vector<rocsparse_int>&       bsr_row_ptr_C,
                                  std::vector<rocsparse_int>&       bsr_col_ind_C,
                                  rocsparse_int                     row_block_dim_C,
                                  rocsparse_int                     col_block_dim_C,
                                  rocsparse_index_base              base_C);

template void host_bsr_to_bsc(rocsparse_int               mb,
                              rocsparse_int               nb,
                              rocsparse_int               nnzb,
                              rocsparse_int               bsr_dim,
                              const rocsparse_int*        bsr_row_ptr,
                              const rocsparse_int*        bsr_col_ind,
                              const float*                bsr_val,
                              std::vector<rocsparse_int>& bsc_row_ind,
                              std::vector<rocsparse_int>& bsc_col_ptr,
                              std::vector<float>&         bsc_val,
                              rocsparse_index_base        bsr_base,
                              rocsparse_index_base        bsc_base);

template void host_csr_to_hyb(rocsparse_int                     M,
                              rocsparse_int                     nnz,
                              const std::vector<rocsparse_int>& csr_row_ptr,
                              const std::vector<rocsparse_int>& csr_col_ind,
                              const std::vector<float>&         csr_val,
                              std::vector<rocsparse_int>&       ell_col_ind,
                              std::vector<float>&               ell_val,
                              rocsparse_int&                    ell_width,
                              rocsparse_int&                    ell_nnz,
                              std::vector<rocsparse_int>&       coo_row_ind,
                              std::vector<rocsparse_int>&       coo_col_ind,
                              std::vector<float>&               coo_val,
                              rocsparse_int&                    coo_nnz,
                              rocsparse_hyb_partition           part,
                              rocsparse_index_base              base);

template void host_csr_to_csr_compress(rocsparse_int                     M,
                                       rocsparse_int                     N,
                                       rocsparse_int                     nnz,
                                       const std::vector<rocsparse_int>& csr_row_ptr_A,
                                       const std::vector<rocsparse_int>& csr_col_ind_A,
                                       const std::vector<float>&         csr_val_A,
                                       std::vector<rocsparse_int>&       csr_row_ptr_C,
                                       std::vector<rocsparse_int>&       csr_col_ind_C,
                                       std::vector<float>&               csr_val_C,
                                       rocsparse_index_base              base,
                                       float                             tol);
template void host_prune_csr_to_csr(rocsparse_int                     M,
                                    rocsparse_int                     N,
                                    rocsparse_int                     nnz_A,
                                    const std::vector<rocsparse_int>& csr_row_ptr_A,
                                    const std::vector<rocsparse_int>& csr_col_ind_A,
                                    const std::vector<float>&         csr_val_A,
                                    rocsparse_int&                    nnz_C,
                                    std::vector<rocsparse_int>&       csr_row_ptr_C,
                                    std::vector<rocsparse_int>&       csr_col_ind_C,
                                    std::vector<float>&               csr_val_C,
                                    rocsparse_index_base              csr_base_A,
                                    rocsparse_index_base              csr_base_C,
                                    float                             threshold);

template void host_prune_csr_to_csr_by_percentage(rocsparse_int                     M,
                                                  rocsparse_int                     N,
                                                  rocsparse_int                     nnz_A,
                                                  const std::vector<rocsparse_int>& csr_row_ptr_A,
                                                  const std::vector<rocsparse_int>& csr_col_ind_A,
                                                  const std::vector<float>&         csr_val_A,
                                                  rocsparse_int&                    nnz_C,
                                                  std::vector<rocsparse_int>&       csr_row_ptr_C,
                                                  std::vector<rocsparse_int>&       csr_col_ind_C,
                                                  std::vector<float>&               csr_val_C,
                                                  rocsparse_index_base              csr_base_A,
                                                  rocsparse_index_base              csr_base_C,
                                                  float                             percentage);

template void host_ell_to_csr(rocsparse_int                     M,
                              rocsparse_int                     N,
                              const std::vector<rocsparse_int>& ell_col_ind,
                              const std::vector<float>&         ell_val,
                              rocsparse_int                     ell_width,
                              std::vector<rocsparse_int>&       csr_row_ptr,
                              std::vector<rocsparse_int>&       csr_col_ind,
                              std::vector<float>&               csr_val,
                              rocsparse_int&                    csr_nnz,
                              rocsparse_index_base              ell_base,
                              rocsparse_index_base              csr_base);

template void host_coosort_by_column(rocsparse_int               M,
                                     rocsparse_int               nnz,
                                     std::vector<rocsparse_int>& coo_row_ind,
                                     std::vector<rocsparse_int>& coo_col_ind,
                                     std::vector<float>&         coo_val);

// DOUBLE

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template void host_gthrz(rocsparse_int        nnz,
                         double*              y,
                         double*              x_val,
                         const rocsparse_int* x_ind,
                         rocsparse_index_base base);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template void host_bsrmv(rocsparse_direction  dir,
                         rocsparse_operation  trans,
                         rocsparse_int        mb,
                         rocsparse_int        nb,
                         rocsparse_int        nnzb,
                         double               alpha,
                         const rocsparse_int* bsr_row_ptr,
                         const rocsparse_int* bsr_col_ind,
                         const double*        bsr_val,
                         rocsparse_int        bsr_dim,
                         const double*        x,
                         double               beta,
                         double*              y,
                         rocsparse_index_base base);

template void host_bsrxmv(rocsparse_direction  dir,
                          rocsparse_operation  trans,
                          rocsparse_int        size_of_mask,
                          rocsparse_int        mb,
                          rocsparse_int        nb,
                          rocsparse_int        nnzb,
                          double               alpha,
                          const rocsparse_int* bsr_mask_ptr,
                          const rocsparse_int* bsr_row_ptr,
                          const rocsparse_int* bsr_end_ptr,
                          const rocsparse_int* bsr_col_ind,
                          const double*        bsr_val,
                          rocsparse_int        bsr_dim,
                          const double*        x,
                          double               beta,
                          double*              y,
                          rocsparse_index_base base);

template void host_bsrsv(rocsparse_operation  trans,
                         rocsparse_direction  dir,
                         rocsparse_int        mb,
                         rocsparse_int        nnzb,
                         double               alpha,
                         const rocsparse_int* bsr_row_ptr,
                         const rocsparse_int* bsr_col_ind,
                         const double*        bsr_val,
                         rocsparse_int        bsr_dim,
                         const double*        x,
                         double*              y,
                         rocsparse_diag_type  diag_type,
                         rocsparse_fill_mode  fill_mode,
                         rocsparse_index_base base,
                         rocsparse_int*       struct_pivot,
                         rocsparse_int*       numeric_pivot);

template void host_hybmv(rocsparse_operation  trans,
                         rocsparse_int        M,
                         rocsparse_int        N,
                         double               alpha,
                         rocsparse_int        ell_nnz,
                         const rocsparse_int* ell_col_ind,
                         const double*        ell_val,
                         rocsparse_int        ell_width,
                         rocsparse_int        coo_nnz,
                         const rocsparse_int* coo_row_ind,
                         const rocsparse_int* coo_col_ind,
                         const double*        coo_val,
                         const double*        x,
                         double               beta,
                         double*              y,
                         rocsparse_index_base base);

template void host_gebsrmv(rocsparse_direction  dir,
                           rocsparse_operation  trans,
                           rocsparse_int        mb,
                           rocsparse_int        nb,
                           rocsparse_int        nnzb,
                           double               alpha,
                           const rocsparse_int* bsr_row_ptr,
                           const rocsparse_int* bsr_col_ind,
                           const double*        bsr_val,
                           rocsparse_int        row_block_dim,
                           rocsparse_int        col_block_dim,
                           const double*        x,
                           double               beta,
                           double*              y,
                           rocsparse_index_base base);

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template void host_bsrmm(rocsparse_handle          handle,
                         rocsparse_direction       dir,
                         rocsparse_operation       transA,
                         rocsparse_operation       transB,
                         rocsparse_int             Mb,
                         rocsparse_int             N,
                         rocsparse_int             Kb,
                         rocsparse_int             nnzb,
                         const double*             alpha,
                         const rocsparse_mat_descr descr,
                         const double*             bsr_val_A,
                         const rocsparse_int*      bsr_row_ptr_A,
                         const rocsparse_int*      bsr_col_ind_A,
                         rocsparse_int             block_dim,
                         const double*             B,
                         rocsparse_int             ldb,
                         const double*             beta,
                         double*                   C,
                         rocsparse_int             ldc);

template void host_bsrsm(rocsparse_int        mb,
                         rocsparse_int        nrhs,
                         rocsparse_int        nnzb,
                         rocsparse_direction  dir,
                         rocsparse_operation  transA,
                         rocsparse_operation  transX,
                         double               alpha,
                         const rocsparse_int* bsr_row_ptr,
                         const rocsparse_int* bsr_col_ind,
                         const double*        bsr_val,
                         rocsparse_int        bsr_dim,
                         const double*        B,
                         rocsparse_int        ldb,
                         double*              X,
                         rocsparse_int        ldx,
                         rocsparse_diag_type  diag_type,
                         rocsparse_fill_mode  fill_mode,
                         rocsparse_index_base base,
                         rocsparse_int*       struct_pivot,
                         rocsparse_int*       numeric_pivot);
template void host_gemmi(rocsparse_int        M,
                         rocsparse_int        N,
                         rocsparse_operation  transA,
                         rocsparse_operation  transB,
                         double               alpha,
                         const double*        A,
                         rocsparse_int        lda,
                         const rocsparse_int* csr_row_ptr,
                         const rocsparse_int* csr_col_ind,
                         const double*        csr_val,
                         double               beta,
                         double*              C,
                         rocsparse_int        ldc,
                         rocsparse_index_base base);

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template void host_csrgeam_nnz(rocsparse_int                     M,
                               rocsparse_int                     N,
                               double                            alpha,
                               const std::vector<rocsparse_int>& csr_row_ptr_A,
                               const std::vector<rocsparse_int>& csr_col_ind_A,
                               double                            beta,
                               const std::vector<rocsparse_int>& csr_row_ptr_B,
                               const std::vector<rocsparse_int>& csr_col_ind_B,
                               std::vector<rocsparse_int>&       csr_row_ptr_C,
                               rocsparse_int*                    nnz_C,
                               rocsparse_index_base              base_A,
                               rocsparse_index_base              base_B,
                               rocsparse_index_base              base_C);

template void host_csrgeam(rocsparse_int                     M,
                           rocsparse_int                     N,
                           double                            alpha,
                           const std::vector<rocsparse_int>& csr_row_ptr_A,
                           const std::vector<rocsparse_int>& csr_col_ind_A,
                           const std::vector<double>&        csr_val_A,
                           double                            beta,
                           const std::vector<rocsparse_int>& csr_row_ptr_B,
                           const std::vector<rocsparse_int>& csr_col_ind_B,
                           const std::vector<double>&        csr_val_B,
                           const std::vector<rocsparse_int>& csr_row_ptr_C,
                           std::vector<rocsparse_int>&       csr_col_ind_C,
                           std::vector<double>&              csr_val_C,
                           rocsparse_index_base              base_A,
                           rocsparse_index_base              base_B,
                           rocsparse_index_base              base_C);

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template void host_bsric0(rocsparse_direction               direction,
                          rocsparse_int                     Mb,
                          rocsparse_int                     block_dim,
                          const std::vector<rocsparse_int>& bsr_row_ptr,
                          const std::vector<rocsparse_int>& bsr_col_ind,
                          std::vector<double>&              bsr_val,
                          rocsparse_index_base              base,
                          rocsparse_int*                    struct_pivot,
                          rocsparse_int*                    numeric_pivot);

template void host_bsrilu0(rocsparse_direction               dir,
                           rocsparse_int                     mb,
                           const std::vector<rocsparse_int>& bsr_row_ptr,
                           const std::vector<rocsparse_int>& bsr_col_ind,
                           std::vector<double>&              bsr_val,
                           rocsparse_int                     bsr_dim,
                           rocsparse_index_base              base,
                           rocsparse_int*                    struct_pivot,
                           rocsparse_int*                    numeric_pivot,
                           bool                              boost,
                           double                            boost_tol,
                           double                            boost_val);

template void host_csric0(rocsparse_int                     M,
                          const std::vector<rocsparse_int>& csr_row_ptr,
                          const std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<double>&              csr_val,
                          rocsparse_index_base              base,
                          rocsparse_int*                    struct_pivot,
                          rocsparse_int*                    numeric_pivot);

template void host_csrilu0(rocsparse_int                     M,
                           const std::vector<rocsparse_int>& csr_row_ptr,
                           const std::vector<rocsparse_int>& csr_col_ind,
                           std::vector<double>&              csr_val,
                           rocsparse_index_base              base,
                           rocsparse_int*                    struct_pivot,
                           rocsparse_int*                    numeric_pivot,
                           bool                              boost,
                           double                            boost_tol,
                           double                            boost_val);

template void host_gtsv_no_pivot(rocsparse_int              m,
                                 rocsparse_int              n,
                                 const std::vector<double>& dl,
                                 const std::vector<double>& d,
                                 const std::vector<double>& du,
                                 std::vector<double>&       B,
                                 rocsparse_int              ldb);

template void host_gtsv_no_pivot_strided_batch(rocsparse_int              m,
                                               const std::vector<double>& dl,
                                               const std::vector<double>& d,
                                               const std::vector<double>& du,
                                               std::vector<double>&       x,
                                               rocsparse_int              batch_count,
                                               rocsparse_int              batch_stride);

template void host_gtsv_interleaved_batch(rocsparse_gtsv_interleaved_alg algo,
                                          rocsparse_int                  m,
                                          const double*                  dl,
                                          const double*                  d,
                                          const double*                  du,
                                          double*                        x,
                                          rocsparse_int                  batch_count,
                                          rocsparse_int                  batch_stride);

template void host_gpsv_interleaved_batch(rocsparse_gpsv_interleaved_alg algo,
                                          rocsparse_int                  m,
                                          double*                        ds,
                                          double*                        dl,
                                          double*                        d,
                                          double*                        du,
                                          double*                        dw,
                                          double*                        x,
                                          rocsparse_int                  batch_count,
                                          rocsparse_int                  batch_stride);

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template rocsparse_status host_nnz(rocsparse_direction dirA,
                                   rocsparse_int       m,
                                   rocsparse_int       n,
                                   const double*       A,
                                   rocsparse_int       lda,
                                   rocsparse_int*      nnz_per_row_columns,
                                   rocsparse_int*      nnz_total_dev_host_ptr);

template void host_prune_dense2csr(rocsparse_int               m,
                                   rocsparse_int               n,
                                   const std::vector<double>&  A,
                                   rocsparse_int               lda,
                                   rocsparse_index_base        base,
                                   double                      threshold,
                                   rocsparse_int&              nnz,
                                   std::vector<double>&        csr_val,
                                   std::vector<rocsparse_int>& csr_row_ptr,
                                   std::vector<rocsparse_int>& csr_col_ind);

template void host_prune_dense2csr_by_percentage(rocsparse_int               m,
                                                 rocsparse_int               n,
                                                 const std::vector<double>&  A,
                                                 rocsparse_int               lda,
                                                 rocsparse_index_base        base,
                                                 double                      percentage,
                                                 rocsparse_int&              nnz,
                                                 std::vector<double>&        csr_val,
                                                 std::vector<rocsparse_int>& csr_row_ptr,
                                                 std::vector<rocsparse_int>& csr_col_ind);

template void host_csr_to_gebsr(rocsparse_direction               direction,
                                rocsparse_int                     m,
                                rocsparse_int                     n,
                                rocsparse_int                     nnz,
                                const std::vector<double>&        csr_val,
                                const std::vector<rocsparse_int>& csr_row_ptr,
                                const std::vector<rocsparse_int>& csr_col_ind,
                                rocsparse_int                     row_block_dim,
                                rocsparse_int                     col_block_dim,
                                rocsparse_index_base              csr_base,
                                std::vector<double>&              bsr_val,
                                std::vector<rocsparse_int>&       bsr_row_ptr,
                                std::vector<rocsparse_int>&       bsr_col_ind,
                                rocsparse_index_base              bsr_base);

template void host_gebsr_to_gebsc(rocsparse_int                     Mb,
                                  rocsparse_int                     Nb,
                                  rocsparse_int                     nnzb,
                                  const std::vector<rocsparse_int>& bsr_row_ptr,
                                  const std::vector<rocsparse_int>& bsr_col_ind,
                                  const std::vector<double>&        bsr_val,
                                  rocsparse_int                     row_block_dim,
                                  rocsparse_int                     col_block_dim,
                                  std::vector<rocsparse_int>&       bsc_row_ind,
                                  std::vector<rocsparse_int>&       bsc_col_ptr,
                                  std::vector<double>&              bsc_val,
                                  rocsparse_action                  action,
                                  rocsparse_index_base              base);

template void host_gebsr_to_csr(rocsparse_direction               direction,
                                rocsparse_int                     mb,
                                rocsparse_int                     nb,
                                rocsparse_int                     nnzb,
                                const std::vector<double>&        bsr_val,
                                const std::vector<rocsparse_int>& bsr_row_ptr,
                                const std::vector<rocsparse_int>& bsr_col_ind,
                                rocsparse_int                     row_block_dim,
                                rocsparse_int                     col_block_dim,
                                rocsparse_index_base              bsr_base,
                                std::vector<double>&              csr_val,
                                std::vector<rocsparse_int>&       csr_row_ptr,
                                std::vector<rocsparse_int>&       csr_col_ind,
                                rocsparse_index_base              csr_base);

template void host_gebsr_to_gebsr(rocsparse_direction               direction,
                                  rocsparse_int                     mb,
                                  rocsparse_int                     nb,
                                  rocsparse_int                     nnzb,
                                  const std::vector<double>&        bsr_val_A,
                                  const std::vector<rocsparse_int>& bsr_row_ptr_A,
                                  const std::vector<rocsparse_int>& bsr_col_ind_A,
                                  rocsparse_int                     row_block_dim_A,
                                  rocsparse_int                     col_block_dim_A,
                                  rocsparse_index_base              base_A,
                                  std::vector<double>&              bsr_val_C,
                                  std::vector<rocsparse_int>&       bsr_row_ptr_C,
                                  std::vector<rocsparse_int>&       bsr_col_ind_C,
                                  rocsparse_int                     row_block_dim_C,
                                  rocsparse_int                     col_block_dim_C,
                                  rocsparse_index_base              base_C);

template void host_bsr_to_bsc(rocsparse_int               mb,
                              rocsparse_int               nb,
                              rocsparse_int               nnzb,
                              rocsparse_int               bsr_dim,
                              const rocsparse_int*        bsr_row_ptr,
                              const rocsparse_int*        bsr_col_ind,
                              const double*               bsr_val,
                              std::vector<rocsparse_int>& bsc_row_ind,
                              std::vector<rocsparse_int>& bsc_col_ptr,
                              std::vector<double>&        bsc_val,
                              rocsparse_index_base        bsr_base,
                              rocsparse_index_base        bsc_base);

template void host_csr_to_hyb(rocsparse_int                     M,
                              rocsparse_int                     nnz,
                              const std::vector<rocsparse_int>& csr_row_ptr,
                              const std::vector<rocsparse_int>& csr_col_ind,
                              const std::vector<double>&        csr_val,
                              std::vector<rocsparse_int>&       ell_col_ind,
                              std::vector<double>&              ell_val,
                              rocsparse_int&                    ell_width,
                              rocsparse_int&                    ell_nnz,
                              std::vector<rocsparse_int>&       coo_row_ind,
                              std::vector<rocsparse_int>&       coo_col_ind,
                              std::vector<double>&              coo_val,
                              rocsparse_int&                    coo_nnz,
                              rocsparse_hyb_partition           part,
                              rocsparse_index_base              base);

template void host_csr_to_csr_compress(rocsparse_int                     M,
                                       rocsparse_int                     N,
                                       rocsparse_int                     nnz,
                                       const std::vector<rocsparse_int>& csr_row_ptr_A,
                                       const std::vector<rocsparse_int>& csr_col_ind_A,
                                       const std::vector<double>&        csr_val_A,
                                       std::vector<rocsparse_int>&       csr_row_ptr_C,
                                       std::vector<rocsparse_int>&       csr_col_ind_C,
                                       std::vector<double>&              csr_val_C,
                                       rocsparse_index_base              base,
                                       double                            tol);
template void host_prune_csr_to_csr(rocsparse_int                     M,
                                    rocsparse_int                     N,
                                    rocsparse_int                     nnz_A,
                                    const std::vector<rocsparse_int>& csr_row_ptr_A,
                                    const std::vector<rocsparse_int>& csr_col_ind_A,
                                    const std::vector<double>&        csr_val_A,
                                    rocsparse_int&                    nnz_C,
                                    std::vector<rocsparse_int>&       csr_row_ptr_C,
                                    std::vector<rocsparse_int>&       csr_col_ind_C,
                                    std::vector<double>&              csr_val_C,
                                    rocsparse_index_base              csr_base_A,
                                    rocsparse_index_base              csr_base_C,
                                    double                            threshold);

template void host_prune_csr_to_csr_by_percentage(rocsparse_int                     M,
                                                  rocsparse_int                     N,
                                                  rocsparse_int                     nnz_A,
                                                  const std::vector<rocsparse_int>& csr_row_ptr_A,
                                                  const std::vector<rocsparse_int>& csr_col_ind_A,
                                                  const std::vector<double>&        csr_val_A,
                                                  rocsparse_int&                    nnz_C,
                                                  std::vector<rocsparse_int>&       csr_row_ptr_C,
                                                  std::vector<rocsparse_int>&       csr_col_ind_C,
                                                  std::vector<double>&              csr_val_C,
                                                  rocsparse_index_base              csr_base_A,
                                                  rocsparse_index_base              csr_base_C,
                                                  double                            percentage);

template void host_ell_to_csr(rocsparse_int                     M,
                              rocsparse_int                     N,
                              const std::vector<rocsparse_int>& ell_col_ind,
                              const std::vector<double>&        ell_val,
                              rocsparse_int                     ell_width,
                              std::vector<rocsparse_int>&       csr_row_ptr,
                              std::vector<rocsparse_int>&       csr_col_ind,
                              std::vector<double>&              csr_val,
                              rocsparse_int&                    csr_nnz,
                              rocsparse_index_base              ell_base,
                              rocsparse_index_base              csr_base);

template void host_coosort_by_column(rocsparse_int               M,
                                     rocsparse_int               nnz,
                                     std::vector<rocsparse_int>& coo_row_ind,
                                     std::vector<rocsparse_int>& coo_col_ind,
                                     std::vector<double>&        coo_val);

// ROCSPARSE_DOUBLE_COMPLEX

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template void host_gthrz(rocsparse_int             nnz,
                         rocsparse_double_complex* y,
                         rocsparse_double_complex* x_val,
                         const rocsparse_int*      x_ind,
                         rocsparse_index_base      base);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template void host_bsrmv(rocsparse_direction             dir,
                         rocsparse_operation             trans,
                         rocsparse_int                   mb,
                         rocsparse_int                   nb,
                         rocsparse_int                   nnzb,
                         rocsparse_double_complex        alpha,
                         const rocsparse_int*            bsr_row_ptr,
                         const rocsparse_int*            bsr_col_ind,
                         const rocsparse_double_complex* bsr_val,
                         rocsparse_int                   bsr_dim,
                         const rocsparse_double_complex* x,
                         rocsparse_double_complex        beta,
                         rocsparse_double_complex*       y,
                         rocsparse_index_base            base);

template void host_bsrxmv(rocsparse_direction             dir,
                          rocsparse_operation             trans,
                          rocsparse_int                   size_of_mask,
                          rocsparse_int                   mb,
                          rocsparse_int                   nb,
                          rocsparse_int                   nnzb,
                          rocsparse_double_complex        alpha,
                          const rocsparse_int*            bsr_mask_ptr,
                          const rocsparse_int*            bsr_row_ptr,
                          const rocsparse_int*            bsr_end_ptr,
                          const rocsparse_int*            bsr_col_ind,
                          const rocsparse_double_complex* bsr_val,
                          rocsparse_int                   bsr_dim,
                          const rocsparse_double_complex* x,
                          rocsparse_double_complex        beta,
                          rocsparse_double_complex*       y,
                          rocsparse_index_base            base);

template void host_bsrsv(rocsparse_operation             trans,
                         rocsparse_direction             dir,
                         rocsparse_int                   mb,
                         rocsparse_int                   nnzb,
                         rocsparse_double_complex        alpha,
                         const rocsparse_int*            bsr_row_ptr,
                         const rocsparse_int*            bsr_col_ind,
                         const rocsparse_double_complex* bsr_val,
                         rocsparse_int                   bsr_dim,
                         const rocsparse_double_complex* x,
                         rocsparse_double_complex*       y,
                         rocsparse_diag_type             diag_type,
                         rocsparse_fill_mode             fill_mode,
                         rocsparse_index_base            base,
                         rocsparse_int*                  struct_pivot,
                         rocsparse_int*                  numeric_pivot);

template void host_hybmv(rocsparse_operation             trans,
                         rocsparse_int                   M,
                         rocsparse_int                   N,
                         rocsparse_double_complex        alpha,
                         rocsparse_int                   ell_nnz,
                         const rocsparse_int*            ell_col_ind,
                         const rocsparse_double_complex* ell_val,
                         rocsparse_int                   ell_width,
                         rocsparse_int                   coo_nnz,
                         const rocsparse_int*            coo_row_ind,
                         const rocsparse_int*            coo_col_ind,
                         const rocsparse_double_complex* coo_val,
                         const rocsparse_double_complex* x,
                         rocsparse_double_complex        beta,
                         rocsparse_double_complex*       y,
                         rocsparse_index_base            base);

template void host_gebsrmv(rocsparse_direction             dir,
                           rocsparse_operation             trans,
                           rocsparse_int                   mb,
                           rocsparse_int                   nb,
                           rocsparse_int                   nnzb,
                           rocsparse_double_complex        alpha,
                           const rocsparse_int*            bsr_row_ptr,
                           const rocsparse_int*            bsr_col_ind,
                           const rocsparse_double_complex* bsr_val,
                           rocsparse_int                   row_block_dim,
                           rocsparse_int                   col_block_dim,
                           const rocsparse_double_complex* x,
                           rocsparse_double_complex        beta,
                           rocsparse_double_complex*       y,
                           rocsparse_index_base            base);

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */

template void host_bsrmm(rocsparse_handle                handle,
                         rocsparse_direction             dir,
                         rocsparse_operation             transA,
                         rocsparse_operation             transB,
                         rocsparse_int                   Mb,
                         rocsparse_int                   N,
                         rocsparse_int                   Kb,
                         rocsparse_int                   nnzb,
                         const rocsparse_double_complex* alpha,
                         const rocsparse_mat_descr       descr,
                         const rocsparse_double_complex* bsr_val_A,
                         const rocsparse_int*            bsr_row_ptr_A,
                         const rocsparse_int*            bsr_col_ind_A,
                         rocsparse_int                   block_dim,
                         const rocsparse_double_complex* B,
                         rocsparse_int                   ldb,
                         const rocsparse_double_complex* beta,
                         rocsparse_double_complex*       C,
                         rocsparse_int                   ldc);

template void host_bsrsm(rocsparse_int                   mb,
                         rocsparse_int                   nrhs,
                         rocsparse_int                   nnzb,
                         rocsparse_direction             dir,
                         rocsparse_operation             transA,
                         rocsparse_operation             transX,
                         rocsparse_double_complex        alpha,
                         const rocsparse_int*            bsr_row_ptr,
                         const rocsparse_int*            bsr_col_ind,
                         const rocsparse_double_complex* bsr_val,
                         rocsparse_int                   bsr_dim,
                         const rocsparse_double_complex* B,
                         rocsparse_int                   ldb,
                         rocsparse_double_complex*       X,
                         rocsparse_int                   ldx,
                         rocsparse_diag_type             diag_type,
                         rocsparse_fill_mode             fill_mode,
                         rocsparse_index_base            base,
                         rocsparse_int*                  struct_pivot,
                         rocsparse_int*                  numeric_pivot);
template void host_gemmi(rocsparse_int                   M,
                         rocsparse_int                   N,
                         rocsparse_operation             transA,
                         rocsparse_operation             transB,
                         rocsparse_double_complex        alpha,
                         const rocsparse_double_complex* A,
                         rocsparse_int                   lda,
                         const rocsparse_int*            csr_row_ptr,
                         const rocsparse_int*            csr_col_ind,
                         const rocsparse_double_complex* csr_val,
                         rocsparse_double_complex        beta,
                         rocsparse_double_complex*       C,
                         rocsparse_int                   ldc,
                         rocsparse_index_base            base);

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template void host_csrgeam_nnz(rocsparse_int                     M,
                               rocsparse_int                     N,
                               rocsparse_double_complex          alpha,
                               const std::vector<rocsparse_int>& csr_row_ptr_A,
                               const std::vector<rocsparse_int>& csr_col_ind_A,
                               rocsparse_double_complex          beta,
                               const std::vector<rocsparse_int>& csr_row_ptr_B,
                               const std::vector<rocsparse_int>& csr_col_ind_B,
                               std::vector<rocsparse_int>&       csr_row_ptr_C,
                               rocsparse_int*                    nnz_C,
                               rocsparse_index_base              base_A,
                               rocsparse_index_base              base_B,
                               rocsparse_index_base              base_C);

template void host_csrgeam(rocsparse_int                                M,
                           rocsparse_int                                N,
                           rocsparse_double_complex                     alpha,
                           const std::vector<rocsparse_int>&            csr_row_ptr_A,
                           const std::vector<rocsparse_int>&            csr_col_ind_A,
                           const std::vector<rocsparse_double_complex>& csr_val_A,
                           rocsparse_double_complex                     beta,
                           const std::vector<rocsparse_int>&            csr_row_ptr_B,
                           const std::vector<rocsparse_int>&            csr_col_ind_B,
                           const std::vector<rocsparse_double_complex>& csr_val_B,
                           const std::vector<rocsparse_int>&            csr_row_ptr_C,
                           std::vector<rocsparse_int>&                  csr_col_ind_C,
                           std::vector<rocsparse_double_complex>&       csr_val_C,
                           rocsparse_index_base                         base_A,
                           rocsparse_index_base                         base_B,
                           rocsparse_index_base                         base_C);

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template void host_bsric0(rocsparse_direction                    direction,
                          rocsparse_int                          Mb,
                          rocsparse_int                          block_dim,
                          const std::vector<rocsparse_int>&      bsr_row_ptr,
                          const std::vector<rocsparse_int>&      bsr_col_ind,
                          std::vector<rocsparse_double_complex>& bsr_val,
                          rocsparse_index_base                   base,
                          rocsparse_int*                         struct_pivot,
                          rocsparse_int*                         numeric_pivot);

template void host_bsrilu0(rocsparse_direction                    dir,
                           rocsparse_int                          mb,
                           const std::vector<rocsparse_int>&      bsr_row_ptr,
                           const std::vector<rocsparse_int>&      bsr_col_ind,
                           std::vector<rocsparse_double_complex>& bsr_val,
                           rocsparse_int                          bsr_dim,
                           rocsparse_index_base                   base,
                           rocsparse_int*                         struct_pivot,
                           rocsparse_int*                         numeric_pivot,
                           bool                                   boost,
                           double                                 boost_tol,
                           rocsparse_double_complex               boost_val);

template void host_csric0(rocsparse_int                          M,
                          const std::vector<rocsparse_int>&      csr_row_ptr,
                          const std::vector<rocsparse_int>&      csr_col_ind,
                          std::vector<rocsparse_double_complex>& csr_val,
                          rocsparse_index_base                   base,
                          rocsparse_int*                         struct_pivot,
                          rocsparse_int*                         numeric_pivot);

template void host_csrilu0(rocsparse_int                          M,
                           const std::vector<rocsparse_int>&      csr_row_ptr,
                           const std::vector<rocsparse_int>&      csr_col_ind,
                           std::vector<rocsparse_double_complex>& csr_val,
                           rocsparse_index_base                   base,
                           rocsparse_int*                         struct_pivot,
                           rocsparse_int*                         numeric_pivot,
                           bool                                   boost,
                           double                                 boost_tol,
                           rocsparse_double_complex               boost_val);

template void host_gtsv_no_pivot(rocsparse_int                                m,
                                 rocsparse_int                                n,
                                 const std::vector<rocsparse_double_complex>& dl,
                                 const std::vector<rocsparse_double_complex>& d,
                                 const std::vector<rocsparse_double_complex>& du,
                                 std::vector<rocsparse_double_complex>&       B,
                                 rocsparse_int                                ldb);

template void host_gtsv_no_pivot_strided_batch(rocsparse_int                                m,
                                               const std::vector<rocsparse_double_complex>& dl,
                                               const std::vector<rocsparse_double_complex>& d,
                                               const std::vector<rocsparse_double_complex>& du,
                                               std::vector<rocsparse_double_complex>&       x,
                                               rocsparse_int batch_count,
                                               rocsparse_int batch_stride);

template void host_gtsv_interleaved_batch(rocsparse_gtsv_interleaved_alg  algo,
                                          rocsparse_int                   m,
                                          const rocsparse_double_complex* dl,
                                          const rocsparse_double_complex* d,
                                          const rocsparse_double_complex* du,
                                          rocsparse_double_complex*       x,
                                          rocsparse_int                   batch_count,
                                          rocsparse_int                   batch_stride);

template void host_gpsv_interleaved_batch(rocsparse_gpsv_interleaved_alg algo,
                                          rocsparse_int                  m,
                                          rocsparse_double_complex*      ds,
                                          rocsparse_double_complex*      dl,
                                          rocsparse_double_complex*      d,
                                          rocsparse_double_complex*      du,
                                          rocsparse_double_complex*      dw,
                                          rocsparse_double_complex*      x,
                                          rocsparse_int                  batch_count,
                                          rocsparse_int                  batch_stride);

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template rocsparse_status host_nnz(rocsparse_direction             dirA,
                                   rocsparse_int                   m,
                                   rocsparse_int                   n,
                                   const rocsparse_double_complex* A,
                                   rocsparse_int                   lda,
                                   rocsparse_int*                  nnz_per_row_columns,
                                   rocsparse_int*                  nnz_total_dev_host_ptr);

template void host_csr_to_gebsr(rocsparse_direction                          direction,
                                rocsparse_int                                m,
                                rocsparse_int                                n,
                                rocsparse_int                                nnz,
                                const std::vector<rocsparse_double_complex>& csr_val,
                                const std::vector<rocsparse_int>&            csr_row_ptr,
                                const std::vector<rocsparse_int>&            csr_col_ind,
                                rocsparse_int                                row_block_dim,
                                rocsparse_int                                col_block_dim,
                                rocsparse_index_base                         csr_base,
                                std::vector<rocsparse_double_complex>&       bsr_val,
                                std::vector<rocsparse_int>&                  bsr_row_ptr,
                                std::vector<rocsparse_int>&                  bsr_col_ind,
                                rocsparse_index_base                         bsr_base);

template void host_gebsr_to_gebsc(rocsparse_int                                Mb,
                                  rocsparse_int                                Nb,
                                  rocsparse_int                                nnzb,
                                  const std::vector<rocsparse_int>&            bsr_row_ptr,
                                  const std::vector<rocsparse_int>&            bsr_col_ind,
                                  const std::vector<rocsparse_double_complex>& bsr_val,
                                  rocsparse_int                                row_block_dim,
                                  rocsparse_int                                col_block_dim,
                                  std::vector<rocsparse_int>&                  bsc_row_ind,
                                  std::vector<rocsparse_int>&                  bsc_col_ptr,
                                  std::vector<rocsparse_double_complex>&       bsc_val,
                                  rocsparse_action                             action,
                                  rocsparse_index_base                         base);

template void host_gebsr_to_csr(rocsparse_direction                          direction,
                                rocsparse_int                                mb,
                                rocsparse_int                                nb,
                                rocsparse_int                                nnzb,
                                const std::vector<rocsparse_double_complex>& bsr_val,
                                const std::vector<rocsparse_int>&            bsr_row_ptr,
                                const std::vector<rocsparse_int>&            bsr_col_ind,
                                rocsparse_int                                row_block_dim,
                                rocsparse_int                                col_block_dim,
                                rocsparse_index_base                         bsr_base,
                                std::vector<rocsparse_double_complex>&       csr_val,
                                std::vector<rocsparse_int>&                  csr_row_ptr,
                                std::vector<rocsparse_int>&                  csr_col_ind,
                                rocsparse_index_base                         csr_base);

template void host_gebsr_to_gebsr(rocsparse_direction                          direction,
                                  rocsparse_int                                mb,
                                  rocsparse_int                                nb,
                                  rocsparse_int                                nnzb,
                                  const std::vector<rocsparse_double_complex>& bsr_val_A,
                                  const std::vector<rocsparse_int>&            bsr_row_ptr_A,
                                  const std::vector<rocsparse_int>&            bsr_col_ind_A,
                                  rocsparse_int                                row_block_dim_A,
                                  rocsparse_int                                col_block_dim_A,
                                  rocsparse_index_base                         base_A,
                                  std::vector<rocsparse_double_complex>&       bsr_val_C,
                                  std::vector<rocsparse_int>&                  bsr_row_ptr_C,
                                  std::vector<rocsparse_int>&                  bsr_col_ind_C,
                                  rocsparse_int                                row_block_dim_C,
                                  rocsparse_int                                col_block_dim_C,
                                  rocsparse_index_base                         base_C);

template void host_bsr_to_bsc(rocsparse_int                          mb,
                              rocsparse_int                          nb,
                              rocsparse_int                          nnzb,
                              rocsparse_int                          bsr_dim,
                              const rocsparse_int*                   bsr_row_ptr,
                              const rocsparse_int*                   bsr_col_ind,
                              const rocsparse_double_complex*        bsr_val,
                              std::vector<rocsparse_int>&            bsc_row_ind,
                              std::vector<rocsparse_int>&            bsc_col_ptr,
                              std::vector<rocsparse_double_complex>& bsc_val,
                              rocsparse_index_base                   bsr_base,
                              rocsparse_index_base                   bsc_base);

template void host_csr_to_hyb(rocsparse_int                                M,
                              rocsparse_int                                nnz,
                              const std::vector<rocsparse_int>&            csr_row_ptr,
                              const std::vector<rocsparse_int>&            csr_col_ind,
                              const std::vector<rocsparse_double_complex>& csr_val,
                              std::vector<rocsparse_int>&                  ell_col_ind,
                              std::vector<rocsparse_double_complex>&       ell_val,
                              rocsparse_int&                               ell_width,
                              rocsparse_int&                               ell_nnz,
                              std::vector<rocsparse_int>&                  coo_row_ind,
                              std::vector<rocsparse_int>&                  coo_col_ind,
                              std::vector<rocsparse_double_complex>&       coo_val,
                              rocsparse_int&                               coo_nnz,
                              rocsparse_hyb_partition                      part,
                              rocsparse_index_base                         base);

template void host_csr_to_csr_compress(rocsparse_int                                M,
                                       rocsparse_int                                N,
                                       rocsparse_int                                nnz,
                                       const std::vector<rocsparse_int>&            csr_row_ptr_A,
                                       const std::vector<rocsparse_int>&            csr_col_ind_A,
                                       const std::vector<rocsparse_double_complex>& csr_val_A,
                                       std::vector<rocsparse_int>&                  csr_row_ptr_C,
                                       std::vector<rocsparse_int>&                  csr_col_ind_C,
                                       std::vector<rocsparse_double_complex>&       csr_val_C,
                                       rocsparse_index_base                         base,
                                       rocsparse_double_complex                     tol);

template void host_ell_to_csr(rocsparse_int                                M,
                              rocsparse_int                                N,
                              const std::vector<rocsparse_int>&            ell_col_ind,
                              const std::vector<rocsparse_double_complex>& ell_val,
                              rocsparse_int                                ell_width,
                              std::vector<rocsparse_int>&                  csr_row_ptr,
                              std::vector<rocsparse_int>&                  csr_col_ind,
                              std::vector<rocsparse_double_complex>&       csr_val,
                              rocsparse_int&                               csr_nnz,
                              rocsparse_index_base                         ell_base,
                              rocsparse_index_base                         csr_base);

template void host_coosort_by_column(rocsparse_int                          M,
                                     rocsparse_int                          nnz,
                                     std::vector<rocsparse_int>&            coo_row_ind,
                                     std::vector<rocsparse_int>&            coo_col_ind,
                                     std::vector<rocsparse_double_complex>& coo_val);

// ROCSPARSE_FLOAT_COMPLEX

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template void host_gthrz(rocsparse_int            nnz,
                         rocsparse_float_complex* y,
                         rocsparse_float_complex* x_val,
                         const rocsparse_int*     x_ind,
                         rocsparse_index_base     base);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template void host_bsrmv(rocsparse_direction            dir,
                         rocsparse_operation            trans,
                         rocsparse_int                  mb,
                         rocsparse_int                  nb,
                         rocsparse_int                  nnzb,
                         rocsparse_float_complex        alpha,
                         const rocsparse_int*           bsr_row_ptr,
                         const rocsparse_int*           bsr_col_ind,
                         const rocsparse_float_complex* bsr_val,
                         rocsparse_int                  bsr_dim,
                         const rocsparse_float_complex* x,
                         rocsparse_float_complex        beta,
                         rocsparse_float_complex*       y,
                         rocsparse_index_base           base);

template void host_bsrxmv(rocsparse_direction            dir,
                          rocsparse_operation            trans,
                          rocsparse_int                  size_of_mask,
                          rocsparse_int                  mb,
                          rocsparse_int                  nb,
                          rocsparse_int                  nnzb,
                          rocsparse_float_complex        alpha,
                          const rocsparse_int*           bsr_mask_ptr,
                          const rocsparse_int*           bsr_row_ptr,
                          const rocsparse_int*           bsr_end_ptr,
                          const rocsparse_int*           bsr_col_ind,
                          const rocsparse_float_complex* bsr_val,
                          rocsparse_int                  bsr_dim,
                          const rocsparse_float_complex* x,
                          rocsparse_float_complex        beta,
                          rocsparse_float_complex*       y,
                          rocsparse_index_base           base);

template void host_bsrsv(rocsparse_operation            trans,
                         rocsparse_direction            dir,
                         rocsparse_int                  mb,
                         rocsparse_int                  nnzb,
                         rocsparse_float_complex        alpha,
                         const rocsparse_int*           bsr_row_ptr,
                         const rocsparse_int*           bsr_col_ind,
                         const rocsparse_float_complex* bsr_val,
                         rocsparse_int                  bsr_dim,
                         const rocsparse_float_complex* x,
                         rocsparse_float_complex*       y,
                         rocsparse_diag_type            diag_type,
                         rocsparse_fill_mode            fill_mode,
                         rocsparse_index_base           base,
                         rocsparse_int*                 struct_pivot,
                         rocsparse_int*                 numeric_pivot);

template void host_hybmv(rocsparse_operation            trans,
                         rocsparse_int                  M,
                         rocsparse_int                  N,
                         rocsparse_float_complex        alpha,
                         rocsparse_int                  ell_nnz,
                         const rocsparse_int*           ell_col_ind,
                         const rocsparse_float_complex* ell_val,
                         rocsparse_int                  ell_width,
                         rocsparse_int                  coo_nnz,
                         const rocsparse_int*           coo_row_ind,
                         const rocsparse_int*           coo_col_ind,
                         const rocsparse_float_complex* coo_val,
                         const rocsparse_float_complex* x,
                         rocsparse_float_complex        beta,
                         rocsparse_float_complex*       y,
                         rocsparse_index_base           base);

template void host_gebsrmv(rocsparse_direction            dir,
                           rocsparse_operation            trans,
                           rocsparse_int                  mb,
                           rocsparse_int                  nb,
                           rocsparse_int                  nnzb,
                           rocsparse_float_complex        alpha,
                           const rocsparse_int*           bsr_row_ptr,
                           const rocsparse_int*           bsr_col_ind,
                           const rocsparse_float_complex* bsr_val,
                           rocsparse_int                  row_block_dim,
                           rocsparse_int                  col_block_dim,
                           const rocsparse_float_complex* x,
                           rocsparse_float_complex        beta,
                           rocsparse_float_complex*       y,
                           rocsparse_index_base           base);

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */

template void host_bsrmm(rocsparse_handle               handle,
                         rocsparse_direction            dir,
                         rocsparse_operation            transA,
                         rocsparse_operation            transB,
                         rocsparse_int                  Mb,
                         rocsparse_int                  N,
                         rocsparse_int                  Kb,
                         rocsparse_int                  nnzb,
                         const rocsparse_float_complex* alpha,
                         const rocsparse_mat_descr      descr,
                         const rocsparse_float_complex* bsr_val_A,
                         const rocsparse_int*           bsr_row_ptr_A,
                         const rocsparse_int*           bsr_col_ind_A,
                         rocsparse_int                  block_dim,
                         const rocsparse_float_complex* B,
                         rocsparse_int                  ldb,
                         const rocsparse_float_complex* beta,
                         rocsparse_float_complex*       C,
                         rocsparse_int                  ldc);

template void host_bsrsm(rocsparse_int                  mb,
                         rocsparse_int                  nrhs,
                         rocsparse_int                  nnzb,
                         rocsparse_direction            dir,
                         rocsparse_operation            transA,
                         rocsparse_operation            transX,
                         rocsparse_float_complex        alpha,
                         const rocsparse_int*           bsr_row_ptr,
                         const rocsparse_int*           bsr_col_ind,
                         const rocsparse_float_complex* bsr_val,
                         rocsparse_int                  bsr_dim,
                         const rocsparse_float_complex* B,
                         rocsparse_int                  ldb,
                         rocsparse_float_complex*       X,
                         rocsparse_int                  ldx,
                         rocsparse_diag_type            diag_type,
                         rocsparse_fill_mode            fill_mode,
                         rocsparse_index_base           base,
                         rocsparse_int*                 struct_pivot,
                         rocsparse_int*                 numeric_pivot);
template void host_gemmi(rocsparse_int                  M,
                         rocsparse_int                  N,
                         rocsparse_operation            transA,
                         rocsparse_operation            transB,
                         rocsparse_float_complex        alpha,
                         const rocsparse_float_complex* A,
                         rocsparse_int                  lda,
                         const rocsparse_int*           csr_row_ptr,
                         const rocsparse_int*           csr_col_ind,
                         const rocsparse_float_complex* csr_val,
                         rocsparse_float_complex        beta,
                         rocsparse_float_complex*       C,
                         rocsparse_int                  ldc,
                         rocsparse_index_base           base);

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template void host_csrgeam_nnz(rocsparse_int                     M,
                               rocsparse_int                     N,
                               rocsparse_float_complex           alpha,
                               const std::vector<rocsparse_int>& csr_row_ptr_A,
                               const std::vector<rocsparse_int>& csr_col_ind_A,
                               rocsparse_float_complex           beta,
                               const std::vector<rocsparse_int>& csr_row_ptr_B,
                               const std::vector<rocsparse_int>& csr_col_ind_B,
                               std::vector<rocsparse_int>&       csr_row_ptr_C,
                               rocsparse_int*                    nnz_C,
                               rocsparse_index_base              base_A,
                               rocsparse_index_base              base_B,
                               rocsparse_index_base              base_C);

template void host_csrgeam(rocsparse_int                               M,
                           rocsparse_int                               N,
                           rocsparse_float_complex                     alpha,
                           const std::vector<rocsparse_int>&           csr_row_ptr_A,
                           const std::vector<rocsparse_int>&           csr_col_ind_A,
                           const std::vector<rocsparse_float_complex>& csr_val_A,
                           rocsparse_float_complex                     beta,
                           const std::vector<rocsparse_int>&           csr_row_ptr_B,
                           const std::vector<rocsparse_int>&           csr_col_ind_B,
                           const std::vector<rocsparse_float_complex>& csr_val_B,
                           const std::vector<rocsparse_int>&           csr_row_ptr_C,
                           std::vector<rocsparse_int>&                 csr_col_ind_C,
                           std::vector<rocsparse_float_complex>&       csr_val_C,
                           rocsparse_index_base                        base_A,
                           rocsparse_index_base                        base_B,
                           rocsparse_index_base                        base_C);

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template void host_bsric0(rocsparse_direction                   direction,
                          rocsparse_int                         Mb,
                          rocsparse_int                         block_dim,
                          const std::vector<rocsparse_int>&     bsr_row_ptr,
                          const std::vector<rocsparse_int>&     bsr_col_ind,
                          std::vector<rocsparse_float_complex>& bsr_val,
                          rocsparse_index_base                  base,
                          rocsparse_int*                        struct_pivot,
                          rocsparse_int*                        numeric_pivot);

template void host_bsrilu0(rocsparse_direction                   dir,
                           rocsparse_int                         mb,
                           const std::vector<rocsparse_int>&     bsr_row_ptr,
                           const std::vector<rocsparse_int>&     bsr_col_ind,
                           std::vector<rocsparse_float_complex>& bsr_val,
                           rocsparse_int                         bsr_dim,
                           rocsparse_index_base                  base,
                           rocsparse_int*                        struct_pivot,
                           rocsparse_int*                        numeric_pivot,
                           bool                                  boost,
                           float                                 boost_tol,
                           rocsparse_float_complex               boost_val);

template void host_csric0(rocsparse_int                         M,
                          const std::vector<rocsparse_int>&     csr_row_ptr,
                          const std::vector<rocsparse_int>&     csr_col_ind,
                          std::vector<rocsparse_float_complex>& csr_val,
                          rocsparse_index_base                  base,
                          rocsparse_int*                        struct_pivot,
                          rocsparse_int*                        numeric_pivot);

template void host_csrilu0(rocsparse_int                         M,
                           const std::vector<rocsparse_int>&     csr_row_ptr,
                           const std::vector<rocsparse_int>&     csr_col_ind,
                           std::vector<rocsparse_float_complex>& csr_val,
                           rocsparse_index_base                  base,
                           rocsparse_int*                        struct_pivot,
                           rocsparse_int*                        numeric_pivot,
                           bool                                  boost,
                           float                                 boost_tol,
                           rocsparse_float_complex               boost_val);

template void host_gtsv_no_pivot(rocsparse_int                               m,
                                 rocsparse_int                               n,
                                 const std::vector<rocsparse_float_complex>& dl,
                                 const std::vector<rocsparse_float_complex>& d,
                                 const std::vector<rocsparse_float_complex>& du,
                                 std::vector<rocsparse_float_complex>&       B,
                                 rocsparse_int                               ldb);

template void host_gtsv_no_pivot_strided_batch(rocsparse_int                               m,
                                               const std::vector<rocsparse_float_complex>& dl,
                                               const std::vector<rocsparse_float_complex>& d,
                                               const std::vector<rocsparse_float_complex>& du,
                                               std::vector<rocsparse_float_complex>&       x,
                                               rocsparse_int batch_count,
                                               rocsparse_int batch_stride);

template void host_gtsv_interleaved_batch(rocsparse_gtsv_interleaved_alg algo,
                                          rocsparse_int                  m,
                                          const rocsparse_float_complex* dl,
                                          const rocsparse_float_complex* d,
                                          const rocsparse_float_complex* du,
                                          rocsparse_float_complex*       x,
                                          rocsparse_int                  batch_count,
                                          rocsparse_int                  batch_stride);

template void host_gpsv_interleaved_batch(rocsparse_gpsv_interleaved_alg algo,
                                          rocsparse_int                  m,
                                          rocsparse_float_complex*       ds,
                                          rocsparse_float_complex*       dl,
                                          rocsparse_float_complex*       d,
                                          rocsparse_float_complex*       du,
                                          rocsparse_float_complex*       dw,
                                          rocsparse_float_complex*       x,
                                          rocsparse_int                  batch_count,
                                          rocsparse_int                  batch_stride);

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template rocsparse_status host_nnz(rocsparse_direction            dirA,
                                   rocsparse_int                  m,
                                   rocsparse_int                  n,
                                   const rocsparse_float_complex* A,
                                   rocsparse_int                  lda,
                                   rocsparse_int*                 nnz_per_row_columns,
                                   rocsparse_int*                 nnz_total_dev_host_ptr);

template void host_csr_to_gebsr(rocsparse_direction                         direction,
                                rocsparse_int                               m,
                                rocsparse_int                               n,
                                rocsparse_int                               nnz,
                                const std::vector<rocsparse_float_complex>& csr_val,
                                const std::vector<rocsparse_int>&           csr_row_ptr,
                                const std::vector<rocsparse_int>&           csr_col_ind,
                                rocsparse_int                               row_block_dim,
                                rocsparse_int                               col_block_dim,
                                rocsparse_index_base                        csr_base,
                                std::vector<rocsparse_float_complex>&       bsr_val,
                                std::vector<rocsparse_int>&                 bsr_row_ptr,
                                std::vector<rocsparse_int>&                 bsr_col_ind,
                                rocsparse_index_base                        bsr_base);

template void host_gebsr_to_gebsc(rocsparse_int                               Mb,
                                  rocsparse_int                               Nb,
                                  rocsparse_int                               nnzb,
                                  const std::vector<rocsparse_int>&           bsr_row_ptr,
                                  const std::vector<rocsparse_int>&           bsr_col_ind,
                                  const std::vector<rocsparse_float_complex>& bsr_val,
                                  rocsparse_int                               row_block_dim,
                                  rocsparse_int                               col_block_dim,
                                  std::vector<rocsparse_int>&                 bsc_row_ind,
                                  std::vector<rocsparse_int>&                 bsc_col_ptr,
                                  std::vector<rocsparse_float_complex>&       bsc_val,
                                  rocsparse_action                            action,
                                  rocsparse_index_base                        base);

template void host_gebsr_to_csr(rocsparse_direction                         direction,
                                rocsparse_int                               mb,
                                rocsparse_int                               nb,
                                rocsparse_int                               nnzb,
                                const std::vector<rocsparse_float_complex>& bsr_val,
                                const std::vector<rocsparse_int>&           bsr_row_ptr,
                                const std::vector<rocsparse_int>&           bsr_col_ind,
                                rocsparse_int                               row_block_dim,
                                rocsparse_int                               col_block_dim,
                                rocsparse_index_base                        bsr_base,
                                std::vector<rocsparse_float_complex>&       csr_val,
                                std::vector<rocsparse_int>&                 csr_row_ptr,
                                std::vector<rocsparse_int>&                 csr_col_ind,
                                rocsparse_index_base                        csr_base);

template void host_gebsr_to_gebsr(rocsparse_direction                         direction,
                                  rocsparse_int                               mb,
                                  rocsparse_int                               nb,
                                  rocsparse_int                               nnzb,
                                  const std::vector<rocsparse_float_complex>& bsr_val_A,
                                  const std::vector<rocsparse_int>&           bsr_row_ptr_A,
                                  const std::vector<rocsparse_int>&           bsr_col_ind_A,
                                  rocsparse_int                               row_block_dim_A,
                                  rocsparse_int                               col_block_dim_A,
                                  rocsparse_index_base                        base_A,
                                  std::vector<rocsparse_float_complex>&       bsr_val_C,
                                  std::vector<rocsparse_int>&                 bsr_row_ptr_C,
                                  std::vector<rocsparse_int>&                 bsr_col_ind_C,
                                  rocsparse_int                               row_block_dim_C,
                                  rocsparse_int                               col_block_dim_C,
                                  rocsparse_index_base                        base_C);

template void host_bsr_to_bsc(rocsparse_int                         mb,
                              rocsparse_int                         nb,
                              rocsparse_int                         nnzb,
                              rocsparse_int                         bsr_dim,
                              const rocsparse_int*                  bsr_row_ptr,
                              const rocsparse_int*                  bsr_col_ind,
                              const rocsparse_float_complex*        bsr_val,
                              std::vector<rocsparse_int>&           bsc_row_ind,
                              std::vector<rocsparse_int>&           bsc_col_ptr,
                              std::vector<rocsparse_float_complex>& bsc_val,
                              rocsparse_index_base                  bsr_base,
                              rocsparse_index_base                  bsc_base);

template void host_csr_to_hyb(rocsparse_int                               M,
                              rocsparse_int                               nnz,
                              const std::vector<rocsparse_int>&           csr_row_ptr,
                              const std::vector<rocsparse_int>&           csr_col_ind,
                              const std::vector<rocsparse_float_complex>& csr_val,
                              std::vector<rocsparse_int>&                 ell_col_ind,
                              std::vector<rocsparse_float_complex>&       ell_val,
                              rocsparse_int&                              ell_width,
                              rocsparse_int&                              ell_nnz,
                              std::vector<rocsparse_int>&                 coo_row_ind,
                              std::vector<rocsparse_int>&                 coo_col_ind,
                              std::vector<rocsparse_float_complex>&       coo_val,
                              rocsparse_int&                              coo_nnz,
                              rocsparse_hyb_partition                     part,
                              rocsparse_index_base                        base);

template void host_csr_to_csr_compress(rocsparse_int                               M,
                                       rocsparse_int                               N,
                                       rocsparse_int                               nnz,
                                       const std::vector<rocsparse_int>&           csr_row_ptr_A,
                                       const std::vector<rocsparse_int>&           csr_col_ind_A,
                                       const std::vector<rocsparse_float_complex>& csr_val_A,
                                       std::vector<rocsparse_int>&                 csr_row_ptr_C,
                                       std::vector<rocsparse_int>&                 csr_col_ind_C,
                                       std::vector<rocsparse_float_complex>&       csr_val_C,
                                       rocsparse_index_base                        base,
                                       rocsparse_float_complex                     tol);

template void host_ell_to_csr(rocsparse_int                               M,
                              rocsparse_int                               N,
                              const std::vector<rocsparse_int>&           ell_col_ind,
                              const std::vector<rocsparse_float_complex>& ell_val,
                              rocsparse_int                               ell_width,
                              std::vector<rocsparse_int>&                 csr_row_ptr,
                              std::vector<rocsparse_int>&                 csr_col_ind,
                              std::vector<rocsparse_float_complex>&       csr_val,
                              rocsparse_int&                              csr_nnz,
                              rocsparse_index_base                        ell_base,
                              rocsparse_index_base                        csr_base);

template void host_coosort_by_column(rocsparse_int                         M,
                                     rocsparse_int                         nnz,
                                     std::vector<rocsparse_int>&           coo_row_ind,
                                     std::vector<rocsparse_int>&           coo_col_ind,
                                     std::vector<rocsparse_float_complex>& coo_val);

#define INSTANTIATE2(ITYPE, TTYPE)                                                               \
    template void host_gemvi<ITYPE, TTYPE>(ITYPE                M,                               \
                                           ITYPE                N,                               \
                                           TTYPE                alpha,                           \
                                           const TTYPE*         A,                               \
                                           ITYPE                lda,                             \
                                           ITYPE                nnz,                             \
                                           const TTYPE*         x_val,                           \
                                           const ITYPE*         x_ind,                           \
                                           TTYPE                beta,                            \
                                           TTYPE*               y,                               \
                                           rocsparse_index_base base);                           \
    template void host_coo_to_dense<ITYPE, TTYPE>(ITYPE                     m,                   \
                                                  ITYPE                     n,                   \
                                                  ITYPE                     nnz,                 \
                                                  rocsparse_index_base      base,                \
                                                  const std::vector<TTYPE>& coo_val,             \
                                                  const std::vector<ITYPE>& coo_row_ind,         \
                                                  const std::vector<ITYPE>& coo_col_ind,         \
                                                  std::vector<TTYPE>&       A,                   \
                                                  ITYPE                     ld,                  \
                                                  rocsparse_order           order);                        \
    template void host_dense_to_coo<ITYPE, TTYPE>(ITYPE                     m,                   \
                                                  ITYPE                     n,                   \
                                                  rocsparse_index_base      base,                \
                                                  const std::vector<TTYPE>& A,                   \
                                                  ITYPE                     ld,                  \
                                                  rocsparse_order           order,               \
                                                  const std::vector<ITYPE>& nnz_per_row,         \
                                                  std::vector<TTYPE>&       coo_val,             \
                                                  std::vector<ITYPE>&       coo_row_ind,         \
                                                  std::vector<ITYPE>&       coo_col_ind);              \
    template void host_coomv<ITYPE, TTYPE>(rocsparse_operation  trans,                           \
                                           ITYPE                M,                               \
                                           ITYPE                N,                               \
                                           ITYPE                nnz,                             \
                                           TTYPE                alpha,                           \
                                           const ITYPE*         coo_row_ind,                     \
                                           const ITYPE*         coo_col_ind,                     \
                                           const TTYPE*         coo_val,                         \
                                           const TTYPE*         x,                               \
                                           TTYPE                beta,                            \
                                           TTYPE*               y,                               \
                                           rocsparse_index_base base);                           \
    template void host_coomv_aos<ITYPE, TTYPE>(rocsparse_operation  trans,                       \
                                               ITYPE                M,                           \
                                               ITYPE                N,                           \
                                               ITYPE                nnz,                         \
                                               TTYPE                alpha,                       \
                                               const ITYPE*         coo_ind,                     \
                                               const TTYPE*         coo_val,                     \
                                               const TTYPE*         x,                           \
                                               TTYPE                beta,                        \
                                               TTYPE*               y,                           \
                                               rocsparse_index_base base);                       \
    template void host_ellmv<ITYPE, TTYPE>(rocsparse_operation  trans,                           \
                                           ITYPE                M,                               \
                                           ITYPE                N,                               \
                                           TTYPE                alpha,                           \
                                           const ITYPE*         ell_col_ind,                     \
                                           const TTYPE*         ell_val,                         \
                                           ITYPE                ell_width,                       \
                                           const TTYPE*         x,                               \
                                           TTYPE                beta,                            \
                                           TTYPE*               y,                               \
                                           rocsparse_index_base base);                           \
    template void host_coosv<ITYPE, TTYPE>(rocsparse_operation       trans,                      \
                                           ITYPE                     M,                          \
                                           ITYPE                     nnz,                        \
                                           TTYPE                     alpha,                      \
                                           const std::vector<ITYPE>& coo_row_ind,                \
                                           const std::vector<ITYPE>& coo_col_ind,                \
                                           const std::vector<TTYPE>& coo_val,                    \
                                           const std::vector<TTYPE>& x,                          \
                                           std::vector<TTYPE>&       y,                          \
                                           rocsparse_diag_type       diag_type,                  \
                                           rocsparse_fill_mode       fill_mode,                  \
                                           rocsparse_index_base      base,                       \
                                           ITYPE*                    struct_pivot,               \
                                           ITYPE*                    numeric_pivot);                                \
    template void host_coomm<TTYPE, ITYPE>(rocsparse_spmm_alg   alg,                             \
                                           ITYPE                M,                               \
                                           ITYPE                N,                               \
                                           ITYPE                NNZ,                             \
                                           rocsparse_operation  transB,                          \
                                           TTYPE                alpha,                           \
                                           const ITYPE*         coo_row_ind_A,                   \
                                           const ITYPE*         coo_col_ind_A,                   \
                                           const TTYPE*         coo_val_A,                       \
                                           const TTYPE*         B,                               \
                                           ITYPE                ldb,                             \
                                           TTYPE                beta,                            \
                                           TTYPE*               C,                               \
                                           ITYPE                ldc,                             \
                                           rocsparse_order      order,                           \
                                           rocsparse_index_base base);                           \
    template void host_coosm<ITYPE, TTYPE>(ITYPE                M,                               \
                                           ITYPE                nrhs,                            \
                                           ITYPE                nnz,                             \
                                           rocsparse_operation  transA,                          \
                                           rocsparse_operation  transB,                          \
                                           TTYPE                alpha,                           \
                                           const ITYPE*         coo_row_ind,                     \
                                           const ITYPE*         coo_col_ind,                     \
                                           const TTYPE*         coo_val,                         \
                                           TTYPE*               B,                               \
                                           ITYPE                ldb,                             \
                                           rocsparse_diag_type  diag_type,                       \
                                           rocsparse_fill_mode  fill_mode,                       \
                                           rocsparse_index_base base,                            \
                                           ITYPE*               struct_pivot,                    \
                                           ITYPE*               numeric_pivot);                                \
    template void host_axpby<ITYPE, TTYPE>(ITYPE                size,                            \
                                           ITYPE                nnz,                             \
                                           TTYPE                alpha,                           \
                                           const TTYPE*         x_val,                           \
                                           const ITYPE*         x_ind,                           \
                                           TTYPE                beta,                            \
                                           TTYPE*               y,                               \
                                           rocsparse_index_base base);                           \
    template void host_gthr<ITYPE, TTYPE>(                                                       \
        ITYPE nnz, const TTYPE* y, TTYPE* x_val, const ITYPE* x_ind, rocsparse_index_base base); \
    template void host_roti<ITYPE, TTYPE>(ITYPE nnz,                                             \
                                          TTYPE * x_val,                                         \
                                          const ITYPE*         x_ind,                            \
                                          TTYPE*               y,                                \
                                          const TTYPE*         c,                                \
                                          const TTYPE*         s,                                \
                                          rocsparse_index_base base);                            \
    template void host_doti<ITYPE, TTYPE>(ITYPE                nnz,                              \
                                          const TTYPE*         x_val,                            \
                                          const ITYPE*         x_ind,                            \
                                          const TTYPE*         y,                                \
                                          TTYPE*               result,                           \
                                          rocsparse_index_base base);                            \
    template void host_dotci<ITYPE, TTYPE>(ITYPE                nnz,                             \
                                           const TTYPE*         x_val,                           \
                                           const ITYPE*         x_ind,                           \
                                           const TTYPE*         y,                               \
                                           TTYPE*               result,                          \
                                           rocsparse_index_base base);                           \
    template void host_sctr<ITYPE, TTYPE>(                                                       \
        ITYPE nnz, const TTYPE* x_val, const ITYPE* x_ind, TTYPE* y, rocsparse_index_base base)

#define INSTANTIATE3(ITYPE, JTYPE, TTYPE)                                                   \
    template void host_csr_to_csc<ITYPE, JTYPE, TTYPE>(JTYPE                M,              \
                                                       JTYPE                N,              \
                                                       ITYPE                nnz,            \
                                                       const ITYPE*         csr_row_ptr,    \
                                                       const JTYPE*         csr_col_ind,    \
                                                       const TTYPE*         csr_val,        \
                                                       std::vector<JTYPE>&  csc_row_ind,    \
                                                       std::vector<ITYPE>&  csc_col_ptr,    \
                                                       std::vector<TTYPE>&  csc_val,        \
                                                       rocsparse_action     action,         \
                                                       rocsparse_index_base base);          \
    template void host_csrsv<ITYPE, JTYPE, TTYPE>(rocsparse_operation  trans,               \
                                                  JTYPE                M,                   \
                                                  ITYPE                nnz,                 \
                                                  TTYPE                alpha,               \
                                                  const ITYPE*         csr_row_ptr,         \
                                                  const JTYPE*         csr_col_ind,         \
                                                  const TTYPE*         csr_val,             \
                                                  const TTYPE*         x,                   \
                                                  TTYPE*               y,                   \
                                                  rocsparse_diag_type  diag_type,           \
                                                  rocsparse_fill_mode  fill_mode,           \
                                                  rocsparse_index_base base,                \
                                                  JTYPE*               struct_pivot,        \
                                                  JTYPE*               numeric_pivot);                    \
    template void host_csrmv<ITYPE, JTYPE, TTYPE>(rocsparse_operation   trans,              \
                                                  JTYPE                 M,                  \
                                                  JTYPE                 N,                  \
                                                  ITYPE                 nnz,                \
                                                  TTYPE                 alpha,              \
                                                  const ITYPE*          csr_row_ptr,        \
                                                  const JTYPE*          csr_col_ind,        \
                                                  const TTYPE*          csr_val,            \
                                                  const TTYPE*          x,                  \
                                                  TTYPE                 beta,               \
                                                  TTYPE*                y,                  \
                                                  rocsparse_index_base  base,               \
                                                  rocsparse_matrix_type matrix_type,        \
                                                  rocsparse_spmv_alg    algo);                 \
    template void host_csrmm<TTYPE, ITYPE, JTYPE>(JTYPE                M,                   \
                                                  JTYPE                N,                   \
                                                  JTYPE                K,                   \
                                                  rocsparse_operation  transA,              \
                                                  rocsparse_operation  transB,              \
                                                  TTYPE                alpha,               \
                                                  const ITYPE*         csr_row_ptr_A,       \
                                                  const JTYPE*         csr_col_ind_A,       \
                                                  const TTYPE*         csr_val_A,           \
                                                  const TTYPE*         B,                   \
                                                  JTYPE                ldb,                 \
                                                  TTYPE                beta,                \
                                                  TTYPE*               C,                   \
                                                  JTYPE                ldc,                 \
                                                  rocsparse_order      order,               \
                                                  rocsparse_index_base base);               \
    template void host_csrsm<ITYPE, JTYPE, TTYPE>(JTYPE                M,                   \
                                                  JTYPE                nrhs,                \
                                                  ITYPE                nnz,                 \
                                                  rocsparse_operation  transA,              \
                                                  rocsparse_operation  transB,              \
                                                  TTYPE                alpha,               \
                                                  const ITYPE*         csr_row_ptr,         \
                                                  const JTYPE*         csr_col_ind,         \
                                                  const TTYPE*         csr_val,             \
                                                  TTYPE*               B,                   \
                                                  JTYPE                ldb,                 \
                                                  rocsparse_diag_type  diag_type,           \
                                                  rocsparse_fill_mode  fill_mode,           \
                                                  rocsparse_index_base base,                \
                                                  JTYPE*               struct_pivot,        \
                                                  JTYPE*               numeric_pivot);                    \
    template void host_csrgemm_nnz<TTYPE, ITYPE, JTYPE>(JTYPE                M,             \
                                                        JTYPE                N,             \
                                                        JTYPE                K,             \
                                                        const TTYPE*         alpha,         \
                                                        const ITYPE*         csr_row_ptr_A, \
                                                        const JTYPE*         csr_col_ind_A, \
                                                        const ITYPE*         csr_row_ptr_B, \
                                                        const JTYPE*         csr_col_ind_B, \
                                                        const TTYPE*         beta,          \
                                                        const ITYPE*         csr_row_ptr_D, \
                                                        const JTYPE*         csr_col_ind_D, \
                                                        ITYPE*               csr_row_ptr_C, \
                                                        ITYPE*               nnz_C,         \
                                                        rocsparse_index_base base_A,        \
                                                        rocsparse_index_base base_B,        \
                                                        rocsparse_index_base base_C,        \
                                                        rocsparse_index_base base_D);       \
    template void host_csrgemm<TTYPE, ITYPE, JTYPE>(JTYPE                M,                 \
                                                    JTYPE                N,                 \
                                                    JTYPE                L,                 \
                                                    const TTYPE*         alpha,             \
                                                    const ITYPE*         csr_row_ptr_A,     \
                                                    const JTYPE*         csr_col_ind_A,     \
                                                    const TTYPE*         csr_val_A,         \
                                                    const ITYPE*         csr_row_ptr_B,     \
                                                    const JTYPE*         csr_col_ind_B,     \
                                                    const TTYPE*         csr_val_B,         \
                                                    const TTYPE*         beta,              \
                                                    const ITYPE*         csr_row_ptr_D,     \
                                                    const JTYPE*         csr_col_ind_D,     \
                                                    const TTYPE*         csr_val_D,         \
                                                    const ITYPE*         csr_row_ptr_C,     \
                                                    JTYPE*               csr_col_ind_C,     \
                                                    TTYPE*               csr_val_C,         \
                                                    rocsparse_index_base base_A,            \
                                                    rocsparse_index_base base_B,            \
                                                    rocsparse_index_base base_C,            \
                                                    rocsparse_index_base base_D);

#define INSTANTIATE4(DIR, ITYPE, JTYPE, TTYPE)                                                       \
    template void host_dense2csx<DIR, TTYPE, ITYPE, JTYPE>(JTYPE                m,                   \
                                                           JTYPE                n,                   \
                                                           rocsparse_index_base base,                \
                                                           const TTYPE*         A,                   \
                                                           ITYPE                ld,                  \
                                                           rocsparse_order      order,               \
                                                           const ITYPE*         nnz_per_row_columns, \
                                                           TTYPE*               csx_val,             \
                                                           ITYPE*               csx_row_col_ptr,     \
                                                           JTYPE*               csx_col_row_ind);                  \
    template void host_csx2dense<DIR, TTYPE, ITYPE, JTYPE>(JTYPE                m,                   \
                                                           JTYPE                n,                   \
                                                           rocsparse_index_base base,                \
                                                           rocsparse_order      order,               \
                                                           const TTYPE*         csx_val,             \
                                                           const ITYPE*         csx_row_col_ptr,     \
                                                           const JTYPE*         csx_col_row_ind,     \
                                                           TTYPE*               A,                   \
                                                           ITYPE                ld);

INSTANTIATE2(int32_t, float);
INSTANTIATE2(int32_t, double);
INSTANTIATE2(int32_t, rocsparse_float_complex);
INSTANTIATE2(int32_t, rocsparse_double_complex);
INSTANTIATE2(int64_t, float);
INSTANTIATE2(int64_t, double);
INSTANTIATE2(int64_t, rocsparse_float_complex);
INSTANTIATE2(int64_t, rocsparse_double_complex);

INSTANTIATE3(int32_t, int32_t, float);
INSTANTIATE3(int32_t, int32_t, double);
INSTANTIATE3(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE3(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE3(int64_t, int32_t, float);
INSTANTIATE3(int64_t, int32_t, double);
INSTANTIATE3(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE3(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE3(int64_t, int64_t, float);
INSTANTIATE3(int64_t, int64_t, double);
INSTANTIATE3(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE3(int64_t, int64_t, rocsparse_double_complex);

INSTANTIATE4(rocsparse_direction_row, int32_t, int32_t, float);
INSTANTIATE4(rocsparse_direction_row, int32_t, int32_t, double);
INSTANTIATE4(rocsparse_direction_row, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE4(rocsparse_direction_row, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE4(rocsparse_direction_row, int64_t, int32_t, float);
INSTANTIATE4(rocsparse_direction_row, int64_t, int32_t, double);
INSTANTIATE4(rocsparse_direction_row, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE4(rocsparse_direction_row, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE4(rocsparse_direction_row, int64_t, int64_t, float);
INSTANTIATE4(rocsparse_direction_row, int64_t, int64_t, double);
INSTANTIATE4(rocsparse_direction_row, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE4(rocsparse_direction_row, int64_t, int64_t, rocsparse_double_complex);
INSTANTIATE4(rocsparse_direction_column, int32_t, int32_t, float);
INSTANTIATE4(rocsparse_direction_column, int32_t, int32_t, double);
INSTANTIATE4(rocsparse_direction_column, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE4(rocsparse_direction_column, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE4(rocsparse_direction_column, int64_t, int32_t, float);
INSTANTIATE4(rocsparse_direction_column, int64_t, int32_t, double);
INSTANTIATE4(rocsparse_direction_column, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE4(rocsparse_direction_column, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE4(rocsparse_direction_column, int64_t, int64_t, float);
INSTANTIATE4(rocsparse_direction_column, int64_t, int64_t, double);
INSTANTIATE4(rocsparse_direction_column, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE4(rocsparse_direction_column, int64_t, int64_t, rocsparse_double_complex);
