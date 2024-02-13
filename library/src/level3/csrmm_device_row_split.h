/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"

namespace rocsparse
{
    // See Yang C., Bulu? A., Owens J.D. (2018) Design Principles for Sparse Matrix Multiplication on the GPU.
    // In: Aldinucci M., Padovani L., Torquati M. (eds) Euro-Par 2018: Parallel Processing. Euro-Par 2018.
    // Lecture Notes in Computer Science, vol 11014. Springer, Cham. https://doi.org/10.1007/978-3-319-96983-1_48
    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              unsigned int LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnn_row_split_device(bool conj_A,
                                                       bool conj_B,
                                                       J    offset,
                                                       J    M,
                                                       J    N,
                                                       J    K,
                                                       I    nnz,
                                                       T    alpha,
                                                       const I* __restrict__ csr_row_ptr,
                                                       const J* __restrict__ csr_col_ind,
                                                       const A* __restrict__ csr_val,
                                                       const B* __restrict__ dense_B,
                                                       int64_t ldb,
                                                       T       beta,
                                                       C* __restrict__ dense_C,
                                                       int64_t              ldc,
                                                       rocsparse_order      order_C,
                                                       rocsparse_index_base idx_base)
    {
        const int tid  = hipThreadIdx_x;
        const J   gid  = hipBlockIdx_x * BLOCKSIZE + tid;
        const int lid  = gid & (WF_SIZE - 1);
        const J   row  = gid / WF_SIZE;
        const J   colB = offset + LOOPS * hipBlockIdx_y;

        if(row >= M)
        {
            return;
        }

        const I row_start = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        T sum[LOOPS]{};

        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            const J col = csr_col_ind[j] - idx_base;
            const T val = conj_val(csr_val[j], conj_A);

            for(unsigned int p = 0; p < LOOPS; p++)
            {
                sum[p] = rocsparse::fma<T>(
                    val,
                    conj_val(rocsparse::ldg(dense_B + col + (colB + p) * ldb), conj_B),
                    sum[p]);
            }
        }

        for(unsigned int p = 0; p < LOOPS; p++)
        {
            sum[p] = rocsparse::wfreduce_sum<WF_SIZE>(sum[p]);
        }

        if(lid == WF_SIZE - 1)
        {
            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row + (colB + p) * ldc] = alpha * sum[p];
                    }
                }
                else
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + (colB + p)] = alpha * sum[p];
                    }
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row + (colB + p) * ldc] = rocsparse::fma<T>(
                            beta, dense_C[row + (colB + p) * ldc], alpha * sum[p]);
                    }
                }
                else
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + (colB + p)] = rocsparse::fma<T>(
                            beta, dense_C[row * ldc + (colB + p)], alpha * sum[p]);
                    }
                }
            }
        }
    }

    // See Yang C., Bulu? A., Owens J.D. (2018) Design Principles for Sparse Matrix Multiplication on the GPU.
    // In: Aldinucci M., Padovani L., Torquati M. (eds) Euro-Par 2018: Parallel Processing. Euro-Par 2018.
    // Lecture Notes in Computer Science, vol 11014. Springer, Cham. https://doi.org/10.1007/978-3-319-96983-1_48
    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              unsigned int LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnt_row_split_main_device(bool conj_A,
                                                            bool conj_B,
                                                            J    offset,
                                                            J    ncol,
                                                            J    M,
                                                            J    N,
                                                            J    K,
                                                            I    nnz,
                                                            T    alpha,
                                                            const I* __restrict__ csr_row_ptr,
                                                            const J* __restrict__ csr_col_ind,
                                                            const A* __restrict__ csr_val,
                                                            const B* __restrict__ dense_B,
                                                            int64_t ldb,
                                                            T       beta,
                                                            C* __restrict__ dense_C,
                                                            int64_t              ldc,
                                                            rocsparse_order      order_C,
                                                            rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const J   row = gid / WF_SIZE;
        const int lid = tid & (WF_SIZE - 1);

        if(row >= M)
        {
            return;
        }

        const I row_start = rocsparse::nontemporal_load(csr_row_ptr + row) - idx_base;
        const I row_end   = rocsparse::nontemporal_load(csr_row_ptr + row + 1) - idx_base;

        T sum[LOOPS];

        for(J l = 0; l < ncol; l += WF_SIZE * LOOPS)
        {
            const J colB = l + lid;

            for(unsigned int p = 0; p < LOOPS; p++)
            {
                sum[p] = static_cast<T>(0);
            }

            for(I j = row_start; j < row_end; j += WF_SIZE)
            {
                const I k = j + lid;

                const int64_t col
                    = (k < row_end)
                          ? (ldb * (rocsparse::nontemporal_load(csr_col_ind + k) - idx_base))
                          : 0;

                const T val = (k < row_end)
                                  ? conj_val(rocsparse::nontemporal_load(csr_val + k), conj_A)
                                  : static_cast<T>(0);

                for(unsigned int i = 0; i < WF_SIZE; ++i)
                {
                    const T       v = rocsparse::shfl(val, i, WF_SIZE);
                    const int64_t c = __shfl(col, i, WF_SIZE);

                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse::fma<T>(
                            v,
                            conj_val(rocsparse::ldg(dense_B + colB + p * WF_SIZE + c), conj_B),
                            sum[p]);
                    }
                }
            }

            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row + (colB + p * WF_SIZE) * ldc] = alpha * sum[p];
                    }
                }
                else
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + colB + p * WF_SIZE] = alpha * sum[p];
                    }
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row + (colB + p * WF_SIZE) * ldc] = rocsparse::fma<T>(
                            beta, dense_C[row + (colB + p * WF_SIZE) * ldc], alpha * sum[p]);
                    }
                }
                else
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + colB + p * WF_SIZE] = rocsparse::fma<T>(
                            beta, dense_C[row * ldc + colB + p * WF_SIZE], alpha * sum[p]);
                    }
                }
            }
        }
    }

    // See Yang C., Bulu? A., Owens J.D. (2018) Design Principles for Sparse Matrix Multiplication on the GPU.
    // In: Aldinucci M., Padovani L., Torquati M. (eds) Euro-Par 2018: Parallel Processing. Euro-Par 2018.
    // Lecture Notes in Computer Science, vol 11014. Springer, Cham. https://doi.org/10.1007/978-3-319-96983-1_48
    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnt_row_split_remainder_device(bool conj_A,
                                                                 bool conj_B,
                                                                 J    offset,
                                                                 J    ncol,
                                                                 J    M,
                                                                 J    N,
                                                                 J    K,
                                                                 I    nnz,
                                                                 T    alpha,
                                                                 const I* __restrict__ csr_row_ptr,
                                                                 const J* __restrict__ csr_col_ind,
                                                                 const A* __restrict__ csr_val,
                                                                 const B* __restrict__ dense_B,
                                                                 int64_t ldb,
                                                                 T       beta,
                                                                 C* __restrict__ dense_C,
                                                                 int64_t              ldc,
                                                                 rocsparse_order      order_C,
                                                                 rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const J   row = gid / WF_SIZE;
        const int lid = tid & (WF_SIZE - 1);

        if(row >= M)
        {
            return;
        }

        const I row_start = rocsparse::nontemporal_load(csr_row_ptr + row) - idx_base;
        const I row_end   = rocsparse::nontemporal_load(csr_row_ptr + row + 1) - idx_base;

        for(J l = offset; l < ncol; l += WF_SIZE)
        {
            const J colB = l + lid;
            T       sum  = static_cast<T>(0);

            for(I j = row_start; j < row_end; j += WF_SIZE)
            {
                I k = j + lid;

                const int64_t col
                    = (k < row_end)
                          ? (ldb * (rocsparse::nontemporal_load(csr_col_ind + k) - idx_base))
                          : 0;
                const T val = (k < row_end)
                                  ? conj_val(rocsparse::nontemporal_load(csr_val + k), conj_A)
                                  : static_cast<T>(0);

                for(unsigned int i = 0; i < WF_SIZE; ++i)
                {
                    const T       v = rocsparse::shfl(val, i, WF_SIZE);
                    const int64_t c = __shfl(col, i, WF_SIZE);
                    sum             = rocsparse::fma<T>(
                        v,
                        (colB < ncol) ? conj_val(rocsparse::ldg(dense_B + colB + c), conj_B)
                                                  : static_cast<T>(0),
                        sum);
                }
            }

            if(colB < ncol)
            {
                if(beta == static_cast<T>(0))
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[row + colB * ldc] = alpha * sum;
                    }
                    else
                    {
                        dense_C[row * ldc + colB] = alpha * sum;
                    }
                }
                else
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[row + colB * ldc]
                            = rocsparse::fma<T>(beta, dense_C[row + colB * ldc], alpha * sum);
                    }
                    else
                    {
                        dense_C[row * ldc + colB]
                            = rocsparse::fma<T>(beta, dense_C[row * ldc + colB], alpha * sum);
                    }
                }
            }
        }
    }
}
