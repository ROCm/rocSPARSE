/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef CSRMM_DEVICE_ROW_SPLIT_H
#define CSRMM_DEVICE_ROW_SPLIT_H

#include "common.h"

// See Yang C., Bulu? A., Owens J.D. (2018) Design Principles for Sparse Matrix Multiplication on the GPU.
// In: Aldinucci M., Padovani L., Torquati M. (eds) Euro-Par 2018: Parallel Processing. Euro-Par 2018.
// Lecture Notes in Computer Science, vol 11014. Springer, Cham. https://doi.org/10.1007/978-3-319-96983-1_48
template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          typename I,
          typename J,
          typename T>
static __device__ void csrmmnn_row_split_device(bool conj_A,
                                                bool conj_B,
                                                J    offset,
                                                J    M,
                                                J    N,
                                                J    K,
                                                I    nnz,
                                                T    alpha,
                                                const I* __restrict__ csr_row_ptr,
                                                const J* __restrict__ csr_col_ind,
                                                const T* __restrict__ csr_val,
                                                const T* __restrict__ B,
                                                J ldb,
                                                T beta,
                                                T* __restrict__ C,
                                                J                    ldc,
                                                rocsparse_order      order,
                                                rocsparse_index_base idx_base)
{
    int tid  = hipThreadIdx_x;
    J   gid  = hipBlockIdx_x * BLOCKSIZE + tid;
    int lid  = gid & (WF_SIZE - 1);
    J   row  = gid / WF_SIZE;
    J   colB = offset + LOOPS * hipBlockIdx_y;

    if(row >= M)
    {
        return;
    }

    I row_start = csr_row_ptr[row] - idx_base;
    I row_end   = csr_row_ptr[row + 1] - idx_base;

    T sum[LOOPS]{};

    for(I j = row_start + lid; j < row_end; j += WF_SIZE)
    {
        J col = csr_col_ind[j] - idx_base;
        T val = conj_val(csr_val[j], conj_A);

        for(J p = 0; p < LOOPS; p++)
        {
            sum[p] = rocsparse_fma(
                val, conj_val(rocsparse_ldg(B + col + (colB + p) * ldb), conj_B), sum[p]);
        }
    }

    for(J p = 0; p < LOOPS; p++)
    {
        sum[p] = rocsparse_wfreduce_sum<WF_SIZE>(sum[p]);
    }

    if(lid == WF_SIZE - 1)
    {
        if(beta == static_cast<T>(0))
        {
            if(order == rocsparse_order_column)
            {
                for(J p = 0; p < LOOPS; p++)
                {
                    C[row + (colB + p) * ldc] = alpha * sum[p];
                }
            }
            else
            {
                for(J p = 0; p < LOOPS; p++)
                {
                    C[row * ldc + (colB + p)] = alpha * sum[p];
                }
            }
        }
        else
        {
            if(order == rocsparse_order_column)
            {
                for(J p = 0; p < LOOPS; p++)
                {
                    C[row + (colB + p) * ldc]
                        = rocsparse_fma(beta, C[row + (colB + p) * ldc], alpha * sum[p]);
                }
            }
            else
            {
                for(J p = 0; p < LOOPS; p++)
                {
                    C[row * ldc + (colB + p)]
                        = rocsparse_fma(beta, C[row * ldc + (colB + p)], alpha * sum[p]);
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
          typename I,
          typename J,
          typename T>
static __device__ void csrmmnt_row_split_main_device(bool conj_A,
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
                                                     const T* __restrict__ csr_val,
                                                     const T* __restrict__ B,
                                                     J ldb,
                                                     T beta,
                                                     T* __restrict__ C,
                                                     J                    ldc,
                                                     rocsparse_order      order,
                                                     rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
    J   row = gid / WF_SIZE;
    int lid = tid & (WF_SIZE - 1);

    if(row >= M)
    {
        return;
    }

    I row_start = rocsparse_nontemporal_load(csr_row_ptr + row) - idx_base;
    I row_end   = rocsparse_nontemporal_load(csr_row_ptr + row + 1) - idx_base;

    T sum[LOOPS];

    for(J l = 0; l < ncol; l += WF_SIZE * LOOPS)
    {
        J colB = l + lid;

        for(J p = 0; p < LOOPS; p++)
        {
            sum[p] = static_cast<T>(0);
        }

        for(I j = row_start; j < row_end; j += WF_SIZE)
        {
            I k = j + lid;

            I col;
            T val;

            if(k < row_end)
            {
                col = ldb * (rocsparse_nontemporal_load(csr_col_ind + k) - idx_base);
                val = conj_val(rocsparse_nontemporal_load(csr_val + k), conj_A);
            }
            else
            {
                col = 0;
                val = static_cast<T>(0);
            }

            for(J i = 0; i < WF_SIZE; ++i)
            {
                T v = rocsparse_shfl(val, i, WF_SIZE);
                J c = __shfl(col, i, WF_SIZE);

                for(J p = 0; p < LOOPS; p++)
                {
                    sum[p] = rocsparse_fma(
                        v, conj_val(rocsparse_ldg(B + colB + p * WF_SIZE + c), conj_B), sum[p]);
                }
            }
        }

        if(beta == static_cast<T>(0))
        {
            if(order == rocsparse_order_column)
            {
                for(J p = 0; p < LOOPS; p++)
                {
                    C[row + (colB + p * WF_SIZE) * ldc] = alpha * sum[p];
                }
            }
            else
            {
                for(J p = 0; p < LOOPS; p++)
                {
                    C[row * ldc + colB + p * WF_SIZE] = alpha * sum[p];
                }
            }
        }
        else
        {
            if(order == rocsparse_order_column)
            {
                for(J p = 0; p < LOOPS; p++)
                {
                    C[row + (colB + p * WF_SIZE) * ldc]
                        = rocsparse_fma(beta, C[row + (colB + p * WF_SIZE) * ldc], alpha * sum[p]);
                }
            }
            else
            {
                for(J p = 0; p < LOOPS; p++)
                {
                    C[row * ldc + colB + p * WF_SIZE]
                        = rocsparse_fma(beta, C[row * ldc + colB + p * WF_SIZE], alpha * sum[p]);
                }
            }
        }
    }
}

// See Yang C., Bulu? A., Owens J.D. (2018) Design Principles for Sparse Matrix Multiplication on the GPU.
// In: Aldinucci M., Padovani L., Torquati M. (eds) Euro-Par 2018: Parallel Processing. Euro-Par 2018.
// Lecture Notes in Computer Science, vol 11014. Springer, Cham. https://doi.org/10.1007/978-3-319-96983-1_48
template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename J, typename T>
static __device__ void csrmmnt_row_split_remainder_device(bool conj_A,
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
                                                          const T* __restrict__ csr_val,
                                                          const T* __restrict__ B,
                                                          J ldb,
                                                          T beta,
                                                          T* __restrict__ C,
                                                          J                    ldc,
                                                          rocsparse_order      order,
                                                          rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
    J   row = gid / WF_SIZE;
    int lid = tid & (WF_SIZE - 1);

    if(row >= M)
    {
        return;
    }

    I row_start = rocsparse_nontemporal_load(csr_row_ptr + row) - idx_base;
    I row_end   = rocsparse_nontemporal_load(csr_row_ptr + row + 1) - idx_base;

    for(J l = offset; l < ncol; l += WF_SIZE)
    {
        J colB = l + lid;
        T sum  = static_cast<T>(0);

        for(I j = row_start; j < row_end; j += WF_SIZE)
        {
            I k = j + lid;

            I col;
            T val;

            if(k < row_end)
            {
                col = ldb * (rocsparse_nontemporal_load(csr_col_ind + k) - idx_base);
                val = conj_val(rocsparse_nontemporal_load(csr_val + k), conj_A);
            }
            else
            {
                col = 0;
                val = static_cast<T>(0);
            }

            for(J i = 0; i < WF_SIZE; ++i)
            {
                T v = rocsparse_shfl(val, i, WF_SIZE);
                J c = __shfl(col, i, WF_SIZE);
                sum = rocsparse_fma(v,
                                    (colB < ncol) ? conj_val(rocsparse_ldg(B + colB + c), conj_B)
                                                  : static_cast<T>(0),
                                    sum);
            }
        }

        if(colB < ncol)
        {
            if(beta == static_cast<T>(0))
            {
                if(order == rocsparse_order_column)
                {
                    C[row + colB * ldc] = alpha * sum;
                }
                else
                {
                    C[row * ldc + colB] = alpha * sum;
                }
            }
            else
            {
                if(order == rocsparse_order_column)
                {
                    C[row + colB * ldc] = rocsparse_fma(beta, C[row + colB * ldc], alpha * sum);
                }
                else
                {
                    C[row * ldc + colB] = rocsparse_fma(beta, C[row * ldc + colB], alpha * sum);
                }
            }
        }
    }
}

#endif // CSRMM_DEVICE_ROW_SPLIT_H
