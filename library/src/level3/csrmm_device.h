/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnn_general_device(bool    conj_A,
                                                     bool    conj_B,
                                                     J       M,
                                                     J       N,
                                                     J       K,
                                                     I       nnz,
                                                     int64_t offsets_batch_stride_A,
                                                     int64_t columns_values_batch_stride_A,
                                                     T       alpha,
                                                     const I* __restrict__ csr_row_ptr,
                                                     const J* __restrict__ csr_col_ind,
                                                     const A* __restrict__ csr_val,
                                                     const B* __restrict__ dense_B,
                                                     int64_t ldb,
                                                     int64_t batch_stride_B,
                                                     T       beta,
                                                     C* __restrict__ dense_C,
                                                     int64_t              ldc,
                                                     int64_t              batch_stride_C,
                                                     rocsparse_order      order_C,
                                                     rocsparse_index_base idx_base)
    {
        int tid = hipThreadIdx_x;
        J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
        int lid = gid & (WF_SIZE - 1);
        int wid = tid / WF_SIZE;
        J   row = gid / WF_SIZE;
        J   col = lid + hipBlockIdx_y * WF_SIZE;

        J batch = hipBlockIdx_z;

        if(row >= M)
        {
            return;
        }

        int64_t colB = col * ldb;

        __shared__ J shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
        __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

        I row_start = csr_row_ptr[row + offsets_batch_stride_A * batch] - idx_base;
        I row_end   = csr_row_ptr[row + 1 + offsets_batch_stride_A * batch] - idx_base;

        T sum = static_cast<T>(0);

        for(I j = row_start; j < row_end; j += WF_SIZE)
        {
            I k = j + lid;

            __syncthreads();

            if(k < row_end)
            {
                shared_col[wid][lid]
                    = csr_col_ind[k + columns_values_batch_stride_A * batch] - idx_base;
                shared_val[wid][lid]
                    = conj_val(csr_val[k + columns_values_batch_stride_A * batch], conj_A);
            }
            else
            {
                shared_col[wid][lid] = 0;
                shared_val[wid][lid] = static_cast<T>(0);
            }

            __syncthreads();

            if(col < N)
            {
                for(unsigned int i = 0; i < WF_SIZE; ++i)
                {
                    sum = rocsparse_fma<T>(
                        shared_val[wid][i],
                        conj_val(dense_B[shared_col[wid][i] + colB + batch_stride_B * batch],
                                 conj_B),
                        sum);
                }
            }
        }

        if(col < N)
        {
            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    dense_C[row + col * ldc + batch_stride_C * batch] = alpha * sum;
                }
                else
                {
                    dense_C[row * ldc + col + batch_stride_C * batch] = alpha * sum;
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    dense_C[row + col * ldc + batch_stride_C * batch] = rocsparse_fma<T>(
                        beta, dense_C[row + col * ldc + batch_stride_C * batch], alpha * sum);
                }
                else
                {
                    dense_C[row * ldc + col + batch_stride_C * batch] = rocsparse_fma<T>(
                        beta, dense_C[row * ldc + col + batch_stride_C * batch], alpha * sum);
                }
            }
        }
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              unsigned int LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnt_general_main_device(bool    conj_A,
                                                          bool    conj_B,
                                                          J       offset,
                                                          J       ncol,
                                                          J       M,
                                                          J       N,
                                                          J       K,
                                                          I       nnz,
                                                          int64_t offsets_batch_stride_A,
                                                          int64_t columns_values_batch_stride_A,
                                                          T       alpha,
                                                          const I* __restrict__ csr_row_ptr,
                                                          const J* __restrict__ csr_col_ind,
                                                          const A* __restrict__ csr_val,
                                                          const B* __restrict__ dense_B,
                                                          int64_t ldb,
                                                          int64_t batch_stride_B,
                                                          T       beta,
                                                          C* __restrict__ dense_C,
                                                          int64_t              ldc,
                                                          int64_t              batch_stride_C,
                                                          rocsparse_order      order_C,
                                                          rocsparse_index_base idx_base)
    {
        int tid = hipThreadIdx_x;
        J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
        J   row = gid / WF_SIZE;
        int lid = tid & (WF_SIZE - 1);
        int wid = tid / WF_SIZE;

        J batch = hipBlockIdx_y;

        if(row >= M)
        {
            return;
        }

        __shared__ J shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
        __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

        I row_start = rocsparse_nontemporal_load(csr_row_ptr + row + offsets_batch_stride_A * batch)
                      - idx_base;
        I row_end
            = rocsparse_nontemporal_load(csr_row_ptr + row + 1 + offsets_batch_stride_A * batch)
              - idx_base;

        for(J l = 0; l < ncol; l += WF_SIZE * LOOPS)
        {
            J col = l + lid;

            T sum[LOOPS];

            for(unsigned int p = 0; p < LOOPS; p++)
            {
                sum[p] = static_cast<T>(0);
            }

            for(I j = row_start; j < row_end; j += WF_SIZE)
            {
                I k = j + lid;

                __threadfence_block();

                if(k < row_end)
                {
                    shared_col[wid][lid]
                        = (rocsparse_nontemporal_load(csr_col_ind + k
                                                      + columns_values_batch_stride_A * batch)
                           - idx_base);
                    shared_val[wid][lid]
                        = conj_val(rocsparse_nontemporal_load(
                                       csr_val + k + columns_values_batch_stride_A * batch),
                                   conj_A);
                }
                else
                {
                    shared_col[wid][lid] = 0;
                    shared_val[wid][lid] = static_cast<T>(0);
                }

                __threadfence_block();

                for(unsigned int i = 0; i < WF_SIZE; ++i)
                {
                    int64_t sc = ldb * shared_col[wid][i];
                    T       sv = shared_val[wid][i];

                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse_fma<T>(
                            sv,
                            conj_val(rocsparse_ldg(dense_B + col + p * WF_SIZE + sc
                                                   + batch_stride_B * batch),
                                     conj_B),
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
                        dense_C[row + (col + p * WF_SIZE) * ldc + batch_stride_C * batch]
                            = alpha * sum[p];
                    }
                }
                else
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + col + p * WF_SIZE + batch_stride_C * batch]
                            = alpha * sum[p];
                    }
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row + (col + p * WF_SIZE) * ldc + batch_stride_C * batch]
                            = rocsparse_fma<T>(
                                beta,
                                dense_C[row + (col + p * WF_SIZE) * ldc + batch_stride_C * batch],
                                alpha * sum[p]);
                    }
                }
                else
                {
                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + col + p * WF_SIZE + batch_stride_C * batch]
                            = rocsparse_fma<T>(
                                beta,
                                dense_C[row * ldc + col + p * WF_SIZE + batch_stride_C * batch],
                                alpha * sum[p]);
                    }
                }
            }
        }
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void
        csrmmnt_general_remainder_device(bool    conj_A,
                                         bool    conj_B,
                                         J       offset,
                                         J       ncol,
                                         J       M,
                                         J       N,
                                         J       K,
                                         I       nnz,
                                         int64_t offsets_batch_stride_A,
                                         int64_t columns_values_batch_stride_A,
                                         T       alpha,
                                         const I* __restrict__ csr_row_ptr,
                                         const J* __restrict__ csr_col_ind,
                                         const A* __restrict__ csr_val,
                                         const B* __restrict__ dense_B,
                                         int64_t ldb,
                                         int64_t batch_stride_B,
                                         T       beta,
                                         C* __restrict__ dense_C,
                                         int64_t              ldc,
                                         int64_t              batch_stride_C,
                                         rocsparse_order      order_C,
                                         rocsparse_index_base idx_base)
    {
        int tid = hipThreadIdx_x;
        J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
        J   row = gid / WF_SIZE;
        int lid = tid & (WF_SIZE - 1);
        int wid = tid / WF_SIZE;

        J batch = hipBlockIdx_y;

        if(row >= M)
        {
            return;
        }

        __shared__ J shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
        __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

        I row_start = rocsparse_nontemporal_load(csr_row_ptr + row + offsets_batch_stride_A * batch)
                      - idx_base;
        I row_end
            = rocsparse_nontemporal_load(csr_row_ptr + row + 1 + offsets_batch_stride_A * batch)
              - idx_base;

        for(J l = offset; l < ncol; l += WF_SIZE)
        {
            J col = l + lid;
            T sum = static_cast<T>(0);

            for(I j = row_start; j < row_end; j += WF_SIZE)
            {
                I k = j + lid;

                __syncthreads();

                if(k < row_end)
                {
                    shared_col[wid][lid]
                        = (rocsparse_nontemporal_load(csr_col_ind + k
                                                      + columns_values_batch_stride_A * batch)
                           - idx_base);
                    shared_val[wid][lid]
                        = conj_val(rocsparse_nontemporal_load(
                                       csr_val + k + columns_values_batch_stride_A * batch),
                                   conj_A);
                }
                else
                {
                    shared_col[wid][lid] = 0;
                    shared_val[wid][lid] = static_cast<T>(0);
                }

                __syncthreads();

                if(col < ncol)
                {
                    for(unsigned int i = 0; i < WF_SIZE; ++i)
                    {
                        sum = rocsparse_fma<T>(
                            shared_val[wid][i],
                            conj_val(rocsparse_ldg(dense_B + col + ldb * shared_col[wid][i]
                                                   + batch_stride_B * batch),
                                     conj_B),
                            sum);
                    }
                }
            }

            if(col < ncol)
            {
                if(beta == static_cast<T>(0))
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[row + col * ldc + batch_stride_C * batch] = alpha * sum;
                    }
                    else
                    {
                        dense_C[row * ldc + col + batch_stride_C * batch] = alpha * sum;
                    }
                }
                else
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[row + col * ldc + batch_stride_C * batch] = rocsparse_fma<T>(
                            beta, dense_C[row + col * ldc + batch_stride_C * batch], alpha * sum);
                    }
                    else
                    {
                        dense_C[row * ldc + col + batch_stride_C * batch] = rocsparse_fma<T>(
                            beta, dense_C[row * ldc + col + batch_stride_C * batch], alpha * sum);
                    }
                }
            }
        }
    }

    // Scale kernel for beta != 1.0
    template <typename I, typename T>
    ROCSPARSE_DEVICE_ILF void csrmm_scale_device(
        I m, I n, T beta, T* __restrict__ data, int64_t ld, int64_t stride, rocsparse_order order)
    {
        I gidx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        I gidy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

        I batch = hipBlockIdx_z;

        if(gidx >= m || gidy >= n)
        {
            return;
        }

        if(order == rocsparse_order_column)
        {
            data[gidx + ld * gidy + stride * batch]
                = data[gidx + ld * gidy + stride * batch] * beta;
        }
        else
        {
            data[gidy + ld * gidx + stride * batch]
                = data[gidy + ld * gidx + stride * batch] * beta;
        }
    }

    // See Y. Tao et al., "Atomic reduction based sparse matrix-transpose vector multiplication on GPUs,"
    // 2014 20th IEEE International Conference on Parallel and Distributed Systems (ICPADS), 2014, pp. 987-992,
    // doi: 10.1109/PADSW.2014.7097920.
    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmtn_general_device(bool    conj_A,
                                                     bool    conj_B,
                                                     J       M,
                                                     J       N,
                                                     J       K,
                                                     I       nnz,
                                                     int64_t offsets_batch_stride_A,
                                                     int64_t columns_values_batch_stride_A,
                                                     T       alpha,
                                                     const I* __restrict__ csr_row_ptr,
                                                     const J* __restrict__ csr_col_ind,
                                                     const A* __restrict__ csr_val,
                                                     const B* __restrict__ dense_B,
                                                     int64_t ldb,
                                                     int64_t batch_stride_B,
                                                     T       beta,
                                                     C* __restrict__ dense_C,
                                                     int64_t              ldc,
                                                     int64_t              batch_stride_C,
                                                     rocsparse_order      order_C,
                                                     rocsparse_index_base idx_base)
    {
        int tid = hipThreadIdx_x;
        J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
        int lid = gid & (WF_SIZE - 1);
        int wid = tid / WF_SIZE;

        J row = gid / WF_SIZE;

        J batch = hipBlockIdx_z;

        if(row >= M)
        {
            return;
        }

        J       cid  = lid + hipBlockIdx_y * WF_SIZE;
        int64_t colB = cid * ldb;

        __shared__ T shared_B[BLOCKSIZE / WF_SIZE][WF_SIZE];
        shared_B[wid][lid] = (cid < N)
                                 ? conj_val(dense_B[row + colB + batch_stride_B * batch], conj_B)
                                 : static_cast<T>(0);

        __threadfence_block();
        I row_start = csr_row_ptr[row + offsets_batch_stride_A * batch] - idx_base;
        I row_end   = csr_row_ptr[row + 1 + offsets_batch_stride_A * batch] - idx_base;

        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            J col = csr_col_ind[j + columns_values_batch_stride_A * batch] - idx_base;
            T val = alpha * conj_val(csr_val[j + columns_values_batch_stride_A * batch], conj_A);

            if(order_C == rocsparse_order_column)
            {
                for(J i = 0; i < WF_SIZE && (i + hipBlockIdx_y * WF_SIZE) < N; ++i)
                {
                    rocsparse_atomic_add(&dense_C[col + (i + hipBlockIdx_y * WF_SIZE) * ldc
                                                  + batch_stride_C * batch],
                                         val * shared_B[wid][i]);
                }
            }
            else
            {
                for(J i = 0; i < WF_SIZE && (i + hipBlockIdx_y * WF_SIZE) < N; ++i)
                {
                    rocsparse_atomic_add(&dense_C[col * ldc + (i + hipBlockIdx_y * WF_SIZE)
                                                  + batch_stride_C * batch],
                                         val * shared_B[wid][i]);
                }
            }
        }
    }

    // See Y. Tao et al., "Atomic reduction based sparse matrix-transpose vector multiplication on GPUs,"
    // 2014 20th IEEE International Conference on Parallel and Distributed Systems (ICPADS), 2014, pp. 987-992,
    // doi: 10.1109/PADSW.2014.7097920.
    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmtt_general_device(bool    conj_A,
                                                     bool    conj_B,
                                                     J       M,
                                                     J       N,
                                                     J       K,
                                                     I       nnz,
                                                     int64_t offsets_batch_stride_A,
                                                     int64_t columns_values_batch_stride_A,
                                                     T       alpha,
                                                     const I* __restrict__ csr_row_ptr,
                                                     const J* __restrict__ csr_col_ind,
                                                     const A* __restrict__ csr_val,
                                                     const B* __restrict__ dense_B,
                                                     int64_t ldb,
                                                     int64_t batch_stride_B,
                                                     T       beta,
                                                     C* __restrict__ dense_C,
                                                     int64_t              ldc,
                                                     int64_t              batch_stride_C,
                                                     rocsparse_order      order_C,
                                                     rocsparse_index_base idx_base)
    {
        int tid = hipThreadIdx_x;
        J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
        int lid = gid & (WF_SIZE - 1);
        int wid = tid / WF_SIZE;

        J row = gid / WF_SIZE;
        J cid = lid + hipBlockIdx_y * WF_SIZE;

        J batch = hipBlockIdx_z;

        if(row >= M)
        {
            return;
        }

        __shared__ T shared_B[BLOCKSIZE / WF_SIZE][WF_SIZE];

        shared_B[wid][lid]
            = (cid < N) ? conj_val(dense_B[ldb * row + cid + batch_stride_B * batch], conj_B)
                        : static_cast<T>(0);

        __threadfence_block();
        I row_start = csr_row_ptr[row + offsets_batch_stride_A * batch] - idx_base;
        I row_end   = csr_row_ptr[row + 1 + offsets_batch_stride_A * batch] - idx_base;

        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            J col = csr_col_ind[j + columns_values_batch_stride_A * batch] - idx_base;
            T val = alpha * conj_val(csr_val[j + columns_values_batch_stride_A * batch], conj_A);

            if(order_C == rocsparse_order_column)
            {
                for(J i = 0; i < WF_SIZE && (i + hipBlockIdx_y * WF_SIZE) < N; ++i)
                {
                    rocsparse_atomic_add(&dense_C[col + (i + hipBlockIdx_y * WF_SIZE) * ldc
                                                  + batch_stride_C * batch],
                                         val * shared_B[wid][i]);
                }
            }
            else
            {
                for(J i = 0; i < WF_SIZE && (i + hipBlockIdx_y * WF_SIZE) < N; ++i)
                {
                    rocsparse_atomic_add(&dense_C[col * ldc + (i + hipBlockIdx_y * WF_SIZE)
                                                  + batch_stride_C * batch],
                                         val * shared_B[wid][i]);
                }
            }
        }
    }
}
