/*! \file */
/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef BELLMM_GENERAL_DEVICE_H
#define BELLMM_GENERAL_DEVICE_H

#include "common.h"

template <rocsparse_int BELL_BLOCK_DIM, rocsparse_int BLK_SIZE_Y, typename T, typename I>
static ROCSPARSE_DEVICE_ILF void bellmm_general_blockdim_device(rocsparse_operation trans_A,
                                                                rocsparse_operation trans_B,
                                                                rocsparse_order     order_B,
                                                                rocsparse_order     order_C,
                                                                rocsparse_direction dir_A,
                                                                I                   Mb,
                                                                I                   N,
                                                                T                   alpha,
                                                                I                   bell_cols,
                                                                I                   block_dim,
                                                                const I* __restrict__ bell_col_ind,
                                                                const T* __restrict__ bell_val,
                                                                const T* __restrict__ B,
                                                                I ldb,
                                                                T beta,
                                                                T* __restrict__ C,
                                                                I                    ldc,
                                                                rocsparse_index_base idx_base)
{
    const I tidx       = hipThreadIdx_x;
    const I tidy       = hipThreadIdx_y;
    const I block_row  = hipBlockIdx_x;
    const I bell_width = (block_row < Mb) ? (bell_cols / block_dim) : 0;

    __shared__ T shared_B[BELL_BLOCK_DIM * BLK_SIZE_Y];
    __shared__ T shared_A[BELL_BLOCK_DIM * BELL_BLOCK_DIM];

    const I global_col = tidy + hipBlockIdx_y * BLK_SIZE_Y;
    const I colB       = global_col * ldb;

    for(I x = 0; x < block_dim; x += BELL_BLOCK_DIM)
    {
        const I global_row = tidx + x + hipBlockIdx_x * block_dim;

        T sum = static_cast<T>(0);

        for(I j = 0; j < bell_width; ++j)
        {
            const I ell_idx   = j * Mb + block_row;
            const I block_col = (bell_col_ind[ell_idx] - idx_base);

            for(I y = 0; y < block_dim; y += BLK_SIZE_Y)
            {
                const bool is_A_valid
                    = ((tidx + x) < block_dim && (tidy + y) < block_dim) && (block_col >= 0);
                const bool is_B_valid
                    = ((global_col < N) && ((tidx + y) < block_dim)) && (block_col >= 0);

                if((trans_B == rocsparse_operation_none && order_B == rocsparse_order_column)
                   || (trans_B != rocsparse_operation_none && order_B != rocsparse_order_column))
                {
                    shared_B[BELL_BLOCK_DIM * tidy + tidx]
                        = (is_B_valid) ? B[block_dim * block_col + (tidx + y) + colB]
                                       : static_cast<T>(0);
                }
                else
                {
                    shared_B[BELL_BLOCK_DIM * tidy + tidx]
                        = (is_B_valid) ? B[global_col + ldb * (block_dim * block_col + (tidx + y))]
                                       : static_cast<T>(0);
                }
                if(dir_A == rocsparse_direction_row)
                {
                    shared_A[BELL_BLOCK_DIM * tidy + tidx]
                        = (is_A_valid) ? bell_val[block_dim * block_dim * ell_idx
                                                  + block_dim * (tidx + x) + (tidy + y)]
                                       : static_cast<T>(0);
                }
                else
                {
                    shared_A[BELL_BLOCK_DIM * tidy + tidx]
                        = (is_A_valid) ? bell_val[block_dim * block_dim * ell_idx
                                                  + block_dim * (tidy + y) + (tidx + x)]
                                       : static_cast<T>(0);
                }

                __syncthreads();

                if(block_col >= 0)
                {
                    if((trans_A == rocsparse_operation_conjugate_transpose)
                       && (trans_B == rocsparse_operation_conjugate_transpose))
                    {
                        for(I j = 0; j < BELL_BLOCK_DIM; j++)
                        {
                            sum = rocsparse_fma(rocsparse_conj(shared_A[BELL_BLOCK_DIM * j + tidx]),
                                                rocsparse_conj(shared_B[BELL_BLOCK_DIM * tidy + j]),
                                                sum);
                        }
                    }
                    else if((trans_A != rocsparse_operation_conjugate_transpose)
                            && (trans_B == rocsparse_operation_conjugate_transpose))
                    {
                        for(I j = 0; j < BELL_BLOCK_DIM; j++)
                        {
                            sum = rocsparse_fma(shared_A[BELL_BLOCK_DIM * j + tidx],
                                                rocsparse_conj(shared_B[BELL_BLOCK_DIM * tidy + j]),
                                                sum);
                        }
                    }
                    else if((trans_A == rocsparse_operation_conjugate_transpose)
                            && (trans_B != rocsparse_operation_conjugate_transpose))
                    {
                        for(I j = 0; j < BELL_BLOCK_DIM; j++)
                        {
                            sum = rocsparse_fma(rocsparse_conj(shared_A[BELL_BLOCK_DIM * j + tidx]),
                                                shared_B[BELL_BLOCK_DIM * tidy + j],
                                                sum);
                        }
                    }
                    else
                    {
                        for(I j = 0; j < BELL_BLOCK_DIM; j++)
                        {
                            sum = rocsparse_fma(shared_A[BELL_BLOCK_DIM * j + tidx],
                                                shared_B[BELL_BLOCK_DIM * tidy + j],
                                                sum);
                        }
                    }
                }

                __syncthreads();
            }
        }

        const I shift_C = (order_C == rocsparse_order_column) ? (global_row + ldc * global_col)
                                                              : (global_row * ldc + global_col);
        if(block_row < Mb && global_col < N && (tidx + x) < block_dim)
        {
            if(beta == static_cast<T>(0))
            {
                C[shift_C] = alpha * sum;
            }
            else
            {
                C[shift_C] = rocsparse_fma(beta, C[shift_C], alpha * sum);
            }
        }
    }
}

#endif // BELLMM_GENERAL_DEVICE_H
