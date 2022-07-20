/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

template <rocsparse_int BSR_BLOCK_DIM,
          rocsparse_int BLK_SIZE_Y,
          rocsparse_int UNROLL_SIZE_Y,
          typename T>
static __device__ void
    bsrmm_large_blockdim_device_ext(rocsparse_direction direction,
                                    rocsparse_operation trans_B,
                                    rocsparse_int       Mb,
                                    rocsparse_int       N,
                                    T                   alpha,
                                    const rocsparse_int* __restrict__ bsr_row_ptr,
                                    const rocsparse_int* __restrict__ bsr_col_ind,
                                    const T* __restrict__ bsr_val,
                                    rocsparse_int block_dim,
                                    const T* __restrict__ B,
                                    rocsparse_int ldb,
                                    T             beta,
                                    T* __restrict__ C,
                                    rocsparse_int        ldc,
                                    rocsparse_index_base idx_base)
{
    rocsparse_int tidx = hipThreadIdx_x, tidy = hipThreadIdx_y;

    rocsparse_int global_row = tidx + hipBlockIdx_x * block_dim;

    __shared__ T shared_B[BSR_BLOCK_DIM * (BLK_SIZE_Y * UNROLL_SIZE_Y)];
    __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

    T             sum[UNROLL_SIZE_Y]{};
    bool          col_valid[UNROLL_SIZE_Y]{};
    rocsparse_int cols[UNROLL_SIZE_Y]{};
    for(rocsparse_int l = 0; l < UNROLL_SIZE_Y; ++l)
    {
        cols[l]      = (tidy + BLK_SIZE_Y * l) + hipBlockIdx_y * (BLK_SIZE_Y * UNROLL_SIZE_Y);
        col_valid[l] = (cols[l] < N);
    }

    rocsparse_int block_row       = hipBlockIdx_x;
    rocsparse_int block_row_start = 0;
    rocsparse_int block_row_end   = 0;
    if(block_row < Mb)
    {
        block_row_start = bsr_row_ptr[block_row] - idx_base;
        block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;
    }

    for(rocsparse_int l = 0; l < UNROLL_SIZE_Y; ++l)
    {
        sum[l] = static_cast<T>(0);
    }

    const rocsparse_int block_dim_sqr = block_dim * block_dim;
    const bool          is_tidx       = tidx < block_dim;
    const bool          is_tidy       = tidy < block_dim;
    const bool          is_tidx_tidy  = is_tidx && is_tidy;

    for(rocsparse_int k = block_row_start; k < block_row_end; k++)
    {
        rocsparse_int block_col = (bsr_col_ind[k] - idx_base);
        if(is_tidx)
        {
            if(trans_B == rocsparse_operation_none)
            {
                for(rocsparse_int l = 0; l < UNROLL_SIZE_Y; ++l)
                {
                    if(col_valid[l])
                    {
                        shared_B[BSR_BLOCK_DIM * (BLK_SIZE_Y * l + tidy) + tidx]
                            = B[block_dim * block_col + tidx + cols[l] * ldb];
                    }
                }
            }
            else
            {
                for(rocsparse_int l = 0; l < UNROLL_SIZE_Y; ++l)
                {
                    if(col_valid[l])
                    {
                        shared_B[BSR_BLOCK_DIM * (BLK_SIZE_Y * l + tidy) + tidx]
                            = B[cols[l] + ldb * (block_dim * block_col + tidx)];
                    }
                }
            }
        }

        if(is_tidx_tidy)
        {
            if(direction == rocsparse_direction_row)
            {
                shared_A[BSR_BLOCK_DIM * tidx + tidy]
                    = bsr_val[block_dim_sqr * k + block_dim * tidx + tidy];
            }
            else
            {
                shared_A[BSR_BLOCK_DIM * tidx + tidy]
                    = bsr_val[block_dim_sqr * k + block_dim * tidy + tidx];
            }
        }

        __syncthreads();

        if(is_tidx)
        {
            for(rocsparse_int l = 0; l < UNROLL_SIZE_Y; ++l)
            {
                if(col_valid[l])
                {
                    for(rocsparse_int j = 0; j < block_dim; j++)
                    {
                        sum[l]
                            = rocsparse_fma(shared_A[BSR_BLOCK_DIM * tidx + j],
                                            shared_B[BSR_BLOCK_DIM * (BLK_SIZE_Y * l + tidy) + j],
                                            sum[l]);
                    }
                }
            }
        }

        __syncthreads();
    }

    if(block_row < Mb && is_tidx)
    {
        for(rocsparse_int l = 0; l < UNROLL_SIZE_Y; ++l)
        {
            if(col_valid[l])
            {
                if(beta == static_cast<T>(0))
                {
                    C[global_row + cols[l] * ldc] = alpha * sum[l];
                }
                else
                {
                    C[global_row + cols[l] * ldc]
                        = rocsparse_fma(beta, C[global_row + cols[l] * ldc], alpha * sum[l]);
                }
            }
        }
    }
}
