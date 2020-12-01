/*! \file */
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

#include "rocsparse_bsrmv_spzl.hpp"

// BSRMV kernel for BSR block dimension of 2
template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T>
__device__ void bsrmvn_2x2_device(rocsparse_int       mb,
                                  rocsparse_direction dir,
                                  T                   alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  const T* __restrict__ x,
                                  T beta,
                                  T* __restrict__ y,
                                  rocsparse_index_base idx_base)
{
    // BSR block dimension
    static constexpr int BSRDIM = 2;

    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows
    rocsparse_int row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

    // Do not run out of bounds
    if(row >= mb)
    {
        return;
    }

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // BSR block row accumulator
    T sum0 = static_cast<T>(0);
    T sum1 = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block
    for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
    {
        // Do not exceed the row
        if(j + lid < row_end)
        {
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[j + lid] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            if(dir == rocsparse_direction_column)
            {
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 0], x[col + 0], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 1], x[col + 0], sum1);

                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 2], x[col + 1], sum0);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 3], x[col + 1], sum1);
            }
            else
            {
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 0], x[col + 0], sum0);
                sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 1], x[col + 1], sum0);

                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 2], x[col + 0], sum1);
                sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * (j + lid) + 3], x[col + 1], sum1);
            }
        }
    }

    // Each wavefront accumulates its BSR block row sum
    sum0 = rocsparse_wfreduce_sum<WFSIZE>(sum0);
    sum1 = rocsparse_wfreduce_sum<WFSIZE>(sum1);

    // Last lane of each wavefront writes the two row sums to global memory
    if(lid == WFSIZE - 1)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + 0] = rocsparse_fma(beta, y[row * BSRDIM + 0], alpha * sum0);
            y[row * BSRDIM + 1] = rocsparse_fma(beta, y[row * BSRDIM + 1], alpha * sum1);
        }
        else
        {
            y[row * BSRDIM + 0] = alpha * sum0;
            y[row * BSRDIM + 1] = alpha * sum1;
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T, typename U>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_2x2_kernel(rocsparse_int       mb,
                           rocsparse_direction dir,
                           U                   alpha_device_host,
                           const rocsparse_int* __restrict__ bsr_row_ptr,
                           const rocsparse_int* __restrict__ bsr_col_ind,
                           const T* __restrict__ bsr_val,
                           const T* __restrict__ x,
                           U beta_device_host,
                           T* __restrict__ y,
                           rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
    {
        bsrmvn_2x2_device<BLOCKSIZE, WFSIZE>(
            mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
    }
}

template <typename T, typename U>
void bsrmvn_2x2(rocsparse_handle     handle,
                rocsparse_direction  dir,
                rocsparse_int        mb,
                rocsparse_int        nnzb,
                U                    alpha_device_host,
                const rocsparse_int* bsr_row_ptr,
                const rocsparse_int* bsr_col_ind,
                const T*             bsr_val,
                const T*             x,
                U                    beta_device_host,
                T*                   y,
                rocsparse_index_base base)
{
    rocsparse_int blocks_per_row = nnzb / mb;

#define BSRMVN_DIM 128
    if(blocks_per_row < 8)
    {
        hipLaunchKernelGGL((bsrmvn_2x2_kernel<BSRMVN_DIM, 4>),
                           dim3((mb - 1) / (BSRMVN_DIM / 4) + 1),
                           dim3(BSRMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else if(blocks_per_row < 16)
    {
        hipLaunchKernelGGL((bsrmvn_2x2_kernel<BSRMVN_DIM, 8>),
                           dim3((mb - 1) / (BSRMVN_DIM / 8) + 1),
                           dim3(BSRMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else if(blocks_per_row < 32)
    {
        hipLaunchKernelGGL((bsrmvn_2x2_kernel<BSRMVN_DIM, 16>),
                           dim3((mb - 1) / (BSRMVN_DIM / 16) + 1),
                           dim3(BSRMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else if(blocks_per_row < 64)
    {
        hipLaunchKernelGGL((bsrmvn_2x2_kernel<BSRMVN_DIM, 32>),
                           dim3((mb - 1) / (BSRMVN_DIM / 32) + 1),
                           dim3(BSRMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else
    {
        hipLaunchKernelGGL((bsrmvn_2x2_kernel<BSRMVN_DIM, 64>),
                           dim3((mb - 1) / (BSRMVN_DIM / 64) + 1),
                           dim3(BSRMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }

#undef BSRMVN_DIM
}

//
// INSTANTIATE.
//
#define INSTANTIATE(TYPE)                                            \
    template void bsrmvn_2x2(rocsparse_handle     handle,            \
                             rocsparse_direction  dir,               \
                             rocsparse_int        mb,                \
                             rocsparse_int        nnzb,              \
                             const TYPE*          alpha_device_host, \
                             const rocsparse_int* bsr_row_ptr,       \
                             const rocsparse_int* bsr_col_ind,       \
                             const TYPE*          bsr_val,           \
                             const TYPE*          x,                 \
                             const TYPE*          beta_device_host,  \
                             TYPE*                y,                 \
                             rocsparse_index_base base);             \
    template void bsrmvn_2x2(rocsparse_handle     handle,            \
                             rocsparse_direction  dir,               \
                             rocsparse_int        mb,                \
                             rocsparse_int        nnzb,              \
                             TYPE                 alpha_device_host, \
                             const rocsparse_int* bsr_row_ptr,       \
                             const rocsparse_int* bsr_col_ind,       \
                             const TYPE*          bsr_val,           \
                             const TYPE*          x,                 \
                             TYPE                 beta_device_host,  \
                             TYPE*                y,                 \
                             rocsparse_index_base base)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE
