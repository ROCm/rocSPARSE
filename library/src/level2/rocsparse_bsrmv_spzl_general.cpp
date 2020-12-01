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

// General BSRMV that works for any BSR block dimensions
template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T>
__device__ void bsrmvn_general_device(rocsparse_direction dir,
                                      T                   alpha,
                                      const rocsparse_int* __restrict__ bsr_row_ptr,
                                      const rocsparse_int* __restrict__ bsr_col_ind,
                                      const T* __restrict__ bsr_val,
                                      rocsparse_int bsr_dim,
                                      const T* __restrict__ x,
                                      T beta,
                                      T* __restrict__ y,
                                      rocsparse_index_base idx_base)
{
    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each thread block processes a BSR row
    rocsparse_int row = hipBlockIdx_x;

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

    // Each wavefront processes a row of the BSR block.
    // If the number of BSR block rows exceed the number of wavefronts, each wavefront
    // processes multiple rows. 'bi' is the row index into the BSR block and 'bj' is
    // the column index.
    // BLOCKSIZE must be the square of WFSIZE.

    // Loop over the rows of the BSR block in chunks of WFSIZE, such that each
    // wavefront will process a row
    for(rocsparse_int bi = wid; bi < bsr_dim; bi += WFSIZE)
    {
        // BSR block row accumulator
        T sum = static_cast<T>(0);

        // Loop over all BSR blocks in the current row
        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            // BSR column index
            rocsparse_int col = bsr_col_ind[j] - idx_base;

            // Loop over the columns of the BSR block in chunks of WFSIZE, such that
            // each lane will process a single value of the BSR block
            for(rocsparse_int bj = lid; bj < bsr_dim; bj += WFSIZE)
            {
                // Each lane computes the sum of a specific entry over all BSR blocks in
                // the current row
                sum = rocsparse_fma(bsr_val[BSR_IND(j, bi, bj, dir)], x[bsr_dim * col + bj], sum);
            }
        }

        // Each wavefront accumulates its BSR block row sum
        sum = rocsparse_wfreduce_sum<WFSIZE>(sum);

        // Last lane of each wavefront writes its result to global memory
        if(lid == WFSIZE - 1)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * bsr_dim + bi] = rocsparse_fma(beta, y[row * bsr_dim + bi], alpha * sum);
            }
            else
            {
                y[row * bsr_dim + bi] = alpha * sum;
            }
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T, typename U>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_general_kernel(rocsparse_direction dir,
                               U                   alpha_device_host,
                               const rocsparse_int* __restrict__ bsr_row_ptr,
                               const rocsparse_int* __restrict__ bsr_col_ind,
                               const T* __restrict__ bsr_val,
                               rocsparse_int bsr_dim,
                               const T* __restrict__ x,
                               U beta_device_host,
                               T* __restrict__ y,
                               rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
    {
        bsrmvn_general_device<BLOCKSIZE, WFSIZE>(
            dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, bsr_dim, x, beta, y, idx_base);
    }
}

template <typename T, typename U>
void bsrmvn_general(rocsparse_handle     handle,
                    rocsparse_direction  dir,
                    rocsparse_int        mb,
                    U                    alpha_device_host,
                    const rocsparse_int* bsr_row_ptr,
                    const rocsparse_int* bsr_col_ind,
                    const T*             bsr_val,
                    rocsparse_int        bsr_dim,
                    const T*             x,
                    U                    beta_device_host,
                    T*                   y,
                    rocsparse_index_base base)
{

    // Differentiate BSR block dimensions
    if(bsr_dim <= 8)
    {
        hipLaunchKernelGGL((bsrmvn_general_kernel<64, 8>),
                           dim3(mb),
                           dim3(8 * 8),
                           0,
                           handle->stream,
                           dir,
                           alpha_device_host,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           bsr_dim,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else if(bsr_dim <= 16)
    {
        hipLaunchKernelGGL((bsrmvn_general_kernel<256, 16>),
                           dim3(mb),
                           dim3(16 * 16),
                           0,
                           handle->stream,
                           dir,
                           alpha_device_host,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           bsr_dim,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else
    {
        hipLaunchKernelGGL((bsrmvn_general_kernel<1024, 32>),
                           dim3(mb),
                           dim3(32 * 32),
                           0,
                           handle->stream,
                           dir,
                           alpha_device_host,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           bsr_dim,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
}

//
// INSTANTIATE.
//
#define INSTANTIATE(TYPE)                                                \
    template void bsrmvn_general(rocsparse_handle     handle,            \
                                 rocsparse_direction  dir,               \
                                 rocsparse_int        mb,                \
                                 const TYPE*          alpha_device_host, \
                                 const rocsparse_int* bsr_row_ptr,       \
                                 const rocsparse_int* bsr_col_ind,       \
                                 const TYPE*          bsr_val,           \
                                 rocsparse_int        bsr_dim,           \
                                 const TYPE*          x,                 \
                                 const TYPE*          beta_device_host,  \
                                 TYPE*                y,                 \
                                 rocsparse_index_base base);             \
    template void bsrmvn_general(rocsparse_handle     handle,            \
                                 rocsparse_direction  dir,               \
                                 rocsparse_int        mb,                \
                                 TYPE                 alpha_device_host, \
                                 const rocsparse_int* bsr_row_ptr,       \
                                 const rocsparse_int* bsr_col_ind,       \
                                 const TYPE*          bsr_val,           \
                                 rocsparse_int        bsr_dim,           \
                                 const TYPE*          x,                 \
                                 TYPE                 beta_device_host,  \
                                 TYPE*                y,                 \
                                 rocsparse_index_base base)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE
