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

#include "rocsparse_bsrxmv_spzl.hpp"

// General BSRXMV that works for any BSR block dimensions
template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T>
__device__ void bsrxmvn_general_device(rocsparse_direction dir,
                                       T                   alpha,
                                       rocsparse_int       size_of_mask,
                                       const rocsparse_int* __restrict__ bsr_mask_ptr,
                                       const rocsparse_int* __restrict__ bsr_row_ptr,
                                       const rocsparse_int* __restrict__ bsr_end_ptr,
                                       const rocsparse_int* __restrict__ bsr_col_ind,
                                       const T* __restrict__ bsr_val,
                                       rocsparse_int block_dim,
                                       const T* __restrict__ x,
                                       T beta,
                                       T* __restrict__ y,
                                       rocsparse_index_base idx_base)
{
    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each thread block processes a single BSR row
    rocsparse_int row = hipBlockIdx_x;

    if(bsr_mask_ptr != nullptr)
    {
        row = bsr_mask_ptr[row] - idx_base;
    }

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = (bsr_end_ptr == nullptr) ? (bsr_row_ptr[row + 1] - idx_base)
                                                       : (bsr_end_ptr[row] - idx_base);
#if 0
    // Each thread block processes a BSR row
    rocsparse_int row = bsr_mask_ptr[hipBlockIdx_x] - idx_base;


    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_end_ptr[row] - idx_base;
#endif
    // Each wavefront processes a row of the BSR block.
    // If the number of BSR block rows exceed the number of wavefronts, each wavefront
    // processes multiple rows. 'bi' is the row index into the BSR block and 'bj' is
    // the column index.
    // BLOCKSIZE must be the square of WFSIZE.

    // Loop over the rows of the BSR block in chunks of WFSIZE, such that each
    // wavefront will process a row
    for(rocsparse_int bi = wid; bi < block_dim; bi += WFSIZE)
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
            for(rocsparse_int bj = lid; bj < block_dim; bj += WFSIZE)
            {
                // Each lane computes the sum of a specific entry over all BSR blocks in
                // the current row
                sum = rocsparse_fma(bsr_val[BSR_IND(j, bi, bj, dir)], x[block_dim * col + bj], sum);
            }
        }

        // Each wavefront accumulates its BSR block row sum
        sum = rocsparse_wfreduce_sum<WFSIZE>(sum);

        // Last lane of each wavefront writes its result to global memory
        if(lid == WFSIZE - 1)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * block_dim + bi] = rocsparse_fma(beta, y[row * block_dim + bi], alpha * sum);
            }
            else
            {
                y[row * block_dim + bi] = alpha * sum;
            }
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void bsrxmvn_general_kernel(rocsparse_direction dir,
                                U                   alpha_device_host,
                                rocsparse_int       size_of_mask,
                                const rocsparse_int* __restrict__ bsr_mask_ptr,
                                const rocsparse_int* __restrict__ bsr_row_ptr,
                                const rocsparse_int* __restrict__ bsr_end_ptr,
                                const rocsparse_int* __restrict__ bsr_col_ind,
                                const T* __restrict__ bsr_val,
                                rocsparse_int block_dim,
                                const T* __restrict__ x,
                                U beta_device_host,
                                T* __restrict__ y,
                                rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
    {
        bsrxmvn_general_device<BLOCKSIZE, WFSIZE>(dir,
                                                  alpha,
                                                  size_of_mask,
                                                  bsr_mask_ptr,
                                                  bsr_row_ptr,
                                                  bsr_end_ptr,
                                                  bsr_col_ind,
                                                  bsr_val,
                                                  block_dim,
                                                  x,
                                                  beta,
                                                  y,
                                                  idx_base);
    }
}

template <typename T, typename U>
void bsrxmvn_general(rocsparse_handle     handle,
                     rocsparse_direction  dir,
                     rocsparse_int        mb,
                     U                    alpha_device_host,
                     rocsparse_int        size_of_mask,
                     const rocsparse_int* bsr_mask_ptr,
                     const rocsparse_int* bsr_row_ptr,
                     const rocsparse_int* bsr_end_ptr,
                     const rocsparse_int* bsr_col_ind,
                     const T*             bsr_val,
                     rocsparse_int        block_dim,
                     const T*             x,
                     U                    beta_device_host,
                     T*                   y,
                     rocsparse_index_base base)
{
    const rocsparse_int size = (bsr_mask_ptr == nullptr) ? mb : size_of_mask;
    // Differentiate BSR block dimensions
    if(block_dim <= 8)
    {
        hipLaunchKernelGGL((bsrxmvn_general_kernel<64, 8>),
                           dim3(size),
                           dim3(8 * 8),
                           0,
                           handle->stream,
                           dir,
                           alpha_device_host,
                           size_of_mask,
                           bsr_mask_ptr,
                           bsr_row_ptr,
                           bsr_end_ptr,
                           bsr_col_ind,
                           bsr_val,
                           block_dim,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else if(block_dim <= 16)
    {
        hipLaunchKernelGGL((bsrxmvn_general_kernel<256, 16>),
                           dim3(size),
                           dim3(16 * 16),
                           0,
                           handle->stream,
                           dir,
                           alpha_device_host,
                           size_of_mask,
                           bsr_mask_ptr,
                           bsr_row_ptr,
                           bsr_end_ptr,
                           bsr_col_ind,
                           bsr_val,
                           block_dim,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else
    {
        hipLaunchKernelGGL((bsrxmvn_general_kernel<1024, 32>),
                           dim3(size),
                           dim3(32 * 32),
                           0,
                           handle->stream,
                           dir,
                           alpha_device_host,
                           size_of_mask,
                           bsr_mask_ptr,
                           bsr_row_ptr,
                           bsr_end_ptr,
                           bsr_col_ind,
                           bsr_val,
                           block_dim,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
}

//
// INSTANTIATE.
//
#define INSTANTIATE(TYPE)                                                 \
    template void bsrxmvn_general(rocsparse_handle     handle,            \
                                  rocsparse_direction  dir,               \
                                  rocsparse_int        mb,                \
                                  const TYPE*          alpha_device_host, \
                                  rocsparse_int        size_of_mask,      \
                                  const rocsparse_int* bsr_mask_ptr,      \
                                  const rocsparse_int* bsr_row_ptr,       \
                                  const rocsparse_int* bsr_end_ptr,       \
                                  const rocsparse_int* bsr_col_ind,       \
                                  const TYPE*          bsr_val,           \
                                  rocsparse_int        block_dim,         \
                                  const TYPE*          x,                 \
                                  const TYPE*          beta_device_host,  \
                                  TYPE*                y,                 \
                                  rocsparse_index_base base);             \
    template void bsrxmvn_general(rocsparse_handle     handle,            \
                                  rocsparse_direction  dir,               \
                                  rocsparse_int        mb,                \
                                  TYPE                 alpha_device_host, \
                                  rocsparse_int        size_of_mask,      \
                                  const rocsparse_int* bsr_mask_ptr,      \
                                  const rocsparse_int* bsr_row_ptr,       \
                                  const rocsparse_int* bsr_end_ptr,       \
                                  const rocsparse_int* bsr_col_ind,       \
                                  const TYPE*          bsr_val,           \
                                  rocsparse_int        block_dim,         \
                                  const TYPE*          x,                 \
                                  TYPE                 beta_device_host,  \
                                  TYPE*                y,                 \
                                  rocsparse_index_base base)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE
