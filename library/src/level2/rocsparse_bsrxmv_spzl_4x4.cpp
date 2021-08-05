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

// BSRXMV kernel for BSR block dimension of 4
template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T>
__device__ void bsrxmvn_4x4_device(rocsparse_int       mb,
                                   rocsparse_direction dir,
                                   T                   alpha,
                                   rocsparse_int       size_of_mask,
                                   const rocsparse_int* __restrict__ bsr_mask_ptr,
                                   const rocsparse_int* __restrict__ bsr_row_ptr,
                                   const rocsparse_int* __restrict__ bsr_end_ptr,
                                   const rocsparse_int* __restrict__ bsr_col_ind,
                                   const T* __restrict__ bsr_val,
                                   const T* __restrict__ x,
                                   T beta,
                                   T* __restrict__ y,
                                   rocsparse_index_base idx_base)
{
    // BSR block dimension
    static constexpr int BSRDIM = 4;

    // Lane id
    rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    rocsparse_int wid = hipThreadIdx_x / WFSIZE;

    // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows
    rocsparse_int row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

    // Do not run out of bounds
    if(bsr_mask_ptr == nullptr)
    {
        if(row >= mb)
        {
            return;
        }
    }
    else
    {
        if(row >= size_of_mask)
        {
            return;
        }
        row = bsr_mask_ptr[row] - idx_base;
    }

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = (bsr_end_ptr == nullptr) ? (bsr_row_ptr[row + 1] - idx_base)
                                                       : (bsr_end_ptr[row] - idx_base);
#if 0
    // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows


    rocsparse_int mask_idx = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;
    // Do not run out of bounds
    if(mask_idx >= size_of_mask)
    {
        return;
    }
    rocsparse_int row = bsr_mask_ptr[mask_idx] - idx_base;

    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_end_ptr[row] - idx_base;
#endif
    // BSR block row accumulator
    T sum0 = static_cast<T>(0);
    T sum1 = static_cast<T>(0);
    T sum2 = static_cast<T>(0);
    T sum3 = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block
    if(dir == rocsparse_direction_column)
    {
        for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Do not exceed the row
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[j] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 0], x[col + 0], sum0);
            sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 1], x[col + 0], sum1);
            sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 2], x[col + 0], sum2);
            sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 3], x[col + 0], sum3);

            sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 4], x[col + 1], sum0);
            sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 5], x[col + 1], sum1);
            sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 6], x[col + 1], sum2);
            sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 7], x[col + 1], sum3);

            sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 8], x[col + 2], sum0);
            sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 9], x[col + 2], sum1);
            sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 10], x[col + 2], sum2);
            sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 11], x[col + 2], sum3);

            sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 12], x[col + 3], sum0);
            sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 13], x[col + 3], sum1);
            sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 14], x[col + 3], sum2);
            sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 15], x[col + 3], sum3);
        }
    }
    else
    {
        for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Do not exceed the row
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[j] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 0], x[col + 0], sum0);
            sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 1], x[col + 1], sum0);
            sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 2], x[col + 2], sum0);
            sum0 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 3], x[col + 3], sum0);

            sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 4], x[col + 0], sum1);
            sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 5], x[col + 1], sum1);
            sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 6], x[col + 2], sum1);
            sum1 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 7], x[col + 3], sum1);

            sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 8], x[col + 0], sum2);
            sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 9], x[col + 1], sum2);
            sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 10], x[col + 2], sum2);
            sum2 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 11], x[col + 3], sum2);

            sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 12], x[col + 0], sum3);
            sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 13], x[col + 1], sum3);
            sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 14], x[col + 2], sum3);
            sum3 = rocsparse_fma(bsr_val[BSRDIM * BSRDIM * j + 15], x[col + 3], sum3);
        }
    }
    // Each wavefront accumulates its BSR block row sum
    sum0 = rocsparse_wfreduce_sum<WFSIZE>(sum0);
    sum1 = rocsparse_wfreduce_sum<WFSIZE>(sum1);
    sum2 = rocsparse_wfreduce_sum<WFSIZE>(sum2);
    sum3 = rocsparse_wfreduce_sum<WFSIZE>(sum3);

    // Last lane of each wavefront writes the two row sums to global memory
    if(lid == WFSIZE - 1)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + 0] = rocsparse_fma(beta, y[row * BSRDIM + 0], alpha * sum0);
            y[row * BSRDIM + 1] = rocsparse_fma(beta, y[row * BSRDIM + 1], alpha * sum1);
            y[row * BSRDIM + 2] = rocsparse_fma(beta, y[row * BSRDIM + 2], alpha * sum2);
            y[row * BSRDIM + 3] = rocsparse_fma(beta, y[row * BSRDIM + 3], alpha * sum3);
        }
        else
        {
            y[row * BSRDIM + 0] = alpha * sum0;
            y[row * BSRDIM + 1] = alpha * sum1;
            y[row * BSRDIM + 2] = alpha * sum2;
            y[row * BSRDIM + 3] = alpha * sum3;
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void bsrxmvn_4x4_kernel(rocsparse_int       mb,
                            rocsparse_direction dir,
                            U                   alpha_device_host,
                            rocsparse_int       size_of_mask,
                            const rocsparse_int* __restrict__ bsr_mask_ptr,
                            const rocsparse_int* __restrict__ bsr_row_ptr,
                            const rocsparse_int* __restrict__ bsr_end_ptr,
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
        bsrxmvn_4x4_device<BLOCKSIZE, WFSIZE>(mb,
                                              dir,
                                              alpha,
                                              size_of_mask,
                                              bsr_mask_ptr,
                                              bsr_row_ptr,
                                              bsr_end_ptr,
                                              bsr_col_ind,
                                              bsr_val,
                                              x,
                                              beta,
                                              y,
                                              idx_base);
    }
}

template <typename T, typename U>
void bsrxmvn_4x4(rocsparse_handle     handle,
                 rocsparse_direction  dir,
                 rocsparse_int        mb,
                 rocsparse_int        nnzb,
                 U                    alpha_device_host,
                 rocsparse_int        size_of_mask,
                 const rocsparse_int* bsr_mask_ptr,
                 const rocsparse_int* bsr_row_ptr,
                 const rocsparse_int* bsr_end_ptr,
                 const rocsparse_int* bsr_col_ind,
                 const T*             bsr_val,
                 const T*             x,
                 U                    beta_device_host,
                 T*                   y,
                 rocsparse_index_base base)
{
    const rocsparse_int blocks_per_row = nnzb / mb;
    const rocsparse_int size           = (bsr_mask_ptr == nullptr) ? mb : size_of_mask;

#define BSRXMVN_DIM 128
    if(blocks_per_row < 8)
    {
        hipLaunchKernelGGL((bsrxmvn_4x4_kernel<BSRXMVN_DIM, 4>),
                           dim3((size - 1) / (BSRXMVN_DIM / 4) + 1),
                           dim3(BSRXMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           size_of_mask,
                           bsr_mask_ptr,
                           bsr_row_ptr,
                           bsr_end_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else if(blocks_per_row < 16)
    {
        hipLaunchKernelGGL((bsrxmvn_4x4_kernel<BSRXMVN_DIM, 8>),
                           dim3((size - 1) / (BSRXMVN_DIM / 8) + 1),
                           dim3(BSRXMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           size_of_mask,
                           bsr_mask_ptr,
                           bsr_row_ptr,
                           bsr_end_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else if(blocks_per_row < 32)
    {
        hipLaunchKernelGGL((bsrxmvn_4x4_kernel<BSRXMVN_DIM, 16>),
                           dim3((size - 1) / (BSRXMVN_DIM / 16) + 1),
                           dim3(BSRXMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           size_of_mask,
                           bsr_mask_ptr,
                           bsr_row_ptr,
                           bsr_end_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else if(blocks_per_row < 64)
    {
        hipLaunchKernelGGL((bsrxmvn_4x4_kernel<BSRXMVN_DIM, 32>),
                           dim3((size - 1) / (BSRXMVN_DIM / 32) + 1),
                           dim3(BSRXMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           size_of_mask,
                           bsr_mask_ptr,
                           bsr_row_ptr,
                           bsr_end_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }
    else
    {
        hipLaunchKernelGGL((bsrxmvn_4x4_kernel<BSRXMVN_DIM, 64>),
                           dim3((size - 1) / (BSRXMVN_DIM / 64) + 1),
                           dim3(BSRXMVN_DIM),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha_device_host,
                           size_of_mask,
                           bsr_mask_ptr,
                           bsr_row_ptr,
                           bsr_end_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta_device_host,
                           y,
                           base);
    }

#undef BSRXMVN_DIM
}

//
// INSTANTIATE.
//
#define INSTANTIATE(TYPE)                                             \
    template void bsrxmvn_4x4(rocsparse_handle     handle,            \
                              rocsparse_direction  dir,               \
                              rocsparse_int        mb,                \
                              rocsparse_int        nnzb,              \
                              const TYPE*          alpha_device_host, \
                              rocsparse_int        size_of_mask,      \
                              const rocsparse_int* bsr_mask_ptr,      \
                              const rocsparse_int* bsr_row_ptr,       \
                              const rocsparse_int* bsr_end_ptr,       \
                              const rocsparse_int* bsr_col_ind,       \
                              const TYPE*          bsr_val,           \
                              const TYPE*          x,                 \
                              const TYPE*          beta_device_host,  \
                              TYPE*                y,                 \
                              rocsparse_index_base base);             \
    template void bsrxmvn_4x4(rocsparse_handle     handle,            \
                              rocsparse_direction  dir,               \
                              rocsparse_int        mb,                \
                              rocsparse_int        nnzb,              \
                              TYPE                 alpha_device_host, \
                              rocsparse_int        size_of_mask,      \
                              const rocsparse_int* bsr_mask_ptr,      \
                              const rocsparse_int* bsr_row_ptr,       \
                              const rocsparse_int* bsr_end_ptr,       \
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
