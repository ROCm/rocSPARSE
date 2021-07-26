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

// BSRXMV kernel for BSR block dimension of 17 to 32
template <unsigned int BSRDIM, typename T>
__device__ void bsrxmvn_17_32_device(rocsparse_int       mb,
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
    // BSR block lane id
    rocsparse_int lid = hipThreadIdx_x % BSRDIM;

    // Offset into x vector
    rocsparse_int idx
        = (dir == rocsparse_direction_column) ? ((hipThreadIdx_x / BSRDIM) % BSRDIM) : lid;

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
    // Each thread block processes a single BSR row
    rocsparse_int row = bsr_mask_ptr[hipBlockIdx_x] - idx_base;
    // BSR row entry and exit point
    rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = bsr_end_ptr[row] - idx_base;
#endif

    // BSR block row accumulator
    T sum = static_cast<T>(0);

    // Loop over all BSR blocks in the current row where each lane
    // processes a BSR block value
    for(rocsparse_int j = row_begin; j < row_end; ++j)
    {
        rocsparse_int k = j + hipThreadIdx_x / (BSRDIM * BSRDIM);

        // Do not exceed the row
        if(k < row_end)
        {
            // Column index into x vector
            rocsparse_int col = (bsr_col_ind[k] - idx_base) * BSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            sum = rocsparse_fma(bsr_val[j * BSRDIM * BSRDIM + hipThreadIdx_x], x[col + idx], sum);
        }
    }

    // Accumulate each row sum of the BSR block
    __shared__ T sdata[BSRDIM * BSRDIM];

    sdata[hipThreadIdx_x] = sum;

    __syncthreads();

    if(dir == rocsparse_direction_column)
    {
        if(hipThreadIdx_x < BSRDIM * BSRDIM - BSRDIM * 16)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 16];
        __syncthreads();
        if(hipThreadIdx_x < BSRDIM * 8)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 8];
        __syncthreads();
        if(hipThreadIdx_x < BSRDIM * 4)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 4];
        __syncthreads();
        if(hipThreadIdx_x < BSRDIM * 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 2];
        __threadfence_block();
        if(hipThreadIdx_x < BSRDIM * 1)
            sum = sdata[hipThreadIdx_x] + sdata[hipThreadIdx_x + BSRDIM * 1];
    }
    else
    {
        // Reduce the intra block row sum
        if(lid < BSRDIM - 16)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 16];
        __syncthreads();
        if(lid < 8)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 8];
        __syncthreads();
        if(lid < 4)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
        __syncthreads();
        if(lid < 2)
            sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];
        __syncthreads();

        // Final reduction
        if(hipThreadIdx_x < BSRDIM)
            sum = sdata[hipThreadIdx_x * BSRDIM] + sdata[hipThreadIdx_x * BSRDIM + 1];
    }

    // First bunch of threads write row sums to global memory
    if(hipThreadIdx_x < BSRDIM)
    {
        if(beta != static_cast<T>(0))
        {
            y[row * BSRDIM + hipThreadIdx_x]
                = rocsparse_fma(beta, y[row * BSRDIM + hipThreadIdx_x], alpha * sum);
        }
        else
        {
            y[row * BSRDIM + hipThreadIdx_x] = alpha * sum;
        }
    }
}

// Kernels for BSR block dimensions of 17 to 32
template <unsigned int BSRDIM, typename T, typename U>
__launch_bounds__(BSRDIM* BSRDIM) ROCSPARSE_KERNEL
    void bsrxmvn_17_32_kernel(rocsparse_int       mb,
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
        bsrxmvn_17_32_device<BSRDIM>(mb,
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
void bsrxmvn_17_32(rocsparse_handle     handle,
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
                   rocsparse_int        block_dim,
                   const T*             x,
                   U                    beta_device_host,
                   T*                   y,
                   rocsparse_index_base base)
{
    const rocsparse_int size = (bsr_mask_ptr == nullptr) ? mb : size_of_mask;
    if(block_dim == 17)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<17>),
                           dim3(size),
                           dim3(17 * 17),
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
    else if(block_dim == 18)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<18>),
                           dim3(size),
                           dim3(18 * 18),
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
    else if(block_dim == 19)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<19>),
                           dim3(size),
                           dim3(19 * 19),
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
    else if(block_dim == 20)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<20>),
                           dim3(size),
                           dim3(20 * 20),
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
    else if(block_dim == 21)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<21>),
                           dim3(size),
                           dim3(21 * 21),
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
    else if(block_dim == 22)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<22>),
                           dim3(size),
                           dim3(22 * 22),
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
    else if(block_dim == 23)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<23>),
                           dim3(size),
                           dim3(23 * 23),
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
    else if(block_dim == 24)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<24>),
                           dim3(size),
                           dim3(24 * 24),
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
    else if(block_dim == 25)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<25>),
                           dim3(size),
                           dim3(25 * 25),
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
    else if(block_dim == 26)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<26>),
                           dim3(size),
                           dim3(26 * 26),
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
    else if(block_dim == 27)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<27>),
                           dim3(size),
                           dim3(27 * 27),
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
    else if(block_dim == 28)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<28>),
                           dim3(size),
                           dim3(28 * 28),
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
    else if(block_dim == 29)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<29>),
                           dim3(size),
                           dim3(29 * 29),
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
    else if(block_dim == 30)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<30>),
                           dim3(size),
                           dim3(30 * 30),
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
    else if(block_dim == 31)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<31>),
                           dim3(size),
                           dim3(31 * 31),
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
    else if(block_dim == 32)
    {
        hipLaunchKernelGGL((bsrxmvn_17_32_kernel<32>),
                           dim3(size),
                           dim3(32 * 32),
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
}

//
// INSTANTIATE.
//
#define INSTANTIATE(TYPE)                                               \
    template void bsrxmvn_17_32(rocsparse_handle     handle,            \
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
                                rocsparse_int        block_dim,         \
                                const TYPE*          x,                 \
                                const TYPE*          beta_device_host,  \
                                TYPE*                y,                 \
                                rocsparse_index_base base);             \
    template void bsrxmvn_17_32(rocsparse_handle     handle,            \
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
