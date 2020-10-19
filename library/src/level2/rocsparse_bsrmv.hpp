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

#pragma once
#ifndef ROCSPARSE_BSRMV_HPP
#define ROCSPARSE_BSRMV_HPP

#include "bsrmv_device.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "rocsparse_csrmv.hpp"
#include "utility.h"

#include <hip/hip_runtime.h>

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_general_kernel_host_pointer(rocsparse_direction dir,
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
    bsrmvn_general_device<T, BLOCKSIZE, WFSIZE>(
        dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, bsr_dim, x, beta, y, idx_base);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_general_kernel_device_pointer(rocsparse_direction dir,
                                              const T*            alpha,
                                              const rocsparse_int* __restrict__ bsr_row_ptr,
                                              const rocsparse_int* __restrict__ bsr_col_ind,
                                              const T* __restrict__ bsr_val,
                                              rocsparse_int bsr_dim,
                                              const T* __restrict__ x,
                                              const T* beta,
                                              T* __restrict__ y,
                                              rocsparse_index_base idx_base)
{
    bsrmvn_general_device<T, BLOCKSIZE, WFSIZE>(
        dir, *alpha, bsr_row_ptr, bsr_col_ind, bsr_val, bsr_dim, x, *beta, y, idx_base);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_2x2_kernel_host_pointer(rocsparse_int       mb,
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
    bsrmvn_2x2_device<T, BLOCKSIZE, WFSIZE>(
        mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_2x2_kernel_device_pointer(rocsparse_int       mb,
                                          rocsparse_direction dir,
                                          const T*            alpha,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          const T* __restrict__ bsr_val,
                                          const T* __restrict__ x,
                                          const T* beta,
                                          T* __restrict__ y,
                                          rocsparse_index_base idx_base)
{
    bsrmvn_2x2_device<T, BLOCKSIZE, WFSIZE>(
        mb, dir, *alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, *beta, y, idx_base);
}

template <typename T>
static void bsrmvn_2x2(rocsparse_handle     handle,
                       rocsparse_direction  dir,
                       rocsparse_int        mb,
                       rocsparse_int        nnzb,
                       const T*             alpha,
                       const rocsparse_int* bsr_row_ptr,
                       const rocsparse_int* bsr_col_ind,
                       const T*             bsr_val,
                       const T*             x,
                       const T*             beta,
                       T*                   y,
                       rocsparse_index_base base)
{
    rocsparse_int blocks_per_row = nnzb / mb;

#define BSRMVN_DIM 128
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        if(blocks_per_row < 8)
        {
            hipLaunchKernelGGL((bsrmvn_2x2_kernel_device_pointer<T, BSRMVN_DIM, 4>),
                               dim3((mb - 1) / (BSRMVN_DIM / 4) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(blocks_per_row < 16)
        {
            hipLaunchKernelGGL((bsrmvn_2x2_kernel_device_pointer<T, BSRMVN_DIM, 8>),
                               dim3((mb - 1) / (BSRMVN_DIM / 8) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(blocks_per_row < 32)
        {
            hipLaunchKernelGGL((bsrmvn_2x2_kernel_device_pointer<T, BSRMVN_DIM, 16>),
                               dim3((mb - 1) / (BSRMVN_DIM / 16) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(blocks_per_row < 64)
        {
            hipLaunchKernelGGL((bsrmvn_2x2_kernel_device_pointer<T, BSRMVN_DIM, 32>),
                               dim3((mb - 1) / (BSRMVN_DIM / 32) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else
        {
            hipLaunchKernelGGL((bsrmvn_2x2_kernel_device_pointer<T, BSRMVN_DIM, 64>),
                               dim3((mb - 1) / (BSRMVN_DIM / 64) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
    }
    else
    {
        if(*alpha != static_cast<T>(0) || *beta != static_cast<T>(1))
        {
            if(blocks_per_row < 8)
            {
                hipLaunchKernelGGL((bsrmvn_2x2_kernel_host_pointer<T, BSRMVN_DIM, 4>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 4) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(blocks_per_row < 16)
            {
                hipLaunchKernelGGL((bsrmvn_2x2_kernel_host_pointer<T, BSRMVN_DIM, 8>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 8) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(blocks_per_row < 32)
            {
                hipLaunchKernelGGL((bsrmvn_2x2_kernel_host_pointer<T, BSRMVN_DIM, 16>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 16) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(blocks_per_row < 64)
            {
                hipLaunchKernelGGL((bsrmvn_2x2_kernel_host_pointer<T, BSRMVN_DIM, 32>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 32) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else
            {
                hipLaunchKernelGGL((bsrmvn_2x2_kernel_host_pointer<T, BSRMVN_DIM, 64>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 64) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
        }
    }
#undef BSRMVN_DIM
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_3x3_kernel_host_pointer(rocsparse_int       mb,
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
    bsrmvn_3x3_device<T, BLOCKSIZE, WFSIZE>(
        mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_3x3_kernel_device_pointer(rocsparse_int       mb,
                                          rocsparse_direction dir,
                                          const T*            alpha,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          const T* __restrict__ bsr_val,
                                          const T* __restrict__ x,
                                          const T* beta,
                                          T* __restrict__ y,
                                          rocsparse_index_base idx_base)
{
    bsrmvn_3x3_device<T, BLOCKSIZE, WFSIZE>(
        mb, dir, *alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, *beta, y, idx_base);
}

template <typename T>
static void bsrmvn_3x3(rocsparse_handle     handle,
                       rocsparse_direction  dir,
                       rocsparse_int        mb,
                       rocsparse_int        nnzb,
                       const T*             alpha,
                       const rocsparse_int* bsr_row_ptr,
                       const rocsparse_int* bsr_col_ind,
                       const T*             bsr_val,
                       const T*             x,
                       const T*             beta,
                       T*                   y,
                       rocsparse_index_base base)
{
    rocsparse_int blocks_per_row = nnzb / mb;

#define BSRMVN_DIM 256
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        if(blocks_per_row < 8)
        {
            hipLaunchKernelGGL((bsrmvn_3x3_kernel_device_pointer<T, BSRMVN_DIM, 4>),
                               dim3((mb - 1) / (BSRMVN_DIM / 4) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(blocks_per_row < 16)
        {
            hipLaunchKernelGGL((bsrmvn_3x3_kernel_device_pointer<T, BSRMVN_DIM, 8>),
                               dim3((mb - 1) / (BSRMVN_DIM / 8) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(blocks_per_row < 32)
        {
            hipLaunchKernelGGL((bsrmvn_3x3_kernel_device_pointer<T, BSRMVN_DIM, 16>),
                               dim3((mb - 1) / (BSRMVN_DIM / 16) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(blocks_per_row < 64)
        {
            hipLaunchKernelGGL((bsrmvn_3x3_kernel_device_pointer<T, BSRMVN_DIM, 32>),
                               dim3((mb - 1) / (BSRMVN_DIM / 32) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else
        {
            hipLaunchKernelGGL((bsrmvn_3x3_kernel_device_pointer<T, BSRMVN_DIM, 64>),
                               dim3((mb - 1) / (BSRMVN_DIM / 64) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
    }
    else
    {
        if(*alpha != static_cast<T>(0) || *beta != static_cast<T>(1))
        {
            if(blocks_per_row <= 4)
            {
                hipLaunchKernelGGL((bsrmvn_3x3_kernel_host_pointer<T, BSRMVN_DIM, 4>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 4) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(blocks_per_row <= 8)
            {
                hipLaunchKernelGGL((bsrmvn_3x3_kernel_host_pointer<T, BSRMVN_DIM, 8>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 8) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(blocks_per_row <= 16)
            {
                hipLaunchKernelGGL((bsrmvn_3x3_kernel_host_pointer<T, BSRMVN_DIM, 16>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 16) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(blocks_per_row <= 32)
            {
                hipLaunchKernelGGL((bsrmvn_3x3_kernel_host_pointer<T, BSRMVN_DIM, 32>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 32) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else
            {
                hipLaunchKernelGGL((bsrmvn_3x3_kernel_host_pointer<T, BSRMVN_DIM, 64>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 64) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
        }
    }
#undef BSRMVN_DIM
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_4x4_kernel_host_pointer(rocsparse_int       mb,
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
    bsrmvn_4x4_device<T, BLOCKSIZE, WFSIZE>(
        mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_4x4_kernel_device_pointer(rocsparse_int       mb,
                                          rocsparse_direction dir,
                                          const T*            alpha,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          const T* __restrict__ bsr_val,
                                          const T* __restrict__ x,
                                          const T* beta,
                                          T* __restrict__ y,
                                          rocsparse_index_base idx_base)
{
    bsrmvn_4x4_device<T, BLOCKSIZE, WFSIZE>(
        mb, dir, *alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, *beta, y, idx_base);
}

template <typename T>
static void bsrmvn_4x4(rocsparse_handle     handle,
                       rocsparse_direction  dir,
                       rocsparse_int        mb,
                       rocsparse_int        nnzb,
                       const T*             alpha,
                       const rocsparse_int* bsr_row_ptr,
                       const rocsparse_int* bsr_col_ind,
                       const T*             bsr_val,
                       const T*             x,
                       const T*             beta,
                       T*                   y,
                       rocsparse_index_base base)
{
    rocsparse_int blocks_per_row = nnzb / mb;

#define BSRMVN_DIM 128
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        if(blocks_per_row < 8)
        {
            hipLaunchKernelGGL((bsrmvn_4x4_kernel_device_pointer<T, BSRMVN_DIM, 4>),
                               dim3((mb - 1) / (BSRMVN_DIM / 4) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(blocks_per_row < 16)
        {
            hipLaunchKernelGGL((bsrmvn_4x4_kernel_device_pointer<T, BSRMVN_DIM, 8>),
                               dim3((mb - 1) / (BSRMVN_DIM / 8) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(blocks_per_row < 32)
        {
            hipLaunchKernelGGL((bsrmvn_4x4_kernel_device_pointer<T, BSRMVN_DIM, 16>),
                               dim3((mb - 1) / (BSRMVN_DIM / 16) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(blocks_per_row < 64)
        {
            hipLaunchKernelGGL((bsrmvn_4x4_kernel_device_pointer<T, BSRMVN_DIM, 32>),
                               dim3((mb - 1) / (BSRMVN_DIM / 32) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
        else
        {
            hipLaunchKernelGGL((bsrmvn_4x4_kernel_device_pointer<T, BSRMVN_DIM, 64>),
                               dim3((mb - 1) / (BSRMVN_DIM / 64) + 1),
                               dim3(BSRMVN_DIM),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        }
    }
    else
    {
        if(*alpha != static_cast<T>(0) || *beta != static_cast<T>(1))
        {
            if(blocks_per_row < 8)
            {
                hipLaunchKernelGGL((bsrmvn_4x4_kernel_host_pointer<T, BSRMVN_DIM, 4>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 4) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(blocks_per_row < 16)
            {
                hipLaunchKernelGGL((bsrmvn_4x4_kernel_host_pointer<T, BSRMVN_DIM, 8>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 8) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(blocks_per_row < 32)
            {
                hipLaunchKernelGGL((bsrmvn_4x4_kernel_host_pointer<T, BSRMVN_DIM, 16>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 16) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(blocks_per_row < 64)
            {
                hipLaunchKernelGGL((bsrmvn_4x4_kernel_host_pointer<T, BSRMVN_DIM, 32>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 32) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else
            {
                hipLaunchKernelGGL((bsrmvn_4x4_kernel_host_pointer<T, BSRMVN_DIM, 64>),
                                   dim3((mb - 1) / (BSRMVN_DIM / 64) + 1),
                                   dim3(BSRMVN_DIM),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
        }
    }
#undef BSRMVN_DIM
}

template <typename T, unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_5x5_kernel_host_pointer(rocsparse_int       mb,
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
    bsrmvn_5x5_device<T, BLOCKSIZE>(
        mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
}

template <typename T, unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_5x5_kernel_device_pointer(rocsparse_int       mb,
                                          rocsparse_direction dir,
                                          const T*            alpha,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          const T* __restrict__ bsr_val,
                                          const T* __restrict__ x,
                                          const T* beta,
                                          T* __restrict__ y,
                                          rocsparse_index_base idx_base)
{
    bsrmvn_5x5_device<T, BLOCKSIZE>(
        mb, dir, *alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, *beta, y, idx_base);
}

template <typename T>
static void bsrmvn_5x5(rocsparse_handle     handle,
                       rocsparse_direction  dir,
                       rocsparse_int        mb,
                       rocsparse_int        nnzb,
                       const T*             alpha,
                       const rocsparse_int* bsr_row_ptr,
                       const rocsparse_int* bsr_col_ind,
                       const T*             bsr_val,
                       const T*             x,
                       const T*             beta,
                       T*                   y,
                       rocsparse_index_base base)
{
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((bsrmvn_5x5_kernel_device_pointer<T, 50>),
                           dim3(mb),
                           dim3(50),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta,
                           y,
                           base);
    }
    else
    {
        if(*alpha != static_cast<T>(0) || *beta != static_cast<T>(1))
        {
            hipLaunchKernelGGL((bsrmvn_5x5_kernel_host_pointer<T, 50>),
                               dim3(mb),
                               dim3(50),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               *alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               *beta,
                               y,
                               base);
        }
    }
}

template <typename T, unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_8x8_kernel_host_pointer(rocsparse_int       mb,
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
    bsrmvn_8x8_device<T, BLOCKSIZE>(
        mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
}

template <typename T, unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_8x8_kernel_device_pointer(rocsparse_int       mb,
                                          rocsparse_direction dir,
                                          const T*            alpha,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          const T* __restrict__ bsr_val,
                                          const T* __restrict__ x,
                                          const T* beta,
                                          T* __restrict__ y,
                                          rocsparse_index_base idx_base)
{
    bsrmvn_8x8_device<T, BLOCKSIZE>(
        mb, dir, *alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, *beta, y, idx_base);
}

template <typename T>
static void bsrmvn_8x8(rocsparse_handle     handle,
                       rocsparse_direction  dir,
                       rocsparse_int        mb,
                       rocsparse_int        nnzb,
                       const T*             alpha,
                       const rocsparse_int* bsr_row_ptr,
                       const rocsparse_int* bsr_col_ind,
                       const T*             bsr_val,
                       const T*             x,
                       const T*             beta,
                       T*                   y,
                       rocsparse_index_base base)
{
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((bsrmvn_8x8_kernel_device_pointer<T, 128>),
                           dim3(mb),
                           dim3(128),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta,
                           y,
                           base);
    }
    else
    {
        if(*alpha != static_cast<T>(0) || *beta != static_cast<T>(1))
        {
            hipLaunchKernelGGL((bsrmvn_8x8_kernel_host_pointer<T, 128>),
                               dim3(mb),
                               dim3(128),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               *alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               *beta,
                               y,
                               base);
        }
    }
}

template <typename T, unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_16x16_kernel_host_pointer(rocsparse_int       mb,
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
    bsrmvn_16x16_device<T, BLOCKSIZE>(
        mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
}

template <typename T, unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmvn_16x16_kernel_device_pointer(rocsparse_int       mb,
                                            rocsparse_direction dir,
                                            const T*            alpha,
                                            const rocsparse_int* __restrict__ bsr_row_ptr,
                                            const rocsparse_int* __restrict__ bsr_col_ind,
                                            const T* __restrict__ bsr_val,
                                            const T* __restrict__ x,
                                            const T* beta,
                                            T* __restrict__ y,
                                            rocsparse_index_base idx_base)
{
    bsrmvn_16x16_device<T, BLOCKSIZE>(
        mb, dir, *alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, *beta, y, idx_base);
}

template <typename T>
static void bsrmvn_16x16(rocsparse_handle     handle,
                         rocsparse_direction  dir,
                         rocsparse_int        mb,
                         rocsparse_int        nnzb,
                         const T*             alpha,
                         const rocsparse_int* bsr_row_ptr,
                         const rocsparse_int* bsr_col_ind,
                         const T*             bsr_val,
                         const T*             x,
                         const T*             beta,
                         T*                   y,
                         rocsparse_index_base base)
{
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((bsrmvn_16x16_kernel_device_pointer<T, 256>),
                           dim3(mb),
                           dim3(256),
                           0,
                           handle->stream,
                           mb,
                           dir,
                           alpha,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           x,
                           beta,
                           y,
                           base);
    }
    else
    {
        if(*alpha != static_cast<T>(0) || *beta != static_cast<T>(1))
        {
            hipLaunchKernelGGL((bsrmvn_16x16_kernel_host_pointer<T, 256>),
                               dim3(mb),
                               dim3(256),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               *alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               *beta,
                               y,
                               base);
        }
    }
}

// Kernels for BSR block dimensions of 17 to 32
template <typename T, unsigned int BSRDIM>
__launch_bounds__(BSRDIM* BSRDIM) __global__
    void bsrmvn_17_32_kernel_host_pointer(rocsparse_int       mb,
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
    bsrmvn_17_32_device<T, BSRDIM>(
        mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
}

template <typename T, unsigned int BSRDIM>
__launch_bounds__(BSRDIM* BSRDIM) __global__
    void bsrmvn_17_32_kernel_device_pointer(rocsparse_int       mb,
                                            rocsparse_direction dir,
                                            const T*            alpha,
                                            const rocsparse_int* __restrict__ bsr_row_ptr,
                                            const rocsparse_int* __restrict__ bsr_col_ind,
                                            const T* __restrict__ bsr_val,
                                            const T* __restrict__ x,
                                            const T* beta,
                                            T* __restrict__ y,
                                            rocsparse_index_base idx_base)
{
    bsrmvn_17_32_device<T, BSRDIM>(
        mb, dir, *alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, *beta, y, idx_base);
}

template <typename T>
static void bsrmvn_17_32(rocsparse_handle     handle,
                         rocsparse_direction  dir,
                         rocsparse_int        mb,
                         rocsparse_int        nnzb,
                         const T*             alpha,
                         const rocsparse_int* bsr_row_ptr,
                         const rocsparse_int* bsr_col_ind,
                         const T*             bsr_val,
                         rocsparse_int        bsr_dim,
                         const T*             x,
                         const T*             beta,
                         T*                   y,
                         rocsparse_index_base base)
{
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        if(bsr_dim == 17)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 17>),
                               dim3(mb),
                               dim3(17 * 17),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 18)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 18>),
                               dim3(mb),
                               dim3(18 * 18),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 19)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 19>),
                               dim3(mb),
                               dim3(19 * 19),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 20)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 20>),
                               dim3(mb),
                               dim3(20 * 20),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 21)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 21>),
                               dim3(mb),
                               dim3(21 * 21),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 22)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 22>),
                               dim3(mb),
                               dim3(22 * 22),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 23)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 23>),
                               dim3(mb),
                               dim3(23 * 23),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 24)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 24>),
                               dim3(mb),
                               dim3(24 * 24),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 25)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 25>),
                               dim3(mb),
                               dim3(25 * 25),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 26)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 26>),
                               dim3(mb),
                               dim3(26 * 26),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 27)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 27>),
                               dim3(mb),
                               dim3(27 * 27),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 28)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 28>),
                               dim3(mb),
                               dim3(28 * 28),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 29)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 29>),
                               dim3(mb),
                               dim3(29 * 29),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 30)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 30>),
                               dim3(mb),
                               dim3(30 * 30),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 31)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 31>),
                               dim3(mb),
                               dim3(31 * 31),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
        else if(bsr_dim == 32)
            hipLaunchKernelGGL((bsrmvn_17_32_kernel_device_pointer<T, 32>),
                               dim3(mb),
                               dim3(32 * 32),
                               0,
                               handle->stream,
                               mb,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta,
                               y,
                               base);
    }
    else
    {
        if(*alpha != static_cast<T>(0) || *beta != static_cast<T>(1))
        {
            if(bsr_dim == 17)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 17>),
                                   dim3(mb),
                                   dim3(17 * 17),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 18)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 18>),
                                   dim3(mb),
                                   dim3(18 * 18),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 19)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 19>),
                                   dim3(mb),
                                   dim3(19 * 19),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 20)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 20>),
                                   dim3(mb),
                                   dim3(20 * 20),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 21)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 21>),
                                   dim3(mb),
                                   dim3(21 * 21),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 22)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 22>),
                                   dim3(mb),
                                   dim3(22 * 22),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 23)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 23>),
                                   dim3(mb),
                                   dim3(23 * 23),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 24)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 24>),
                                   dim3(mb),
                                   dim3(24 * 24),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 25)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 25>),
                                   dim3(mb),
                                   dim3(25 * 25),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 26)
            {
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 26>),
                                   dim3(mb),
                                   dim3(26 * 26),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(bsr_dim == 27)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 27>),
                                   dim3(mb),
                                   dim3(27 * 27),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 28)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 28>),
                                   dim3(mb),
                                   dim3(28 * 28),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 29)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 29>),
                                   dim3(mb),
                                   dim3(29 * 29),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 30)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 30>),
                                   dim3(mb),
                                   dim3(30 * 30),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 31)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 31>),
                                   dim3(mb),
                                   dim3(31 * 31),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
            else if(bsr_dim == 32)
                hipLaunchKernelGGL((bsrmvn_17_32_kernel_host_pointer<T, 32>),
                                   dim3(mb),
                                   dim3(32 * 32),
                                   0,
                                   handle->stream,
                                   mb,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   x,
                                   *beta,
                                   y,
                                   base);
        }
    }
}

template <typename T>
static void bsrmvn_general(rocsparse_handle     handle,
                           rocsparse_direction  dir,
                           rocsparse_int        mb,
                           const T*             alpha,
                           const rocsparse_int* bsr_row_ptr,
                           const rocsparse_int* bsr_col_ind,
                           const T*             bsr_val,
                           rocsparse_int        bsr_dim,
                           const T*             x,
                           const T*             beta,
                           T*                   y,
                           rocsparse_index_base base)
{
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // Differentiate BSR block dimensions
        if(bsr_dim <= 8)
        {
            hipLaunchKernelGGL((bsrmvn_general_kernel_device_pointer<T, 64, 8>),
                               dim3(mb),
                               dim3(8 * 8),
                               0,
                               handle->stream,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               bsr_dim,
                               x,
                               beta,
                               y,
                               base);
        }
        else if(bsr_dim <= 16)
        {
            hipLaunchKernelGGL((bsrmvn_general_kernel_device_pointer<T, 256, 16>),
                               dim3(mb),
                               dim3(16 * 16),
                               0,
                               handle->stream,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               bsr_dim,
                               x,
                               beta,
                               y,
                               base);
        }
        else
        {
            hipLaunchKernelGGL((bsrmvn_general_kernel_device_pointer<T, 1024, 32>),
                               dim3(mb),
                               dim3(32 * 32),
                               0,
                               handle->stream,
                               dir,
                               alpha,
                               bsr_row_ptr,
                               bsr_col_ind,
                               bsr_val,
                               bsr_dim,
                               x,
                               beta,
                               y,
                               base);
        }
    }
    else
    {
        if(*alpha != static_cast<T>(0) || *beta != static_cast<T>(1))
        {
            // Differentiate BSR block dimensions
            if(bsr_dim <= 8)
            {
                hipLaunchKernelGGL((bsrmvn_general_kernel_host_pointer<T, 64, 8>),
                                   dim3(mb),
                                   dim3(8 * 8),
                                   0,
                                   handle->stream,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   bsr_dim,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else if(bsr_dim <= 16)
            {
                hipLaunchKernelGGL((bsrmvn_general_kernel_host_pointer<T, 256, 16>),
                                   dim3(mb),
                                   dim3(16 * 16),
                                   0,
                                   handle->stream,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   bsr_dim,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
            else
            {
                hipLaunchKernelGGL((bsrmvn_general_kernel_host_pointer<T, 1024, 32>),
                                   dim3(mb),
                                   dim3(32 * 32),
                                   0,
                                   handle->stream,
                                   dir,
                                   *alpha,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   bsr_dim,
                                   x,
                                   *beta,
                                   y,
                                   base);
            }
        }
    }
}

template <typename T>
rocsparse_status rocsparse_bsrmv_template(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans,
                                          rocsparse_int             mb,
                                          rocsparse_int             nb,
                                          rocsparse_int             nnzb,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  bsr_val,
                                          const rocsparse_int*      bsr_row_ptr,
                                          const rocsparse_int*      bsr_col_ind,
                                          rocsparse_int             bsr_dim,
                                          const T*                  x,
                                          const T*                  beta,
                                          T*                        y)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xbsrmv"),
                  dir,
                  trans,
                  mb,
                  nb,
                  nnzb,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)bsr_val,
                  (const void*&)bsr_row_ptr,
                  (const void*&)bsr_col_ind,
                  bsr_dim,
                  (const void*&)x,
                  *beta,
                  (const void*&)y);

        log_bench(handle,
                  "./rocsparse-bench -f bsrmv -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx> "
                  "--bsrdim",
                  bsr_dim,
                  "--alpha",
                  *alpha,
                  "--beta",
                  *beta);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xbsrmv"),
                  dir,
                  trans,
                  mb,
                  nb,
                  nnzb,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)bsr_val,
                  (const void*&)bsr_row_ptr,
                  (const void*&)bsr_col_ind,
                  bsr_dim,
                  (const void*&)x,
                  (const void*&)beta,
                  (const void*&)y);
    }

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nb < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnzb < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(bsr_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0 || nnzb == 0 || bsr_dim == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(x == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // bsr_dim == 1 is the CSR case
    if(bsr_dim == 1)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrmv_template(handle,
                                                           trans,
                                                           mb,
                                                           nb,
                                                           nnzb,
                                                           alpha,
                                                           descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           nullptr,
                                                           x,
                                                           beta,
                                                           y));

        return rocsparse_status_success;
    }

    // Run different bsrmv kernels
    if(trans == rocsparse_operation_none)
    {
        if(handle->wavefront_size == 32)
        {
            bsrmvn_general(handle,
                           dir,
                           mb,
                           alpha,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           bsr_dim,
                           x,
                           beta,
                           y,
                           descr->base);

            return rocsparse_status_success;
        }

        if(bsr_dim == 2)
        {
            bsrmvn_2x2(handle,
                       dir,
                       mb,
                       nnzb,
                       alpha,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_val,
                       x,
                       beta,
                       y,
                       descr->base);
        }
        else if(bsr_dim == 3)
        {
            bsrmvn_3x3(handle,
                       dir,
                       mb,
                       nnzb,
                       alpha,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_val,
                       x,
                       beta,
                       y,
                       descr->base);
        }
        else if(bsr_dim == 4)
        {
            bsrmvn_4x4(handle,
                       dir,
                       mb,
                       nnzb,
                       alpha,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_val,
                       x,
                       beta,
                       y,
                       descr->base);
        }
        else if(bsr_dim == 5)
        {
            bsrmvn_5x5(handle,
                       dir,
                       mb,
                       nnzb,
                       alpha,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_val,
                       x,
                       beta,
                       y,
                       descr->base);
        }
        else if(bsr_dim == 8)
        {
            bsrmvn_8x8(handle,
                       dir,
                       mb,
                       nnzb,
                       alpha,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_val,
                       x,
                       beta,
                       y,
                       descr->base);
        }
        else if(bsr_dim == 16)
        {
            bsrmvn_16x16(handle,
                         dir,
                         mb,
                         nnzb,
                         alpha,
                         bsr_row_ptr,
                         bsr_col_ind,
                         bsr_val,
                         x,
                         beta,
                         y,
                         descr->base);
        }
        else if(bsr_dim > 16 && bsr_dim <= 32)
        {
            bsrmvn_17_32(handle,
                         dir,
                         mb,
                         nnzb,
                         alpha,
                         bsr_row_ptr,
                         bsr_col_ind,
                         bsr_val,
                         bsr_dim,
                         x,
                         beta,
                         y,
                         descr->base);
        }
        else
        {
            bsrmvn_general(handle,
                           dir,
                           mb,
                           alpha,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           bsr_dim,
                           x,
                           beta,
                           y,
                           descr->base);
        }
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_BSRMV_HPP
