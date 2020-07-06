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
#ifndef ROCSPARSE_BSRMM_HPP
#define ROCSPARSE_BSRMM_HPP

#include "bsrmm_device.h"
#include "handle.h"
#include "rocsparse.h"
#include "rocsparse_csrmm.hpp"
#include "../level2/rocsparse_bsrmv.hpp"
#include "utility.h"

#include <hip/hip_runtime.h>

#define launch_bsrmmnn_small_blockdim_kernel_host_pointer(T, block_size, wf_size, bsr_block_dim) \
    hipLaunchKernelGGL(                                                                         \
        (bsrmmnn_small_blockdim_kernel_host_pointer<T, block_size, wf_size, bsr_block_dim>),     \
        bsrmmnn_blocks,                                                                         \
        bsrmmnn_threads,                                                                        \
        0,                                                                                      \
        stream,                                                                                 \
        dir,                                                                                    \
        mb,                                                                                     \
        n,                                                                                      \
        *alpha,                                                                                 \
        bsr_row_ptr,                                                                            \
        bsr_col_ind,                                                                            \
        bsr_val,                                                                                \
        B,                                                                                      \
        ldb,                                                                                    \
        *beta,                                                                                  \
        C,                                                                                      \
        ldc,                                                                                    \
        descr->base);

#define launch_bsrmmnn_small_blockdim_kernel_device_pointer(T, block_size, wf_size, bsr_block_dim) \
    hipLaunchKernelGGL(                                                                           \
        (bsrmmnn_small_blockdim_kernel_device_pointer<T, block_size, wf_size, bsr_block_dim>),     \
        bsrmmnn_blocks,                                                                           \
        bsrmmnn_threads,                                                                          \
        0,                                                                                        \
        stream,                                                                                   \
        dir,                                                                                      \
        mb,                                                                                       \
        n,                                                                                        \
        alpha,                                                                                    \
        bsr_row_ptr,                                                                              \
        bsr_col_ind,                                                                              \
        bsr_val,                                                                                  \
        B,                                                                                        \
        ldb,                                                                                      \
        beta,                                                                                     \
        C,                                                                                        \
        ldc,                                                                                      \
        descr->base);

#define launch_bsrmmnt_small_blockdim_kernel_host_pointer(T, block_size, wf_size, bsr_block_dim) \
    hipLaunchKernelGGL(                                                                         \
        (bsrmmnt_small_blockdim_kernel_host_pointer<T, block_size, wf_size, bsr_block_dim>),     \
        bsrmmnt_blocks,                                                                         \
        bsrmmnt_threads,                                                                        \
        0,                                                                                      \
        stream,                                                                                 \
        dir,                                                                                    \
        mb,                                                                                     \
        n,                                                                                      \
        *alpha,                                                                                 \
        bsr_row_ptr,                                                                            \
        bsr_col_ind,                                                                            \
        bsr_val,                                                                                \
        B,                                                                                      \
        ldb,                                                                                    \
        *beta,                                                                                  \
        C,                                                                                      \
        ldc,                                                                                    \
        descr->base);

#define launch_bsrmmnt_small_blockdim_kernel_device_pointer(T, block_size, wf_size, bsr_block_dim) \
    hipLaunchKernelGGL(                                                                           \
        (bsrmmnt_small_blockdim_kernel_device_pointer<T, block_size, wf_size, bsr_block_dim>),     \
        bsrmmnt_blocks,                                                                           \
        bsrmmnt_threads,                                                                          \
        0,                                                                                        \
        stream,                                                                                   \
        dir,                                                                                      \
        mb,                                                                                       \
        n,                                                                                        \
        alpha,                                                                                    \
        bsr_row_ptr,                                                                              \
        bsr_col_ind,                                                                              \
        bsr_val,                                                                                  \
        B,                                                                                        \
        ldb,                                                                                      \
        beta,                                                                                     \
        C,                                                                                        \
        ldc,                                                                                      \
        descr->base);    

 #define launch_bsrmm_large_blockdim_kernel_host_pointer(T, bsr_block_dim, blk_size_y) \
    hipLaunchKernelGGL(                                                     \
        (bsrmm_large_blockdim_kernel_host_pointer<T, bsr_block_dim, blk_size_y>),     \
        bsrmm_blocks,                                                     \
        bsrmm_threads,                                                    \
        0,                                                                  \
        stream,                                                             \
        dir,                                                                \
        trans_B,                                                            \
        mb,                                                                 \
        n,                                                                  \
        *alpha,                                                             \
        bsr_row_ptr,                                                        \
        bsr_col_ind,                                                        \
        bsr_val,                                                            \
        block_dim,                                                          \
        B,                                                                  \
        ldb,                                                                \
        *beta,                                                              \
        C,                                                                  \
        ldc,                                                                \
        descr->base);

#define launch_bsrmm_large_blockdim_kernel_device_pointer(T, bsr_block_dim, blk_size_y) \
    hipLaunchKernelGGL(                                                       \
        (bsrmm_large_blockdim_kernel_device_pointer<T, bsr_block_dim, blk_size_y>),     \
        bsrmm_blocks,                                                      \
        bsrmm_threads,                                                      \
        0,                                                                    \
        stream,                                                               \
        dir,                                                                  \
        trans_B,                                                              \
        mb,                                                                   \
        n,                                                                    \
        alpha,                                                                \
        bsr_row_ptr,                                                          \
        bsr_col_ind,                                                          \
        bsr_val,                                                              \
        block_dim,                                                            \
        B,                                                                    \
        ldb,                                                                  \
        beta,                                                                 \
        C,                                                                    \
        ldc,                                                                  \
        descr->base);

#define launch_bsrmm_general_blockdim_kernel_host_pointer(T, bsr_block_dim, blk_size_y) \
    hipLaunchKernelGGL(                                                     \
        (bsrmm_general_blockdim_kernel_host_pointer<T, bsr_block_dim, blk_size_y>),     \
        bsrmm_blocks,                                                     \
        bsrmm_threads,                                                    \
        0,                                                                  \
        stream,                                                             \
        dir,                                                                \
        trans_B,                                                            \
        mb,                                                                 \
        n,                                                                  \
        *alpha,                                                             \
        bsr_row_ptr,                                                        \
        bsr_col_ind,                                                        \
        bsr_val,                                                            \
        block_dim,                                                          \
        B,                                                                  \
        ldb,                                                                \
        *beta,                                                              \
        C,                                                                  \
        ldc,                                                                \
        descr->base);

#define launch_bsrmm_general_blockdim_kernel_device_pointer(T, bsr_block_dim, blk_size_y) \
    hipLaunchKernelGGL(                                                       \
        (bsrmm_general_blockdim_kernel_device_pointer<T, bsr_block_dim, blk_size_y>),     \
        bsrmm_blocks,                                                      \
        bsrmm_threads,                                                      \
        0,                                                                    \
        stream,                                                               \
        dir,                                                                  \
        trans_B,                                                              \
        mb,                                                                   \
        n,                                                                    \
        alpha,                                                                \
        bsr_row_ptr,                                                          \
        bsr_col_ind,                                                          \
        bsr_val,                                                              \
        block_dim,                                                            \
        B,                                                                    \
        ldb,                                                                  \
        beta,                                                                 \
        C,                                                                    \
        ldc,                                                                  \
        descr->base);

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE, rocsparse_int BSR_BLOCK_DIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmmnn_small_blockdim_kernel_host_pointer(rocsparse_direction direction,
                                                   rocsparse_int       mb,
                                                   rocsparse_int       n,
                                                   T                   alpha,
                                                   const rocsparse_int* __restrict__ bsr_row_ptr,
                                                   const rocsparse_int* __restrict__ bsr_col_ind,
                                                   const T* __restrict__ bsr_val,
                                                   const T* __restrict__ B,
                                                   rocsparse_int ldb,
                                                   T             beta,
                                                   T* __restrict__ C,
                                                   rocsparse_int        ldc,
                                                   rocsparse_index_base idx_base)
{
    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    bsrmmnn_small_blockdim_device<T, BLOCKSIZE, WF_SIZE, BSR_BLOCK_DIM>(
      direction, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, B, ldb, beta, C, ldc, idx_base);
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE, rocsparse_int BSR_BLOCK_DIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmmnn_small_blockdim_kernel_device_pointer(rocsparse_direction direction,
                                                     rocsparse_int       mb,
                                                     rocsparse_int       n,
                                                     const T*            alpha,
                                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                                     const rocsparse_int* __restrict__ bsr_col_ind,
                                                     const T* __restrict__ bsr_val,
                                                     const T* __restrict__ B,
                                                     rocsparse_int ldb,
                                                     const T*      beta,
                                                     T* __restrict__ C,
                                                     rocsparse_int        ldc,
                                                     rocsparse_index_base idx_base)
{
    if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
    {
        return;
    }

    bsrmmnn_small_blockdim_device<T, BLOCKSIZE, WF_SIZE, BSR_BLOCK_DIM>(direction,
                                                                       mb,
                                                                       n,
                                                                       *alpha,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       bsr_val,
                                                                       B,
                                                                       ldb,
                                                                       *beta,
                                                                       C,
                                                                       ldc,
                                                                       idx_base);
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE, rocsparse_int BSR_BLOCK_DIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmmnt_small_blockdim_kernel_host_pointer(rocsparse_direction direction,
                                                   rocsparse_int       mb,
                                                   rocsparse_int       n,
                                                   T                   alpha,
                                                   const rocsparse_int* __restrict__ bsr_row_ptr,
                                                   const rocsparse_int* __restrict__ bsr_col_ind,
                                                   const T* __restrict__ bsr_val,
                                                   const T* __restrict__ B,
                                                   rocsparse_int ldb,
                                                   T             beta,
                                                   T* __restrict__ C,
                                                   rocsparse_int        ldc,
                                                   rocsparse_index_base idx_base)
{
    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    bsrmmnt_small_blockdim_device<T, BLOCKSIZE, WF_SIZE, BSR_BLOCK_DIM>(direction,
                                                                       mb,
                                                                       n,
                                                                       alpha,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       bsr_val,
                                                                       B,
                                                                       ldb,
                                                                       beta,
                                                                       C,
                                                                       ldc,
                                                                       idx_base);
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE, rocsparse_int BSR_BLOCK_DIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrmmnt_small_blockdim_kernel_device_pointer(rocsparse_direction direction,
                                                     rocsparse_int       mb,
                                                     rocsparse_int       n,
                                                     const T*            alpha,
                                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                                     const rocsparse_int* __restrict__ bsr_col_ind,
                                                     const T* __restrict__ bsr_val,
                                                     const T* __restrict__ B,
                                                     rocsparse_int ldb,
                                                     const T*      beta,
                                                     T* __restrict__ C,
                                                     rocsparse_int        ldc,
                                                     rocsparse_index_base idx_base)
{
    if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
    {
        return;
    }

    bsrmmnt_small_blockdim_device<T, BLOCKSIZE, WF_SIZE, BSR_BLOCK_DIM>(direction,
                                                                       mb,
                                                                       n,
                                                                       *alpha,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       bsr_val,
                                                                       B,
                                                                       ldb,
                                                                       *beta,
                                                                       C,
                                                                       ldc,
                                                                       idx_base);
}

template <typename T, rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y>
__launch_bounds__(BSR_BLOCK_DIM * BLK_SIZE_Y) __global__
    void bsrmm_large_blockdim_kernel_host_pointer(rocsparse_direction direction,
                                                    rocsparse_operation trans_B,
                                                    rocsparse_int       mb,
                                                    rocsparse_int       n,
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
    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    bsrmm_large_blockdim_device<T, BSR_BLOCK_DIM, BLK_SIZE_Y>(
        direction, trans_B, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, block_dim, B, ldb, beta, C, ldc, idx_base);
}

template <typename T, rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y>
__launch_bounds__(BSR_BLOCK_DIM * BLK_SIZE_Y) __global__
    void bsrmm_large_blockdim_kernel_device_pointer(rocsparse_direction direction,
                                                      rocsparse_operation trans_B,
                                                     rocsparse_int       mb,
                                                     rocsparse_int       n,
                                                     const T*            alpha,
                                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                                     const rocsparse_int* __restrict__ bsr_col_ind,
                                                     const T* __restrict__ bsr_val,
                                                     rocsparse_int block_dim,
                                                     const T* __restrict__ B,
                                                     rocsparse_int ldb,
                                                     const T*      beta,
                                                     T* __restrict__ C,
                                                     rocsparse_int        ldc,
                                                     rocsparse_index_base idx_base)
{
    if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
    {
        return;
    }

    bsrmm_large_blockdim_device<T, BSR_BLOCK_DIM, BLK_SIZE_Y>(direction,
                                                    trans_B,
                                                    mb,
                                                    n,
                                                    *alpha,
                                                    bsr_row_ptr,
                                                    bsr_col_ind,
                                                    bsr_val,
                                                    block_dim,
                                                    B,
                                                    ldb,
                                                    *beta,
                                                    C,
                                                    ldc,
                                                    idx_base);
}

template <typename T, rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y>
__launch_bounds__(BSR_BLOCK_DIM * BLK_SIZE_Y) __global__
    void bsrmm_general_blockdim_kernel_host_pointer(rocsparse_direction direction,
                                                      rocsparse_operation trans_B,
                                                    rocsparse_int       mb,
                                                    rocsparse_int       n,
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
    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    bsrmm_general_blockdim_device<T, BSR_BLOCK_DIM, BLK_SIZE_Y>(
        direction, trans_B, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, block_dim, B, ldb, beta, C, ldc, idx_base);
}

template <typename T, rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y>
__launch_bounds__(BSR_BLOCK_DIM * BLK_SIZE_Y) __global__
    void bsrmm_general_blockdim_kernel_device_pointer(rocsparse_direction direction,
                                                        rocsparse_operation trans_B,
                                                        rocsparse_int       mb,
                                                        rocsparse_int       n,
                                                        const T*            alpha,
                                                        const rocsparse_int* __restrict__ bsr_row_ptr,
                                                        const rocsparse_int* __restrict__ bsr_col_ind,
                                                        const T* __restrict__ bsr_val,
                                                        rocsparse_int block_dim,
                                                        const T* __restrict__ B,
                                                        rocsparse_int ldb,
                                                        const T*      beta,
                                                        T* __restrict__ C,
                                                        rocsparse_int        ldc,
                                                        rocsparse_index_base idx_base)
{
    if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
    {
        return;
    }

    bsrmm_general_blockdim_device<T, BSR_BLOCK_DIM, BLK_SIZE_Y>(direction,
                                                                  trans_B,
                                                                mb,
                                                                n,
                                                                *alpha,
                                                                bsr_row_ptr,
                                                                bsr_col_ind,
                                                                bsr_val,
                                                                block_dim,
                                                                B,
                                                                ldb,
                                                                *beta,
                                                                C,
                                                                ldc,
                                                                idx_base);
}

template <typename T>
rocsparse_status rocsparse_bsrmm_template(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_int             mb,
                                          rocsparse_int             n,
                                          rocsparse_int             kb,
                                          rocsparse_int             nnzb,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const T*                  bsr_val,
                                          const rocsparse_int*      bsr_row_ptr,
                                          const rocsparse_int*      bsr_col_ind,
                                          rocsparse_int             block_dim,
                                          const T*                  B,
                                          rocsparse_int             ldb,
                                          const T*                  beta,
                                          T*                        C,
                                          rocsparse_int             ldc)
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

    // Logging TODO bench logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xbsrmm"),
                  dir,
                  trans_A,
                  trans_B,
                  mb,
                  n,
                  kb,
                  nnzb,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)bsr_val,
                  (const void*&)bsr_row_ptr,
                  (const void*&)bsr_col_ind,
                  block_dim,
                  (const void*&)B,
                  ldb,
                  *beta,
                  (const void*&)C,
                  ldc);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xbsrmm"),
                  dir,
                  trans_A,
                  trans_B,
                  mb,
                  n,
                  kb,
                  nnzb,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)bsr_val,
                  (const void*&)bsr_row_ptr,
                  (const void*&)bsr_col_ind,
                  block_dim,
                  (const void*&)B,
                  ldb,
                  (const void*&)beta,
                  (const void*&)C,
                  ldc);
    }

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check operation
    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }
    else if(trans_B != rocsparse_operation_none && trans_B != rocsparse_operation_transpose)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0 || n < 0 || kb < 0 || nnzb < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || n == 0 || kb == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_val == nullptr || bsr_row_ptr == nullptr || bsr_col_ind == nullptr || B == nullptr
       || C == nullptr || alpha == nullptr || beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check leading dimension of B
    if(trans_B == rocsparse_operation_none)
    {
        if(ldb < kb)
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(ldb < n)
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check leading dimension of C
    if(ldc < mb)
    {
        return rocsparse_status_invalid_size;
    }

    // Stream
    hipStream_t stream = handle->stream;

    rocsparse_int m   = mb * block_dim;
    rocsparse_int k   = kb * block_dim;
    rocsparse_int nnz = nnzb * block_dim;

    // If n is only 1 and B are non-transposed, then call bsrmv
    if(n == 1)
    {
        if(trans_B == rocsparse_operation_none)
        {
            return rocsparse_bsrmv_template(handle,
                                        dir,
                                        trans_A,
                                        mb,
                                        kb,
                                        nnzb,
                                        alpha,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        block_dim,
                                        B,
                                        beta,
                                        C);
        }
    }

    // If block dimension is one we can simply call csrmm
    if(block_dim == 1)
    {
        return rocsparse_csrmm_template(handle,
                                        trans_A,
                                        trans_B,
                                        m,
                                        n,
                                        k,
                                        nnz,
                                        alpha,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        B,
                                        ldb,
                                        beta,
                                        C,
                                        ldc);
    }

    if(block_dim == 2)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            if(trans_B == rocsparse_operation_none)
            {
                constexpr rocsparse_int BSRMMNN_DIM = 64;
                constexpr rocsparse_int SUB_WF_SIZE = 8;

                dim3 bsrmmnn_blocks((SUB_WF_SIZE * m - 1) / BSRMMNN_DIM + 1, (n - 1) / SUB_WF_SIZE + 1);
                dim3 bsrmmnn_threads(BSRMMNN_DIM);
                launch_bsrmmnn_small_blockdim_kernel_device_pointer(T, BSRMMNN_DIM, SUB_WF_SIZE, 2);
            }
            else
            {
                constexpr rocsparse_int BSRMMNT_DIM = 64;

                // Average nnzb per row of A
                rocsparse_int avg_row_nnzb = (nnzb - 1) / mb + 1;

                // Launch appropriate kernel depending on row nnz of A
                if(avg_row_nnzb < 16)
                {
                    dim3 bsrmmnt_blocks((8 * m - 1) / BSRMMNT_DIM + 1);
                    dim3 bsrmmnt_threads(BSRMMNT_DIM);
                    launch_bsrmmnt_small_blockdim_kernel_device_pointer(T, BSRMMNT_DIM, 8, 2);
                }
                else if(avg_row_nnzb < 32)
                {
                    dim3 bsrmmnt_blocks((16 * m - 1) / BSRMMNT_DIM + 1);
                    dim3 bsrmmnt_threads(BSRMMNT_DIM);
                    launch_bsrmmnt_small_blockdim_kernel_device_pointer(T, BSRMMNT_DIM, 16, 2);
                }
                else if(avg_row_nnzb < 64 || handle->wavefront_size == 32)
                {
                    dim3 bsrmmnt_blocks((32 * m - 1) / BSRMMNT_DIM + 1);
                    dim3 bsrmmnt_threads(BSRMMNT_DIM);
                    launch_bsrmmnt_small_blockdim_kernel_device_pointer(T, BSRMMNT_DIM, 32, 2);
                }
                else if(handle->wavefront_size == 64)
                {
                    dim3 bsrmmnt_blocks((64 * m - 1) / BSRMMNT_DIM + 1);
                    dim3 bsrmmnt_threads(BSRMMNT_DIM);
                    launch_bsrmmnt_small_blockdim_kernel_device_pointer(T, BSRMMNT_DIM, 64, 2);
                }
                else
                {
                    return rocsparse_status_arch_mismatch;
                }
            }
        }
        else
        {
            if(trans_B == rocsparse_operation_none)
            {
                constexpr rocsparse_int BSRMMNN_DIM = 64;
                constexpr rocsparse_int SUB_WF_SIZE = 8;

                dim3 bsrmmnn_blocks((SUB_WF_SIZE * m - 1) / BSRMMNN_DIM + 1, (n - 1) / SUB_WF_SIZE + 1);
                dim3 bsrmmnn_threads(BSRMMNN_DIM);
                launch_bsrmmnn_small_blockdim_kernel_host_pointer(T, BSRMMNN_DIM, SUB_WF_SIZE, 2);
            }
            else
            {
                constexpr rocsparse_int BSRMMNT_DIM = 64;

                // Average nnzb per row of A
                rocsparse_int avg_row_nnzb = (nnzb - 1) / mb + 1;

                // Launch appropriate kernel depending on row nnz of A
                if(avg_row_nnzb < 16)
                {
                    dim3 bsrmmnt_blocks((8 * m - 1) / BSRMMNT_DIM + 1);
                    dim3 bsrmmnt_threads(BSRMMNT_DIM);
                    launch_bsrmmnt_small_blockdim_kernel_host_pointer(T, BSRMMNT_DIM, 8, 2);
                }
                else if(avg_row_nnzb < 32)
                {
                    dim3 bsrmmnt_blocks((16 * m - 1) / BSRMMNT_DIM + 1);
                    dim3 bsrmmnt_threads(BSRMMNT_DIM);
                    launch_bsrmmnt_small_blockdim_kernel_host_pointer(T, BSRMMNT_DIM, 16, 2);
                }
                else if(avg_row_nnzb < 64 || handle->wavefront_size == 32)
                {
                    dim3 bsrmmnt_blocks((32 * m - 1) / BSRMMNT_DIM + 1);
                    dim3 bsrmmnt_threads(BSRMMNT_DIM);
                    launch_bsrmmnt_small_blockdim_kernel_host_pointer(T, BSRMMNT_DIM, 32, 2);
                }
                else if(handle->wavefront_size == 64)
                {
                    dim3 bsrmmnt_blocks((64 * m - 1) / BSRMMNT_DIM + 1);
                    dim3 bsrmmnt_threads(BSRMMNT_DIM);
                    launch_bsrmmnt_small_blockdim_kernel_host_pointer(T, BSRMMNT_DIM, 64, 2);
                }
                else
                {
                    return rocsparse_status_arch_mismatch;
                }
            }
        }

        return rocsparse_status_success;
    }

    // Run different bsrmm kernels for block dim > 2
    if(n <= 16 && block_dim > 4 && block_dim <= 8)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 16 + 1);
            dim3 bsrmm_threads(8, 16, 1);
            launch_bsrmm_large_blockdim_kernel_device_pointer(T, 8, 16);
        }
        else
        {
            dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 16 + 1);
            dim3 bsrmm_threads(8, 16, 1);
            launch_bsrmm_large_blockdim_kernel_host_pointer(T, 8, 16);
        }
    }
    else
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            if(block_dim <= 4)
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 16 + 1);
                dim3 bsrmm_threads(4, 16, 1);
                launch_bsrmm_large_blockdim_kernel_device_pointer(T, 4, 16);
            }
            else if(block_dim <= 8)
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
                dim3 bsrmm_threads(8, 32, 1);
                launch_bsrmm_large_blockdim_kernel_device_pointer(T, 8, 32);
            }
            else if(block_dim <= 16)
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 16 + 1);
                dim3 bsrmm_threads(16, 16, 1);
                launch_bsrmm_large_blockdim_kernel_device_pointer(T, 16, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
                dim3 bsrmm_threads(32, 32, 1);
                launch_bsrmm_large_blockdim_kernel_device_pointer(T, 32, 32);
            }
            else
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
                dim3 bsrmm_threads(32, 32, 1);
                launch_bsrmm_general_blockdim_kernel_device_pointer(T, 32, 32);
            }
        }
        else
        {
            if(block_dim <= 4)
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 16 + 1);
                dim3 bsrmm_threads(4, 16, 1);
                launch_bsrmm_large_blockdim_kernel_host_pointer(T, 4, 16);
            }
            else if(block_dim <= 8)
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
                dim3 bsrmm_threads(8, 32, 1);
                launch_bsrmm_large_blockdim_kernel_host_pointer(T, 8, 32);
            }
            else if(block_dim <= 16)
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 16 + 1);
                dim3 bsrmm_threads(16, 16, 1);
                launch_bsrmm_large_blockdim_kernel_host_pointer(T, 16, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
                dim3 bsrmm_threads(32, 32, 1);
                launch_bsrmm_large_blockdim_kernel_host_pointer(T, 32, 32);
            }
            else
            {
                dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
                dim3 bsrmm_threads(32, 32, 1);
                launch_bsrmm_general_blockdim_kernel_host_pointer(T, 32, 32);
            }
        }
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_BSRMM_HPP