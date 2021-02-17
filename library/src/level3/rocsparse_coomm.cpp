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

#include "rocsparse_coomm.hpp"

#include "coomm_device.h"
#include "definitions.h"
#include "utility.h"

template <unsigned int DIM_X, unsigned int DIM_Y, typename I, typename T, typename U>
__launch_bounds__(DIM_X* DIM_Y) __global__ void coomm_scale(
    I m, I n, U beta_device_host, T* __restrict__ data, I ld, rocsparse_order order)
{
    auto beta = load_scalar_device_host(beta_device_host);
    if(beta != static_cast<T>(1))
    {
        coomm_scale_device(m, n, beta, data, ld, order);
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          bool         TRANSB,
          typename I,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) __global__ void coommnn_wf_segmented(I nnz,
                                                                  I n,
                                                                  I nloops,
                                                                  U alpha_device_host,
                                                                  const I* __restrict__ coo_row_ind,
                                                                  const I* __restrict__ coo_col_ind,
                                                                  const T* __restrict__ coo_val,
                                                                  const T* __restrict__ B,
                                                                  I ldb,
                                                                  T* __restrict__ C,
                                                                  I ldc,
                                                                  I* __restrict__ row_block_red,
                                                                  T* __restrict__ val_block_red,
                                                                  rocsparse_order      order,
                                                                  rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    coommnn_general_wf_segmented<BLOCKSIZE, WF_SIZE, TRANSB>(nnz,
                                                             n,
                                                             nloops,
                                                             alpha,
                                                             coo_row_ind,
                                                             coo_col_ind,
                                                             coo_val,
                                                             B,
                                                             ldb,
                                                             C,
                                                             ldc,
                                                             row_block_red,
                                                             val_block_red,
                                                             order,
                                                             idx_base);
}

template <unsigned int BLK_SIZE_X,
          unsigned int BLK_SIZE_Y,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T,
          typename U>
__launch_bounds__(BLK_SIZE_X* BLK_SIZE_Y) __global__
    void coommnn_wf_atomic(I nnz,
                           I n,
                           I nblocks,
                           U alpha_device_host,
                           const I* __restrict__ coo_row_ind,
                           const I* __restrict__ coo_col_ind,
                           const T* __restrict__ coo_val,
                           const T* __restrict__ B,
                           I ldb,
                           T* __restrict__ C,
                           I                    ldc,
                           rocsparse_order      order,
                           rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    coommnn_general_wf_atomic<BLK_SIZE_X, BLK_SIZE_Y, LOOPS, TRANSB>(
        nnz, n, nblocks, alpha, coo_row_ind, coo_col_ind, coo_val, B, ldb, C, ldc, order, idx_base);
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_segmented(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    rocsparse_order           order,
                                                    I                         m,
                                                    I                         n,
                                                    I                         k,
                                                    I                         nnz,
                                                    U                         alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  coo_val,
                                                    const I*                  coo_row_ind,
                                                    const I*                  coo_col_ind,
                                                    const T*                  B,
                                                    I                         ldb,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    I                         ldc)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Run different coomm kernels
    if(trans_A == rocsparse_operation_none)
    {
#define COOMMN_DIM 256
        I nloops  = 16;
        I nblocks = (nnz - 1) / (handle->wavefront_size * nloops) + 1;

        size_t required_size = 256 + ((sizeof(I) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM
                               + ((sizeof(T) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= required_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMalloc(&temp_storage_ptr, required_size));
            temp_alloc = true;
        }

        dim3 coommn_blocks(nblocks, (n - 1) / (COOMMN_DIM / handle->wavefront_size) + 1);
        dim3 coommn_threads(COOMMN_DIM);

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_storage_ptr);
        ptr += 256;

        // row block reduction buffer
        I* row_block_red = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

        // val block reduction buffer
        T* val_block_red = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

        int len1 = ((sizeof(I) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

        hipMemset(row_block_red, 0XFF, len1);

        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose))
        {
            if(handle->wavefront_size == 32)
            {
                hipLaunchKernelGGL((coommnn_wf_segmented<COOMMN_DIM, 32, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nloops,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   row_block_red,
                                   val_block_red,
                                   order,
                                   descr->base);
            }
            else if(handle->wavefront_size == 64)
            {
                hipLaunchKernelGGL((coommnn_wf_segmented<COOMMN_DIM, 64, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nloops,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   row_block_red,
                                   val_block_red,
                                   order,
                                   descr->base);
            }
        }
        else if((order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            if(handle->wavefront_size == 32)
            {
                hipLaunchKernelGGL((coommnn_wf_segmented<COOMMN_DIM, 32, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nloops,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   row_block_red,
                                   val_block_red,
                                   order,
                                   descr->base);
            }
            else if(handle->wavefront_size == 64)
            {
                hipLaunchKernelGGL((coommnn_wf_segmented<COOMMN_DIM, 64, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nloops,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   row_block_red,
                                   val_block_red,
                                   order,
                                   descr->base);
            }
        }
#undef COOMMN_DIM

        hipLaunchKernelGGL((coommn_general_block_reduce<1024>),
                           dim3(n),
                           1024,
                           0,
                           stream,
                           nblocks,
                           row_block_red,
                           val_block_red,
                           C,
                           ldc,
                           order);

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(hipFree(temp_storage_ptr));
        }
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_atomic(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 rocsparse_order           order,
                                                 I                         m,
                                                 I                         n,
                                                 I                         k,
                                                 I                         nnz,
                                                 U                         alpha_device_host,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  coo_val,
                                                 const I*                  coo_row_ind,
                                                 const I*                  coo_col_ind,
                                                 const T*                  B,
                                                 I                         ldb,
                                                 U                         beta_device_host,
                                                 T*                        C,
                                                 I                         ldc)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Run different coomm kernels
    if(trans_A == rocsparse_operation_none)
    {
        rocsparse_int ncolsB     = trans_B == rocsparse_operation_none ? n : k;
        rocsparse_int BLK_SIZE_X = 1;
        if(ncolsB <= 1)
        {
            BLK_SIZE_X = 1;
        }
        if(ncolsB <= 2)
        {
            BLK_SIZE_X = 2;
        }
        else if(ncolsB <= 4)
        {
            BLK_SIZE_X = 4;
        }
        else if(ncolsB <= 8)
        {
            BLK_SIZE_X = 8;
        }
        else if(ncolsB <= 16)
        {
            BLK_SIZE_X = 16;
        }
        else if(ncolsB <= 32)
        {
            BLK_SIZE_X = 32;
        }
        else if(ncolsB <= 64)
        {
            BLK_SIZE_X = 64;
        }
        else
        {
            BLK_SIZE_X = 128;
        }

        rocsparse_int BLK_SIZE_Y = 256 / BLK_SIZE_X;

        constexpr I LOOPS   = 4;
        I           nblocks = (nnz - 1) / (BLK_SIZE_Y * LOOPS) + 1;

        dim3 coommn_blocks(nblocks, (n - 1) / BLK_SIZE_X + 1);
        dim3 coommn_threads(BLK_SIZE_X, BLK_SIZE_Y);

        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose))
        {
            if(BLK_SIZE_X == 1)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<1, 256 / 1, LOOPS, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 2)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<2, 256 / 2, LOOPS, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 4)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<4, 256 / 4, LOOPS, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 8)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<8, 256 / 8, LOOPS, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 16)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<16, 256 / 16, LOOPS, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 32)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<32, 256 / 32, LOOPS, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 64)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<64, 256 / 64, LOOPS, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<128, 256 / 128, LOOPS, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
        }
        else if((order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            if(BLK_SIZE_X == 1)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<1, 256 / 1, LOOPS, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 2)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<2, 256 / 2, LOOPS, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 4)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<4, 256 / 4, LOOPS, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 8)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<8, 256 / 8, LOOPS, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 16)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<16, 256 / 16, LOOPS, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 32)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<32, 256 / 32, LOOPS, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else if(BLK_SIZE_X == 64)
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<64, 256 / 64, LOOPS, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
            else
            {
                hipLaunchKernelGGL((coommnn_wf_atomic<128, 256 / 128, LOOPS, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nblocks,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   order,
                                   descr->base);
            }
        }
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_order           order,
                                                   rocsparse_spmm_alg        alg,
                                                   I                         m,
                                                   I                         n,
                                                   I                         k,
                                                   I                         nnz,
                                                   U                         alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  coo_val,
                                                   const I*                  coo_row_ind,
                                                   const I*                  coo_col_ind,
                                                   const T*                  B,
                                                   I                         ldb,
                                                   U                         beta_device_host,
                                                   T*                        C,
                                                   I                         ldc)
{
    // Scale C with beta
    hipLaunchKernelGGL((coomm_scale<256, 4>),
                       dim3((m - 1) / 256 + 1, (n - 1) / 4 + 1),
                       dim3(256, 4),
                       0,
                       handle->stream,
                       m,
                       n,
                       beta_device_host,
                       C,
                       ldc,
                       order);

    if(alg == rocsparse_spmm_alg_coo_segmented)
    {
        return rocsparse_coomm_template_segmented(handle,
                                                  trans_A,
                                                  trans_B,
                                                  order,
                                                  m,
                                                  n,
                                                  k,
                                                  nnz,
                                                  alpha_device_host,
                                                  descr,
                                                  coo_val,
                                                  coo_row_ind,
                                                  coo_col_ind,
                                                  B,
                                                  ldb,
                                                  beta_device_host,
                                                  C,
                                                  ldc);
    }
    else if(alg == rocsparse_spmm_alg_coo_atomic)
    {
        return rocsparse_coomm_template_atomic(handle,
                                               trans_A,
                                               trans_B,
                                               order,
                                               m,
                                               n,
                                               k,
                                               nnz,
                                               alpha_device_host,
                                               descr,
                                               coo_val,
                                               coo_row_ind,
                                               coo_col_ind,
                                               B,
                                               ldb,
                                               beta_device_host,
                                               C,
                                               ldc);
    }

    return rocsparse_status_not_implemented;
}

template <typename I, typename T>
rocsparse_status rocsparse_coomm_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_order           order_B,
                                          rocsparse_order           order_C,
                                          rocsparse_spmm_alg        alg,
                                          I                         m,
                                          I                         n,
                                          I                         k,
                                          I                         nnz,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  coo_val,
                                          const I*                  coo_row_ind,
                                          const I*                  coo_col_ind,
                                          const T*                  B,
                                          I                         ldb,
                                          const T*                  beta_device_host,
                                          T*                        C,
                                          I                         ldc)
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
                  replaceX<T>("rocsparse_Xcoomm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  nnz,
                  *alpha_device_host,
                  (const void*&)descr,
                  (const void*&)coo_val,
                  (const void*&)coo_row_ind,
                  (const void*&)coo_col_ind,
                  (const void*&)B,
                  ldb,
                  *beta_device_host,
                  (const void*&)C,
                  ldc);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcoomm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  nnz,
                  (const void*&)alpha_device_host,
                  (const void*&)descr,
                  (const void*&)coo_val,
                  (const void*&)coo_row_ind,
                  (const void*&)coo_col_ind,
                  (const void*&)B,
                  ldb,
                  (const void*&)beta_device_host,
                  (const void*&)C,
                  ldc);
    }

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(order_B != order_C)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    if(alpha_device_host == nullptr || beta_device_host == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    if(coo_val == nullptr || coo_row_ind == nullptr || coo_col_ind == nullptr || B == nullptr
       || C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check leading dimension of B
    I one = 1;
    if(trans_B == rocsparse_operation_none)
    {
        if(ldb < std::max(one, order_B == rocsparse_order_column ? k : n))
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(ldb < std::max(one, order_B == rocsparse_order_column ? n : k))
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check leading dimension of C
    if(ldc < std::max(one, order_C == rocsparse_order_column ? m : n))
    {
        return rocsparse_status_invalid_size;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_coomm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
                                                 alg,
                                                 m,
                                                 n,
                                                 k,
                                                 nnz,
                                                 alpha_device_host,
                                                 descr,
                                                 coo_val,
                                                 coo_row_ind,
                                                 coo_col_ind,
                                                 B,
                                                 ldb,
                                                 beta_device_host,
                                                 C,
                                                 ldc);
    }
    else
    {
        return rocsparse_coomm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
                                                 alg,
                                                 m,
                                                 n,
                                                 k,
                                                 nnz,
                                                 *alpha_device_host,
                                                 descr,
                                                 coo_val,
                                                 coo_row_ind,
                                                 coo_col_ind,
                                                 B,
                                                 ldb,
                                                 *beta_device_host,
                                                 C,
                                                 ldc);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                     \
    template rocsparse_status rocsparse_coomm_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                             \
        rocsparse_operation       trans_A,                            \
        rocsparse_operation       trans_B,                            \
        rocsparse_order           order_B,                            \
        rocsparse_order           order_C,                            \
        rocsparse_spmm_alg        alg,                                \
        ITYPE                     m,                                  \
        ITYPE                     n,                                  \
        ITYPE                     k,                                  \
        ITYPE                     nnz,                                \
        const TTYPE*              alpha_device_host,                  \
        const rocsparse_mat_descr descr,                              \
        const TTYPE*              coo_val,                            \
        const ITYPE*              coo_row_ind,                        \
        const ITYPE*              coo_col_ind,                        \
        const TTYPE*              B,                                  \
        ITYPE                     ldb,                                \
        const TTYPE*              beta_device_host,                   \
        TTYPE*                    C,                                  \
        ITYPE                     ldc);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
