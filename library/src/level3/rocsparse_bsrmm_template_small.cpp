/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "bsrmm_device_small.h"
#include "utility.h"

template <rocsparse_int BLOCKSIZE,
          rocsparse_int WF_SIZE,
          rocsparse_int BSR_BLOCK_DIM,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void bsrmmnn_small_blockdim_kernel(rocsparse_direction direction,
                                       rocsparse_int       mb,
                                       rocsparse_int       n,
                                       U                   alpha_device_host,
                                       const rocsparse_int* __restrict__ bsr_row_ptr,
                                       const rocsparse_int* __restrict__ bsr_col_ind,
                                       const T* __restrict__ bsr_val,
                                       const T* __restrict__ B,
                                       rocsparse_int ldb,
                                       U             beta_device_host,
                                       T* __restrict__ C,
                                       rocsparse_int        ldc,
                                       rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    bsrmmnn_small_blockdim_device<BLOCKSIZE, WF_SIZE, BSR_BLOCK_DIM>(
        direction, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, B, ldb, beta, C, ldc, idx_base);
}

template <rocsparse_int BLOCKSIZE,
          rocsparse_int WF_SIZE,
          rocsparse_int BSR_BLOCK_DIM,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void bsrmmnt_small_blockdim_kernel(rocsparse_direction direction,
                                       rocsparse_int       mb,
                                       rocsparse_int       n,
                                       U                   alpha_device_host,
                                       const rocsparse_int* __restrict__ bsr_row_ptr,
                                       const rocsparse_int* __restrict__ bsr_col_ind,
                                       const T* __restrict__ bsr_val,
                                       const T* __restrict__ B,
                                       rocsparse_int ldb,
                                       U             beta_device_host,
                                       T* __restrict__ C,
                                       rocsparse_int        ldc,
                                       rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    bsrmmnt_small_blockdim_device<BLOCKSIZE, WF_SIZE, BSR_BLOCK_DIM>(
        direction, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, B, ldb, beta, C, ldc, idx_base);
}

template <typename T, typename U>
rocsparse_status rocsparse_bsrmm_template_small(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_int             mb,
                                                rocsparse_int             n,
                                                rocsparse_int             kb,
                                                rocsparse_int             nnzb,
                                                U                         alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                const T*                  B,
                                                rocsparse_int             ldb,
                                                U                         beta,
                                                T*                        C,
                                                rocsparse_int             ldc)

{
    hipStream_t stream = handle->stream;
    //      rocsparse_int nnz = nnzb * block_dim;
    rocsparse_int m = mb * block_dim;
    //      rocsparse_int k   = kb * block_dim;
    assert(block_dim == 2);
    if(trans_B == rocsparse_operation_none)
    {
        constexpr rocsparse_int BSRMMNN_DIM = 64;
        constexpr rocsparse_int SUB_WF_SIZE = 8;

        dim3 bsrmm_blocks((SUB_WF_SIZE * m - 1) / BSRMMNN_DIM + 1, (n - 1) / SUB_WF_SIZE + 1);
        dim3 bsrmm_threads(BSRMMNN_DIM);
        hipLaunchKernelGGL((bsrmmnn_small_blockdim_kernel<BSRMMNN_DIM, SUB_WF_SIZE, 2>),
                           bsrmm_blocks,
                           bsrmm_threads,
                           0,
                           stream,
                           dir,
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
                           descr->base);
    }
    else
    {
        constexpr rocsparse_int BSRMMNT_DIM = 64;

#define UNROLL_SMALL_TRANSPOSE_KERNEL(M_)                                   \
    dim3 bsrmm_blocks((M_ * m - 1) / BSRMMNT_DIM + 1);                      \
    dim3 bsrmm_threads(BSRMMNT_DIM);                                        \
    hipLaunchKernelGGL((bsrmmnt_small_blockdim_kernel<BSRMMNT_DIM, M_, 2>), \
                       bsrmm_blocks,                                        \
                       bsrmm_threads,                                       \
                       0,                                                   \
                       stream,                                              \
                       dir,                                                 \
                       mb,                                                  \
                       n,                                                   \
                       alpha,                                               \
                       bsr_row_ptr,                                         \
                       bsr_col_ind,                                         \
                       bsr_val,                                             \
                       B,                                                   \
                       ldb,                                                 \
                       beta,                                                \
                       C,                                                   \
                       ldc,                                                 \
                       descr->base)

        // Average nnzb per row of A
        rocsparse_int avg_row_nnzb = (nnzb - 1) / mb + 1;

        // Launch appropriate kernel depending on row nnz of A
        if(avg_row_nnzb < 16)
        {
            UNROLL_SMALL_TRANSPOSE_KERNEL(8);
        }
        else if(avg_row_nnzb < 32)
        {
            UNROLL_SMALL_TRANSPOSE_KERNEL(16);
        }
        else if(avg_row_nnzb < 64 || handle->wavefront_size == 32)
        {
            UNROLL_SMALL_TRANSPOSE_KERNEL(32);
        }
        else if(handle->wavefront_size == 64)
        {
            UNROLL_SMALL_TRANSPOSE_KERNEL(64);
        }
        else
        {
            return rocsparse_status_arch_mismatch;
        }

#undef UNROLL_SMALL_TRANSPOSE_KERNEL
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(RT, ST)                                                                         \
                                                                                                    \
    template rocsparse_status rocsparse_bsrmm_template_small(rocsparse_handle          handle,      \
                                                             rocsparse_direction       dir,         \
                                                             rocsparse_operation       trans_A,     \
                                                             rocsparse_operation       trans_B,     \
                                                             rocsparse_int             mb,          \
                                                             rocsparse_int             n,           \
                                                             rocsparse_int             kb,          \
                                                             rocsparse_int             nnzb,        \
                                                             ST                        alpha,       \
                                                             const rocsparse_mat_descr descr,       \
                                                             const RT*                 bsr_val,     \
                                                             const rocsparse_int*      bsr_row_ptr, \
                                                             const rocsparse_int*      bsr_col_ind, \
                                                             rocsparse_int             block_dim,   \
                                                             const RT*                 B,           \
                                                             rocsparse_int             ldb,         \
                                                             ST                        beta,        \
                                                             RT*                       C,           \
                                                             rocsparse_int             ldc)

INSTANTIATE(float, float);
INSTANTIATE(float, const float*);

INSTANTIATE(double, double);
INSTANTIATE(double, const double*);

INSTANTIATE(rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex, const rocsparse_float_complex*);

INSTANTIATE(rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex, const rocsparse_double_complex*);

#undef INSTANTIATE
