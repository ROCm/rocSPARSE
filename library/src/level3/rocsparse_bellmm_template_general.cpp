/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "bellmm_device_general.h"
#include "utility.h"

template <rocsparse_int BELL_BLOCK_DIM,
          rocsparse_int BLK_SIZE_Y,
          typename T,
          typename U,
          typename I>
__launch_bounds__(BELL_BLOCK_DIM* BLK_SIZE_Y) ROCSPARSE_KERNEL
    void bellmm_general_blockdim_kernel(rocsparse_operation trans_A,
                                        rocsparse_operation trans_B,
                                        rocsparse_order     order_B,
                                        rocsparse_order     order_C,
                                        rocsparse_direction dir_A,
                                        I                   Mb,
                                        I                   N,
                                        U                   alpha_device_host,
                                        I                   bell_cols,
                                        I                   block_dim,
                                        const I* __restrict__ bell_col_ind,
                                        const T* __restrict__ bell_val,
                                        const T* __restrict__ B,
                                        I ldb,
                                        U beta_device_host,
                                        T* __restrict__ C,
                                        I                    ldc,
                                        rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    bellmm_general_blockdim_device<BELL_BLOCK_DIM, BLK_SIZE_Y>(trans_A,
                                                               trans_B,
                                                               order_B,
                                                               order_C,
                                                               dir_A,
                                                               Mb,
                                                               N,
                                                               alpha,
                                                               bell_cols,
                                                               block_dim,
                                                               bell_col_ind,
                                                               bell_val,
                                                               B,
                                                               ldb,
                                                               beta,
                                                               C,
                                                               ldc,
                                                               idx_base);
}

template <typename T, typename U, typename I>
rocsparse_status rocsparse_bellmm_template_general(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_order           order_B,
                                                   rocsparse_order           order_C,
                                                   rocsparse_direction       dir_A,
                                                   I                         mb,
                                                   I                         n,
                                                   I                         kb,
                                                   I                         bell_cols,
                                                   I                         block_dim,
                                                   U                         alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const I*                  bell_col_ind,
                                                   const T*                  bell_val,
                                                   const T*                  B,
                                                   I                         ldb,
                                                   U                         beta,
                                                   T*                        C,
                                                   I                         ldc)
{
    hipStream_t stream = handle->stream;
    dim3        bellmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
    dim3        bellmm_threads(32, 32, 1);
    assert(trans_A == rocsparse_operation_none);
    //
    // What happends if A needs to be transposed?
    //
    hipLaunchKernelGGL((bellmm_general_blockdim_kernel<32, 32>),
                       bellmm_blocks,
                       bellmm_threads,
                       0,
                       stream,
                       trans_A,
                       trans_B,
                       order_B,
                       order_C,
                       dir_A,
                       mb,
                       n,
                       alpha,
                       bell_cols,
                       block_dim,
                       bell_col_ind,
                       bell_val,
                       B,
                       ldb,
                       beta,
                       C,
                       ldc,
                       descr->base);

    return rocsparse_status_success;
}

#define INSTANTIATE(TTYPE, UTYPE, ITYPE)                                                            \
    template rocsparse_status rocsparse_bellmm_template_general(rocsparse_handle    handle,         \
                                                                rocsparse_operation trans_A,        \
                                                                rocsparse_operation trans_B,        \
                                                                rocsparse_order     order_B,        \
                                                                rocsparse_order     order_C,        \
                                                                rocsparse_direction dir_A,          \
                                                                ITYPE               mb,             \
                                                                ITYPE               n,              \
                                                                ITYPE               kb,             \
                                                                ITYPE               bell_cols,      \
                                                                ITYPE               bell_block_dim, \
                                                                UTYPE               alpha,          \
                                                                const rocsparse_mat_descr descr,    \
                                                                const ITYPE* bell_col_ind,          \
                                                                const TTYPE* bell_val,              \
                                                                const TTYPE* B,                     \
                                                                ITYPE        ldb,                   \
                                                                UTYPE        beta,                  \
                                                                TTYPE*       C,                     \
                                                                ITYPE        ldc)

INSTANTIATE(float, float, int32_t);
INSTANTIATE(float, const float*, int32_t);

INSTANTIATE(double, double, int32_t);
INSTANTIATE(double, const double*, int32_t);

INSTANTIATE(rocsparse_float_complex, rocsparse_float_complex, int32_t);
INSTANTIATE(rocsparse_float_complex, const rocsparse_float_complex*, int32_t);

INSTANTIATE(rocsparse_double_complex, rocsparse_double_complex, int32_t);
INSTANTIATE(rocsparse_double_complex, const rocsparse_double_complex*, int32_t);

INSTANTIATE(float, float, int64_t);
INSTANTIATE(float, const float*, int64_t);

INSTANTIATE(double, double, int64_t);
INSTANTIATE(double, const double*, int64_t);

INSTANTIATE(rocsparse_float_complex, rocsparse_float_complex, int64_t);
INSTANTIATE(rocsparse_float_complex, const rocsparse_float_complex*, int64_t);

INSTANTIATE(rocsparse_double_complex, rocsparse_double_complex, int64_t);
INSTANTIATE(rocsparse_double_complex, const rocsparse_double_complex*, int64_t);

#undef INSTANTIATE
