/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

namespace rocsparse
{
    template <rocsparse_int BELL_BLOCK_DIM,
              rocsparse_int BLK_SIZE_Y,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C,
              typename U>
    ROCSPARSE_KERNEL(BELL_BLOCK_DIM* BLK_SIZE_Y)
    void bellmm_general_blockdim_kernel(rocsparse_operation trans_A,
                                        rocsparse_operation trans_B,
                                        rocsparse_direction dir_A,
                                        I                   Mb,
                                        I                   N,
                                        U                   alpha_device_host,
                                        I                   bell_cols,
                                        I                   block_dim,
                                        const I* __restrict__ bell_col_ind,
                                        const A* __restrict__ bell_val,
                                        const B* __restrict__ dense_B,
                                        int64_t         ldb,
                                        rocsparse_order order_B,
                                        U               beta_device_host,
                                        C* __restrict__ dense_C,
                                        int64_t              ldc,
                                        rocsparse_order      order_C,
                                        rocsparse_index_base idx_base)
    {
        auto alpha = load_scalar_device_host(alpha_device_host);
        auto beta  = load_scalar_device_host(beta_device_host);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        rocsparse::bellmm_general_blockdim_device<BELL_BLOCK_DIM, BLK_SIZE_Y, T>(trans_A,
                                                                                 trans_B,
                                                                                 dir_A,
                                                                                 Mb,
                                                                                 N,
                                                                                 alpha,
                                                                                 bell_cols,
                                                                                 block_dim,
                                                                                 bell_col_ind,
                                                                                 bell_val,
                                                                                 dense_B,
                                                                                 ldb,
                                                                                 order_B,
                                                                                 beta,
                                                                                 dense_C,
                                                                                 ldc,
                                                                                 order_C,
                                                                                 idx_base);
    }

    template <typename T, typename I, typename A, typename B, typename C, typename U>
    rocsparse_status bellmm_template_general(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             rocsparse_direction       dir_A,
                                             I                         mb,
                                             I                         n,
                                             I                         kb,
                                             I                         bell_cols,
                                             I                         block_dim,
                                             U                         alpha,
                                             const rocsparse_mat_descr descr,
                                             const I*                  bell_col_ind,
                                             const A*                  bell_val,
                                             const B*                  dense_B,
                                             int64_t                   ldb,
                                             rocsparse_order           order_B,
                                             U                         beta,
                                             C*                        dense_C,
                                             int64_t                   ldc,
                                             rocsparse_order           order_C)
    {
        hipStream_t stream = handle->stream;
        dim3        bellmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
        dim3        bellmm_threads(32, 32, 1);
        assert(trans_A == rocsparse_operation_none);
        //
        // What happends if A needs to be transposed?
        //
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bellmm_general_blockdim_kernel<32, 32, T>),
                                           bellmm_blocks,
                                           bellmm_threads,
                                           0,
                                           stream,
                                           trans_A,
                                           trans_B,
                                           dir_A,
                                           mb,
                                           n,
                                           alpha,
                                           bell_cols,
                                           block_dim,
                                           bell_col_ind,
                                           bell_val,
                                           dense_B,
                                           ldb,
                                           order_B,
                                           beta,
                                           dense_C,
                                           ldc,
                                           order_C,
                                           descr->base);

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(TTYPE, ITYPE, ATYPE, BTYPE, CTYPE, UTYPE)            \
    template rocsparse_status rocsparse::bellmm_template_general<TTYPE>( \
        rocsparse_handle          handle,                                \
        rocsparse_operation       trans_A,                               \
        rocsparse_operation       trans_B,                               \
        rocsparse_direction       dir_A,                                 \
        ITYPE                     mb,                                    \
        ITYPE                     n,                                     \
        ITYPE                     kb,                                    \
        ITYPE                     bell_cols,                             \
        ITYPE                     bell_block_dim,                        \
        UTYPE                     alpha,                                 \
        const rocsparse_mat_descr descr,                                 \
        const ITYPE*              bell_col_ind,                          \
        const ATYPE*              bell_val,                              \
        const BTYPE*              dense_B,                               \
        int64_t                   ldb,                                   \
        rocsparse_order           order_B,                               \
        UTYPE                     beta,                                  \
        CTYPE*                    dense_C,                               \
        int64_t                   ldc,                                   \
        rocsparse_order           order_C)

INSTANTIATE(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int32_t, int32_t, int32_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int32_t, int32_t, int32_t, const int32_t*);

INSTANTIATE(float, int32_t, float, float, float, float);
INSTANTIATE(float, int64_t, float, float, float, float);
INSTANTIATE(float, int32_t, float, float, float, const float*);
INSTANTIATE(float, int64_t, float, float, float, const float*);

INSTANTIATE(double, int32_t, double, double, double, double);
INSTANTIATE(double, int64_t, double, double, double, double);
INSTANTIATE(double, int32_t, double, double, double, const double*);
INSTANTIATE(double, int64_t, double, double, double, const double*);

INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);

INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);

#undef INSTANTIATE
