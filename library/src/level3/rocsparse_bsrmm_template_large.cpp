/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "utility.h"

#include "bsrmm_device_large.h"
#include "rocsparse_csrmm.hpp"

namespace rocsparse
{
    template <uint32_t BSR_BLOCK_DIM,
              uint32_t BLK_SIZE_Y,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename U>
    ROCSPARSE_KERNEL(BSR_BLOCK_DIM* BLK_SIZE_Y)
    void bsrmm_large_blockdim_kernel(rocsparse_direction direction,
                                     rocsparse_operation trans_B,
                                     J                   mb,
                                     J                   n,
                                     U                   alpha_device_host,
                                     const I* __restrict__ bsr_row_ptr,
                                     const J* __restrict__ bsr_col_ind,
                                     const A* __restrict__ bsr_val,
                                     J block_dim,
                                     const B* __restrict__ dense_B,
                                     int64_t ldb,
                                     U       beta_device_host,
                                     C* __restrict__ dense_C,
                                     int64_t              ldc,
                                     rocsparse_index_base idx_base)
    {
        const auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        const auto beta  = rocsparse::load_scalar_device_host(beta_device_host);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        rocsparse::bsrmm_large_blockdim_device<BSR_BLOCK_DIM, BLK_SIZE_Y>(direction,
                                                                          trans_B,
                                                                          mb,
                                                                          n,
                                                                          alpha,
                                                                          bsr_row_ptr,
                                                                          bsr_col_ind,
                                                                          bsr_val,
                                                                          block_dim,
                                                                          dense_B,
                                                                          ldb,
                                                                          beta,
                                                                          dense_C,
                                                                          ldc,
                                                                          idx_base);
    }

    typedef enum
    {
        large_config_4_16 = 1,
        large_config_8_32,
        large_config_8_16,
        large_config_16_16,
        large_config_32_32
    } enum_large_config;

    //
    // Select which tuned kernel to apply.
    //
    template <typename J>
    enum_large_config get_large_config(J block_dim, J n)
    {
        if(block_dim <= 4)
        {
            return large_config_4_16;
        }
        else if(block_dim <= 8)
        {
            if(n <= 16)
            {
                return large_config_8_16;
            }
            else
            {
                return large_config_8_32;
            }
        }
        else if(block_dim <= 16)
        {
            return large_config_16_16;
        }
        else
        {
            rocsparse_host_assert(block_dim <= 32, "Wrong logical dispatch.");
            return large_config_32_32;
        }
    }

    template <typename T, typename I, typename J, typename A, typename B, typename C, typename U>
    rocsparse_status bsrmm_template_large(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          J                         mb,
                                          J                         n,
                                          J                         kb,
                                          I                         nnzb,
                                          U                         alpha,
                                          const rocsparse_mat_descr descr,
                                          const A*                  bsr_val,
                                          const I*                  bsr_row_ptr,
                                          const J*                  bsr_col_ind,
                                          J                         block_dim,
                                          const B*                  dense_B,
                                          int64_t                   ldb,
                                          U                         beta,
                                          C*                        dense_C,
                                          int64_t                   ldc)
    {
        hipStream_t stream = handle->stream;
        rocsparse_host_assert(block_dim <= 32, "This function is designed for block_dim <= 32.");

#define LAUNCH_LARGE_KERNEL(M_, N_)                                                      \
    const dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / N_ + 1);                         \
    const dim3 bsrmm_threads(M_, N_);                                                    \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrmm_large_blockdim_kernel<M_, N_>), \
                                       bsrmm_blocks,                                     \
                                       bsrmm_threads,                                    \
                                       0,                                                \
                                       stream,                                           \
                                       dir,                                              \
                                       trans_B,                                          \
                                       mb,                                               \
                                       n,                                                \
                                       alpha,                                            \
                                       bsr_row_ptr,                                      \
                                       bsr_col_ind,                                      \
                                       bsr_val,                                          \
                                       block_dim,                                        \
                                       dense_B,                                          \
                                       ldb,                                              \
                                       beta,                                             \
                                       dense_C,                                          \
                                       ldc,                                              \
                                       descr->base)

        //
        // Select which tuned kernel to apply.
        //
        const enum_large_config config = rocsparse::get_large_config(block_dim, n);
        switch(config)
        {
#define DEFINE_CASE(i, j, k)       \
    case i:                        \
    {                              \
        LAUNCH_LARGE_KERNEL(j, k); \
        break;                     \
    }
            DEFINE_CASE(large_config_8_32, 8, 32);
            DEFINE_CASE(large_config_4_16, 4, 16);
            DEFINE_CASE(large_config_8_16, 8, 16);
            DEFINE_CASE(large_config_16_16, 16, 16);
            DEFINE_CASE(large_config_32_32, 32, 32);
#undef DEFINE_CASE
        }

#undef LAUNCH_LARGE_KERNEL
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE, UTYPE)  \
                                                                      \
    template rocsparse_status rocsparse::bsrmm_template_large<TTYPE>( \
        rocsparse_handle          handle,                             \
        rocsparse_direction       dir,                                \
        rocsparse_operation       trans_A,                            \
        rocsparse_operation       trans_B,                            \
        JTYPE                     mb,                                 \
        JTYPE                     n,                                  \
        JTYPE                     kb,                                 \
        ITYPE                     nnzb,                               \
        UTYPE                     alpha,                              \
        const rocsparse_mat_descr descr,                              \
        const ATYPE*              bsr_val,                            \
        const ITYPE*              bsr_row_ptr,                        \
        const JTYPE*              bsr_col_ind,                        \
        JTYPE                     block_dim,                          \
        const BTYPE*              dense_B,                            \
        int64_t                   ldb,                                \
        UTYPE                     beta,                               \
        CTYPE*                    dense_C,                            \
        int64_t                   ldc)

// Uniform precisions
INSTANTIATE(float, int32_t, int32_t, float, float, float, float);
INSTANTIATE(float, int64_t, int32_t, float, float, float, float);
INSTANTIATE(float, int64_t, int64_t, float, float, float, float);
INSTANTIATE(double, int32_t, int32_t, double, double, double, double);
INSTANTIATE(double, int64_t, int32_t, double, double, double, double);
INSTANTIATE(double, int64_t, int64_t, double, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

INSTANTIATE(float, int32_t, int32_t, float, float, float, const float*);
INSTANTIATE(float, int64_t, int32_t, float, float, float, const float*);
INSTANTIATE(float, int64_t, int64_t, float, float, float, const float*);
INSTANTIATE(double, int32_t, int32_t, double, double, double, const double*);
INSTANTIATE(double, int64_t, int32_t, double, double, double, const double*);
INSTANTIATE(double, int64_t, int64_t, double, double, double, const double*);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);

// Mixed Precisions
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float, float);

INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float, const float*);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float, const float*);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float, const float*);
#undef INSTANTIATE
