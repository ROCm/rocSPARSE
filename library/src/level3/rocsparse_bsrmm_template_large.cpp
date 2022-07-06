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

#include "utility.h"

#include "bsrmm_device_large.h"
#include "rocsparse_csrmm.hpp"

template <rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y, typename T, typename U>
__launch_bounds__(BSR_BLOCK_DIM* BLK_SIZE_Y) ROCSPARSE_KERNEL
    void bsrmm_large_blockdim_kernel(rocsparse_direction direction,
                                     rocsparse_operation trans_B,
                                     rocsparse_int       mb,
                                     rocsparse_int       n,
                                     U                   alpha_device_host,
                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                     const rocsparse_int* __restrict__ bsr_col_ind,
                                     const T* __restrict__ bsr_val,
                                     rocsparse_int block_dim,
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

    bsrmm_large_blockdim_device<BSR_BLOCK_DIM, BLK_SIZE_Y>(direction,
                                                           trans_B,
                                                           mb,
                                                           n,
                                                           alpha,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           bsr_val,
                                                           block_dim,
                                                           B,
                                                           ldb,
                                                           beta,
                                                           C,
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
enum_large_config get_large_config(rocsparse_int block_dim, rocsparse_int n)
{
    enum_large_config config;
    if(block_dim <= 4)
    {
        config = large_config_4_16;
    }
    else if(block_dim <= 8)
    {
        if(n <= 16)
        {
            config = large_config_8_16;
        }
        else
        {
            config = large_config_8_32;
        }
    }
    else if(block_dim <= 16)
    {
        config = large_config_16_16;
    }
    else
    {
        assert(block_dim <= 32);
        config = large_config_32_32;
    }
    return config;
}

template <typename T, typename U>
rocsparse_status rocsparse_bsrmm_template_large(rocsparse_handle          handle,
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
    assert(block_dim <= 32);

#define LAUNCH_LARGE_KERNEL(M_, N_)                           \
    dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / N_ + 1);    \
    dim3 bsrmm_threads(M_, N_);                               \
    hipLaunchKernelGGL((bsrmm_large_blockdim_kernel<M_, N_>), \
                       bsrmm_blocks,                          \
                       bsrmm_threads,                         \
                       0,                                     \
                       stream,                                \
                       dir,                                   \
                       trans_B,                               \
                       mb,                                    \
                       n,                                     \
                       alpha,                                 \
                       bsr_row_ptr,                           \
                       bsr_col_ind,                           \
                       bsr_val,                               \
                       block_dim,                             \
                       B,                                     \
                       ldb,                                   \
                       beta,                                  \
                       C,                                     \
                       ldc,                                   \
                       descr->base)

    //
    // Select which tuned kernel to apply.
    //
    const enum_large_config config = get_large_config(block_dim, n);
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

#define INSTANTIATE(real_type_, scalar_type_)                                                       \
                                                                                                    \
    template rocsparse_status rocsparse_bsrmm_template_large(rocsparse_handle          handle,      \
                                                             rocsparse_direction       dir,         \
                                                             rocsparse_operation       trans_A,     \
                                                             rocsparse_operation       trans_B,     \
                                                             rocsparse_int             mb,          \
                                                             rocsparse_int             n,           \
                                                             rocsparse_int             kb,          \
                                                             rocsparse_int             nnzb,        \
                                                             scalar_type_              alpha,       \
                                                             const rocsparse_mat_descr descr,       \
                                                             const real_type_*         bsr_val,     \
                                                             const rocsparse_int*      bsr_row_ptr, \
                                                             const rocsparse_int*      bsr_col_ind, \
                                                             rocsparse_int             block_dim,   \
                                                             const real_type_*         B,           \
                                                             rocsparse_int             ldb,         \
                                                             scalar_type_              beta,        \
                                                             real_type_*               C,           \
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
