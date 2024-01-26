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

#include "gebsrmm_device_small.h"
#include "rocsparse_csrmm.hpp"

namespace rocsparse
{
    template <rocsparse_int ROW_BLOCK_DIM,
              rocsparse_int COL_BLOCK_DIM,
              rocsparse_int BLOCK_DIM,
              rocsparse_int BLK_SIZE_Y,
              typename T,
              typename U>
    ROCSPARSE_KERNEL(BLOCK_DIM* BLK_SIZE_Y)
    void gebsrmm_small_blockdim_kernel(rocsparse_direction direction,
                                       rocsparse_operation trans_B,
                                       rocsparse_int       mb,
                                       rocsparse_int       n,
                                       U                   alpha_device_host,
                                       const rocsparse_int* __restrict__ bsr_row_ptr,
                                       const rocsparse_int* __restrict__ bsr_col_ind,
                                       const T* __restrict__ bsr_val,
                                       rocsparse_int row_block_dim,
                                       rocsparse_int col_block_dim,
                                       const T* __restrict__ B,
                                       int64_t ldb,
                                       U       beta_device_host,
                                       T* __restrict__ C,
                                       int64_t              ldc,
                                       rocsparse_index_base idx_base)
    {
        auto alpha = load_scalar_device_host(alpha_device_host);
        auto beta  = load_scalar_device_host(beta_device_host);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        rocsparse::
            gebsrmm_small_blockdim_device<ROW_BLOCK_DIM, COL_BLOCK_DIM, BLOCK_DIM, BLK_SIZE_Y>(
                direction,
                trans_B,
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

    typedef enum
    {
        small_config_1_2_16 = 1,
        small_config_1_3_16,
        small_config_1_4_16,
        small_config_2_1_16,
        small_config_2_3_16,
        small_config_2_4_16,
        small_config_3_1_16,
        small_config_3_2_16,
        small_config_3_4_16,
        small_config_4_1_16,
        small_config_4_2_16,
        small_config_4_3_16,
    } enum_small_config;

    //
    // Select which tuned kernel to apply.
    //
    static enum_small_config
        get_small_config(rocsparse_int row_block_dim, rocsparse_int col_block_dim, rocsparse_int n)
    {
        enum_small_config config;
        if(row_block_dim == 1 && col_block_dim == 2)
            config = small_config_1_2_16;
        else if(row_block_dim == 1 && col_block_dim == 3)
            config = small_config_1_3_16;
        else if(row_block_dim == 1 && col_block_dim == 4)
            config = small_config_1_4_16;

        else if(row_block_dim == 2 && col_block_dim == 1)
            config = small_config_2_1_16;
        else if(row_block_dim == 2 && col_block_dim == 3)
            config = small_config_2_3_16;
        else if(row_block_dim == 2 && col_block_dim == 4)
            config = small_config_2_4_16;

        else if(row_block_dim == 3 && col_block_dim == 1)
            config = small_config_3_1_16;
        else if(row_block_dim == 3 && col_block_dim == 2)
            config = small_config_3_2_16;
        else if(row_block_dim == 3 && col_block_dim == 4)
            config = small_config_3_4_16;

        else if(row_block_dim == 4 && col_block_dim == 1)
            config = small_config_4_1_16;
        else if(row_block_dim == 4 && col_block_dim == 2)
            config = small_config_4_2_16;
        else
        {
            assert(row_block_dim == 4 && col_block_dim == 3);
            config = small_config_4_3_16;
        }

        return config;
    }

    template <typename T, typename U>
    rocsparse_status gebsrmm_template_small(rocsparse_handle          handle,
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
                                            rocsparse_int             row_block_dim,
                                            rocsparse_int             col_block_dim,
                                            const T*                  B,
                                            int64_t                   ldb,
                                            U                         beta,
                                            T*                        C,
                                            int64_t                   ldc)
    {
        hipStream_t stream = handle->stream;
        assert(row_block_dim <= 4);
        assert(col_block_dim <= 4);

#define LAUNCH_SMALL_KERNEL(M_, K_, BLOCK_DIM_, N_)                         \
    dim3 gebsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / N_ + 1);                \
    dim3 gebsrmm_threads(BLOCK_DIM_, N_);                                   \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                     \
        (rocsparse::gebsrmm_small_blockdim_kernel<M_, K_, BLOCK_DIM_, N_>), \
        gebsrmm_blocks,                                                     \
        gebsrmm_threads,                                                    \
        0,                                                                  \
        stream,                                                             \
        dir,                                                                \
        trans_B,                                                            \
        mb,                                                                 \
        n,                                                                  \
        alpha,                                                              \
        bsr_row_ptr,                                                        \
        bsr_col_ind,                                                        \
        bsr_val,                                                            \
        row_block_dim,                                                      \
        col_block_dim,                                                      \
        B,                                                                  \
        ldb,                                                                \
        beta,                                                               \
        C,                                                                  \
        ldc,                                                                \
        descr->base)

        //
        // Select which tuned kernel to apply.
        // where the block max size is the maximum block dimension.
        //
        const enum_small_config config
            = rocsparse::get_small_config(row_block_dim, col_block_dim, n);
        switch(config)
        {
#define DEFINE_CASE(i, j, k, block_dim, l)       \
    case i:                                      \
    {                                            \
        LAUNCH_SMALL_KERNEL(j, k, block_dim, l); \
        break;                                   \
    }

            DEFINE_CASE(small_config_1_2_16, 1, 2, 2, 16);
            DEFINE_CASE(small_config_1_3_16, 1, 3, 3, 16);
            DEFINE_CASE(small_config_1_4_16, 1, 4, 4, 16);
            DEFINE_CASE(small_config_2_1_16, 2, 1, 2, 16);
            DEFINE_CASE(small_config_2_3_16, 2, 3, 3, 16);
            DEFINE_CASE(small_config_2_4_16, 2, 4, 4, 16);
            DEFINE_CASE(small_config_3_1_16, 3, 1, 3, 16);
            DEFINE_CASE(small_config_3_2_16, 3, 2, 3, 16);
            DEFINE_CASE(small_config_3_4_16, 3, 4, 4, 16);
            DEFINE_CASE(small_config_4_1_16, 4, 1, 4, 16);
            DEFINE_CASE(small_config_4_2_16, 4, 2, 4, 16);
            DEFINE_CASE(small_config_4_3_16, 4, 3, 4, 16);
#undef DEFINE_CASE
        }

#undef LAUNCH_SMALL_KERNEL
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(real_type_, scalar_type_)                                                       \
                                                                                                    \
    template rocsparse_status rocsparse::gebsrmm_template_small(rocsparse_handle          handle,   \
                                                                rocsparse_direction       dir,      \
                                                                rocsparse_operation       trans_A,  \
                                                                rocsparse_operation       trans_B,  \
                                                                rocsparse_int             mb,       \
                                                                rocsparse_int             n,        \
                                                                rocsparse_int             kb,       \
                                                                rocsparse_int             nnzb,     \
                                                                scalar_type_              alpha,    \
                                                                const rocsparse_mat_descr descr,    \
                                                                const real_type_*         bsr_val,  \
                                                                const rocsparse_int* bsr_row_ptr,   \
                                                                const rocsparse_int* bsr_col_ind,   \
                                                                rocsparse_int        row_block_dim, \
                                                                rocsparse_int        col_block_dim, \
                                                                const real_type_*    B,             \
                                                                int64_t              ldb,           \
                                                                scalar_type_         beta,          \
                                                                real_type_*          C,             \
                                                                int64_t              ldc)

INSTANTIATE(float, float);
INSTANTIATE(float, const float*);

INSTANTIATE(double, double);
INSTANTIATE(double, const double*);

INSTANTIATE(rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex, const rocsparse_float_complex*);

INSTANTIATE(rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex, const rocsparse_double_complex*);

#undef INSTANTIATE
