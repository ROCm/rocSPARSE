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

#include "bsrmm_device_general.h"
#include "utility.h"

namespace rocsparse
{
    template <rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y, typename T, typename U>
    ROCSPARSE_KERNEL(BSR_BLOCK_DIM* BLK_SIZE_Y)
    void bsrmm_general_blockdim_kernel(rocsparse_direction direction,
                                       rocsparse_operation trans_B,
                                       rocsparse_int       mb,
                                       rocsparse_int       n,
                                       U                   alpha_device_host,
                                       const rocsparse_int* __restrict__ bsr_row_ptr,
                                       const rocsparse_int* __restrict__ bsr_col_ind,
                                       const T* __restrict__ bsr_val,
                                       rocsparse_int block_dim,
                                       const T* __restrict__ B,
                                       int64_t ldb,
                                       U       beta_device_host,
                                       T* __restrict__ C,
                                       int64_t              ldc,
                                       rocsparse_index_base idx_base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        auto beta  = rocsparse::load_scalar_device_host(beta_device_host);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        rocsparse::bsrmm_general_blockdim_device<BSR_BLOCK_DIM, BLK_SIZE_Y>(direction,
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

    template <typename T, typename U>
    rocsparse_status bsrmm_template_general(rocsparse_handle          handle,
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
                                            int64_t                   ldb,
                                            U                         beta,
                                            T*                        C,
                                            int64_t                   ldc)
    {
        hipStream_t stream = handle->stream;
        assert(block_dim > 32);
        dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
        dim3 bsrmm_threads(32, 32, 1);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrmm_general_blockdim_kernel<32, 32>),
                                           bsrmm_blocks,
                                           bsrmm_threads,
                                           0,
                                           stream,
                                           dir,
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
                                           descr->base);

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(real_type_, scalar_type_)                                                      \
    template rocsparse_status rocsparse::bsrmm_template_general(rocsparse_handle          handle,  \
                                                                rocsparse_direction       dir,     \
                                                                rocsparse_operation       trans_A, \
                                                                rocsparse_operation       trans_B, \
                                                                rocsparse_int             mb,      \
                                                                rocsparse_int             n,       \
                                                                rocsparse_int             kb,      \
                                                                rocsparse_int             nnzb,    \
                                                                scalar_type_              alpha,   \
                                                                const rocsparse_mat_descr descr,   \
                                                                const real_type_*         bsr_val, \
                                                                const rocsparse_int* bsr_row_ptr,  \
                                                                const rocsparse_int* bsr_col_ind,  \
                                                                rocsparse_int        block_dim,    \
                                                                const real_type_*    B,            \
                                                                int64_t              ldb,          \
                                                                scalar_type_         beta,         \
                                                                real_type_*          C,            \
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
