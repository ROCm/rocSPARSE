/*! \file */
/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "csrgemm_device.h"
#include "definitions.h"
#include "rocsparse_csrgemm_nnz_calc.hpp"
#include "utility.h"

template <typename I>
rocsparse_status rocsparse_csrgemm_multadd_nnz_quickreturn(rocsparse_handle          handle,
                                                           rocsparse_operation       trans_A,
                                                           rocsparse_operation       trans_B,
                                                           int64_t                   m,
                                                           int64_t                   n,
                                                           int64_t                   k,
                                                           const rocsparse_mat_descr descr_A,
                                                           int64_t                   nnz_A,
                                                           const void*               csr_row_ptr_A,
                                                           const void*               csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           int64_t                   nnz_B,
                                                           const void*               csr_row_ptr_B,
                                                           const void*               csr_col_ind_B,
                                                           const rocsparse_mat_descr descr_D,
                                                           int64_t                   nnz_D,
                                                           const void*               csr_row_ptr_D,
                                                           const void*               csr_col_ind_D,
                                                           const rocsparse_mat_descr descr_C,
                                                           I*                        csr_row_ptr_C,
                                                           I*                        nnz_C,
                                                           const rocsparse_mat_info  info_C,
                                                           void*                     temp_buffer)
{
    if(m == 0 || n == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), handle->stream));
        }
        else
        {
            *nnz_C = 0;
        }

        if(m > 0)
        {
#define CSRGEMM_DIM 1024
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrgemm_set_base<CSRGEMM_DIM>),
                                               dim3((m + 1) / CSRGEMM_DIM + 1),
                                               dim3(CSRGEMM_DIM),
                                               0,
                                               handle->stream,
                                               m + 1,
                                               csr_row_ptr_C,
                                               descr_C->base);
#undef CSRGEMM_DIM
        }
        return rocsparse_status_success;
    }

#if 0
    // k == 0 || nnz_A == 0 || nnz_B == 0 - scale D with beta
    if(k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_csrgemm_nnz_scal(handle,
                                          m,
                                          n,
                                          descr_D,
                                          nnz_D,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          descr_C,
                                          csr_row_ptr_C,
                                          nnz_C,
                                          info_C,
                                          temp_buffer);
    }

    if((trans_A != rocsparse_operation_none) || (trans_B != rocsparse_operation_none))
    {
        return rocsparse_status_not_implemented;
    }

    // nnz_D == 0 - compute alpha * A * B
    if(nnz_D == 0)
    {
        return rocsparse_csrgemm_nnz_mult(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          descr_A,
                                          nnz_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          descr_C,
                                          csr_row_ptr_C,
                                          nnz_C,
                                          info_C,
                                          temp_buffer);
    }
#endif

    return rocsparse_status_continue;
}

#define INSTANTIATE(I)                                                   \
    template rocsparse_status rocsparse_csrgemm_multadd_nnz_quickreturn( \
        rocsparse_handle          handle,                                \
        rocsparse_operation       trans_A,                               \
        rocsparse_operation       trans_B,                               \
        int64_t                   m,                                     \
        int64_t                   n,                                     \
        int64_t                   k,                                     \
        const rocsparse_mat_descr descr_A,                               \
        int64_t                   nnz_A,                                 \
        const void*               csr_row_ptr_A,                         \
        const void*               csr_col_ind_A,                         \
        const rocsparse_mat_descr descr_B,                               \
        int64_t                   nnz_B,                                 \
        const void*               csr_row_ptr_B,                         \
        const void*               csr_col_ind_B,                         \
        const rocsparse_mat_descr descr_D,                               \
        int64_t                   nnz_D,                                 \
        const void*               csr_row_ptr_D,                         \
        const void*               csr_col_ind_D,                         \
        const rocsparse_mat_descr descr_C,                               \
        I*                        csr_row_ptr_C,                         \
        I*                        nnz_C,                                 \
        const rocsparse_mat_info  info_C,                                \
        void*                     temp_buffer)

INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
#undef INSTANTIATE

template <typename I, typename J>
rocsparse_status rocsparse_csrgemm_multadd_nnz_core(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    J                         m,
                                                    J                         n,
                                                    J                         k,
                                                    const rocsparse_mat_descr descr_A,
                                                    I                         nnz_A,
                                                    const I*                  csr_row_ptr_A,
                                                    const J*                  csr_col_ind_A,
                                                    const rocsparse_mat_descr descr_B,
                                                    I                         nnz_B,
                                                    const I*                  csr_row_ptr_B,
                                                    const J*                  csr_col_ind_B,
                                                    const rocsparse_mat_descr descr_D,
                                                    I                         nnz_D,
                                                    const I*                  csr_row_ptr_D,
                                                    const J*                  csr_col_ind_D,
                                                    const rocsparse_mat_descr descr_C,
                                                    I*                        csr_row_ptr_C,
                                                    I*                        nnz_C,
                                                    const rocsparse_mat_info  info_C,
                                                    void*                     temp_buffer)
{
#if 0
    // k == 0 || nnz_A == 0 || nnz_B == 0 - scale D with beta
    if(k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_csrgemm_nnz_scal(handle,
                                          m,
                                          n,
                                          descr_D,
                                          nnz_D,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          descr_C,
                                          csr_row_ptr_C,
                                          nnz_C,
                                          info_C,
                                          temp_buffer);
    }

    // nnz_D == 0 - compute alpha * A * B
    if(nnz_D == 0)
    {
        return rocsparse_csrgemm_nnz_mult(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          descr_A,
                                          nnz_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          descr_C,
                                          csr_row_ptr_C,
                                          nnz_C,
                                          info_C,
                                          temp_buffer);
    }
#endif

    // Perform nnz calculation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz_calc(handle,
                                                         trans_A,
                                                         trans_B,
                                                         m,
                                                         n,
                                                         k,
                                                         descr_A,
                                                         nnz_A,
                                                         csr_row_ptr_A,
                                                         csr_col_ind_A,
                                                         descr_B,
                                                         nnz_B,
                                                         csr_row_ptr_B,
                                                         csr_col_ind_B,
                                                         descr_D,
                                                         nnz_D,
                                                         csr_row_ptr_D,
                                                         csr_col_ind_D,
                                                         descr_C,
                                                         csr_row_ptr_C,
                                                         nnz_C,
                                                         info_C,
                                                         temp_buffer));
    return rocsparse_status_success;
}

#define INSTANTIATE(I, J)                                         \
    template rocsparse_status rocsparse_csrgemm_multadd_nnz_core( \
        rocsparse_handle          handle,                         \
        rocsparse_operation       trans_A,                        \
        rocsparse_operation       trans_B,                        \
        J                         m,                              \
        J                         n,                              \
        J                         k,                              \
        const rocsparse_mat_descr descr_A,                        \
        I                         nnz_A,                          \
        const I*                  csr_row_ptr_A,                  \
        const J*                  csr_col_ind_A,                  \
        const rocsparse_mat_descr descr_B,                        \
        I                         nnz_B,                          \
        const I*                  csr_row_ptr_B,                  \
        const J*                  csr_col_ind_B,                  \
        const rocsparse_mat_descr descr_D,                        \
        I                         nnz_D,                          \
        const I*                  csr_row_ptr_D,                  \
        const J*                  csr_col_ind_D,                  \
        const rocsparse_mat_descr descr_C,                        \
        I*                        csr_row_ptr_C,                  \
        I*                        nnz_C,                          \
        const rocsparse_mat_info  info_C,                         \
        void*                     temp_buffer)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE