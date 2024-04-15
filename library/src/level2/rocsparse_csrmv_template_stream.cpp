/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"
#include "control.h"
#include "rocsparse_csrmv.hpp"
#include "utility.h"

#include "csrmv_device.h"
#include "csrmv_symm_device.h"

namespace rocsparse
{
#define LAUNCH_CSRMVN_GENERAL(wfsize)                                               \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrmvn_general_kernel<CSRMVN_DIM, wfsize>), \
                                       dim3(csrmvn_blocks),                         \
                                       dim3(csrmvn_threads),                        \
                                       0,                                           \
                                       stream,                                      \
                                       conj,                                        \
                                       m,                                           \
                                       alpha_device_host,                           \
                                       csr_row_ptr_begin,                           \
                                       csr_row_ptr_end,                             \
                                       csr_col_ind,                                 \
                                       csr_val,                                     \
                                       x,                                           \
                                       beta_device_host,                            \
                                       y,                                           \
                                       descr->base)

#define LAUNCH_CSRMVT(wfsize)                                                       \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrmvt_general_kernel<CSRMVT_DIM, wfsize>), \
                                       dim3(csrmvt_blocks),                         \
                                       dim3(csrmvt_threads),                        \
                                       0,                                           \
                                       stream,                                      \
                                       skip_diag,                                   \
                                       conj,                                        \
                                       m,                                           \
                                       alpha_device_host,                           \
                                       csr_row_ptr_begin,                           \
                                       csr_row_ptr_end,                             \
                                       csr_col_ind,                                 \
                                       csr_val,                                     \
                                       x,                                           \
                                       y,                                           \
                                       descr->base)

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmvn_general_kernel(bool     conj,
                               J        m,
                               U        alpha_device_host,
                               const I* csr_row_ptr_begin,
                               const I* csr_row_ptr_end,
                               const J* __restrict__ csr_col_ind,
                               const A* __restrict__ csr_val,
                               const X* __restrict__ x,
                               U beta_device_host,
                               Y* __restrict__ y,
                               rocsparse_index_base idx_base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        auto beta  = rocsparse::load_scalar_device_host(beta_device_host);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmvn_general_device<BLOCKSIZE, WF_SIZE>(conj,
                                                                 m,
                                                                 alpha,
                                                                 csr_row_ptr_begin,
                                                                 csr_row_ptr_end,
                                                                 csr_col_ind,
                                                                 csr_val,
                                                                 x,
                                                                 beta,
                                                                 y,
                                                                 idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, typename J, typename Y, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmvt_scale_kernel(J size, U scalar_device_host, Y* __restrict__ data)
    {
        auto scalar = rocsparse::load_scalar_device_host(scalar_device_host);
        rocsparse::csrmvt_scale_device(size, scalar, data);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmvt_general_kernel(bool     skip_diag,
                               bool     conj,
                               J        m,
                               U        alpha_device_host,
                               const I* csr_row_ptr_begin,
                               const I* csr_row_ptr_end,
                               const J* __restrict__ csr_col_ind,
                               const A* __restrict__ csr_val,
                               const X* __restrict__ x,
                               Y* __restrict__ y,
                               rocsparse_index_base idx_base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        if(alpha != 0)
        {
            rocsparse::csrmvt_general_device<BLOCKSIZE, WF_SIZE>(skip_diag,
                                                                 conj,
                                                                 m,
                                                                 alpha,
                                                                 csr_row_ptr_begin,
                                                                 csr_row_ptr_end,
                                                                 csr_col_ind,
                                                                 csr_val,
                                                                 x,
                                                                 y,
                                                                 idx_base);
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
rocsparse_status rocsparse::csrmv_stream_template_dispatch(rocsparse_handle    handle,
                                                           rocsparse_operation trans,
                                                           J                   m,
                                                           J                   n,
                                                           I                   nnz,
                                                           U                   alpha_device_host,
                                                           const rocsparse_mat_descr descr,
                                                           const A*                  csr_val,
                                                           const I* csr_row_ptr_begin,
                                                           const I* csr_row_ptr_end,
                                                           const J* csr_col_ind,
                                                           const X* x,
                                                           U        beta_device_host,
                                                           Y*       y,
                                                           bool     force_conj)
{
    bool conj = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    // Stream
    hipStream_t stream = handle->stream;

    if(descr->type == rocsparse_matrix_type_hermitian)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    // Average nnz per row
    J nnz_per_row = nnz / m;

    if(trans == rocsparse_operation_none || descr->type == rocsparse_matrix_type_symmetric)
    {
#define CSRMVN_DIM 512
        dim3 csrmvn_blocks((m - 1) / CSRMVN_DIM + 1);
        dim3 csrmvn_threads(CSRMVN_DIM);

        if(nnz_per_row < 4)
        {
            LAUNCH_CSRMVN_GENERAL(2);
        }
        else if(nnz_per_row < 8)
        {
            LAUNCH_CSRMVN_GENERAL(4);
        }
        else if(nnz_per_row < 16)
        {
            LAUNCH_CSRMVN_GENERAL(8);
        }
        else if(nnz_per_row < 32)
        {
            LAUNCH_CSRMVN_GENERAL(16);
        }
        else if(nnz_per_row < 64 || handle->wavefront_size == 32)
        {
            LAUNCH_CSRMVN_GENERAL(32);
        }
        else
        {
            LAUNCH_CSRMVN_GENERAL(64);
        }
#undef CSRMVN_DIM
    }

    if(trans != rocsparse_operation_none || descr->type == rocsparse_matrix_type_symmetric)
    {
#define CSRMVT_DIM 256
        if(descr->type != rocsparse_matrix_type_symmetric)
        {
            // Scale y with beta
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmvt_scale_kernel<CSRMVT_DIM>),
                                               dim3((n - 1) / CSRMVT_DIM + 1),
                                               dim3(CSRMVT_DIM),
                                               0,
                                               stream,
                                               n,
                                               beta_device_host,
                                               y);
        }

        bool skip_diag = (descr->type == rocsparse_matrix_type_symmetric);

        rocsparse_int max_blocks = 1024;
        rocsparse_int min_blocks = (m - 1) / CSRMVT_DIM + 1;

        dim3 csrmvt_blocks(rocsparse::min(min_blocks, max_blocks));
        dim3 csrmvt_threads(CSRMVT_DIM);

        if(nnz_per_row < 4)
        {
            LAUNCH_CSRMVT(4);
        }
        else if(nnz_per_row < 8)
        {
            LAUNCH_CSRMVT(8);
        }
        else if(nnz_per_row < 16)
        {
            LAUNCH_CSRMVT(16);
        }
        else if(nnz_per_row < 32 || handle->wavefront_size == 32)
        {
            LAUNCH_CSRMVT(32);
        }
        else
        {
            LAUNCH_CSRMVT(64);
        }
#undef CSRMVT_DIM
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, UTYPE)            \
    template rocsparse_status rocsparse::csrmv_stream_template_dispatch<TTYPE>( \
        rocsparse_handle          handle,                                       \
        rocsparse_operation       trans,                                        \
        JTYPE                     m,                                            \
        JTYPE                     n,                                            \
        ITYPE                     nnz,                                          \
        UTYPE                     alpha_device_host,                            \
        const rocsparse_mat_descr descr,                                        \
        const ATYPE*              csr_val,                                      \
        const ITYPE*              csr_row_ptr_begin,                            \
        const ITYPE*              csr_row_ptr_end,                              \
        const JTYPE*              csr_col_ind,                                  \
        const XTYPE*              x,                                            \
        UTYPE                     beta_device_host,                             \
        YTYPE*                    y,                                            \
        bool                      force_conj);

// Uniform precision
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

// Mixed percision
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float, float);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(double, int32_t, int32_t, float, double, double, double);
INSTANTIATE(double, int64_t, int32_t, float, double, double, double);
INSTANTIATE(double, int64_t, int64_t, float, double, double, double);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float, const float*);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float, const float*);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float, const float*);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(double, int32_t, int32_t, float, double, double, const double*);
INSTANTIATE(double, int64_t, int32_t, float, double, double, const double*);
INSTANTIATE(double, int64_t, int64_t, float, double, double, const double*);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);

#undef INSTANTIATE
