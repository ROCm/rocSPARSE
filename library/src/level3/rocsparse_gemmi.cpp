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

#include "internal/level3/rocsparse_gemmi.h"
#include "gemmi_device.h"
#include "rocsparse_gemmi.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gemmi_scale_kernel(rocsparse_int size, U alpha_device_host, T* __restrict__ data)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        rocsparse::gemmi_scale_kernel<BLOCKSIZE>(size, alpha, data);
    }

    template <uint32_t BLOCKSIZE, typename T, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gemmit_kernel(rocsparse_int m,
                       U             alpha_device_host,
                       const T* __restrict__ A,
                       rocsparse_int lda,
                       const rocsparse_int* __restrict__ csr_row_ptr,
                       const rocsparse_int* __restrict__ csr_col_ind,
                       const T* __restrict__ csr_val,
                       U beta_device_host,
                       T* __restrict__ C,
                       rocsparse_int        ldc,
                       rocsparse_index_base base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        auto beta  = rocsparse::load_scalar_device_host(beta_device_host);
        rocsparse::gemmit_kernel<BLOCKSIZE>(
            m, alpha, A, lda, csr_row_ptr, csr_col_ind, csr_val, beta, C, ldc, base);
    }

    template <typename T>
    static rocsparse_status gemmi_core(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       rocsparse_int             k,
                                       rocsparse_int             nnz,
                                       const T*                  alpha,
                                       const T*                  A,
                                       rocsparse_int             lda,
                                       const rocsparse_mat_descr descr,
                                       const T*                  csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       const T*                  beta,
                                       T*                        C,
                                       rocsparse_int             ldc)
    {

        // Stream
        hipStream_t stream = handle->stream;

        // If k == 0, scale C with beta
        if(k == 0)
        {
#define SCALE_DIM 256
            dim3 scale_blocks((m * n - 1) / SCALE_DIM + 1);
            dim3 scale_threads(SCALE_DIM);

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gemmi_scale_kernel<SCALE_DIM>),
                                                   scale_blocks,
                                                   scale_threads,
                                                   0,
                                                   stream,
                                                   m * n,
                                                   beta,
                                                   C);
            }
            else
            {
                if(*beta == static_cast<T>(0))
                {
                    RETURN_IF_HIP_ERROR(hipMemsetAsync(C, 0, sizeof(T) * m * n, stream));
                }
                else if(*beta != static_cast<T>(1))
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gemmi_scale_kernel<SCALE_DIM>),
                                                       scale_blocks,
                                                       scale_threads,
                                                       0,
                                                       stream,
                                                       m * n,
                                                       *beta,
                                                       C);
                }
            }
#undef SCALE_DIM

            return rocsparse_status_success;
        }

#define GEMMIT_DIM 256
        dim3 gemmit_blocks((m - 1) / GEMMIT_DIM + 1, n);
        dim3 gemmit_threads(GEMMIT_DIM);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gemmit_kernel<GEMMIT_DIM>),
                                               gemmit_blocks,
                                               gemmit_threads,
                                               0,
                                               stream,
                                               m,
                                               alpha,
                                               A,
                                               lda,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               beta,
                                               C,
                                               ldc,
                                               descr->base);
        }
        else
        {
            // Quick return
            if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }
            else if(*alpha == static_cast<T>(0))
            {
                if(*beta == static_cast<T>(0))
                {
                    RETURN_IF_HIP_ERROR(hipMemsetAsync(C, 0, sizeof(T) * m * n, stream));
                }
                else
                {
#define SCALE_DIM 256
                    dim3 scale_blocks((m * n - 1) / SCALE_DIM + 1);
                    dim3 scale_threads(SCALE_DIM);

                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gemmi_scale_kernel<SCALE_DIM>),
                                                       scale_blocks,
                                                       scale_threads,
                                                       0,
                                                       stream,
                                                       m * n,
                                                       *beta,
                                                       C);
#undef SCALE_DIM
                }

                return rocsparse_status_success;
            }

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gemmit_kernel<GEMMIT_DIM>),
                                               gemmit_blocks,
                                               gemmit_threads,
                                               0,
                                               stream,
                                               m,
                                               *alpha,
                                               A,
                                               lda,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               *beta,
                                               C,
                                               ldc,
                                               descr->base);
        }
#undef GEMMIT_DIM

        return rocsparse_status_success;
    }

    template <typename T>
    static rocsparse_status gemmi_quickreturn(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             k,
                                              rocsparse_int             nnz,
                                              const T*                  alpha,
                                              const T*                  A,
                                              rocsparse_int             lda,
                                              const rocsparse_mat_descr descr,
                                              const T*                  csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              const T*                  beta,
                                              T*                        C,
                                              rocsparse_int             ldc)
    {
        if(m == 0 || n == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename T>
    static rocsparse_status gemmi_checkarg(rocsparse_handle          handle, //0
                                           rocsparse_operation       trans_A, //1
                                           rocsparse_operation       trans_B, //2
                                           rocsparse_int             m, //3
                                           rocsparse_int             n, //4
                                           rocsparse_int             k, //5
                                           rocsparse_int             nnz, //6
                                           const T*                  alpha, //7
                                           const T*                  A, //8
                                           rocsparse_int             lda, //9
                                           const rocsparse_mat_descr descr, //10
                                           const T*                  csr_val, //11
                                           const rocsparse_int*      csr_row_ptr, //12
                                           const rocsparse_int*      csr_col_ind, //13
                                           const T*                  beta, //14
                                           T*                        C, //15
                                           rocsparse_int             ldc) //16
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, n);
        ROCSPARSE_CHECKARG_SIZE(5, k);
        ROCSPARSE_CHECKARG_SIZE(6, nnz);

        const rocsparse_status status = rocsparse::gemmi_quickreturn(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     nnz,
                                                                     alpha,
                                                                     A,
                                                                     lda,
                                                                     descr,
                                                                     csr_val,
                                                                     csr_row_ptr,
                                                                     csr_col_ind,
                                                                     beta,
                                                                     C,
                                                                     ldc);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG(
            1, trans_A, (trans_A != rocsparse_operation_none), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(2,
                           trans_B,
                           (trans_B != rocsparse_operation_transpose),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG_POINTER(10, descr);
        ROCSPARSE_CHECKARG(10,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG_ARRAY(7, k, alpha);
        ROCSPARSE_CHECKARG_ARRAY(8, k, A);

        ROCSPARSE_CHECKARG(9,
                           lda,
                           (lda < rocsparse::max(static_cast<rocsparse_int>(1), m)),
                           rocsparse_status_invalid_size);

        ROCSPARSE_CHECKARG_ARRAY(11, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(12, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(13, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(14, beta);
        ROCSPARSE_CHECKARG_POINTER(15, C);

        ROCSPARSE_CHECKARG(16,
                           ldc,
                           (ldc < rocsparse::max(static_cast<rocsparse_int>(1), m)),
                           rocsparse_status_invalid_size);

        return rocsparse_status_continue;
    }
}

template <typename T>
rocsparse_status rocsparse::gemmi_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           rocsparse_int             k,
                                           rocsparse_int             nnz,
                                           const T*                  alpha,
                                           const T*                  A,
                                           rocsparse_int             lda,
                                           const rocsparse_mat_descr descr,
                                           const T*                  csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           const T*                  beta,
                                           T*                        C,
                                           rocsparse_int             ldc)
{

    const rocsparse_status status = rocsparse::gemmi_quickreturn(handle,
                                                                 trans_A,
                                                                 trans_B,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 nnz,
                                                                 alpha,
                                                                 A,
                                                                 lda,
                                                                 descr,
                                                                 csr_val,
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
                                                                 beta,
                                                                 C,
                                                                 ldc);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gemmi_core(handle,
                                                    trans_A,
                                                    trans_B,
                                                    m,
                                                    n,
                                                    k,
                                                    nnz,
                                                    alpha,
                                                    A,
                                                    lda,
                                                    descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    beta,
                                                    C,
                                                    ldc));

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    rocsparse_status gemmi_impl(rocsparse_handle          handle,
                                rocsparse_operation       trans_A,
                                rocsparse_operation       trans_B,
                                rocsparse_int             m,
                                rocsparse_int             n,
                                rocsparse_int             k,
                                rocsparse_int             nnz,
                                const T*                  alpha,
                                const T*                  A,
                                rocsparse_int             lda,
                                const rocsparse_mat_descr descr,
                                const T*                  csr_val,
                                const rocsparse_int*      csr_row_ptr,
                                const rocsparse_int*      csr_col_ind,
                                const T*                  beta,
                                T*                        C,
                                rocsparse_int             ldc)
    {

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xgemmi"),
                             trans_A,
                             trans_B,
                             m,
                             n,
                             k,
                             nnz,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)A,
                             lda,
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             LOG_TRACE_SCALAR_VALUE(handle, beta),
                             (const void*&)C,
                             ldc);

        const rocsparse_status status = rocsparse::gemmi_checkarg(handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  nnz,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind,
                                                                  beta,
                                                                  C,
                                                                  ldc);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gemmi_core(handle,
                                                        trans_A,
                                                        trans_B,
                                                        m,
                                                        n,
                                                        k,
                                                        nnz,
                                                        alpha,
                                                        A,
                                                        lda,
                                                        descr,
                                                        csr_val,
                                                        csr_row_ptr,
                                                        csr_col_ind,
                                                        beta,
                                                        C,
                                                        ldc));

        return rocsparse_status_success;
    }
}

#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans_A,     \
                                     rocsparse_operation       trans_B,     \
                                     rocsparse_int             m,           \
                                     rocsparse_int             n,           \
                                     rocsparse_int             k,           \
                                     rocsparse_int             nnz,         \
                                     const TYPE*               alpha,       \
                                     const TYPE*               A,           \
                                     rocsparse_int             lda,         \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     const TYPE*               beta,        \
                                     TYPE*                     C,           \
                                     rocsparse_int             ldc)         \
    try                                                                     \
    {                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gemmi_impl(handle,             \
                                                        trans_A,            \
                                                        trans_B,            \
                                                        m,                  \
                                                        n,                  \
                                                        k,                  \
                                                        nnz,                \
                                                        alpha,              \
                                                        A,                  \
                                                        lda,                \
                                                        descr,              \
                                                        csr_val,            \
                                                        csr_row_ptr,        \
                                                        csr_col_ind,        \
                                                        beta,               \
                                                        C,                  \
                                                        ldc));              \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

C_IMPL(rocsparse_sgemmi, float);
C_IMPL(rocsparse_dgemmi, double);
C_IMPL(rocsparse_cgemmi, rocsparse_float_complex);
C_IMPL(rocsparse_zgemmi, rocsparse_double_complex);
#undef C_IMPL
