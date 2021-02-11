/*! \file */
/* ************************************************************************
* Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_csrmm.hpp"

#include "csrmm_device.h"
#include "utility.h"

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) __global__ void csrmmnn_kernel(J m,
                                                            J n,
                                                            J k,
                                                            I nnz,
                                                            U alpha_device_host,
                                                            const I* __restrict__ csr_row_ptr,
                                                            const J* __restrict__ csr_col_ind,
                                                            const T* __restrict__ csr_val,
                                                            const T* __restrict__ B,
                                                            J ldb,
                                                            U beta_device_host,
                                                            T* __restrict__ C,
                                                            J                    ldc,
                                                            rocsparse_order      order,
                                                            rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    csrmmnn_general_device<BLOCKSIZE, WF_SIZE>(m,
                                               n,
                                               k,
                                               nnz,
                                               alpha,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               B,
                                               ldb,
                                               beta,
                                               C,
                                               ldc,
                                               order,
                                               idx_base);
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) __global__ void csrmmnt_kernel(J offset,
                                                            J ncol,
                                                            J m,
                                                            J n,
                                                            J k,
                                                            I nnz,
                                                            U alpha_device_host,
                                                            const I* __restrict__ csr_row_ptr,
                                                            const J* __restrict__ csr_col_ind,
                                                            const T* __restrict__ csr_val,
                                                            const T* __restrict__ B,
                                                            J ldb,
                                                            U beta_device_host,
                                                            T* __restrict__ C,
                                                            J                    ldc,
                                                            rocsparse_order      order,
                                                            rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }
    csrmmnt_general_device<BLOCKSIZE, WF_SIZE>(offset,
                                               ncol,
                                               m,
                                               n,
                                               k,
                                               nnz,
                                               alpha,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               B,
                                               ldb,
                                               beta,
                                               C,
                                               ldc,
                                               order,
                                               idx_base);
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmm_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_order           order,
                                                   J                         m,
                                                   J                         n,
                                                   J                         k,
                                                   I                         nnz,
                                                   U                         alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   const T*                  B,
                                                   J                         ldb,
                                                   U                         beta_device_host,
                                                   T*                        C,
                                                   J                         ldc)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Run different csrmv kernels
    if(trans_A == rocsparse_operation_none)
    {
        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose))
        {
#define CSRMMNN_DIM 256
#define SUB_WF_SIZE 8
            dim3 csrmmnn_blocks((SUB_WF_SIZE * m - 1) / CSRMMNN_DIM + 1, (n - 1) / SUB_WF_SIZE + 1);
            dim3 csrmmnn_threads(CSRMMNN_DIM);

            hipLaunchKernelGGL((csrmmnn_kernel<CSRMMNN_DIM, SUB_WF_SIZE>),
                               csrmmnn_blocks,
                               csrmmnn_threads,
                               0,
                               stream,
                               m,
                               n,
                               k,
                               nnz,
                               alpha_device_host,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               B,
                               ldb,
                               beta_device_host,
                               C,
                               ldc,
                               order,
                               descr->base);
#undef SUB_WF_SIZE
#undef CSRMMNN_DIM
        }
        else if((order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            // Average nnz per row of A
            I avg_row_nnz = (nnz - 1) / m + 1;

#define CSRMMNT_DIM 256
            // Computation is split into two parts, main and remainder
            // First step: Compute main, which is the maximum number of
            //             columns of B that is dividable by the next
            //             power of two of the average row nnz of A.
            // Second step: Compute remainder, which is the remaining
            //              columns of B.
            J main      = 0;
            J remainder = 0;

            // Launch appropriate kernel depending on row nnz of A
            if(avg_row_nnz < 16)
            {
                remainder = n % 8;
                main      = n - remainder;

                // Launch main kernel if enough columns of B
                if(main > 0)
                {
                    hipLaunchKernelGGL((csrmmnt_kernel<CSRMMNT_DIM, 8>),
                                       dim3((8 * m - 1) / CSRMMNT_DIM + 1),
                                       dim3(CSRMMNT_DIM),
                                       0,
                                       stream,
                                       (J)0,
                                       main,
                                       m,
                                       n,
                                       k,
                                       nnz,
                                       alpha_device_host,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       B,
                                       ldb,
                                       beta_device_host,
                                       C,
                                       ldc,
                                       order,
                                       descr->base);
                }
            }
            else if(avg_row_nnz < 32)
            {
                remainder = n % 16;
                main      = n - remainder;

                // Launch main kernel if enough columns of B
                if(main > 0)
                {
                    hipLaunchKernelGGL((csrmmnt_kernel<CSRMMNT_DIM, 16>),
                                       dim3((16 * m - 1) / CSRMMNT_DIM + 1),
                                       dim3(CSRMMNT_DIM),
                                       0,
                                       stream,
                                       (J)0,
                                       main,
                                       m,
                                       n,
                                       k,
                                       nnz,
                                       alpha_device_host,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       B,
                                       ldb,
                                       beta_device_host,
                                       C,
                                       ldc,
                                       order,
                                       descr->base);
                }
            }
            else if(avg_row_nnz < 64 || handle->wavefront_size == 32)
            {
                remainder = n % 32;
                main      = n - remainder;

                // Launch main kernel if enough columns of B
                if(main > 0)
                {
                    hipLaunchKernelGGL((csrmmnt_kernel<CSRMMNT_DIM, 32>),
                                       dim3((32 * m - 1) / CSRMMNT_DIM + 1),
                                       dim3(CSRMMNT_DIM),
                                       0,
                                       stream,
                                       (J)0,
                                       main,
                                       m,
                                       n,
                                       k,
                                       nnz,
                                       alpha_device_host,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       B,
                                       ldb,
                                       beta_device_host,
                                       C,
                                       ldc,
                                       order,
                                       descr->base);
                }
            }
            else if(handle->wavefront_size == 64)
            {
                remainder = n % 64;
                main      = n - remainder;

                // Launch main kernel if enough columns of B
                if(main > 0)
                {
                    hipLaunchKernelGGL((csrmmnt_kernel<CSRMMNT_DIM, 64>),
                                       dim3((64 * m - 1) / CSRMMNT_DIM + 1),
                                       dim3(CSRMMNT_DIM),
                                       0,
                                       stream,
                                       (J)0,
                                       main,
                                       m,
                                       n,
                                       k,
                                       nnz,
                                       alpha_device_host,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       B,
                                       ldb,
                                       beta_device_host,
                                       C,
                                       ldc,
                                       order,
                                       descr->base);
                }
            }
            else
            {
                return rocsparse_status_arch_mismatch;
            }

            // Process remainder
            if(remainder > 0)
            {
                if(remainder <= 8)
                {
                    hipLaunchKernelGGL((csrmmnt_kernel<CSRMMNT_DIM, 8>),
                                       dim3((8 * m - 1) / CSRMMNT_DIM + 1),
                                       dim3(CSRMMNT_DIM),
                                       0,
                                       stream,
                                       main,
                                       n,
                                       m,
                                       n,
                                       k,
                                       nnz,
                                       alpha_device_host,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       B,
                                       ldb,
                                       beta_device_host,
                                       C,
                                       ldc,
                                       order,
                                       descr->base);
                }
                else if(remainder <= 16)
                {
                    hipLaunchKernelGGL((csrmmnt_kernel<CSRMMNT_DIM, 16>),
                                       dim3((16 * m - 1) / CSRMMNT_DIM + 1),
                                       dim3(CSRMMNT_DIM),
                                       0,
                                       stream,
                                       main,
                                       n,
                                       m,
                                       n,
                                       k,
                                       nnz,
                                       alpha_device_host,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       B,
                                       ldb,
                                       beta_device_host,
                                       C,
                                       ldc,
                                       order,
                                       descr->base);
                }
                else if(remainder <= 32 || handle->wavefront_size == 32)
                {
                    hipLaunchKernelGGL((csrmmnt_kernel<CSRMMNT_DIM, 32>),
                                       dim3((32 * m - 1) / CSRMMNT_DIM + 1),
                                       dim3(CSRMMNT_DIM),
                                       0,
                                       stream,
                                       main,
                                       n,
                                       m,
                                       n,
                                       k,
                                       nnz,
                                       alpha_device_host,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       B,
                                       ldb,
                                       beta_device_host,
                                       C,
                                       ldc,
                                       order,
                                       descr->base);
                }
                else if(remainder <= 64)
                {
                    hipLaunchKernelGGL((csrmmnt_kernel<CSRMMNT_DIM, 64>),
                                       dim3((64 * m - 1) / CSRMMNT_DIM + 1),
                                       dim3(CSRMMNT_DIM),
                                       0,
                                       stream,
                                       main,
                                       n,
                                       m,
                                       n,
                                       k,
                                       nnz,
                                       alpha_device_host,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       B,
                                       ldb,
                                       beta_device_host,
                                       C,
                                       ldc,
                                       order,
                                       descr->base);
                }
                else
                {
                    return rocsparse_status_arch_mismatch;
                }
            }
#undef CSRMMNT_DIM
        }
        else
        {
            return rocsparse_status_not_implemented;
        }
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrmm_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_order           order_B,
                                          rocsparse_order           order_C,
                                          J                         m,
                                          J                         n,
                                          J                         k,
                                          I                         nnz,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          const T*                  B,
                                          J                         ldb,
                                          const T*                  beta_device_host,
                                          T*                        C,
                                          J                         ldc)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging TODO bench logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  nnz,
                  *alpha_device_host,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  *beta_device_host,
                  (const void*&)C,
                  ldc);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  nnz,
                  (const void*&)alpha_device_host,
                  (const void*&)descr,
                  (const void*&)csr_val,
                  (const void*&)csr_row_ptr,
                  (const void*&)csr_col_ind,
                  (const void*&)B,
                  ldb,
                  (const void*&)beta_device_host,
                  (const void*&)C,
                  ldc);
    }

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    if(order_B != order_C)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        return rocsparse_status_success;
    }

    //
    // Check the rest of pointer arguments
    //
    if(alpha_device_host == nullptr || beta_device_host == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    //
    // Check the rest of pointer arguments
    //
    if(csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || B == nullptr
       || C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check leading dimension of B
    J one = 1;
    if(trans_B == rocsparse_operation_none)
    {
        if(ldb < std::max(one, order_B == rocsparse_order_column ? k : n))
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(ldb < std::max(one, order_B == rocsparse_order_column ? n : k))
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check leading dimension of C
    if(ldc < std::max(one, order_C == rocsparse_order_column ? m : n))
    {
        return rocsparse_status_invalid_size;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_csrmm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
                                                 m,
                                                 n,
                                                 k,
                                                 nnz,
                                                 alpha_device_host,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 B,
                                                 ldb,
                                                 beta_device_host,
                                                 C,
                                                 ldc);
    }
    else
    {
        return rocsparse_csrmm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
                                                 m,
                                                 n,
                                                 k,
                                                 nnz,
                                                 *alpha_device_host,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 B,
                                                 ldb,
                                                 *beta_device_host,
                                                 C,
                                                 ldc);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                     \
    template rocsparse_status rocsparse_csrmm_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                    \
        rocsparse_operation       trans_A,                                   \
        rocsparse_operation       trans_B,                                   \
        rocsparse_order           order_B,                                   \
        rocsparse_order           order_C,                                   \
        JTYPE                     m,                                         \
        JTYPE                     n,                                         \
        JTYPE                     k,                                         \
        ITYPE                     nnz,                                       \
        const TTYPE*              alpha_device_host,                         \
        const rocsparse_mat_descr descr,                                     \
        const TTYPE*              csr_val,                                   \
        const ITYPE*              csr_row_ptr,                               \
        const JTYPE*              csr_col_ind,                               \
        const TTYPE*              B,                                         \
        JTYPE                     ldb,                                       \
        const TTYPE*              beta_device_host,                          \
        TTYPE*                    C,                                         \
        JTYPE                     ldc);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans_A,     \
                                     rocsparse_operation       trans_B,     \
                                     rocsparse_int             m,           \
                                     rocsparse_int             n,           \
                                     rocsparse_int             k,           \
                                     rocsparse_int             nnz,         \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     const TYPE*               B,           \
                                     rocsparse_int             ldb,         \
                                     const TYPE*               beta,        \
                                     TYPE*                     C,           \
                                     rocsparse_int             ldc)         \
    {                                                                       \
        return rocsparse_csrmm_template(handle,                             \
                                        trans_A,                            \
                                        trans_B,                            \
                                        rocsparse_order_column,             \
                                        rocsparse_order_column,             \
                                        m,                                  \
                                        n,                                  \
                                        k,                                  \
                                        nnz,                                \
                                        alpha,                              \
                                        descr,                              \
                                        csr_val,                            \
                                        csr_row_ptr,                        \
                                        csr_col_ind,                        \
                                        B,                                  \
                                        ldb,                                \
                                        beta,                               \
                                        C,                                  \
                                        ldc);                               \
    }

C_IMPL(rocsparse_scsrmm, float);
C_IMPL(rocsparse_dcsrmm, double);
C_IMPL(rocsparse_ccsrmm, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrmm, rocsparse_double_complex);

#undef C_IMPL
