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

#include "utility.h"

#include "csrmm_device.h"

template <unsigned int DIM_X, unsigned int DIM_Y, typename I, typename T, typename U>
__launch_bounds__(DIM_X* DIM_Y) ROCSPARSE_KERNEL void csrmm_scale(
    I m, I n, U beta_device_host, T* __restrict__ data, I ld, I stride, rocsparse_order order)
{
    auto beta = load_scalar_device_host(beta_device_host);
    if(beta != static_cast<T>(1))
    {
        csrmm_scale_device(m, n, beta, data, ld, stride, order);
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmmnn_general_kernel(bool conj_A,
                                bool conj_B,
                                J    m,
                                J    n,
                                J    k,
                                I    nnz,
                                I    offsets_batch_stride_A,
                                I    columns_values_batch_stride_A,
                                U    alpha_device_host,
                                const I* __restrict__ csr_row_ptr,
                                const J* __restrict__ csr_col_ind,
                                const T* __restrict__ csr_val,
                                const T* __restrict__ B,
                                J ldb,
                                I batch_stride_B,
                                U beta_device_host,
                                T* __restrict__ C,
                                J                    ldc,
                                I                    batch_stride_C,
                                rocsparse_order      order,
                                rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    csrmmnn_general_device<BLOCKSIZE, WF_SIZE>(
        conj_A,
        conj_B,
        m,
        n,
        k,
        nnz,
        alpha,
        load_pointer(csr_row_ptr, hipBlockIdx_z, offsets_batch_stride_A),
        load_pointer(csr_col_ind, hipBlockIdx_z, columns_values_batch_stride_A),
        load_pointer(csr_val, hipBlockIdx_z, columns_values_batch_stride_A),
        load_pointer(B, hipBlockIdx_z, batch_stride_B),
        ldb,
        beta,
        load_pointer(C, hipBlockIdx_z, batch_stride_C),
        ldc,
        order,
        idx_base);
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmmnt_general_main_kernel(bool conj_A,
                                     bool conj_B,
                                     J    offset,
                                     J    ncol,
                                     J    m,
                                     J    n,
                                     J    k,
                                     I    nnz,
                                     I    offsets_batch_stride_A,
                                     I    columns_values_batch_stride_A,
                                     U    alpha_device_host,
                                     const I* __restrict__ csr_row_ptr,
                                     const J* __restrict__ csr_col_ind,
                                     const T* __restrict__ csr_val,
                                     const T* __restrict__ B,
                                     J ldb,
                                     I batch_stride_B,
                                     U beta_device_host,
                                     T* __restrict__ C,
                                     J                    ldc,
                                     I                    batch_stride_C,
                                     rocsparse_order      order,
                                     rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }
    csrmmnt_general_main_device<BLOCKSIZE, WF_SIZE, LOOPS>(
        conj_A,
        conj_B,
        offset,
        ncol,
        m,
        n,
        k,
        nnz,
        alpha,
        load_pointer(csr_row_ptr, hipBlockIdx_y, offsets_batch_stride_A),
        load_pointer(csr_col_ind, hipBlockIdx_y, columns_values_batch_stride_A),
        load_pointer(csr_val, hipBlockIdx_y, columns_values_batch_stride_A),
        load_pointer(B, hipBlockIdx_y, batch_stride_B),
        ldb,
        beta,
        load_pointer(C, hipBlockIdx_y, batch_stride_C),
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
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmmnt_general_remainder_kernel(bool conj_A,
                                          bool conj_B,
                                          J    offset,
                                          J    ncol,
                                          J    m,
                                          J    n,
                                          J    k,
                                          I    nnz,
                                          I    offsets_batch_stride_A,
                                          I    columns_values_batch_stride_A,
                                          U    alpha_device_host,
                                          const I* __restrict__ csr_row_ptr,
                                          const J* __restrict__ csr_col_ind,
                                          const T* __restrict__ csr_val,
                                          const T* __restrict__ B,
                                          J ldb,
                                          I batch_stride_B,
                                          U beta_device_host,
                                          T* __restrict__ C,
                                          J                    ldc,
                                          I                    batch_stride_C,
                                          rocsparse_order      order,
                                          rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }
    csrmmnt_general_remainder_device<BLOCKSIZE, WF_SIZE>(
        conj_A,
        conj_B,
        offset,
        ncol,
        m,
        n,
        k,
        nnz,
        alpha,
        load_pointer(csr_row_ptr, hipBlockIdx_y, offsets_batch_stride_A),
        load_pointer(csr_col_ind, hipBlockIdx_y, columns_values_batch_stride_A),
        load_pointer(csr_val, hipBlockIdx_y, columns_values_batch_stride_A),
        load_pointer(B, hipBlockIdx_y, batch_stride_B),
        ldb,
        beta,
        load_pointer(C, hipBlockIdx_y, batch_stride_C),
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
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmmtn_general_kernel(bool conj_A,
                                bool conj_B,
                                J    m,
                                J    n,
                                J    k,
                                I    nnz,
                                I    offsets_batch_stride_A,
                                I    columns_values_batch_stride_A,
                                U    alpha_device_host,
                                const I* __restrict__ csr_row_ptr,
                                const J* __restrict__ csr_col_ind,
                                const T* __restrict__ csr_val,
                                const T* __restrict__ B,
                                J ldb,
                                I batch_stride_B,
                                U beta_device_host,
                                T* __restrict__ C,
                                J                    ldc,
                                I                    batch_stride_C,
                                rocsparse_order      order,
                                rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }
    csrmmtn_general_device<BLOCKSIZE, WF_SIZE>(
        conj_A,
        conj_B,
        m,
        n,
        k,
        nnz,
        alpha,
        load_pointer(csr_row_ptr, hipBlockIdx_z, offsets_batch_stride_A),
        load_pointer(csr_col_ind, hipBlockIdx_z, columns_values_batch_stride_A),
        load_pointer(csr_val, hipBlockIdx_z, columns_values_batch_stride_A),
        load_pointer(B, hipBlockIdx_z, batch_stride_B),
        ldb,
        beta,
        load_pointer(C, hipBlockIdx_z, batch_stride_C),
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
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmmtt_general_kernel(bool conj_A,
                                bool conj_B,
                                J    m,
                                J    n,
                                J    k,
                                I    nnz,
                                I    offsets_batch_stride_A,
                                I    columns_values_batch_stride_A,
                                U    alpha_device_host,
                                const I* __restrict__ csr_row_ptr,
                                const J* __restrict__ csr_col_ind,
                                const T* __restrict__ csr_val,
                                const T* __restrict__ B,
                                J ldb,
                                I batch_stride_B,
                                U beta_device_host,
                                T* __restrict__ C,
                                J                    ldc,
                                I                    batch_stride_C,
                                rocsparse_order      order,
                                rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }
    csrmmtt_general_device<BLOCKSIZE, WF_SIZE>(
        conj_A,
        conj_B,
        m,
        n,
        k,
        nnz,
        alpha,
        load_pointer(csr_row_ptr, hipBlockIdx_z, offsets_batch_stride_A),
        load_pointer(csr_col_ind, hipBlockIdx_z, columns_values_batch_stride_A),
        load_pointer(csr_val, hipBlockIdx_z, columns_values_batch_stride_A),
        load_pointer(B, hipBlockIdx_z, batch_stride_B),
        ldb,
        beta,
        load_pointer(C, hipBlockIdx_z, batch_stride_C),
        ldc,
        order,
        idx_base);
}

#define LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(CSRMMNT_DIM, WF_SIZE, LOOPS)            \
    hipLaunchKernelGGL((csrmmnt_general_main_kernel<CSRMMNT_DIM, WF_SIZE, LOOPS>), \
                       dim3((m - 1) / (CSRMMNT_DIM / WF_SIZE) + 1, batch_count_C), \
                       dim3(CSRMMNT_DIM),                                          \
                       0,                                                          \
                       handle->stream,                                             \
                       conj_A,                                                     \
                       conj_B,                                                     \
                       (J)0,                                                       \
                       main,                                                       \
                       m,                                                          \
                       n,                                                          \
                       k,                                                          \
                       nnz,                                                        \
                       offsets_batch_stride_A,                                     \
                       columns_values_batch_stride_A,                              \
                       alpha_device_host,                                          \
                       csr_row_ptr,                                                \
                       csr_col_ind,                                                \
                       csr_val,                                                    \
                       B,                                                          \
                       ldb,                                                        \
                       batch_stride_B,                                             \
                       beta_device_host,                                           \
                       C,                                                          \
                       ldc,                                                        \
                       batch_stride_C,                                             \
                       order,                                                      \
                       descr->base);

#define LAUNCH_CSRMMNT_GENERAL_REMAINDER_KERNEL(CSRMMNT_DIM, WF_SIZE)              \
    hipLaunchKernelGGL((csrmmnt_general_remainder_kernel<CSRMMNT_DIM, WF_SIZE>),   \
                       dim3((m - 1) / (CSRMMNT_DIM / WF_SIZE) + 1, batch_count_C), \
                       dim3(CSRMMNT_DIM),                                          \
                       0,                                                          \
                       handle->stream,                                             \
                       conj_A,                                                     \
                       conj_B,                                                     \
                       main,                                                       \
                       n,                                                          \
                       m,                                                          \
                       n,                                                          \
                       k,                                                          \
                       nnz,                                                        \
                       offsets_batch_stride_A,                                     \
                       columns_values_batch_stride_A,                              \
                       alpha_device_host,                                          \
                       csr_row_ptr,                                                \
                       csr_col_ind,                                                \
                       csr_val,                                                    \
                       B,                                                          \
                       ldb,                                                        \
                       batch_stride_B,                                             \
                       beta_device_host,                                           \
                       C,                                                          \
                       ldc,                                                        \
                       batch_stride_C,                                             \
                       order,                                                      \
                       descr->base);

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmmnn_template_general(rocsparse_handle handle,
                                                    rocsparse_order  order,
                                                    bool             conj_A,
                                                    bool             conj_B,
                                                    J                m,
                                                    J                n,
                                                    J                k,
                                                    I                nnz,
                                                    J                batch_count_A,
                                                    I                offsets_batch_stride_A,
                                                    I                columns_values_batch_stride_A,
                                                    U                alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  csr_val,
                                                    const I*                  csr_row_ptr,
                                                    const J*                  csr_col_ind,
                                                    const T*                  B,
                                                    J                         ldb,
                                                    J                         batch_count_B,
                                                    I                         batch_stride_B,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    J                         ldc,
                                                    J                         batch_count_C,
                                                    I                         batch_stride_C)
{
#define CSRMMNN_DIM 256
#define WF_SIZE 8
    hipLaunchKernelGGL(
        (csrmmnn_general_kernel<CSRMMNN_DIM, WF_SIZE>),
        dim3((m - 1) / (CSRMMNN_DIM / WF_SIZE) + 1, (n - 1) / WF_SIZE + 1, batch_count_C),
        dim3(CSRMMNN_DIM),
        0,
        handle->stream,
        conj_A,
        conj_B,
        m,
        n,
        k,
        nnz,
        offsets_batch_stride_A,
        columns_values_batch_stride_A,
        alpha_device_host,
        csr_row_ptr,
        csr_col_ind,
        csr_val,
        B,
        ldb,
        batch_stride_B,
        beta_device_host,
        C,
        ldc,
        batch_stride_C,
        order,
        descr->base);
#undef CSRMMNN_DIM
#undef WF_SIZE

    return rocsparse_status_success;
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmmnt_template_general(rocsparse_handle handle,
                                                    rocsparse_order  order,
                                                    bool             conj_A,
                                                    bool             conj_B,
                                                    J                m,
                                                    J                n,
                                                    J                k,
                                                    I                nnz,
                                                    J                batch_count_A,
                                                    I                offsets_batch_stride_A,
                                                    I                columns_values_batch_stride_A,
                                                    U                alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  csr_val,
                                                    const I*                  csr_row_ptr,
                                                    const J*                  csr_col_ind,
                                                    const T*                  B,
                                                    J                         ldb,
                                                    J                         batch_count_B,
                                                    I                         batch_stride_B,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    J                         ldc,
                                                    J                         batch_count_C,
                                                    I                         batch_stride_C)
{
    // Average nnz per row of A
    I avg_row_nnz = (nnz - 1) / m + 1;

    J main      = 0;
    J remainder = 0;

    // Launch appropriate kernel depending on row nnz of A
    if(avg_row_nnz < 16)
    {
        if(n >= 128)
        {
            remainder = n % (8 * 16);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(128, 8, 16);
        }
        else if(n >= 64)
        {
            remainder = n % (8 * 8);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 8, 8);
        }
        else if(n >= 32)
        {
            remainder = n % (8 * 4);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 8, 4);
        }
        else if(n >= 16)
        {
            remainder = n % (8 * 2);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 8, 2);
        }
        else if(n >= 8)
        {
            remainder = n % (8 * 1);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 8, 1);
        }
        else
        {
            remainder = n;
        }
    }
    else if(avg_row_nnz < 32)
    {
        if(n >= 256)
        {
            remainder = n % (16 * 16);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(128, 16, 16);
        }
        else if(n >= 128)
        {
            remainder = n % (16 * 8);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 16, 8);
        }
        else if(n >= 64)
        {
            remainder = n % (16 * 4);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 16, 4);
        }
        else if(n >= 32)
        {
            remainder = n % (16 * 2);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 16, 2);
        }
        else if(n >= 16)
        {
            remainder = n % (16 * 1);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 16, 1);
        }
        else
        {
            remainder = n;
        }
    }
    else if(avg_row_nnz < 64 || handle->wavefront_size == 32)
    {
        if(n >= 512)
        {
            remainder = n % (32 * 16);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(128, 32, 16);
        }
        else if(n >= 256)
        {
            remainder = n % (32 * 8);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 32, 8);
        }
        else if(n >= 128)
        {
            remainder = n % (32 * 4);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 32, 4);
        }
        else if(n >= 64)
        {
            remainder = n % (32 * 2);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 32, 2);
        }
        else if(n >= 32)
        {
            remainder = n % (32 * 1);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 32, 1);
        }
        else
        {
            remainder = n;
        }
    }
    else if(handle->wavefront_size == 64)
    {
        if(n >= 512)
        {
            remainder = n % (64 * 8);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 64, 8);
        }
        else if(n >= 256)
        {
            remainder = n % (64 * 4);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 64, 4);
        }
        else if(n >= 128)
        {
            remainder = n % (64 * 2);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 64, 2);
        }
        else if(n >= 64)
        {
            remainder = n % (64 * 1);
            main      = n - remainder;

            LAUNCH_CSRMMNT_GENERAL_MAIN_KERNEL(256, 64, 1);
        }
        else
        {
            remainder = n;
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
            LAUNCH_CSRMMNT_GENERAL_REMAINDER_KERNEL(256, 8);
        }
        else if(remainder <= 16)
        {
            LAUNCH_CSRMMNT_GENERAL_REMAINDER_KERNEL(256, 16);
        }
        else if(remainder <= 32 || handle->wavefront_size == 32)
        {
            LAUNCH_CSRMMNT_GENERAL_REMAINDER_KERNEL(256, 32);
        }
        else if(remainder <= 64 || handle->wavefront_size == 64)
        {
            LAUNCH_CSRMMNT_GENERAL_REMAINDER_KERNEL(256, 64);
        }
        else
        {
            return rocsparse_status_arch_mismatch;
        }
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmmtn_template_general(rocsparse_handle handle,
                                                    rocsparse_order  order,
                                                    bool             conj_A,
                                                    bool             conj_B,
                                                    J                m,
                                                    J                n,
                                                    J                k,
                                                    I                nnz,
                                                    J                batch_count_A,
                                                    I                offsets_batch_stride_A,
                                                    I                columns_values_batch_stride_A,
                                                    U                alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  csr_val,
                                                    const I*                  csr_row_ptr,
                                                    const J*                  csr_col_ind,
                                                    const T*                  B,
                                                    J                         ldb,
                                                    J                         batch_count_B,
                                                    I                         batch_stride_B,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    J                         ldc,
                                                    J                         batch_count_C,
                                                    I                         batch_stride_C)
{
#define CSRMMTN_DIM 256
#define WF_SIZE 4

    // Scale C with beta
    hipLaunchKernelGGL((csrmm_scale<CSRMMTN_DIM, WF_SIZE>),
                       dim3((k - 1) / CSRMMTN_DIM + 1, (n - 1) / WF_SIZE + 1, batch_count_C),
                       dim3(CSRMMTN_DIM, WF_SIZE),
                       0,
                       handle->stream,
                       k,
                       n,
                       beta_device_host,
                       C,
                       ldc,
                       (J)batch_stride_C,
                       order);

    hipLaunchKernelGGL(
        (csrmmtn_general_kernel<CSRMMTN_DIM, WF_SIZE>),
        dim3((m - 1) / (CSRMMTN_DIM / WF_SIZE) + 1, (n - 1) / WF_SIZE + 1, batch_count_C),
        dim3(CSRMMTN_DIM),
        0,
        handle->stream,
        conj_A,
        conj_B,
        m,
        n,
        k,
        nnz,
        offsets_batch_stride_A,
        columns_values_batch_stride_A,
        alpha_device_host,
        csr_row_ptr,
        csr_col_ind,
        csr_val,
        B,
        ldb,
        batch_stride_B,
        beta_device_host,
        C,
        ldc,
        batch_stride_C,
        order,
        descr->base);

#undef CSRMMTN_DIM
#undef WF_SIZE

    return rocsparse_status_success;
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmmtt_template_general(rocsparse_handle handle,
                                                    rocsparse_order  order,
                                                    bool             conj_A,
                                                    bool             conj_B,
                                                    J                m,
                                                    J                n,
                                                    J                k,
                                                    I                nnz,
                                                    J                batch_count_A,
                                                    I                offsets_batch_stride_A,
                                                    I                columns_values_batch_stride_A,
                                                    U                alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  csr_val,
                                                    const I*                  csr_row_ptr,
                                                    const J*                  csr_col_ind,
                                                    const T*                  B,
                                                    J                         ldb,
                                                    J                         batch_count_B,
                                                    I                         batch_stride_B,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    J                         ldc,
                                                    J                         batch_count_C,
                                                    I                         batch_stride_C)
{
#define CSRMMTT_DIM 256
#define WF_SIZE 4
    // Scale C with beta
    hipLaunchKernelGGL((csrmm_scale<CSRMMTT_DIM, WF_SIZE>),
                       dim3((k - 1) / CSRMMTT_DIM + 1, (n - 1) / WF_SIZE + 1, batch_count_C),
                       dim3(CSRMMTT_DIM, WF_SIZE),
                       0,
                       handle->stream,
                       k,
                       n,
                       beta_device_host,
                       C,
                       ldc,
                       (J)batch_stride_C,
                       order);

    hipLaunchKernelGGL(
        (csrmmtt_general_kernel<CSRMMTT_DIM, WF_SIZE>),
        dim3((m - 1) / (CSRMMTT_DIM / WF_SIZE) + 1, (n - 1) / WF_SIZE + 1, batch_count_C),
        dim3(CSRMMTT_DIM),
        0,
        handle->stream,
        conj_A,
        conj_B,
        m,
        n,
        k,
        nnz,
        offsets_batch_stride_A,
        columns_values_batch_stride_A,
        alpha_device_host,
        csr_row_ptr,
        csr_col_ind,
        csr_val,
        B,
        ldb,
        batch_stride_B,
        beta_device_host,
        C,
        ldc,
        batch_stride_C,
        order,
        descr->base);

#undef CSRMMTT_DIM
#undef WF_SIZE

    return rocsparse_status_success;
}

#define ROCSPARSE_CSRMM_TEMPLATE_GENERAL_IMPL(NAME) \
    NAME(handle,                                    \
         order,                                     \
         conj_A,                                    \
         conj_B,                                    \
         m,                                         \
         n,                                         \
         k,                                         \
         nnz,                                       \
         batch_count_A,                             \
         offsets_batch_stride_A,                    \
         columns_values_batch_stride_A,             \
         alpha_device_host,                         \
         descr,                                     \
         csr_val,                                   \
         csr_row_ptr,                               \
         csr_col_ind,                               \
         B,                                         \
         ldb,                                       \
         batch_count_B,                             \
         batch_stride_B,                            \
         beta_device_host,                          \
         C,                                         \
         ldc,                                       \
         batch_count_C,                             \
         batch_stride_C);

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmm_template_general(rocsparse_handle    handle,
                                                  rocsparse_operation trans_A,
                                                  rocsparse_operation trans_B,
                                                  rocsparse_order     order,
                                                  J                   m,
                                                  J                   n,
                                                  J                   k,
                                                  I                   nnz,
                                                  J                   batch_count_A,
                                                  I                   offsets_batch_stride_A,
                                                  I                   columns_values_batch_stride_A,
                                                  U                   alpha_device_host,
                                                  const rocsparse_mat_descr descr,
                                                  const T*                  csr_val,
                                                  const I*                  csr_row_ptr,
                                                  const J*                  csr_col_ind,
                                                  const T*                  B,
                                                  J                         ldb,
                                                  J                         batch_count_B,
                                                  I                         batch_stride_B,
                                                  U                         beta_device_host,
                                                  T*                        C,
                                                  J                         ldc,
                                                  J                         batch_count_C,
                                                  I                         batch_stride_C,
                                                  bool                      force_conj_A)
{
    bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose || force_conj_A);
    bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

    // Run different csrmm kernels
    if(trans_A == rocsparse_operation_none)
    {
        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_conjugate_transpose))
        {
            return ROCSPARSE_CSRMM_TEMPLATE_GENERAL_IMPL(rocsparse_csrmmnn_template_general);
        }
        else if((order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_column
                    && trans_B == rocsparse_operation_conjugate_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            return ROCSPARSE_CSRMM_TEMPLATE_GENERAL_IMPL(rocsparse_csrmmnt_template_general);
        }
    }
    else
    {
        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_conjugate_transpose))
        {
            return ROCSPARSE_CSRMM_TEMPLATE_GENERAL_IMPL(rocsparse_csrmmtn_template_general);
        }
        else if((order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_column
                    && trans_B == rocsparse_operation_conjugate_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            return ROCSPARSE_CSRMM_TEMPLATE_GENERAL_IMPL(rocsparse_csrmmtt_template_general);
        }
    }
    return rocsparse_status_not_implemented;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE, UTYPE)                  \
    template rocsparse_status rocsparse_csrmm_template_general(  \
        rocsparse_handle          handle,                        \
        rocsparse_operation       trans_A,                       \
        rocsparse_operation       trans_B,                       \
        rocsparse_order           order,                         \
        JTYPE                     m,                             \
        JTYPE                     n,                             \
        JTYPE                     k,                             \
        ITYPE                     nnz,                           \
        JTYPE                     batch_count_A,                 \
        ITYPE                     offsets_batch_stride_A,        \
        ITYPE                     columns_values_batch_stride_A, \
        UTYPE                     alpha_device_host,             \
        const rocsparse_mat_descr descr,                         \
        const TTYPE*              csr_val,                       \
        const ITYPE*              csr_row_ptr,                   \
        const JTYPE*              csr_col_ind,                   \
        const TTYPE*              B,                             \
        JTYPE                     ldb,                           \
        JTYPE                     batch_count_B,                 \
        ITYPE                     batch_stride_B,                \
        UTYPE                     beta_device_host,              \
        TTYPE*                    C,                             \
        JTYPE                     ldc,                           \
        JTYPE                     batch_count_C,                 \
        ITYPE                     batch_stride_C,                \
        bool                      force_conj_A)

INSTANTIATE(int32_t, int32_t, float, float);
INSTANTIATE(int32_t, int32_t, double, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float, float);
INSTANTIATE(int64_t, int32_t, double, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float, float);
INSTANTIATE(int64_t, int64_t, double, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int32_t, int32_t, float, const float*);
INSTANTIATE(int32_t, int32_t, double, const double*);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex, const rocsparse_double_complex*);
INSTANTIATE(int64_t, int32_t, float, const float*);
INSTANTIATE(int64_t, int32_t, double, const double*);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex, const rocsparse_double_complex*);
INSTANTIATE(int64_t, int64_t, float, const float*);
INSTANTIATE(int64_t, int64_t, double, const double*);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex, const rocsparse_double_complex*);
#undef INSTANTIATE
