/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "rocsparse_reduce.hpp"
#include "rocsparse_sddmm.hpp"
#include "utility.h"

#include "../conversion/rocsparse_ell2dense.hpp"

template <rocsparse_int BLOCKSIZE,
          rocsparse_int NTHREADS_PER_DOTPRODUCT,
          typename I,
          typename J,
          typename T,
          typename U>
ROCSPARSE_KERNEL_W(BLOCKSIZE, 1)
void sddmm_ell_kernel(rocsparse_operation transA,
                      rocsparse_operation transB,
                      rocsparse_order     orderA,
                      rocsparse_order     orderB,
                      J                   M,
                      J                   N,
                      J                   K,
                      I                   nnz,
                      U                   alpha_device_host,
                      const T* __restrict__ A,
                      int64_t lda,
                      const T* __restrict__ B,
                      int64_t ldb,
                      U       beta_device_host,
                      T* __restrict__ val,
                      const J* __restrict__ ind,
                      rocsparse_index_base base,
                      T* __restrict__ workspace)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }
    //
    // Each group treats one row.
    //
    static constexpr rocsparse_int NUM_COEFF          = (BLOCKSIZE / NTHREADS_PER_DOTPRODUCT);
    const I                        local_coeff_index  = hipThreadIdx_x / NTHREADS_PER_DOTPRODUCT;
    const I                        local_thread_index = hipThreadIdx_x % NTHREADS_PER_DOTPRODUCT;
    const int64_t                  incx               = (orderA == rocsparse_order_column)
                                                            ? ((transA == rocsparse_operation_none) ? lda : 1)
                                                            : ((transA == rocsparse_operation_none) ? 1 : lda);

    const int64_t incy = (orderB == rocsparse_order_column)
                             ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                             : ((transB == rocsparse_operation_none) ? ldb : 1);

    __shared__ T s[NUM_COEFF][NTHREADS_PER_DOTPRODUCT];

    const I innz = hipBlockIdx_x * NUM_COEFF + local_coeff_index;
    if(innz >= nnz)
    {
        return;
    }

    const J i = innz % M;
    const J j = ind[innz] - base;
    if(j < 0)
    {
        return;
    }
    const T* x = (orderA == rocsparse_order_column)
                     ? ((transA == rocsparse_operation_none) ? (A + i) : (A + lda * i))
                     : ((transA == rocsparse_operation_none) ? (A + lda * i) : (A + i));

    const T* y = (orderB == rocsparse_order_column)
                     ? ((transB == rocsparse_operation_none) ? (B + ldb * j) : (B + j))
                     : ((transB == rocsparse_operation_none) ? (B + j) : (B + ldb * j));

    T sum = static_cast<T>(0);
    for(J k = local_thread_index; k < K; k += NTHREADS_PER_DOTPRODUCT)
    {
        sum += x[k * incx] * y[k * incy];
    }
    s[local_coeff_index][local_thread_index] = sum;
    __syncthreads();

#pragma unroll
    for(int ipow2_ = 2; ipow2_ <= NTHREADS_PER_DOTPRODUCT; ipow2_ *= 2)
    {
        if(local_thread_index < NTHREADS_PER_DOTPRODUCT / ipow2_)
        {
            s[local_coeff_index][local_thread_index]
                += s[local_coeff_index][local_thread_index + NTHREADS_PER_DOTPRODUCT / ipow2_];
        }
        __syncthreads();
    }

    if(local_thread_index == 0)
    {
        val[innz] = val[innz] * beta + alpha * s[local_coeff_index][0];
    }
}

template <rocsparse_int NUM_ELL_COLUMNS_PER_BLOCK, rocsparse_int WF_SIZE, typename I, typename T>
ROCSPARSE_KERNEL(WF_SIZE* NUM_ELL_COLUMNS_PER_BLOCK)
void sddmm_ell_sample_kernel(I m,
                             I n,
                             const T* __restrict__ dense_val,
                             int64_t ld,
                             I       ell_width,
                             T* __restrict__ ell_val,
                             const I* __restrict__ ell_col_ind,
                             rocsparse_index_base ell_base)
{
    const auto wavefront_index  = hipThreadIdx_x / WF_SIZE;
    const auto lane_index       = hipThreadIdx_x % WF_SIZE;
    const auto ell_column_index = NUM_ELL_COLUMNS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

    if(ell_column_index < ell_width)
    {
        //
        // One wavefront executes one ell column.
        //
        for(I row_index = lane_index; row_index < m; row_index += WF_SIZE)
        {
            const auto ell_idx      = ELL_IND(row_index, ell_column_index, m, ell_width);
            const auto column_index = ell_col_ind[ell_idx] - ell_base;

            if(column_index >= 0 && column_index < n)
            {
                ell_val[ell_idx] = dense_val[column_index * ld + row_index];
            }
        }
    }
}

template <typename I, typename J, typename T>
struct rocsparse_sddmm_st<rocsparse_format_ell, rocsparse_sddmm_alg_default, I, J, T>
{

    static rocsparse_status buffer_size(rocsparse_handle     handle,
                                        rocsparse_operation  trans_A,
                                        rocsparse_operation  trans_B,
                                        rocsparse_order      order_A,
                                        rocsparse_order      order_B,
                                        J                    m,
                                        J                    n,
                                        J                    k,
                                        I                    nnz,
                                        const T*             alpha,
                                        const T*             A_val,
                                        int64_t              A_ld,
                                        const T*             B_val,
                                        int64_t              B_ld,
                                        const T*             beta,
                                        const I*             C_row_data,
                                        const J*             C_col_data,
                                        T*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_mat_descr  C_descr,
                                        rocsparse_sddmm_alg  alg,
                                        size_t*              buffer_size)
    {
        buffer_size[0] = 0;
        return rocsparse_status_success;
    }

    static rocsparse_status preprocess(rocsparse_handle     handle,
                                       rocsparse_operation  trans_A,
                                       rocsparse_operation  trans_B,
                                       rocsparse_order      order_A,
                                       rocsparse_order      order_B,
                                       J                    m,
                                       J                    n,
                                       J                    k,
                                       I                    nnz,
                                       const T*             alpha,
                                       const T*             A_val,
                                       int64_t              A_ld,
                                       const T*             B_val,
                                       int64_t              B_ld,
                                       const T*             beta,
                                       const I*             C_row_data,
                                       const J*             C_col_data,
                                       T*                   C_val_data,
                                       rocsparse_index_base C_base,
                                       rocsparse_mat_descr  C_descr,
                                       rocsparse_sddmm_alg  alg,
                                       void*                buffer)
    {
        return rocsparse_status_success;
    }

    static rocsparse_status compute(rocsparse_handle     handle,
                                    rocsparse_operation  trans_A,
                                    rocsparse_operation  trans_B,
                                    rocsparse_order      order_A,
                                    rocsparse_order      order_B,
                                    J                    m,
                                    J                    n,
                                    J                    k,
                                    I                    nnz,
                                    const T*             alpha,
                                    const T*             A_val,
                                    int64_t              A_ld,
                                    const T*             B_val,
                                    int64_t              B_ld,
                                    const T*             beta,
                                    const I*             C_row_data,
                                    const J*             C_col_data,
                                    T*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_mat_descr  C_descr,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer)
    {

        static constexpr int NB = 512;
#define HLAUNCH(K_)                                                         \
    int64_t num_blocks_x = (nnz - 1) / (NB / K_) + 1;                       \
    dim3    blocks(num_blocks_x);                                           \
    dim3    threads(NB);                                                    \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((sddmm_ell_kernel<NB, K_, I, J, T>), \
                                       blocks,                              \
                                       threads,                             \
                                       0,                                   \
                                       handle->stream,                      \
                                       trans_A,                             \
                                       trans_B,                             \
                                       order_A,                             \
                                       order_B,                             \
                                       m,                                   \
                                       n,                                   \
                                       k,                                   \
                                       nnz,                                 \
                                       *(const T*)alpha,                    \
                                       A_val,                               \
                                       A_ld,                                \
                                       B_val,                               \
                                       B_ld,                                \
                                       *(const T*)beta,                     \
                                       (T*)C_val_data,                      \
                                       (const J*)C_col_data,                \
                                       C_base,                              \
                                       (T*)buffer)

#define DLAUNCH(K_)                                                         \
    int64_t num_blocks_x = (nnz - 1) / (NB / K_) + 1;                       \
    dim3    blocks(num_blocks_x);                                           \
    dim3    threads(NB);                                                    \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((sddmm_ell_kernel<NB, K_, I, J, T>), \
                                       blocks,                              \
                                       threads,                             \
                                       0,                                   \
                                       handle->stream,                      \
                                       trans_A,                             \
                                       trans_B,                             \
                                       order_A,                             \
                                       order_B,                             \
                                       m,                                   \
                                       n,                                   \
                                       k,                                   \
                                       nnz,                                 \
                                       alpha,                               \
                                       A_val,                               \
                                       A_ld,                                \
                                       B_val,                               \
                                       B_ld,                                \
                                       beta,                                \
                                       (T*)C_val_data,                      \
                                       (const J*)C_col_data,                \
                                       C_base,                              \
                                       (T*)buffer)

        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }
            if(k > 4)
            {
                HLAUNCH(8);
            }
            else if(k > 2)
            {
                HLAUNCH(4);
            }
            else if(k > 1)
            {
                HLAUNCH(2);
            }
            else
            {
                HLAUNCH(1);
            }
        }
        else
        {
            if(k > 4)
            {
                DLAUNCH(8);
            }
            else if(k > 2)
            {
                DLAUNCH(4);
            }
            else if(k > 1)
            {
                DLAUNCH(2);
            }
            else
            {
                DLAUNCH(1);
            }
        }
        return rocsparse_status_success;
    }
};

template <typename I, typename J, typename T>
struct rocsparse_sddmm_st<rocsparse_format_ell, rocsparse_sddmm_alg_dense, I, J, T>
{
    static rocsparse_status buffer_size(rocsparse_handle     handle,
                                        rocsparse_operation  trans_A,
                                        rocsparse_operation  trans_B,
                                        rocsparse_order      order_A,
                                        rocsparse_order      order_B,
                                        J                    m,
                                        J                    n,
                                        J                    k,
                                        I                    nnz,
                                        const T*             alpha,
                                        const T*             A_val,
                                        int64_t              A_ld,
                                        const T*             B_val,
                                        int64_t              B_ld,
                                        const T*             beta,
                                        const I*             C_row_data,
                                        const J*             C_col_data,
                                        T*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_mat_descr  C_descr,
                                        rocsparse_sddmm_alg  alg,
                                        size_t*              buffer_size)
    {
        if(nnz == 0)
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }

        *buffer_size = ((sizeof(T) * m * n - 1) / 256 + 1) * 256;
        return rocsparse_status_success;
    }

    static rocsparse_status preprocess(rocsparse_handle     handle,
                                       rocsparse_operation  trans_A,
                                       rocsparse_operation  trans_B,
                                       rocsparse_order      order_A,
                                       rocsparse_order      order_B,
                                       J                    m,
                                       J                    n,
                                       J                    k,
                                       I                    nnz,
                                       const T*             alpha,
                                       const T*             A_val,
                                       int64_t              A_ld,
                                       const T*             B_val,
                                       int64_t              B_ld,
                                       const T*             beta,
                                       const I*             C_row_data,
                                       const J*             C_col_data,
                                       T*                   C_val_data,
                                       rocsparse_index_base C_base,
                                       rocsparse_mat_descr  C_descr,
                                       rocsparse_sddmm_alg  alg,
                                       void*                buffer)
    {
        return rocsparse_status_success;
    }

    static rocsparse_status compute(rocsparse_handle     handle,
                                    rocsparse_operation  trans_A,
                                    rocsparse_operation  trans_B,
                                    rocsparse_order      order_A,
                                    rocsparse_order      order_B,
                                    J                    m,
                                    J                    n,
                                    J                    k,
                                    I                    nnz,
                                    const T*             alpha,
                                    const T*             A_val,
                                    int64_t              A_ld,
                                    const T*             B_val,
                                    int64_t              B_ld,
                                    const T*             beta,
                                    const I*             C_row_data,
                                    const J*             C_col_data,
                                    T*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_mat_descr  C_descr,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer)
    {
        if(nnz == 0)
        {
            return rocsparse_status_success;
        }

        if(buffer == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }

        char* ptr   = reinterpret_cast<char*>(buffer);
        T*    dense = reinterpret_cast<T*>(ptr);

        const auto ell_width = static_cast<J>(nnz / m);

        // Convert to Dense
        RETURN_IF_ROCSPARSE_ERROR((rocsparse_ell2dense_template(handle,
                                                                m,
                                                                n,
                                                                C_descr,
                                                                ell_width,
                                                                C_val_data,
                                                                C_col_data,
                                                                dense,
                                                                m,
                                                                rocsparse_order_column)));

        const bool A_col_major = (order_A == rocsparse_order_column);
        const bool B_col_major = (order_B == rocsparse_order_column);

        const rocsparse_operation trans_A_adjusted
            = (A_col_major != (trans_A == rocsparse_operation_none)) ? rocsparse_operation_transpose
                                                                     : rocsparse_operation_none;
        const rocsparse_operation trans_B_adjusted
            = (B_col_major != (trans_B == rocsparse_operation_none)) ? rocsparse_operation_transpose
                                                                     : rocsparse_operation_none;

        // Compute
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_blas_gemm_ex(handle->blas_handle,
                                                         trans_A_adjusted,
                                                         trans_B_adjusted,
                                                         m,
                                                         n,
                                                         k,
                                                         alpha,
                                                         A_val,
                                                         get_datatype<T>(),
                                                         A_ld,
                                                         B_val,
                                                         get_datatype<T>(),
                                                         B_ld,
                                                         beta,
                                                         dense,
                                                         get_datatype<T>(),
                                                         m,
                                                         dense,
                                                         get_datatype<T>(),
                                                         m,
                                                         get_datatype<T>(),
                                                         rocsparse_blas_gemm_alg_standard,
                                                         0,
                                                         0));

        // Sample dense C
        if(handle->wavefront_size == 32)
        {
            static constexpr rocsparse_int WAVEFRONT_SIZE         = 32;
            static constexpr rocsparse_int NELL_COLUMNS_PER_BLOCK = 16;

            rocsparse_int blocks = (ell_width - 1) / NELL_COLUMNS_PER_BLOCK + 1;
            dim3          k_blocks(blocks), k_threads(WAVEFRONT_SIZE * NELL_COLUMNS_PER_BLOCK);

            hipLaunchKernelGGL(
                (sddmm_ell_sample_kernel<NELL_COLUMNS_PER_BLOCK, WAVEFRONT_SIZE, I, T>),
                k_blocks,
                k_threads,
                0,
                handle->stream,
                m,
                n,
                dense,
                m,
                ell_width,
                C_val_data,
                C_col_data,
                C_base);
        }
        else
        {
            static constexpr rocsparse_int WAVEFRONT_SIZE         = 64;
            static constexpr rocsparse_int NELL_COLUMNS_PER_BLOCK = 16;

            rocsparse_int blocks = (ell_width - 1) / NELL_COLUMNS_PER_BLOCK + 1;
            dim3          k_blocks(blocks), k_threads(WAVEFRONT_SIZE * NELL_COLUMNS_PER_BLOCK);

            hipLaunchKernelGGL(
                (sddmm_ell_sample_kernel<NELL_COLUMNS_PER_BLOCK, WAVEFRONT_SIZE, I, T>),
                k_blocks,
                k_threads,
                0,
                handle->stream,
                m,
                n,
                dense,
                m,
                ell_width,
                C_val_data,
                C_col_data,
                C_base);
        }

        return rocsparse_status_success;
    }
};

#define INSTANTIATE(ITYPE_, JTYPE_, TTYPE_)                         \
    template struct rocsparse_sddmm_st<rocsparse_format_ell,        \
                                       rocsparse_sddmm_alg_default, \
                                       ITYPE_,                      \
                                       JTYPE_,                      \
                                       TTYPE_>;                     \
    template struct rocsparse_sddmm_st<rocsparse_format_ell,        \
                                       rocsparse_sddmm_alg_dense,   \
                                       ITYPE_,                      \
                                       JTYPE_,                      \
                                       TTYPE_>

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

#undef INSTANTIATE
