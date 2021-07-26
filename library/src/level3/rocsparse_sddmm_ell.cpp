/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

template <rocsparse_int BLOCKSIZE,
          rocsparse_int WIN,
          typename I,
          typename J,
          typename T,
          typename U>
ROCSPARSE_KERNEL __launch_bounds__(BLOCKSIZE, 1) void sddmm_ell_kernel(rocsparse_operation transA,
                                                                       rocsparse_operation transB,
                                                                       rocsparse_order     orderA,
                                                                       rocsparse_order     orderB,
                                                                       J                   M,
                                                                       J                   N,
                                                                       J                   K,
                                                                       I                   nnz,
                                                                       U alpha_device_host,
                                                                       const T* __restrict__ A,
                                                                       J lda,
                                                                       const T* __restrict__ B,
                                                                       J ldb,
                                                                       U beta_device_host,
                                                                       T* __restrict__ val,

                                                                       const J* __restrict__ ind,
                                                                       rocsparse_index_base base,
                                                                       T* __restrict__ workspace)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    I at = hipBlockIdx_y;
    I i  = at % M;
    I j  = ind[at] - base;

    const T* x = (orderA == rocsparse_order_column)
                     ? ((transA == rocsparse_operation_none) ? (A + i) : (A + lda * i))
                     : ((transA == rocsparse_operation_none) ? (A + lda * i) : (A + i));
    J incx = (orderA == rocsparse_order_column) ? ((transA == rocsparse_operation_none) ? lda : 1)
                                                : ((transA == rocsparse_operation_none) ? 1 : lda);

    const T* y = (orderB == rocsparse_order_column)
                     ? ((transB == rocsparse_operation_none) ? (B + ldb * j) : (B + j))
                     : ((transB == rocsparse_operation_none) ? (B + j) : (B + ldb * j));
    J incy = (orderB == rocsparse_order_column) ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                                                : ((transB == rocsparse_operation_none) ? ldb : 1);

    size_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    T      sum = static_cast<T>(0);

    // sum WIN elements per thread
    size_t inc = hipBlockDim_x * hipGridDim_x;
    if(j >= 0 && j < N)
    {
        for(J l = 0; l < WIN && gid < K; l++, gid += inc)
        {
            sum += y[gid * incy] * x[gid * incx];
        }
    }

    sum = rocsparse_reduce_block<BLOCKSIZE>(sum);
    if(hipThreadIdx_x == 0)
    {
        workspace[hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x] = sum;
        if(hipGridDim_x == 1) // small N avoid second kernel
        {
            if(j >= 0 && j < N)
            {
                val[at] = val[at] * beta + alpha * sum;
            }
        }
    }
}

template <rocsparse_int NB, rocsparse_int WIN, typename T, typename U>
ROCSPARSE_KERNEL __launch_bounds__(NB) void ell_finalize_kernel(rocsparse_int n_sums,
                                                                U             alpha_device_host,
                                                                U             beta_device_host,
                                                                T* __restrict__ in,
                                                                T* __restrict__ out)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    T sum = static_cast<T>(0);

    int offset = hipBlockIdx_y * n_sums;
    in += offset;

    int inc = hipBlockDim_x * hipGridDim_x * WIN;

    int i         = hipThreadIdx_x * WIN;
    int remainder = n_sums % WIN;
    int end       = n_sums - remainder;
    for(; i < end; i += inc) // cover all sums as 1 block
    {
        for(int j = 0; j < WIN; j++)
            sum += in[i + j];
    }
    if(hipThreadIdx_x < remainder)
    {
        sum += in[n_sums - 1 - hipThreadIdx_x];
    }

    sum = rocsparse_reduce_block<NB>(sum);
    if(hipThreadIdx_x == 0)
        out[hipBlockIdx_y] = out[hipBlockIdx_y] * beta + alpha * sum;
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
                                        J                    A_ld,
                                        const T*             B_val,
                                        J                    B_ld,
                                        const T*             beta,
                                        const I*             C_row_data,
                                        const J*             C_col_data,
                                        T*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_sddmm_alg  alg,
                                        size_t*              buffer_size)
    {
        //
        // TODO: change this weird assumption.
        //
        I width                 = nnz;
        nnz                     = width * m;
        static constexpr int NB = 512;
        buffer_size[0]          = nnz * ((k - 1) / NB + 1) * sizeof(T);
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
                                       J                    A_ld,
                                       const T*             B_val,
                                       J                    B_ld,
                                       const T*             beta,
                                       const I*             C_row_data,
                                       const J*             C_col_data,
                                       T*                   C_val_data,
                                       rocsparse_index_base C_base,
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
                                    J                    A_ld,
                                    const T*             B_val,
                                    J                    B_ld,
                                    const T*             beta,
                                    const I*             C_row_data,
                                    const J*             C_col_data,
                                    T*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer)
    {
        //
        // TODO: change this weird assumption.
        //
        I width = nnz;
        nnz     = width * m;

        static constexpr int WIN          = rocsparse_reduce_WIN<T>();
        int64_t              num_blocks_y = nnz;
        static constexpr int NB           = 512;
        int64_t              num_blocks_x = (k - 1) / NB + 1;
        dim3                 blocks(num_blocks_x, num_blocks_y);
        dim3                 threads(NB);

        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            hipLaunchKernelGGL((sddmm_ell_kernel<NB, WIN, I, J, T>),
                               blocks,
                               threads,
                               0,
                               handle->stream,
                               trans_A,
                               trans_B,
                               order_A,
                               order_B,
                               m,
                               n,
                               k,
                               nnz,
                               *(const T*)alpha,
                               A_val,
                               A_ld,
                               B_val,
                               B_ld,
                               *(const T*)beta,
                               (T*)C_val_data,
                               (const J*)C_col_data,
                               C_base,
                               (T*)buffer);

            if(num_blocks_x > 1) // if single block first kernel did all work
            {
                hipLaunchKernelGGL((ell_finalize_kernel<NB, WIN, T>),
                                   dim3(1, nnz),
                                   threads,
                                   0,
                                   handle->stream,
                                   num_blocks_x,
                                   *(const T*)alpha,
                                   *(const T*)beta,
                                   (T*)buffer,
                                   (T*)C_val_data);
            }
        }
        else
        {

            hipLaunchKernelGGL((sddmm_ell_kernel<NB, WIN, I, J, T>),
                               blocks,
                               threads,
                               0,
                               handle->stream,
                               trans_A,
                               trans_B,
                               order_A,
                               order_B,
                               m,
                               n,
                               k,
                               nnz,
                               alpha,
                               A_val,
                               A_ld,
                               B_val,
                               B_ld,
                               beta,
                               (T*)C_val_data,
                               (const J*)C_col_data,
                               C_base,
                               (T*)buffer);

            if(num_blocks_x > 1) // if single block first kernel did all work
            {
                hipLaunchKernelGGL((ell_finalize_kernel<NB, WIN, T>),
                                   dim3(1, nnz),
                                   threads,
                                   0,
                                   handle->stream,
                                   num_blocks_x,
                                   alpha,
                                   beta,
                                   (T*)buffer,
                                   (T*)C_val_data);
            }
        }
        return rocsparse_status_success;
    }
};

template struct rocsparse_sddmm_st<rocsparse_format_ell,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_ell,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_ell,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_ell,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   rocsparse_double_complex>;

template struct rocsparse_sddmm_st<rocsparse_format_ell,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_ell,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_ell,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_ell,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   rocsparse_double_complex>;
