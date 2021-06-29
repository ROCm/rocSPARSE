/*! \file */
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

#include "rocsparse_coomm_template_atomic.hpp"
#include "common.h"
#include "definitions.h"
#include "utility.h"

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          bool         NT,
          typename I,
          typename T>
static __device__ void coommnn_atomic_main_device(rocsparse_operation  transB,
                                                  I                    offset,
                                                  I                    ncol,
                                                  I                    nnz,
                                                  I                    n,
                                                  T                    alpha,
                                                  const I*             coo_row_ind,
                                                  const I*             coo_col_ind,
                                                  const T*             coo_val,
                                                  const T*             B,
                                                  I                    ldb,
                                                  T*                   C,
                                                  I                    ldc,
                                                  rocsparse_order      order,
                                                  rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    I   gid = hipBlockIdx_x * BLOCKSIZE + tid;
    int lid = tid & (WF_SIZE - 1);

    I row = (gid < nnz) ? rocsparse_nontemporal_load(coo_row_ind + gid) - idx_base : 0;
    I col = (gid < nnz) ? rocsparse_nontemporal_load(coo_col_ind + gid) - idx_base : 0;
    T val = (gid < nnz) ? rocsparse_nontemporal_load(coo_val + gid) : static_cast<T>(0);

    for(I l = 0; l < ncol; l += WF_SIZE * LOOPS)
    {
        I colB = l + lid;

        T sum[LOOPS]{};

        I current_row = __shfl(row, 0, WF_SIZE);

        for(I i = 0; i < WF_SIZE; ++i)
        {
            T v = rocsparse_shfl(val, i, WF_SIZE);
            I c = __shfl(col, i, WF_SIZE);
            I r = __shfl(row, i, WF_SIZE);

            if(r != current_row)
            {
                if(order == rocsparse_order_column)
                {
                    for(I p = 0; p < LOOPS; p++)
                    {
                        atomicAdd(&C[(colB + p * WF_SIZE) * ldc + current_row], alpha * sum[p]);
                    }
                }
                else
                {
                    for(I p = 0; p < LOOPS; p++)
                    {
                        atomicAdd(&C[current_row * ldc + colB + p * WF_SIZE], alpha * sum[p]);
                    }
                }

                for(I p = 0; p < LOOPS; p++)
                {
                    sum[p] = static_cast<T>(0);
                }

                current_row = r;
            }

            if(NT)
            {
                if(transB == rocsparse_operation_conjugate_transpose)
                {
                    for(I p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse_fma(
                            v, rocsparse_conj(B[c * ldb + colB + p * WF_SIZE]), sum[p]);
                    }
                }
                else
                {
                    for(I p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse_fma(v, B[c * ldb + colB + p * WF_SIZE], sum[p]);
                    }
                }
            }
            else
            {
                if(transB == rocsparse_operation_conjugate_transpose)
                {
                    for(I p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse_fma(
                            v, rocsparse_conj(B[(colB + p * WF_SIZE) * ldb + c]), sum[p]);
                    }
                }
                else
                {
                    for(I p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse_fma(v, B[(colB + p * WF_SIZE) * ldb + c], sum[p]);
                    }
                }
            }
        }

        if(order == rocsparse_order_column)
        {
            for(I p = 0; p < LOOPS; p++)
            {
                atomicAdd(&C[(colB + p * WF_SIZE) * ldc + current_row], alpha * sum[p]);
            }
        }
        else
        {
            for(I p = 0; p < LOOPS; p++)
            {
                atomicAdd(&C[current_row * ldc + colB + p * WF_SIZE], alpha * sum[p]);
            }
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, bool NT, typename I, typename T>
static __device__ void coommnn_atomic_remainder_device(rocsparse_operation  transB,
                                                       I                    offset,
                                                       I                    nnz,
                                                       I                    n,
                                                       T                    alpha,
                                                       const I*             coo_row_ind,
                                                       const I*             coo_col_ind,
                                                       const T*             coo_val,
                                                       const T*             B,
                                                       I                    ldb,
                                                       T*                   C,
                                                       I                    ldc,
                                                       rocsparse_order      order,
                                                       rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    I   gid = hipBlockIdx_x * BLOCKSIZE + tid;
    int lid = tid & (WF_SIZE - 1);

    I row = (gid < nnz) ? rocsparse_nontemporal_load(coo_row_ind + gid) - idx_base : 0;
    I col = (gid < nnz) ? rocsparse_nontemporal_load(coo_col_ind + gid) - idx_base : 0;
    T val = (gid < nnz) ? rocsparse_nontemporal_load(coo_val + gid) : static_cast<T>(0);

    for(I l = offset; l < n; l += WF_SIZE)
    {
        I colB = l + lid;

        T sum = static_cast<T>(0);

        I current_row = __shfl(row, 0, WF_SIZE);

        for(I i = 0; i < WF_SIZE; ++i)
        {
            T v = rocsparse_shfl(val, i, WF_SIZE);
            I c = __shfl(col, i, WF_SIZE);
            I r = __shfl(row, i, WF_SIZE);

            if(r != current_row)
            {
                if(colB < n)
                {
                    if(order == rocsparse_order_column)
                    {
                        atomicAdd(&C[colB * ldc + current_row], alpha * sum);
                    }
                    else
                    {
                        atomicAdd(&C[current_row * ldc + colB], alpha * sum);
                    }
                }

                sum = static_cast<T>(0);

                current_row = r;
            }

            if(colB < n)
            {
                if(NT)
                {
                    if(transB == rocsparse_operation_conjugate_transpose)
                    {
                        sum = rocsparse_fma(v, rocsparse_conj(B[c * ldb + colB]), sum);
                    }
                    else
                    {
                        sum = rocsparse_fma(v, B[c * ldb + colB], sum);
                    }
                }
                else
                {
                    if(transB == rocsparse_operation_conjugate_transpose)
                    {
                        sum = rocsparse_fma(v, rocsparse_conj(B[colB * ldb + c]), sum);
                    }
                    else
                    {
                        sum = rocsparse_fma(v, B[colB * ldb + c], sum);
                    }
                }
            }
        }

        if(colB < n)
        {
            if(order == rocsparse_order_column)
            {
                atomicAdd(&C[colB * ldc + current_row], alpha * sum);
            }
            else
            {
                atomicAdd(&C[current_row * ldc + colB], alpha * sum);
            }
        }
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          bool         NT,
          typename I,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) __global__ void coommnn_atomic_main(rocsparse_operation trans_B,
                                                                 I                   offset,
                                                                 I                   ncol,
                                                                 I                   nnz,
                                                                 I                   n,
                                                                 U alpha_device_host,
                                                                 const I* __restrict__ coo_row_ind,
                                                                 const I* __restrict__ coo_col_ind,
                                                                 const T* __restrict__ coo_val,
                                                                 const T* __restrict__ B,
                                                                 I ldb,
                                                                 T* __restrict__ C,
                                                                 I                    ldc,
                                                                 rocsparse_order      order,
                                                                 rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    coommnn_atomic_main_device<BLOCKSIZE, WF_SIZE, LOOPS, NT>(trans_B,
                                                              offset,
                                                              ncol,
                                                              nnz,
                                                              n,
                                                              alpha,
                                                              coo_row_ind,
                                                              coo_col_ind,
                                                              coo_val,
                                                              B,
                                                              ldb,
                                                              C,
                                                              ldc,
                                                              order,
                                                              idx_base);
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, bool NT, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) __global__
    void coommnn_atomic_remainder(rocsparse_operation trans_B,
                                  I                   offset,
                                  I                   nnz,
                                  I                   n,
                                  U                   alpha_device_host,
                                  const I* __restrict__ coo_row_ind,
                                  const I* __restrict__ coo_col_ind,
                                  const T* __restrict__ coo_val,
                                  const T* __restrict__ B,
                                  I ldb,
                                  T* __restrict__ C,
                                  I                    ldc,
                                  rocsparse_order      order,
                                  rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    coommnn_atomic_remainder_device<BLOCKSIZE, WF_SIZE, NT>(trans_B,
                                                            offset,
                                                            nnz,
                                                            n,
                                                            alpha,
                                                            coo_row_ind,
                                                            coo_col_ind,
                                                            coo_val,
                                                            B,
                                                            ldb,
                                                            C,
                                                            ldc,
                                                            order,
                                                            idx_base);
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, bool NT, typename I, typename... Ts>
rocsparse_status
    coommnn_atomic_dispatch(hipStream_t stream, rocsparse_operation trans_B, I nnz, I n, Ts&&... ts)
{
#define TREAT_CONDITION(value_)                                                           \
    remainder = n % value_;                                                               \
    main      = n - remainder;                                                            \
    hipLaunchKernelGGL((coommnn_atomic_main<BLOCKSIZE, WF_SIZE, (value_ / WF_SIZE), NT>), \
                       dim3((nnz - 1) / BLOCKSIZE + 1),                                   \
                       dim3(BLOCKSIZE),                                                   \
                       0,                                                                 \
                       stream,                                                            \
                       trans_B,                                                           \
                       (I)0,                                                              \
                       main,                                                              \
                       nnz,                                                               \
                       n,                                                                 \
                       ts...);

    I main      = 0;
    I remainder = n;

    if(WF_SIZE == 32)
    {
        if(n >= 256)
        {
            TREAT_CONDITION(256);
        }
        else if(n >= 128)
        {
            TREAT_CONDITION(128);
        }
        else if(n >= 64)
        {
            TREAT_CONDITION(64);
        }
        else if(n >= 32)
        {
            TREAT_CONDITION(32);
        }
    }
    else if(WF_SIZE == 64)
    {
        if(n >= 512)
        {
            TREAT_CONDITION(512);
        }
        else if(n >= 256)
        {
            TREAT_CONDITION(256);
        }
        else if(n >= 128)
        {
            TREAT_CONDITION(128);
        }
        else if(n >= 64)
        {
            TREAT_CONDITION(64);
        }
    }

    if(remainder > 0)
    {
        hipLaunchKernelGGL((coommnn_atomic_remainder<BLOCKSIZE, WF_SIZE, NT>),
                           dim3((nnz - 1) / BLOCKSIZE + 1),
                           dim3(BLOCKSIZE),
                           0,
                           stream,
                           trans_B,
                           main,
                           nnz,
                           n,
                           ts...);
    }

    return rocsparse_status_success;
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_atomic(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 rocsparse_order           order,
                                                 I                         m,
                                                 I                         n,
                                                 I                         k,
                                                 I                         nnz,
                                                 U                         alpha_device_host,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  coo_val,
                                                 const I*                  coo_row_ind,
                                                 const I*                  coo_col_ind,
                                                 const T*                  B,
                                                 I                         ldb,
                                                 U                         beta_device_host,
                                                 T*                        C,
                                                 I                         ldc)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Run different coomm kernels
    if(trans_A == rocsparse_operation_none)
    {
        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_conjugate_transpose))
        {
            if(handle->wavefront_size == 32)
            {
                return coommnn_atomic_dispatch<256, 32, false>(stream,
                                                               trans_B,
                                                               nnz,
                                                               n,
                                                               alpha_device_host,
                                                               coo_row_ind,
                                                               coo_col_ind,
                                                               coo_val,
                                                               B,
                                                               ldb,
                                                               C,
                                                               ldc,
                                                               order,
                                                               descr->base);
            }
            else if(handle->wavefront_size == 64)
            {
                return coommnn_atomic_dispatch<256, 64, false>(stream,
                                                               trans_B,
                                                               nnz,
                                                               n,
                                                               alpha_device_host,
                                                               coo_row_ind,
                                                               coo_col_ind,
                                                               coo_val,
                                                               B,
                                                               ldb,
                                                               C,
                                                               ldc,
                                                               order,
                                                               descr->base);
            }
            else
            {
                return rocsparse_status_not_implemented;
            }
        }
        else if((order == rocsparse_order_column
                 && trans_B == rocsparse_operation_conjugate_transpose)
                || (order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            if(handle->wavefront_size == 32)
            {
                return coommnn_atomic_dispatch<256, 32, true>(stream,
                                                              trans_B,
                                                              nnz,
                                                              n,
                                                              alpha_device_host,
                                                              coo_row_ind,
                                                              coo_col_ind,
                                                              coo_val,
                                                              B,
                                                              ldb,
                                                              C,
                                                              ldc,
                                                              order,
                                                              descr->base);
            }
            else if(handle->wavefront_size == 64)
            {
                return coommnn_atomic_dispatch<256, 64, true>(stream,
                                                              trans_B,
                                                              nnz,
                                                              n,
                                                              alpha_device_host,
                                                              coo_row_ind,
                                                              coo_col_ind,
                                                              coo_val,
                                                              B,
                                                              ldb,
                                                              C,
                                                              ldc,
                                                              order,
                                                              descr->base);
            }
            else
            {
                return rocsparse_status_not_implemented;
            }
        }
    }
    else
    {
        return rocsparse_status_not_implemented;
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE, UTYPE)                                            \
    template rocsparse_status rocsparse_coomm_template_atomic<ITYPE, TTYPE, UTYPE>( \
        rocsparse_handle          handle,                                           \
        rocsparse_operation       trans_A,                                          \
        rocsparse_operation       trans_B,                                          \
        rocsparse_order           order,                                            \
        ITYPE                     m,                                                \
        ITYPE                     n,                                                \
        ITYPE                     k,                                                \
        ITYPE                     nnz,                                              \
        UTYPE                     alpha_device_host,                                \
        const rocsparse_mat_descr descr,                                            \
        const TTYPE*              coo_val,                                          \
        const ITYPE*              coo_row_ind,                                      \
        const ITYPE*              coo_col_ind,                                      \
        const TTYPE*              B,                                                \
        ITYPE                     ldb,                                              \
        UTYPE                     beta_device_host,                                 \
        TTYPE*                    C,                                                \
        ITYPE                     ldc);

INSTANTIATE(int32_t, float, float);
INSTANTIATE(int32_t, double, double);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int64_t, float, float);
INSTANTIATE(int64_t, double, double);
INSTANTIATE(int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int32_t, float, const float*);
INSTANTIATE(int32_t, double, const double*);
INSTANTIATE(int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int32_t, rocsparse_double_complex, const rocsparse_double_complex*);

INSTANTIATE(int64_t, float, const float*);
INSTANTIATE(int64_t, double, const double*);
INSTANTIATE(int64_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, rocsparse_double_complex, const rocsparse_double_complex*);

#undef INSTANTIATE
