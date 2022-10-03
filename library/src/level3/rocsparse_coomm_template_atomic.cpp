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

#include "common.h"
#include "definitions.h"
#include "utility.h"

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T>
static ROCSPARSE_DEVICE_ILF void coommnn_atomic_main_device(bool    conj_A,
                                                            bool    conj_B,
                                                            I       ncol,
                                                            int64_t nnz,
                                                            I       n,
                                                            I       batch_stride_A,
                                                            T       alpha,
                                                            const I* __restrict__ coo_row_ind,
                                                            const I* __restrict__ coo_col_ind,
                                                            const T* __restrict__ coo_val,
                                                            const T* __restrict__ B,
                                                            I ldb,
                                                            I batch_stride_B,
                                                            T* __restrict__ C,
                                                            I                    ldc,
                                                            I                    batch_stride_C,
                                                            rocsparse_order      order,
                                                            rocsparse_index_base idx_base)
{
    int     tid = hipThreadIdx_x;
    int64_t gid = hipBlockIdx_x * BLOCKSIZE + tid;
    int     lid = tid & (WF_SIZE - 1);

    int batch = hipBlockIdx_y;

    I row = (gid < nnz)
                ? rocsparse_nontemporal_load(coo_row_ind + gid + batch_stride_A * batch) - idx_base
                : 0;
    I col = (gid < nnz)
                ? rocsparse_nontemporal_load(coo_col_ind + gid + batch_stride_A * batch) - idx_base
                : 0;
    T val = (gid < nnz) ? rocsparse_nontemporal_load(coo_val + gid + batch_stride_A * batch)
                        : static_cast<T>(0);

    for(I l = 0; l < ncol; l += WF_SIZE * LOOPS)
    {
        I colB = l + lid;

        T sum[LOOPS]{};

        I current_row = rocsparse_shfl(row, 0, WF_SIZE);

        for(I i = 0; i < WF_SIZE; ++i)
        {
            T v = rocsparse_shfl(val, i, WF_SIZE);
            I c = rocsparse_shfl(col, i, WF_SIZE);
            I r = rocsparse_shfl(row, i, WF_SIZE);

            if(r != current_row)
            {
                if(order == rocsparse_order_column)
                {
                    for(I p = 0; p < LOOPS; p++)
                    {
                        atomicAdd(
                            &C[(colB + p * WF_SIZE) * ldc + current_row + batch_stride_C * batch],
                            alpha * sum[p]);
                    }
                }
                else
                {
                    for(I p = 0; p < LOOPS; p++)
                    {
                        atomicAdd(
                            &C[current_row * ldc + colB + p * WF_SIZE + batch_stride_C * batch],
                            alpha * sum[p]);
                    }
                }

                for(I p = 0; p < LOOPS; p++)
                {
                    sum[p] = static_cast<T>(0);
                }

                current_row = r;
            }

            if(TRANSB)
            {
                for(I p = 0; p < LOOPS; p++)
                {
                    sum[p] = rocsparse_fma(
                        v,
                        conj_val(B[c * ldb + colB + p * WF_SIZE + batch_stride_B * batch], conj_B),
                        sum[p]);
                }
            }
            else
            {
                for(I p = 0; p < LOOPS; p++)
                {
                    sum[p] = rocsparse_fma(
                        v,
                        conj_val(B[(colB + p * WF_SIZE) * ldb + c + batch_stride_B * batch],
                                 conj_B),
                        sum[p]);
                }
            }
        }

        if(order == rocsparse_order_column)
        {
            for(I p = 0; p < LOOPS; p++)
            {
                atomicAdd(&C[(colB + p * WF_SIZE) * ldc + current_row + batch_stride_C * batch],
                          alpha * sum[p]);
            }
        }
        else
        {
            for(I p = 0; p < LOOPS; p++)
            {
                atomicAdd(&C[current_row * ldc + colB + p * WF_SIZE + batch_stride_C * batch],
                          alpha * sum[p]);
            }
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, bool TRANSB, typename I, typename T>
static ROCSPARSE_DEVICE_ILF void coommnn_atomic_remainder_device(bool    conj_A,
                                                                 bool    conj_B,
                                                                 I       ncol_offset,
                                                                 I       n,
                                                                 int64_t nnz,
                                                                 I       batch_stride_A,
                                                                 T       alpha,
                                                                 const I* __restrict__ coo_row_ind,
                                                                 const I* __restrict__ coo_col_ind,
                                                                 const T* __restrict__ coo_val,
                                                                 const T* __restrict__ B,
                                                                 I ldb,
                                                                 I batch_stride_B,
                                                                 T* __restrict__ C,
                                                                 I               ldc,
                                                                 I               batch_stride_C,
                                                                 rocsparse_order order,
                                                                 rocsparse_index_base idx_base)
{
    int     tid = hipThreadIdx_x;
    int     lid = tid & (WF_SIZE - 1);
    int     wid = tid / WF_SIZE;
    int64_t gid = BLOCKSIZE * hipBlockIdx_x + tid;

    int batch = hipBlockIdx_y;

    __shared__ I shared_row[(BLOCKSIZE / WF_SIZE) * WF_SIZE];
    __shared__ T shared_val[(BLOCKSIZE / WF_SIZE) * WF_SIZE];

    I row = (gid < nnz)
                ? rocsparse_nontemporal_load(&coo_row_ind[gid + batch_stride_A * batch]) - idx_base
                : -1;
    I col = (gid < nnz)
                ? rocsparse_nontemporal_load(&coo_col_ind[gid + batch_stride_A * batch]) - idx_base
                : 0;
    T val = (gid < nnz) ? alpha * rocsparse_nontemporal_load(&coo_val[gid + batch_stride_A * batch])
                        : static_cast<T>(0);

    for(I l = ncol_offset; l < n; l += WF_SIZE)
    {
        I colB = l + lid;

        T sum         = static_cast<T>(0);
        I current_row = rocsparse_shfl(row, 0, WF_SIZE);

        for(I i = 0; i < WF_SIZE; ++i)
        {
            T v = rocsparse_shfl(val, i, WF_SIZE);
            I c = rocsparse_shfl(col, i, WF_SIZE);
            I r = rocsparse_shfl(row, i, WF_SIZE);

            if(r != current_row)
            {
                if(colB < n)
                {
                    if(order == rocsparse_order_column)
                    {
                        atomicAdd(&C[colB * ldc + current_row + batch_stride_C * batch], sum);
                    }
                    else
                    {
                        atomicAdd(&C[current_row * ldc + colB + batch_stride_C * batch], sum);
                    }
                }

                sum = static_cast<T>(0);

                current_row = r;
            }

            if(colB < n)
            {
                if(TRANSB)
                {
                    sum = rocsparse_fma(
                        v, conj_val(B[c * ldb + colB + batch_stride_B * batch], conj_B), sum);
                }
                else
                {
                    sum = rocsparse_fma(
                        v, conj_val(B[colB * ldb + c + batch_stride_B * batch], conj_B), sum);
                }
            }
        }

        __syncthreads();
        shared_row[(BLOCKSIZE / WF_SIZE) * lid + wid] = current_row;
        shared_val[(BLOCKSIZE / WF_SIZE) * lid + wid] = sum;
        __syncthreads();

        current_row = shared_row[tid];
        sum         = shared_val[tid];

        int slid = tid & ((BLOCKSIZE / WF_SIZE) - 1);
        int swid = tid / (BLOCKSIZE / WF_SIZE);

        // segmented reduction
        for(I j = 1; j < (BLOCKSIZE / WF_SIZE); j <<= 1)
        {
            if(slid >= j)
            {
                if(current_row == shared_row[slid - j])
                {
                    sum = sum + shared_val[(BLOCKSIZE / WF_SIZE) * swid + slid - j];
                }
            }
            __syncthreads();
            shared_val[(BLOCKSIZE / WF_SIZE) * swid + slid] = sum;
            __syncthreads();
        }

        if(slid < ((BLOCKSIZE / WF_SIZE) - 1))
        {
            if(current_row != shared_row[slid + 1] && current_row >= 0)
            {
                if((l + swid) < n)
                {
                    if(order == rocsparse_order_column)
                    {
                        atomicAdd(&C[(l + swid) * ldc + current_row + batch_stride_C * batch], sum);
                    }
                    else
                    {
                        atomicAdd(&C[current_row * ldc + (l + swid) + batch_stride_C * batch], sum);
                    }
                }
            }
        }

        if(slid == ((BLOCKSIZE / WF_SIZE) - 1))
        {
            if(current_row >= 0)
            {
                if((l + swid) < n)
                {
                    if(order == rocsparse_order_column)
                    {
                        atomicAdd(&C[(l + swid) * ldc + current_row + batch_stride_C * batch], sum);
                    }
                    else
                    {
                        atomicAdd(&C[current_row * ldc + (l + swid) + batch_stride_C * batch], sum);
                    }
                }
            }
        }
    }
}

template <unsigned int BLOCKSIZE, bool TRANSB, typename I, typename T>
static __device__ void coommtn_atomic_device(bool                 conj_A,
                                             bool                 conj_B,
                                             int64_t              nnz,
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
    int64_t gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

    if(gid >= nnz)
    {
        return;
    }

    I row = coo_row_ind[gid] - idx_base;
    I col = coo_col_ind[gid] - idx_base;
    T val = conj_val(coo_val[gid], conj_A);

    T bval = static_cast<T>(0);

    if(TRANSB)
    {
        bval = conj_val(B[ldb * row + hipBlockIdx_y], conj_B);
    }
    else
    {
        bval = conj_val(B[hipBlockIdx_y * ldb + row], conj_B);
    }

    if(order == rocsparse_order_column)
    {
        atomicAdd(&C[hipBlockIdx_y * ldc + col], alpha * (val * bval));
    }
    else
    {
        atomicAdd(&C[col * ldc + hipBlockIdx_y], alpha * (val * bval));
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coommnn_atomic_main(bool    conj_A,
                             bool    conj_B,
                             I       ncol,
                             int64_t nnz,
                             I       n,
                             I       batch_stride_A,
                             U       alpha_device_host,
                             const I* __restrict__ coo_row_ind,
                             const I* __restrict__ coo_col_ind,
                             const T* __restrict__ coo_val,
                             const T* __restrict__ B,
                             I ldb,
                             I batch_stride_B,
                             T* __restrict__ C,
                             I                    ldc,
                             I                    batch_stride_C,
                             rocsparse_order      order,
                             rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    if(alpha != static_cast<T>(0))
    {
        coommnn_atomic_main_device<BLOCKSIZE, WF_SIZE, LOOPS, TRANSB>(conj_A,
                                                                      conj_B,
                                                                      ncol,
                                                                      nnz,
                                                                      n,
                                                                      batch_stride_A,
                                                                      alpha,
                                                                      coo_row_ind,
                                                                      coo_col_ind,
                                                                      coo_val,
                                                                      B,
                                                                      ldb,
                                                                      batch_stride_B,
                                                                      C,
                                                                      ldc,
                                                                      batch_stride_C,
                                                                      order,
                                                                      idx_base);
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          bool         TRANSB,
          typename I,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coommnn_atomic_remainder(bool    conj_A,
                                  bool    conj_B,
                                  I       ncol_offset,
                                  I       n,
                                  int64_t nnz,
                                  I       batch_stride_A,
                                  U       alpha_device_host,
                                  const I* __restrict__ coo_row_ind,
                                  const I* __restrict__ coo_col_ind,
                                  const T* __restrict__ coo_val,
                                  const T* __restrict__ B,
                                  I ldb,
                                  I batch_stride_B,
                                  T* __restrict__ C,
                                  I                    ldc,
                                  I                    batch_stride_C,
                                  rocsparse_order      order,
                                  rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);

    if(alpha != static_cast<T>(0))
    {
        coommnn_atomic_remainder_device<BLOCKSIZE, WF_SIZE, TRANSB>(conj_A,
                                                                    conj_B,
                                                                    ncol_offset,
                                                                    n,
                                                                    nnz,
                                                                    batch_stride_A,
                                                                    alpha,
                                                                    coo_row_ind,
                                                                    coo_col_ind,
                                                                    coo_val,
                                                                    B,
                                                                    ldb,
                                                                    batch_stride_B,
                                                                    C,
                                                                    ldc,
                                                                    batch_stride_C,
                                                                    order,
                                                                    idx_base);
    }
}

template <unsigned int BLOCKSIZE, bool TRANSB, typename I, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coommtn_atomic_main(bool    conj_A,
                             bool    conj_B,
                             int64_t nnz,
                             I       n,
                             U       alpha_device_host,
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
    if(alpha != static_cast<T>(0))
    {
        coommtn_atomic_device<BLOCKSIZE, TRANSB>(conj_A,
                                                 conj_B,
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
}

#define LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB)     \
    hipLaunchKernelGGL((coommnn_atomic_main<COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB>), \
                       dim3((nnz - 1) / COOMMNN_DIM + 1, batch_count_C),           \
                       dim3(COOMMNN_DIM),                                          \
                       0,                                                          \
                       handle->stream,                                             \
                       conj_A,                                                     \
                       conj_B,                                                     \
                       main,                                                       \
                       nnz,                                                        \
                       n,                                                          \
                       batch_stride_A,                                             \
                       alpha_device_host,                                          \
                       coo_row_ind,                                                \
                       coo_col_ind,                                                \
                       coo_val,                                                    \
                       B,                                                          \
                       ldb,                                                        \
                       batch_stride_B,                                             \
                       C,                                                          \
                       ldc,                                                        \
                       batch_stride_C,                                             \
                       order,                                                      \
                       descr->base);

#define LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(COOMMNN_DIM, WF_SIZE, TRANSB)     \
    hipLaunchKernelGGL((coommnn_atomic_remainder<COOMMNN_DIM, WF_SIZE, TRANSB>), \
                       dim3((nnz - 1) / COOMMNN_DIM + 1, batch_count_C),         \
                       dim3(COOMMNN_DIM),                                        \
                       0,                                                        \
                       handle->stream,                                           \
                       conj_A,                                                   \
                       conj_B,                                                   \
                       main,                                                     \
                       n,                                                        \
                       nnz,                                                      \
                       batch_stride_A,                                           \
                       alpha_device_host,                                        \
                       coo_row_ind,                                              \
                       coo_col_ind,                                              \
                       coo_val,                                                  \
                       B,                                                        \
                       ldb,                                                      \
                       batch_stride_B,                                           \
                       C,                                                        \
                       ldc,                                                      \
                       batch_stride_C,                                           \
                       order,                                                    \
                       descr->base);

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          bool         TRANSB,
          typename I,
          typename T,
          typename U>
rocsparse_status coommnn_atomic_dispatch(rocsparse_handle          handle,
                                         bool                      conj_A,
                                         bool                      conj_B,
                                         rocsparse_order           order,
                                         I                         m,
                                         I                         n,
                                         I                         k,
                                         int64_t                   nnz,
                                         I                         batch_count_A,
                                         I                         batch_stride_A,
                                         U                         alpha_device_host,
                                         const rocsparse_mat_descr descr,
                                         const T*                  coo_val,
                                         const I*                  coo_row_ind,
                                         const I*                  coo_col_ind,
                                         const T*                  B,
                                         I                         ldb,
                                         I                         batch_count_B,
                                         I                         batch_stride_B,
                                         U                         beta_device_host,
                                         T*                        C,
                                         I                         ldc,
                                         I                         batch_count_C,
                                         I                         batch_stride_C)
{
    I main      = 0;
    I remainder = n;

    if(n >= 256)
    {
        remainder = n % 256;
        main      = n - remainder;
        LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (256 / WF_SIZE), TRANSB);
    }
    else if(n >= 192)
    {
        remainder = n % 192;
        main      = n - remainder;
        LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (192 / WF_SIZE), TRANSB);
    }
    else if(n >= 128)
    {
        remainder = n % 128;
        main      = n - remainder;
        LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (128 / WF_SIZE), TRANSB);
    }
    else if(n >= 64)
    {
        remainder = n % 64;
        main      = n - remainder;
        LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (64 / WF_SIZE), TRANSB);
    }

    if(remainder > 0)
    {
        if(remainder <= 1)
        {
            LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 1, TRANSB);
        }
        else if(remainder <= 2)
        {
            LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 2, TRANSB);
        }
        else if(remainder <= 4)
        {
            LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 4, TRANSB);
        }
        else if(remainder <= 8)
        {
            LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 8, TRANSB);
        }
        else if(remainder <= 16)
        {
            LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 16, TRANSB);
        }
        else if(remainder <= 32 || WF_SIZE == 32)
        {
            LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 32, TRANSB);
        }
        else
        {
            LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 64, TRANSB);
        }
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
                                                 int64_t                   nnz,
                                                 I                         batch_count_A,
                                                 I                         batch_stride_A,
                                                 U                         alpha_device_host,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  coo_val,
                                                 const I*                  coo_row_ind,
                                                 const I*                  coo_col_ind,
                                                 const T*                  B,
                                                 I                         ldb,
                                                 I                         batch_count_B,
                                                 I                         batch_stride_B,
                                                 U                         beta_device_host,
                                                 T*                        C,
                                                 I                         ldc,
                                                 I                         batch_count_C,
                                                 I                         batch_stride_C)
{
    bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose);
    bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

    // Run different coomm kernels
    if(trans_A == rocsparse_operation_none)
    {
        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_conjugate_transpose))
        {
            if(handle->wavefront_size == 32)
            {
                return coommnn_atomic_dispatch<256, 32, false>(handle,
                                                               conj_A,
                                                               conj_B,
                                                               order,
                                                               m,
                                                               n,
                                                               k,
                                                               nnz,
                                                               batch_count_A,
                                                               batch_stride_A,
                                                               alpha_device_host,
                                                               descr,
                                                               coo_val,
                                                               coo_row_ind,
                                                               coo_col_ind,
                                                               B,
                                                               ldb,
                                                               batch_count_B,
                                                               batch_stride_B,
                                                               beta_device_host,
                                                               C,
                                                               ldc,
                                                               batch_count_C,
                                                               batch_stride_C);
            }
            else if(handle->wavefront_size == 64)
            {
                return coommnn_atomic_dispatch<256, 64, false>(handle,
                                                               conj_A,
                                                               conj_B,
                                                               order,
                                                               m,
                                                               n,
                                                               k,
                                                               nnz,
                                                               batch_count_A,
                                                               batch_stride_A,
                                                               alpha_device_host,
                                                               descr,
                                                               coo_val,
                                                               coo_row_ind,
                                                               coo_col_ind,
                                                               B,
                                                               ldb,
                                                               batch_count_B,
                                                               batch_stride_B,
                                                               beta_device_host,
                                                               C,
                                                               ldc,
                                                               batch_count_C,
                                                               batch_stride_C);
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
                return coommnn_atomic_dispatch<256, 32, true>(handle,
                                                              conj_A,
                                                              conj_B,
                                                              order,
                                                              m,
                                                              n,
                                                              k,
                                                              nnz,
                                                              batch_count_A,
                                                              batch_stride_A,
                                                              alpha_device_host,
                                                              descr,
                                                              coo_val,
                                                              coo_row_ind,
                                                              coo_col_ind,
                                                              B,
                                                              ldb,
                                                              batch_count_B,
                                                              batch_stride_B,
                                                              beta_device_host,
                                                              C,
                                                              ldc,
                                                              batch_count_C,
                                                              batch_stride_C);
            }
            else if(handle->wavefront_size == 64)
            {
                return coommnn_atomic_dispatch<256, 64, true>(handle,
                                                              conj_A,
                                                              conj_B,
                                                              order,
                                                              m,
                                                              n,
                                                              k,
                                                              nnz,
                                                              batch_count_A,
                                                              batch_stride_A,
                                                              alpha_device_host,
                                                              descr,
                                                              coo_val,
                                                              coo_row_ind,
                                                              coo_col_ind,
                                                              B,
                                                              ldb,
                                                              batch_count_B,
                                                              batch_stride_B,
                                                              beta_device_host,
                                                              C,
                                                              ldc,
                                                              batch_count_C,
                                                              batch_stride_C);
            }
            else
            {
                return rocsparse_status_not_implemented;
            }
        }
    }
    else
    {
        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_conjugate_transpose))
        {
            hipLaunchKernelGGL((coommtn_atomic_main<256, false>),
                               dim3((nnz - 1) / 256 + 1, n),
                               dim3(256),
                               0,
                               handle->stream,
                               conj_A,
                               conj_B,
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
        else if((order == rocsparse_order_column
                 && trans_B == rocsparse_operation_conjugate_transpose)
                || (order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            hipLaunchKernelGGL((coommtn_atomic_main<256, true>),
                               dim3((nnz - 1) / 256 + 1, n),
                               dim3(256),
                               0,
                               handle->stream,
                               conj_A,
                               conj_B,
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
        int64_t                   nnz,                                              \
        ITYPE                     batch_count_A,                                    \
        ITYPE                     batch_stride_A,                                   \
        UTYPE                     alpha_device_host,                                \
        const rocsparse_mat_descr descr,                                            \
        const TTYPE*              coo_val,                                          \
        const ITYPE*              coo_row_ind,                                      \
        const ITYPE*              coo_col_ind,                                      \
        const TTYPE*              B,                                                \
        ITYPE                     ldb,                                              \
        ITYPE                     batch_count_B,                                    \
        ITYPE                     batch_stride_B,                                   \
        UTYPE                     beta_device_host,                                 \
        TTYPE*                    C,                                                \
        ITYPE                     ldc,                                              \
        ITYPE                     batch_count_C,                                    \
        ITYPE                     batch_stride_C);

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
