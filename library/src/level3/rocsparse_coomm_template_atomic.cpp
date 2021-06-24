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

template <unsigned int BLK_SIZE_X,
          unsigned int BLK_SIZE_Y,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T>
static __device__ void coommnn_general_wf_atomic(I                    nnz,
                                                 I                    n,
                                                 I                    nblocks,
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
    I idx = (BLK_SIZE_Y * hipBlockIdx_x + hipThreadIdx_y) * LOOPS;

    I col = hipBlockIdx_y * BLK_SIZE_X + hipThreadIdx_x;

    if(col >= n || idx >= nnz)
    {
        return;
    }

    T temp = static_cast<T>(0);

    I row = coo_row_ind[idx] - idx_base;

    I end = (idx + LOOPS > nnz) ? nnz - 1 : (idx + LOOPS) - 1;
    while(idx < end)
    {
        if(!TRANSB)
        {
            temp += coo_val[idx] * B[col * ldb + (coo_col_ind[idx] - idx_base)];
        }
        else
        {
            temp += coo_val[idx] * B[(coo_col_ind[idx] - idx_base) * ldb + col];
        }

        I nrow = coo_row_ind[idx + 1] - idx_base;
        if(row != nrow)
        {
            if(order == rocsparse_order_column)
            {
                atomicAdd(&C[col * ldc + row], alpha * temp);
            }
            else
            {
                atomicAdd(&C[row * ldc + col], alpha * temp);
            }

            row  = nrow;
            temp = static_cast<T>(0);
        }

        idx++;
    }

    if(!TRANSB)
    {
        temp += coo_val[idx] * B[col * ldb + (coo_col_ind[idx] - idx_base)];
    }
    else
    {
        temp += coo_val[idx] * B[(coo_col_ind[idx] - idx_base) * ldb + col];
    }

    if(order == rocsparse_order_column)
    {
        atomicAdd(&C[col * ldc + row], alpha * temp);
    }
    else
    {
        atomicAdd(&C[row * ldc + col], alpha * temp);
    }
}

template <unsigned int BLK_SIZE_X,
          unsigned int BLK_SIZE_Y,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T,
          typename U>
__launch_bounds__(BLK_SIZE_X* BLK_SIZE_Y) __global__
    void coommnn_wf_atomic(I nnz,
                           I n,
                           I nblocks,
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
    coommnn_general_wf_atomic<BLK_SIZE_X, BLK_SIZE_Y, LOOPS, TRANSB>(
        nnz, n, nblocks, alpha, coo_row_ind, coo_col_ind, coo_val, B, ldb, C, ldc, order, idx_base);
}

template <unsigned int BLK_SIZE, unsigned int LOOPS, bool TRANSB, typename... Ts>
rocsparse_status coommnn_wf_atomic_dispatch(rocsparse_int blk_size_x,
                                            const dim3&   numBlocks,
                                            const dim3&   dimBlocks,
                                            std::uint32_t sharedMemBytes,
                                            hipStream_t   stream,
                                            Ts&&... ts)
{
    switch(blk_size_x)
    {

#define TREAT_CASE(value_)                                                                \
    case value_:                                                                          \
    {                                                                                     \
        hipLaunchKernelGGL((coommnn_wf_atomic<value_, BLK_SIZE / value_, LOOPS, TRANSB>), \
                           numBlocks,                                                     \
                           dimBlocks,                                                     \
                           sharedMemBytes,                                                \
                           stream,                                                        \
                           ts...);                                                        \
        return rocsparse_status_success;                                                  \
    }

        TREAT_CASE(1);
        TREAT_CASE(2);
        TREAT_CASE(4);
        TREAT_CASE(8);
        TREAT_CASE(16);
        TREAT_CASE(32);
        TREAT_CASE(64);
        TREAT_CASE(128);

#undef TREAT_CASE
    }

    return rocsparse_status_invalid_value;
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
        rocsparse_int ncolsB     = trans_B == rocsparse_operation_none ? n : k;
        rocsparse_int BLK_SIZE_X = 1;
        if(ncolsB <= 1)
        {
            BLK_SIZE_X = 1;
        }
        if(ncolsB <= 2)
        {
            BLK_SIZE_X = 2;
        }
        else if(ncolsB <= 4)
        {
            BLK_SIZE_X = 4;
        }
        else if(ncolsB <= 8)
        {
            BLK_SIZE_X = 8;
        }
        else if(ncolsB <= 16)
        {
            BLK_SIZE_X = 16;
        }
        else if(ncolsB <= 32)
        {
            BLK_SIZE_X = 32;
        }
        else if(ncolsB <= 64)
        {
            BLK_SIZE_X = 64;
        }
        else
        {
            BLK_SIZE_X = 128;
        }

        rocsparse_int BLK_SIZE_Y = 256 / BLK_SIZE_X;

        constexpr I LOOPS   = 4;
        I           nblocks = (nnz - 1) / (BLK_SIZE_Y * LOOPS) + 1;

        dim3 coommn_blocks(nblocks, (n - 1) / BLK_SIZE_X + 1);
        dim3 coommn_threads(BLK_SIZE_X, BLK_SIZE_Y);

        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose))
        {
            return coommnn_wf_atomic_dispatch<256, LOOPS, false>(BLK_SIZE_X,
                                                                 coommn_blocks,
                                                                 coommn_threads,
                                                                 0,
                                                                 stream,
                                                                 nnz,
                                                                 n,
                                                                 nblocks,
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
        else if((order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            return coommnn_wf_atomic_dispatch<256, LOOPS, true>(BLK_SIZE_X,
                                                                coommn_blocks,
                                                                coommn_threads,
                                                                0,
                                                                stream,
                                                                nnz,
                                                                n,
                                                                nblocks,
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
