/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "utility.h"

namespace rocsparse
{
    template <unsigned int WF_SIZE,
              unsigned int LOOPS,
              unsigned int COLS,
              bool         NT,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void coommnn_segmented_atomic_device(rocsparse_operation  trans_B,
                                                              int64_t              nnz,
                                                              I                    nstart,
                                                              int64_t              batch_stride_A,
                                                              T                    alpha,
                                                              const I*             coo_row_ind,
                                                              const I*             coo_col_ind,
                                                              const A*             coo_val,
                                                              const B*             dense_B,
                                                              int64_t              ldb,
                                                              int64_t              batch_stride_B,
                                                              C*                   dense_C,
                                                              int64_t              ldc,
                                                              int64_t              batch_stride_C,
                                                              rocsparse_order      order_C,
                                                              rocsparse_index_base idx_base)
    {
        int tid = hipThreadIdx_x;
        int lid = tid & (WF_SIZE - 1);

        int batch = hipBlockIdx_z;

        // Shared memory to hold row indices and values for segmented reduction
        __shared__ I shared_row[WF_SIZE];
        __shared__ T shared_val[COLS][WF_SIZE];

        I       col_offset = nstart + COLS * hipBlockIdx_y;
        int64_t offset     = hipBlockIdx_x * LOOPS * WF_SIZE;

        if(offset >= nnz)
        {
            return;
        }

        // Current threads index into COO structure
        int64_t idx = offset + lid;

        I row;
        T val[COLS];

        // Each thread processes 'loop' COO entries
        while(idx < offset + LOOPS * WF_SIZE)
        {
            // Get corresponding COO entry
            I r = (idx < nnz)
                      ? rocsparse_nontemporal_load(coo_row_ind + idx + batch_stride_A * batch)
                            - idx_base
                      : -1;
            I c = (idx < nnz)
                      ? rocsparse_nontemporal_load(coo_col_ind + idx + batch_stride_A * batch)
                            - idx_base
                      : 0;
            T v = (idx < nnz)
                      ? alpha * rocsparse_nontemporal_load(coo_val + idx + batch_stride_A * batch)
                      : static_cast<T>(0);

            row = r;

            if(NT)
            {
                if(trans_B == rocsparse_operation_conjugate_transpose)
                {
                    for(unsigned int p = 0; p < COLS; p++)
                    {
                        val[p] = v
                                 * rocsparse_conj(
                                     dense_B[c * ldb + (col_offset + p) + batch_stride_B * batch]);
                    }
                }
                else
                {
                    for(unsigned int p = 0; p < COLS; p++)
                    {
                        val[p] = v * dense_B[c * ldb + (col_offset + p) + batch_stride_B * batch];
                    }
                }
            }
            else
            {
                if(trans_B == rocsparse_operation_conjugate_transpose)
                {
                    for(unsigned int p = 0; p < COLS; p++)
                    {
                        val[p] = v
                                 * rocsparse_conj(
                                     dense_B[(col_offset + p) * ldb + c + batch_stride_B * batch]);
                    }
                }
                else
                {
                    for(unsigned int p = 0; p < COLS; p++)
                    {
                        val[p] = v * dense_B[(col_offset + p) * ldb + c + batch_stride_B * batch];
                    }
                }
            }

            // First thread in wavefront checks row index from previous loop
            // if it has been completed or if additional rows have to be
            // appended.
            if(idx > offset && lid == 0)
            {
                I prevrow = shared_row[WF_SIZE - 1];
                if(row == prevrow)
                {
                    for(unsigned int p = 0; p < COLS; p++)
                    {
                        val[p] += shared_val[p][WF_SIZE - 1];
                    }
                }
                else if(prevrow >= 0)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(unsigned int p = 0; p < COLS; p++)
                        {
                            rocsparse_atomic_add(
                                &dense_C[prevrow + (col_offset + p) * ldc + batch_stride_C * batch],
                                shared_val[p][WF_SIZE - 1]);
                        }
                    }
                    else
                    {
                        for(unsigned int p = 0; p < COLS; p++)
                        {
                            rocsparse_atomic_add(
                                &dense_C[(col_offset + p) + prevrow * ldc + batch_stride_C * batch],
                                shared_val[p][WF_SIZE - 1]);
                        }
                    }
                }
            }

            __syncthreads();

            for(unsigned int p = 0; p < COLS; p++)
            {
                shared_val[p][lid] = val[p];
            }
            shared_row[lid] = row;

            __syncthreads();

#pragma unroll
            // Segmented wavefront reduction
            for(unsigned int j = 1; j < WF_SIZE; j <<= 1)
            {
                if(lid >= j)
                {
                    if(row == shared_row[lid - j])
                    {
                        for(unsigned int p = 0; p < COLS; p++)
                        {
                            val[p] += shared_val[p][lid - j];
                        }
                    }
                }
                __syncthreads();

                for(unsigned int p = 0; p < COLS; p++)
                {
                    shared_val[p][lid] = val[p];
                }

                __syncthreads();
            }

            // All lanes but the last one write their result in C.
            // The last value might need to be appended by the next iteration.
            if(lid < WF_SIZE - 1)
            {
                if(row != shared_row[lid + 1] && row >= 0)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(unsigned int p = 0; p < COLS; p++)
                        {
                            rocsparse_atomic_add(
                                &dense_C[row + (col_offset + p) * ldc + batch_stride_C * batch],
                                val[p]);
                        }
                    }
                    else
                    {
                        for(unsigned int p = 0; p < COLS; p++)
                        {
                            rocsparse_atomic_add(
                                &dense_C[(col_offset + p) + row * ldc + batch_stride_C * batch],
                                val[p]);
                        }
                    }
                }
            }

            idx += WF_SIZE;
        }

        // Write last entries into buffers for segmented block reduction
        if(lid == WF_SIZE - 1 && row >= 0)
        {
            if(order_C == rocsparse_order_column)
            {
                for(unsigned int p = 0; p < COLS; p++)
                {
                    rocsparse_atomic_add(
                        &dense_C[row + (col_offset + p) * ldc + batch_stride_C * batch], val[p]);
                }
            }
            else
            {
                for(unsigned int p = 0; p < COLS; p++)
                {
                    rocsparse_atomic_add(
                        &dense_C[(col_offset + p) + row * ldc + batch_stride_C * batch], val[p]);
                }
            }
        }
    }

    template <unsigned int WF_SIZE,
              unsigned int LOOPS,
              unsigned int COLS,
              bool         NT,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C,
              typename U>
    ROCSPARSE_KERNEL(WF_SIZE)
    void coommnn_segmented_atomic(rocsparse_operation trans_B,
                                  int64_t             nnz,
                                  I                   n,
                                  int64_t             batch_stride_A,
                                  U                   alpha_device_host,
                                  const I* __restrict__ coo_row_ind,
                                  const I* __restrict__ coo_col_ind,
                                  const A* __restrict__ coo_val,
                                  const B* __restrict__ dense_B,
                                  int64_t ldb,
                                  int64_t batch_stride_B,
                                  C* __restrict__ dense_C,
                                  int64_t              ldc,
                                  int64_t              batch_stride_C,
                                  rocsparse_order      order_C,
                                  rocsparse_index_base idx_base)
    {
        auto alpha = load_scalar_device_host(alpha_device_host);
        if(alpha != static_cast<T>(0))
        {
            rocsparse::coommnn_segmented_atomic_device<WF_SIZE, LOOPS, COLS, NT>(trans_B,
                                                                                 nnz,
                                                                                 n,
                                                                                 batch_stride_A,
                                                                                 alpha,
                                                                                 coo_row_ind,
                                                                                 coo_col_ind,
                                                                                 coo_val,
                                                                                 dense_B,
                                                                                 ldb,
                                                                                 batch_stride_B,
                                                                                 dense_C,
                                                                                 ldc,
                                                                                 batch_stride_C,
                                                                                 order_C,
                                                                                 idx_base);
        }
    }

#define LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(WF_SIZE, LOOPS, COLS, NT) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                       \
        (rocsparse::coommnn_segmented_atomic<WF_SIZE, LOOPS, COLS, NT, T>),   \
        dim3(nblocks, (main - 1) / COLS + 1, batch_count_C),                  \
        dim3(WF_SIZE),                                                        \
        0,                                                                    \
        stream,                                                               \
        trans_B,                                                              \
        nnz,                                                                  \
        (I)0,                                                                 \
        batch_stride_A,                                                       \
        alpha_device_host,                                                    \
        coo_row_ind,                                                          \
        coo_col_ind,                                                          \
        coo_val,                                                              \
        dense_B,                                                              \
        ldb,                                                                  \
        batch_stride_B,                                                       \
        dense_C,                                                              \
        ldc,                                                                  \
        batch_stride_C,                                                       \
        order_C,                                                              \
        descr->base);

#define LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(WF_SIZE, LOOPS, COLS, NT) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                            \
        (rocsparse::coommnn_segmented_atomic<WF_SIZE, LOOPS, COLS, NT, T>),        \
        dim3(nblocks, 1, batch_count_C),                                           \
        dim3(WF_SIZE),                                                             \
        0,                                                                         \
        stream,                                                                    \
        trans_B,                                                                   \
        nnz,                                                                       \
        main,                                                                      \
        batch_stride_A,                                                            \
        alpha_device_host,                                                         \
        coo_row_ind,                                                               \
        coo_col_ind,                                                               \
        coo_val,                                                                   \
        dense_B,                                                                   \
        ldb,                                                                       \
        batch_stride_B,                                                            \
        dense_C,                                                                   \
        ldc,                                                                       \
        batch_stride_C,                                                            \
        order_C,                                                                   \
        descr->base);

    template <typename T, typename I, typename A, typename B, typename C, typename U>
    rocsparse_status coomm_template_segmented_atomic(rocsparse_handle          handle,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     I                         m,
                                                     I                         n,
                                                     I                         k,
                                                     int64_t                   nnz,
                                                     I                         batch_count_A,
                                                     int64_t                   batch_stride_A,
                                                     U                         alpha_device_host,
                                                     const rocsparse_mat_descr descr,
                                                     const A*                  coo_val,
                                                     const I*                  coo_row_ind,
                                                     const I*                  coo_col_ind,
                                                     const B*                  dense_B,
                                                     int64_t                   ldb,
                                                     I                         batch_count_B,
                                                     int64_t                   batch_stride_B,
                                                     rocsparse_order           order_B,
                                                     U                         beta_device_host,
                                                     C*                        dense_C,
                                                     int64_t                   ldc,
                                                     I                         batch_count_C,
                                                     int64_t                   batch_stride_C,
                                                     rocsparse_order           order_C)
    {
        // Stream
        hipStream_t stream = handle->stream;

        // Run different coomm kernels
        if(trans_A == rocsparse_operation_none)
        {
            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                I main = 0;
                I remainder;

                if(handle->wavefront_size == 32)
                {
                    I nloops  = 16;
                    I nblocks = (nnz - 1) / (32 * nloops) + 1;

                    if(n >= 8)
                    {
                        remainder = n % 8;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(32, 16, 8, false);
                    }
                    else if(n >= 4)
                    {
                        remainder = n % 4;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(32, 16, 4, false);
                    }
                    else
                    {
                        remainder = n;
                    }

                    if(remainder > 0)
                    {
                        if(remainder == 1)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 1, false);
                        }
                        else if(remainder == 2)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 2, false);
                        }
                        else if(remainder == 3)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 3, false);
                        }
                        else if(remainder == 4)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 4, false);
                        }
                        else if(remainder == 5)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 5, false);
                        }
                        else if(remainder == 6)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 6, false);
                        }
                        else if(remainder == 7)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 7, false);
                        }
                    }
                }
                else if(handle->wavefront_size == 64)
                {
                    I nloops  = 16;
                    I nblocks = (nnz - 1) / (64 * nloops) + 1;

                    if(n >= 8)
                    {
                        remainder = n % 8;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(64, 16, 8, false);
                    }
                    else if(n >= 4)
                    {
                        remainder = n % 4;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(64, 16, 4, false);
                    }
                    else
                    {
                        remainder = n;
                    }

                    if(remainder > 0)
                    {
                        if(remainder == 1)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 1, false);
                        }
                        else if(remainder == 2)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 2, false);
                        }
                        else if(remainder == 3)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 3, false);
                        }
                        else if(remainder == 4)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 4, false);
                        }
                        else if(remainder == 5)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 5, false);
                        }
                        else if(remainder == 6)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 6, false);
                        }
                        else if(remainder == 7)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 7, false);
                        }
                    }
                }
            }
            else if((order_B == rocsparse_order_column
                     && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                I main = 0;
                I remainder;

                if(handle->wavefront_size == 32)
                {
                    I nloops  = 16;
                    I nblocks = (nnz - 1) / (32 * nloops) + 1;

                    if(n >= 8)
                    {
                        remainder = n % 8;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(32, 16, 8, true);
                    }
                    else if(n >= 4)
                    {
                        remainder = n % 4;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(32, 16, 4, true);
                    }
                    else
                    {
                        remainder = n;
                    }

                    if(remainder > 0)
                    {
                        if(remainder == 1)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 1, true);
                        }
                        else if(remainder == 2)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 2, true);
                        }
                        else if(remainder == 3)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 3, true);
                        }
                        else if(remainder == 4)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 4, true);
                        }
                        else if(remainder == 5)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 5, true);
                        }
                        else if(remainder == 6)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 6, true);
                        }
                        else if(remainder == 7)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 7, true);
                        }
                    }
                }
                else if(handle->wavefront_size == 64)
                {
                    I nloops  = 16;
                    I nblocks = (nnz - 1) / (64 * nloops) + 1;

                    if(n >= 8)
                    {
                        remainder = n % 8;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(64, 16, 8, true);
                    }
                    else if(n >= 4)
                    {
                        remainder = n % 4;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(64, 16, 4, true);
                    }
                    else
                    {
                        remainder = n;
                    }

                    if(remainder > 0)
                    {
                        if(remainder == 1)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 1, true);
                        }
                        else if(remainder == 2)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 2, true);
                        }
                        else if(remainder == 3)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 3, true);
                        }
                        else if(remainder == 4)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 4, true);
                        }
                        else if(remainder == 5)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 5, true);
                        }
                        else if(remainder == 6)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 6, true);
                        }
                        else if(remainder == 7)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 7, true);
                        }
                    }
                }
            }
#undef COOMMN_DIM
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(TTYPE, ITYPE, ATYPE, BTYPE, CTYPE, UTYPE)                    \
    template rocsparse_status rocsparse::coomm_template_segmented_atomic<TTYPE>( \
        rocsparse_handle          handle,                                        \
        rocsparse_operation       trans_A,                                       \
        rocsparse_operation       trans_B,                                       \
        ITYPE                     m,                                             \
        ITYPE                     n,                                             \
        ITYPE                     k,                                             \
        int64_t                   nnz,                                           \
        ITYPE                     batch_count_A,                                 \
        int64_t                   batch_stride_A,                                \
        UTYPE                     alpha_device_host,                             \
        const rocsparse_mat_descr descr,                                         \
        const ATYPE*              coo_val,                                       \
        const ITYPE*              coo_row_ind,                                   \
        const ITYPE*              coo_col_ind,                                   \
        const BTYPE*              dense_B,                                       \
        int64_t                   ldb,                                           \
        ITYPE                     batch_count_B,                                 \
        int64_t                   batch_stride_B,                                \
        rocsparse_order           order_B,                                       \
        UTYPE                     beta_device_host,                              \
        CTYPE*                    dense_C,                                       \
        int64_t                   ldc,                                           \
        ITYPE                     batch_count_C,                                 \
        int64_t                   batch_stride_C,                                \
        rocsparse_order           order_C);

// Uniform precisions
INSTANTIATE(float, int32_t, float, float, float, float);
INSTANTIATE(float, int64_t, float, float, float, float);
INSTANTIATE(double, int32_t, double, double, double, double);
INSTANTIATE(double, int64_t, double, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

INSTANTIATE(float, int32_t, float, float, float, const float*);
INSTANTIATE(float, int64_t, float, float, float, const float*);
INSTANTIATE(double, int32_t, double, double, double, const double*);
INSTANTIATE(double, int64_t, double, double, double, const double*);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);

// Mixed Precisions
INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(float, int32_t, int8_t, int8_t, float, float);
INSTANTIATE(float, int64_t, int8_t, int8_t, float, float);

INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(float, int32_t, int8_t, int8_t, float, const float*);
INSTANTIATE(float, int64_t, int8_t, int8_t, float, const float*);
#undef INSTANTIATE
