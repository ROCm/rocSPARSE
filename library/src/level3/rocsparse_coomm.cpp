/*! \file */
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

#include "rocsparse_coomm.hpp"
#include "common.h"
#include "definitions.h"
#include "utility.h"

// Scale kernel for beta != 1.0
template <unsigned int BLOCKSIZE, typename I, typename T>
ROCSPARSE_DEVICE_ILF void coommnn_scale_device(
    I m, I n, T beta, T* __restrict__ data, int64_t ld, int64_t stride, rocsparse_order order)
{
    I gid   = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    I batch = hipBlockIdx_y;

    if(gid >= m * n)
    {
        return;
    }

    I wid = (order == rocsparse_order_column) ? gid / m : gid / n;
    I lid = (order == rocsparse_order_column) ? gid % m : gid % n;

    if(beta == static_cast<T>(0))
    {
        data[lid + ld * wid + stride * batch] = static_cast<T>(0);
    }
    else
    {
        data[lid + ld * wid + stride * batch] *= beta;
    }
}

template <unsigned int BLOCKSIZE, typename I, typename T, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void coommnn_scale_kernel(I m,
                          I n,
                          U beta_device_host,
                          T* __restrict__ data,
                          int64_t         ld,
                          int64_t         stride,
                          rocsparse_order order)
{
    auto beta = load_scalar_device_host(beta_device_host);
    if(beta != static_cast<T>(1))
    {
        coommnn_scale_device<BLOCKSIZE>(m, n, beta, data, ld, stride, order);
    }
}

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_coomm_alg value_)
{
    switch(value_)
    {
    case rocsparse_coomm_alg_default:
    case rocsparse_coomm_alg_atomic:
    case rocsparse_coomm_alg_segmented:
    case rocsparse_coomm_alg_segmented_atomic:
    {
        return false;
    }
    }
    return true;
};

template <typename T, typename I, typename A>
rocsparse_status rocsparse_coomm_buffer_size_template_segmented(rocsparse_handle    handle,
                                                                rocsparse_operation trans_A,
                                                                I                   m,
                                                                I                   n,
                                                                I                   k,
                                                                int64_t             nnz,
                                                                I                   batch_count,
                                                                const rocsparse_mat_descr descr,
                                                                const A*                  coo_val,
                                                                const I* coo_row_ind,
                                                                const I* coo_col_ind,
                                                                size_t*  buffer_size);

template <typename T, typename I, typename A>
rocsparse_status rocsparse_coomm_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_coomm_alg       alg,
                                                      I                         m,
                                                      I                         n,
                                                      I                         k,
                                                      int64_t                   nnz,
                                                      I                         batch_count,
                                                      const rocsparse_mat_descr descr,
                                                      const A*                  coo_val,
                                                      const I*                  coo_row_ind,
                                                      const I*                  coo_col_ind,
                                                      size_t*                   buffer_size)
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

    // Logging
    log_trace(handle,
              "rocsparse_coomm_buffer_size",
              trans_A,
              alg,
              m,
              n,
              k,
              nnz,
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)buffer_size);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_requires_sorted_storage;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0 || batch_count < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    switch(alg)
    {
    case rocsparse_coomm_alg_default:
    case rocsparse_coomm_alg_atomic:
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    case rocsparse_coomm_alg_segmented:
    {
        return rocsparse_coomm_buffer_size_template_segmented<T>(handle,
                                                                 trans_A,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 nnz,
                                                                 batch_count,
                                                                 descr,
                                                                 coo_val,
                                                                 coo_row_ind,
                                                                 coo_col_ind,
                                                                 buffer_size);
    }

    case rocsparse_coomm_alg_segmented_atomic:
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename T, typename I, typename A>
rocsparse_status rocsparse_coomm_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_coomm_alg       alg,
                                                   I                         m,
                                                   I                         n,
                                                   I                         k,
                                                   int64_t                   nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const A*                  coo_val,
                                                   const I*                  coo_row_ind,
                                                   const I*                  coo_col_ind,
                                                   void*                     temp_buffer)
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

    // Logging
    log_trace(handle,
              "rocsparse_coomm_analysis",
              trans_A,
              alg,
              m,
              n,
              k,
              nnz,
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)temp_buffer);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_requires_sorted_storage;
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

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    switch(alg)
    {
    case rocsparse_coomm_alg_default:
    case rocsparse_coomm_alg_atomic:
    {
        return rocsparse_status_success;
    }

    case rocsparse_coomm_alg_segmented:
    {
        if(temp_buffer == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }

        return rocsparse_status_success;
    }

    case rocsparse_coomm_alg_segmented_atomic:
    {
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename T, typename I, typename A, typename B, typename C, typename U>
rocsparse_status rocsparse_coomm_template_atomic(rocsparse_handle          handle,
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
                                                 rocsparse_order           order_C);

template <typename T, typename I, typename A, typename B, typename C, typename U>
rocsparse_status rocsparse_coomm_template_segmented_atomic(rocsparse_handle    handle,
                                                           rocsparse_operation trans_A,
                                                           rocsparse_operation trans_B,
                                                           I                   m,
                                                           I                   n,
                                                           I                   k,
                                                           int64_t             nnz,
                                                           I                   batch_count_A,
                                                           int64_t             batch_stride_A,
                                                           U                   alpha_device_host,
                                                           const rocsparse_mat_descr descr,
                                                           const A*                  coo_val,
                                                           const I*                  coo_row_ind,
                                                           const I*                  coo_col_ind,
                                                           const B*                  dense_B,
                                                           int64_t                   ldb,
                                                           I                         batch_count_B,
                                                           int64_t                   batch_stride_B,
                                                           rocsparse_order           order_B,
                                                           U               beta_device_host,
                                                           C*              dense_C,
                                                           int64_t         ldc,
                                                           I               batch_count_C,
                                                           int64_t         batch_stride_C,
                                                           rocsparse_order order_C);

template <typename T, typename I, typename A, typename B, typename C, typename U>
rocsparse_status rocsparse_coomm_template_segmented(rocsparse_handle          handle,
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
                                                    rocsparse_order           order_C,
                                                    void*                     temp_buffer);

template <typename T, typename I, typename A, typename B, typename C, typename U>
rocsparse_status rocsparse_coomm_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_coomm_alg       alg,
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
                                                   rocsparse_order           order_C,
                                                   void*                     temp_buffer)
{
    if(trans_A == rocsparse_operation_none)
    {
        hipLaunchKernelGGL((coommnn_scale_kernel<256>),
                           dim3((int64_t(m) * n - 1) / 256 + 1, batch_count_C),
                           dim3(256),
                           0,
                           handle->stream,
                           m,
                           n,
                           beta_device_host,
                           dense_C,
                           ldc,
                           batch_stride_C,
                           order_C);
    }
    else
    {
        hipLaunchKernelGGL((coommnn_scale_kernel<256>),
                           dim3((int64_t(k) * n - 1) / 256 + 1, batch_count_C),
                           dim3(256),
                           0,
                           handle->stream,
                           k,
                           n,
                           beta_device_host,
                           dense_C,
                           ldc,
                           batch_stride_C,
                           order_C);
    }

    switch(alg)
    {
    case rocsparse_coomm_alg_default:
    case rocsparse_coomm_alg_atomic:
    {
        return rocsparse_coomm_template_atomic<T>(handle,
                                                  trans_A,
                                                  trans_B,
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
                                                  dense_B,
                                                  ldb,
                                                  batch_count_B,
                                                  batch_stride_B,
                                                  order_B,
                                                  beta_device_host,
                                                  dense_C,
                                                  ldc,
                                                  batch_count_C,
                                                  batch_stride_C,
                                                  order_C);
    }

    case rocsparse_coomm_alg_segmented:
    {
        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            if(temp_buffer == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }

            return rocsparse_coomm_template_segmented<T>(handle,
                                                         trans_A,
                                                         trans_B,
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
                                                         dense_B,
                                                         ldb,
                                                         batch_count_B,
                                                         batch_stride_B,
                                                         order_B,
                                                         beta_device_host,
                                                         dense_C,
                                                         ldc,
                                                         batch_count_C,
                                                         batch_stride_C,
                                                         order_C,
                                                         temp_buffer);
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            return rocsparse_coomm_template_atomic<T>(handle,
                                                      trans_A,
                                                      trans_B,
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
                                                      dense_B,
                                                      ldb,
                                                      batch_count_B,
                                                      batch_stride_B,
                                                      order_B,
                                                      beta_device_host,
                                                      dense_C,
                                                      ldc,
                                                      batch_count_C,
                                                      batch_stride_C,
                                                      order_C);
        }
        }
    }

    case rocsparse_coomm_alg_segmented_atomic:
    {
        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            return rocsparse_coomm_template_segmented_atomic<T>(handle,
                                                                trans_A,
                                                                trans_B,
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
                                                                dense_B,
                                                                ldb,
                                                                batch_count_B,
                                                                batch_stride_B,
                                                                order_B,
                                                                beta_device_host,
                                                                dense_C,
                                                                ldc,
                                                                batch_count_C,
                                                                batch_stride_C,
                                                                order_C);
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            return rocsparse_coomm_template_atomic<T>(handle,
                                                      trans_A,
                                                      trans_B,
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
                                                      dense_B,
                                                      ldb,
                                                      batch_count_B,
                                                      batch_stride_B,
                                                      order_B,
                                                      beta_device_host,
                                                      dense_C,
                                                      ldc,
                                                      batch_count_C,
                                                      batch_stride_C,
                                                      order_C);
        }
        }
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename T, typename I, typename A, typename B, typename C>
rocsparse_status rocsparse_coomm_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_coomm_alg       alg,
                                          I                         m,
                                          I                         n,
                                          I                         k,
                                          int64_t                   nnz,
                                          I                         batch_count_A,
                                          int64_t                   batch_stride_A,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const A*                  coo_val,
                                          const I*                  coo_row_ind,
                                          const I*                  coo_col_ind,
                                          const B*                  dense_B,
                                          int64_t                   ldb,
                                          I                         batch_count_B,
                                          int64_t                   batch_stride_B,
                                          rocsparse_order           order_B,
                                          const T*                  beta_device_host,
                                          C*                        dense_C,
                                          int64_t                   ldc,
                                          I                         batch_count_C,
                                          int64_t                   batch_stride_C,
                                          rocsparse_order           order_C,
                                          void*                     temp_buffer)
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
    log_trace(handle,
              replaceX<T>("rocsparse_Xcoomm"),
              trans_A,
              trans_B,
              alg,
              m,
              n,
              k,
              nnz,
              batch_count_A,
              batch_stride_A,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)dense_B,
              ldb,
              batch_count_B,
              batch_stride_B,
              order_B,
              LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
              (const void*&)dense_C,
              ldc,
              batch_count_C,
              batch_stride_C,
              order_C);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(order_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(order_C))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_requires_sorted_storage;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        // matrix never accessed however still need to update C matrix
        rocsparse_int Csize = (trans_A == rocsparse_operation_none) ? m * n : k * n;
        if(Csize > 0)
        {
            if(dense_C == nullptr && beta_device_host == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                hipLaunchKernelGGL((scale_array_2d<256>),
                                   dim3((Csize - 1) / 256 + 1, batch_count_C),
                                   dim3(256),
                                   0,
                                   handle->stream,
                                   (trans_A == rocsparse_operation_none) ? m : k,
                                   n,
                                   ldc,
                                   batch_stride_C,
                                   dense_C,
                                   beta_device_host,
                                   order_C);
            }
            else
            {
                hipLaunchKernelGGL((scale_array_2d<256>),
                                   dim3((Csize - 1) / 256 + 1, batch_count_C),
                                   dim3(256),
                                   0,
                                   handle->stream,
                                   (trans_A == rocsparse_operation_none) ? m : k,
                                   n,
                                   ldc,
                                   batch_stride_C,
                                   dense_C,
                                   *beta_device_host,
                                   order_C);
            }
        }

        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    if(alpha_device_host == nullptr || beta_device_host == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    if(dense_B == nullptr || dense_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check leading dimension of matrices
    static constexpr I s_one = static_cast<I>(1);
    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        // Check leading dimension of C
        if(ldc < std::max(s_one, ((order_C == rocsparse_order_column) ? m : n)))
        {
            return rocsparse_status_invalid_size;
        }

        // Check leading dimension of B
        switch(trans_B)
        {
        case rocsparse_operation_none:
        {
            if(ldb < std::max(s_one, ((order_B == rocsparse_order_column) ? k : n)))
            {
                return rocsparse_status_invalid_size;
            }
            break;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            if(ldb < std::max(s_one, ((order_B == rocsparse_order_column) ? n : k)))
            {
                return rocsparse_status_invalid_size;
            }
            break;
        }
        }
        break;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        // Check leading dimension of C
        if(ldc < std::max(s_one, ((order_C == rocsparse_order_column) ? k : n)))
        {
            return rocsparse_status_invalid_size;
        }

        switch(trans_B)
        {
        case rocsparse_operation_none:
        {
            if(ldb < std::max(s_one, ((order_B == rocsparse_order_column) ? m : n)))
            {
                return rocsparse_status_invalid_size;
            }
            break;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            if(ldb < std::max(s_one, ((order_B == rocsparse_order_column) ? n : m)))
            {
                return rocsparse_status_invalid_size;
            }
            break;
        }
        }
        break;
    }
    }

    // Check batch parameters of matrices
    bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        return rocsparse_status_invalid_value;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_coomm_template_dispatch<T>(handle,
                                                    trans_A,
                                                    trans_B,
                                                    alg,
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
                                                    dense_B,
                                                    ldb,
                                                    batch_count_B,
                                                    batch_stride_B,
                                                    order_B,
                                                    beta_device_host,
                                                    dense_C,
                                                    ldc,
                                                    batch_count_C,
                                                    batch_stride_C,
                                                    order_C,
                                                    temp_buffer);
    }
    else
    {
        return rocsparse_coomm_template_dispatch<T>(handle,
                                                    trans_A,
                                                    trans_B,
                                                    alg,
                                                    m,
                                                    n,
                                                    k,
                                                    nnz,
                                                    batch_count_A,
                                                    batch_stride_A,
                                                    *alpha_device_host,
                                                    descr,
                                                    coo_val,
                                                    coo_row_ind,
                                                    coo_col_ind,
                                                    dense_B,
                                                    ldb,
                                                    batch_count_B,
                                                    batch_stride_B,
                                                    order_B,
                                                    *beta_device_host,
                                                    dense_C,
                                                    ldc,
                                                    batch_count_C,
                                                    batch_stride_C,
                                                    order_C,
                                                    temp_buffer);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE_BUFFER_SIZE(TTYPE, ITYPE, ATYPE)                       \
    template rocsparse_status rocsparse_coomm_buffer_size_template<TTYPE>( \
        rocsparse_handle          handle,                                  \
        rocsparse_operation       trans_A,                                 \
        rocsparse_coomm_alg       alg,                                     \
        ITYPE                     m,                                       \
        ITYPE                     n,                                       \
        ITYPE                     k,                                       \
        int64_t                   nnz,                                     \
        ITYPE                     batch_count,                             \
        const rocsparse_mat_descr descr,                                   \
        const ATYPE*              coo_val,                                 \
        const ITYPE*              coo_row_ind,                             \
        const ITYPE*              coo_col_ind,                             \
        size_t*                   buffer_size);

// Uniform precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, float);
INSTANTIATE_BUFFER_SIZE(double, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, double);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_BUFFER_SIZE(int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int8_t);
#undef INSTANTIATE_BUFFER_SIZE

#define INSTANTIATE_ANALYSIS(TTYPE, ITYPE, ATYPE)                                     \
    template rocsparse_status rocsparse_coomm_analysis_template<TTYPE, ITYPE, ATYPE>( \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans_A,                                            \
        rocsparse_coomm_alg       alg,                                                \
        ITYPE                     m,                                                  \
        ITYPE                     n,                                                  \
        ITYPE                     k,                                                  \
        int64_t                   nnz,                                                \
        const rocsparse_mat_descr descr,                                              \
        const ATYPE*              coo_val,                                            \
        const ITYPE*              coo_row_ind,                                        \
        const ITYPE*              coo_col_ind,                                        \
        void*                     temp_buffer);

// Uniform precisions
INSTANTIATE_ANALYSIS(float, int32_t, float);
INSTANTIATE_ANALYSIS(float, int64_t, float);
INSTANTIATE_ANALYSIS(double, int32_t, double);
INSTANTIATE_ANALYSIS(double, int64_t, double);
INSTANTIATE_ANALYSIS(rocsparse_float_complex, int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(rocsparse_float_complex, int64_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(rocsparse_double_complex, int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(rocsparse_double_complex, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_ANALYSIS(int32_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(int32_t, int64_t, int8_t);
INSTANTIATE_ANALYSIS(float, int32_t, int8_t);
INSTANTIATE_ANALYSIS(float, int64_t, int8_t);
#undef INSTANTIATE_ANALYSIS

#define INSTANTIATE(TTYPE, ITYPE, ATYPE, BTYPE, CTYPE)                                              \
    template rocsparse_status rocsparse_coomm_template(rocsparse_handle          handle,            \
                                                       rocsparse_operation       trans_A,           \
                                                       rocsparse_operation       trans_B,           \
                                                       rocsparse_coomm_alg       alg,               \
                                                       ITYPE                     m,                 \
                                                       ITYPE                     n,                 \
                                                       ITYPE                     k,                 \
                                                       int64_t                   nnz,               \
                                                       ITYPE                     batch_count_A,     \
                                                       int64_t                   batch_stride_A,    \
                                                       const TTYPE*              alpha_device_host, \
                                                       const rocsparse_mat_descr descr,             \
                                                       const ATYPE*              coo_val,           \
                                                       const ITYPE*              coo_row_ind,       \
                                                       const ITYPE*              coo_col_ind,       \
                                                       const BTYPE*              B,                 \
                                                       int64_t                   ldb,               \
                                                       ITYPE                     batch_count_B,     \
                                                       int64_t                   batch_stride_B,    \
                                                       rocsparse_order           order_B,           \
                                                       const TTYPE*              beta_device_host,  \
                                                       CTYPE*                    C,                 \
                                                       int64_t                   ldc,               \
                                                       ITYPE                     batch_count_C,     \
                                                       int64_t                   batch_stride_C,    \
                                                       rocsparse_order           order_C,           \
                                                       void*                     temp_buffer);

// Uniform precisions
INSTANTIATE(float, int32_t, float, float, float);
INSTANTIATE(float, int64_t, float, float, float);
INSTANTIATE(double, int32_t, double, double, double);
INSTANTIATE(double, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed precisions
INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int8_t, int8_t, float);
#undef INSTANTIATE
