/*! \file */
/* ************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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
#include "definitions.h"
#include "utility.h"

// Scale kernel for beta != 1.0
template <unsigned int BLOCKSIZE, typename I, typename T>
static ROCSPARSE_DEVICE_ILF void coommnn_scale_device(
    I m, I n, T beta, T* __restrict__ data, I ld, I stride, rocsparse_order order)
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
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void coommnn_scale_kernel(
    I m, I n, U beta_device_host, T* __restrict__ data, I ld, I stride, rocsparse_order order)
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

template <typename I, typename T>
rocsparse_status rocsparse_coomm_buffer_size_template_segmented(rocsparse_handle    handle,
                                                                rocsparse_operation trans_A,
                                                                I                   m,
                                                                I                   n,
                                                                I                   k,
                                                                I                   nnz,
                                                                I                   batch_count,
                                                                const rocsparse_mat_descr descr,
                                                                const T*                  coo_val,
                                                                const I* coo_row_ind,
                                                                const I* coo_col_ind,
                                                                size_t*  buffer_size);

template <typename I, typename T>
rocsparse_status rocsparse_coomm_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_coomm_alg       alg,
                                                      I                         m,
                                                      I                         n,
                                                      I                         k,
                                                      I                         nnz,
                                                      I                         batch_count,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  coo_val,
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

    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
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
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0 || batch_count < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
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
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    case rocsparse_coomm_alg_segmented:
    {
        return rocsparse_coomm_buffer_size_template_segmented(handle,
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
        *buffer_size = 4;
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename I, typename T>
rocsparse_status rocsparse_coomm_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_coomm_alg       alg,
                                                   I                         m,
                                                   I                         n,
                                                   I                         k,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  coo_val,
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

    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
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
        return rocsparse_status_not_implemented;
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

    if(temp_buffer == nullptr)
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
        return rocsparse_status_success;
    }

    case rocsparse_coomm_alg_segmented:
    {
        return rocsparse_status_success;
    }

    case rocsparse_coomm_alg_segmented_atomic:
    {
        return rocsparse_status_success;
    }
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
                                                 I                         batch_stride_C);

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_segmented_atomic(rocsparse_handle    handle,
                                                           rocsparse_operation trans_A,
                                                           rocsparse_operation trans_B,
                                                           rocsparse_order     order,
                                                           I                   m,
                                                           I                   n,
                                                           I                   k,
                                                           I                   nnz,
                                                           I                   batch_count_A,
                                                           I                   batch_stride_A,
                                                           U                   alpha_device_host,
                                                           const rocsparse_mat_descr descr,
                                                           const T*                  coo_val,
                                                           const I*                  coo_row_ind,
                                                           const I*                  coo_col_ind,
                                                           const T*                  B,
                                                           I                         ldb,
                                                           I                         batch_count_B,
                                                           I                         batch_stride_B,
                                                           U  beta_device_host,
                                                           T* C,
                                                           I  ldc,
                                                           I  batch_count_C,
                                                           I  batch_stride_C);

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_segmented(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    rocsparse_order           order,
                                                    I                         m,
                                                    I                         n,
                                                    I                         k,
                                                    I                         nnz,
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
                                                    I                         batch_stride_C,
                                                    void*                     temp_buffer);

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_order           order,
                                                   rocsparse_coomm_alg       alg,
                                                   I                         m,
                                                   I                         n,
                                                   I                         k,
                                                   I                         nnz,
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
                                                   I                         batch_stride_C,
                                                   void*                     temp_buffer)
{
    // Scale C with beta
    hipLaunchKernelGGL((coommnn_scale_kernel<256>),
                       dim3((m * n - 1) / 256 + 1, batch_count_C),
                       dim3(256),
                       0,
                       handle->stream,
                       m,
                       n,
                       beta_device_host,
                       C,
                       ldc,
                       batch_stride_C,
                       order);

    switch(alg)
    {
    case rocsparse_coomm_alg_default:
    case rocsparse_coomm_alg_atomic:
    {
        return rocsparse_coomm_template_atomic(handle,
                                               trans_A,
                                               trans_B,
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

    case rocsparse_coomm_alg_segmented:
    {
        return rocsparse_coomm_template_segmented(handle,
                                                  trans_A,
                                                  trans_B,
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
                                                  batch_stride_C,
                                                  temp_buffer);
    }

    case rocsparse_coomm_alg_segmented_atomic:
    {
        return rocsparse_coomm_template_segmented_atomic(handle,
                                                         trans_A,
                                                         trans_B,
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
    }
    return rocsparse_status_invalid_value;
}

template <typename I, typename T>
rocsparse_status rocsparse_coomm_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_order           order_B,
                                          rocsparse_order           order_C,
                                          rocsparse_coomm_alg       alg,
                                          I                         m,
                                          I                         n,
                                          I                         k,
                                          I                         nnz,
                                          I                         batch_count_A,
                                          I                         batch_stride_A,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  coo_val,
                                          const I*                  coo_row_ind,
                                          const I*                  coo_col_ind,
                                          const T*                  B,
                                          I                         ldb,
                                          I                         batch_count_B,
                                          I                         batch_stride_B,
                                          const T*                  beta_device_host,
                                          T*                        C,
                                          I                         ldc,
                                          I                         batch_count_C,
                                          I                         batch_stride_C,
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
              order_B,
              order_C,
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
              (const void*&)B,
              ldb,
              batch_count_B,
              batch_stride_B,
              LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
              (const void*&)C,
              ldc,
              batch_count_C,
              batch_stride_C);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
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
    if(B == nullptr || C == nullptr)
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
        return rocsparse_coomm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
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
                                                 B,
                                                 ldb,
                                                 batch_count_B,
                                                 batch_stride_B,
                                                 beta_device_host,
                                                 C,
                                                 ldc,
                                                 batch_count_C,
                                                 batch_stride_C,
                                                 temp_buffer);
    }
    else
    {
        return rocsparse_coomm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
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
                                                 B,
                                                 ldb,
                                                 batch_count_B,
                                                 batch_stride_B,
                                                 *beta_device_host,
                                                 C,
                                                 ldc,
                                                 batch_count_C,
                                                 batch_stride_C,
                                                 temp_buffer);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                                 \
    template rocsparse_status rocsparse_coomm_buffer_size_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                         \
        rocsparse_operation       trans_A,                                        \
        rocsparse_coomm_alg       alg,                                            \
        ITYPE                     m,                                              \
        ITYPE                     n,                                              \
        ITYPE                     k,                                              \
        ITYPE                     nnz,                                            \
        ITYPE                     batch_count,                                    \
        const rocsparse_mat_descr descr,                                          \
        const TTYPE*              coo_val,                                        \
        const ITYPE*              coo_row_ind,                                    \
        const ITYPE*              coo_col_ind,                                    \
        size_t*                   buffer_size);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                                              \
    template rocsparse_status rocsparse_coomm_analysis_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                      \
        rocsparse_operation       trans_A,                                     \
        rocsparse_coomm_alg       alg,                                         \
        ITYPE                     m,                                           \
        ITYPE                     n,                                           \
        ITYPE                     k,                                           \
        ITYPE                     nnz,                                         \
        const rocsparse_mat_descr descr,                                       \
        const TTYPE*              coo_val,                                     \
        const ITYPE*              coo_row_ind,                                 \
        const ITYPE*              coo_col_ind,                                 \
        void*                     temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                                     \
    template rocsparse_status rocsparse_coomm_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                             \
        rocsparse_operation       trans_A,                            \
        rocsparse_operation       trans_B,                            \
        rocsparse_order           order_B,                            \
        rocsparse_order           order_C,                            \
        rocsparse_coomm_alg       alg,                                \
        ITYPE                     m,                                  \
        ITYPE                     n,                                  \
        ITYPE                     k,                                  \
        ITYPE                     nnz,                                \
        ITYPE                     batch_count_A,                      \
        ITYPE                     batch_stride_A,                     \
        const TTYPE*              alpha_device_host,                  \
        const rocsparse_mat_descr descr,                              \
        const TTYPE*              coo_val,                            \
        const ITYPE*              coo_row_ind,                        \
        const ITYPE*              coo_col_ind,                        \
        const TTYPE*              B,                                  \
        ITYPE                     ldb,                                \
        ITYPE                     batch_count_B,                      \
        ITYPE                     batch_stride_B,                     \
        const TTYPE*              beta_device_host,                   \
        TTYPE*                    C,                                  \
        ITYPE                     ldc,                                \
        ITYPE                     batch_count_C,                      \
        ITYPE                     batch_stride_C,                     \
        void*                     temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE
