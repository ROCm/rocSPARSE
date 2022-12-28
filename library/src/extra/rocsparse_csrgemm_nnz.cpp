/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../conversion/rocsparse_identity.hpp"
#include "csrgemm_device.h"
#include "definitions.h"
#include "rocsparse_csrgemm.hpp"
#include "utility.h"

#include <rocprim/rocprim.hpp>

template <unsigned int BLOCKSIZE, typename I, typename J>
ROCSPARSE_KERNEL(BLOCKSIZE)
void csrgemm_set_base(I size, J* __restrict__ out, rocsparse_index_base idx_base_out)
{
    I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    if(idx >= size)
    {
        return;
    }

    out[idx] = idx_base_out;
}

template <typename I, typename J, typename T>
static inline rocsparse_status
    rocsparse_csrgemm_multadd_buffer_size_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   J                         m,
                                                   J                         n,
                                                   J                         k,
                                                   const T*                  alpha,
                                                   const rocsparse_mat_descr descr_A,
                                                   I                         nnz_A,
                                                   const I*                  csr_row_ptr_A,
                                                   const J*                  csr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   I                         nnz_B,
                                                   const I*                  csr_row_ptr_B,
                                                   const J*                  csr_col_ind_B,
                                                   const T*                  beta,
                                                   const rocsparse_mat_descr descr_D,
                                                   I                         nnz_D,
                                                   const I*                  csr_row_ptr_D,
                                                   const J*                  csr_col_ind_D,
                                                   rocsparse_mat_info        info_C,
                                                   size_t*                   buffer_size)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_internal_error;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || descr_B == nullptr || descr_D == nullptr || buffer_size == nullptr
       || alpha == nullptr || beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_D == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(k > 0 && csr_row_ptr_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && csr_col_ind_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_B != 0 && csr_col_ind_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_D != 0 && csr_col_ind_D == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_B->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_D->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Quick return if possible

    // m == 0 || n == 0 - do nothing
    if(m == 0 || n == 0)
    {
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    // k == 0 || nnz_A == 0 || nnz_B == 0 - scale D with beta
    if(k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_csrgemm_scal_buffer_size_template(
            handle, m, n, beta, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, buffer_size);
    }

    if((trans_A != rocsparse_operation_none) || (trans_B != rocsparse_operation_none))
    {
        return rocsparse_status_not_implemented;
    }

    // nnz_D == 0 - compute alpha * A * B
    if(nnz_D == 0)
    {
        return rocsparse_csrgemm_mult_buffer_size_template(handle,
                                                           trans_A,
                                                           trans_B,
                                                           m,
                                                           n,
                                                           k,
                                                           alpha,
                                                           descr_A,
                                                           nnz_A,
                                                           csr_row_ptr_A,
                                                           csr_col_ind_A,
                                                           descr_B,
                                                           nnz_B,
                                                           csr_row_ptr_B,
                                                           csr_col_ind_B,
                                                           info_C,
                                                           buffer_size);
    }

    // Stream
    hipStream_t stream = handle->stream;

    // rocprim buffer
    size_t rocprim_size;
    size_t rocprim_max = 0;

    // rocprim::reduce
    RETURN_IF_HIP_ERROR(rocprim::reduce(
        nullptr, rocprim_size, csr_row_ptr_A, &nnz_A, 0, m, rocprim::maximum<I>(), stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    // rocprim exclusive scan
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
        nullptr, rocprim_size, csr_row_ptr_A, &nnz_A, 0, m + 1, rocprim::plus<I>(), stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    // rocprim::radix_sort_pairs
    rocprim::double_buffer<I> buf1(&nnz_A, &nnz_B);
    rocprim::double_buffer<J> buf2(&n, &k);
    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, rocprim_size, buf1, buf2, m, 0, 3, stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    *buffer_size = ((rocprim_max - 1) / 256 + 1) * 256;

    // Group arrays
    *buffer_size += sizeof(J) * 256 * CSRGEMM_MAXGROUPS;
    *buffer_size += sizeof(J) * 256;
    *buffer_size += ((sizeof(J) * m - 1) / 256 + 1) * 256;

    // Permutation arrays
    *buffer_size += ((sizeof(J) * m - 1) / 256 + 1) * 256;
    *buffer_size += ((sizeof(J) * m - 1) / 256 + 1) * 256;
    *buffer_size += ((sizeof(I) * m - 1) / 256 + 1) * 256;

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
static inline rocsparse_status
    rocsparse_csrgemm_mult_buffer_size_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                J                         m,
                                                J                         n,
                                                J                         k,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr_A,
                                                I                         nnz_A,
                                                const I*                  csr_row_ptr_A,
                                                const J*                  csr_col_ind_A,
                                                const rocsparse_mat_descr descr_B,
                                                I                         nnz_B,
                                                const I*                  csr_row_ptr_B,
                                                const J*                  csr_col_ind_B,
                                                rocsparse_mat_info        info_C,
                                                size_t*                   buffer_size)
{

    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_internal_error;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || descr_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(k > 0 && csr_row_ptr_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(buffer_size == nullptr || alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && csr_col_ind_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_B != 0 && csr_col_ind_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_B->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;

        return rocsparse_status_success;
    }

    if((trans_A != rocsparse_operation_none) || (trans_B != rocsparse_operation_none))
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // rocprim buffer
    size_t rocprim_size;
    size_t rocprim_max = 0;

    // rocprim::reduce
    RETURN_IF_HIP_ERROR(rocprim::reduce(
        nullptr, rocprim_size, csr_row_ptr_A, &nnz_A, 0, m, rocprim::maximum<I>(), stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    // rocprim exclusive scan
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
        nullptr, rocprim_size, csr_row_ptr_A, &nnz_A, 0, m + 1, rocprim::plus<I>(), stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    // rocprim::radix_sort_pairs
    rocprim::double_buffer<I> buf1(&nnz_A, &nnz_B);
    rocprim::double_buffer<J> buf2(&n, &k);
    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, rocprim_size, buf1, buf2, m, 0, 3, stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    *buffer_size = ((rocprim_max - 1) / 256 + 1) * 256;

    // Group arrays
    *buffer_size += sizeof(J) * 256 * CSRGEMM_MAXGROUPS;
    *buffer_size += sizeof(J) * 256;
    *buffer_size += ((sizeof(J) * m - 1) / 256 + 1) * 256;

    // Permutation arrays
    *buffer_size += ((sizeof(J) * m - 1) / 256 + 1) * 256;
    *buffer_size += ((sizeof(J) * m - 1) / 256 + 1) * 256;
    *buffer_size += ((sizeof(I) * m - 1) / 256 + 1) * 256;

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
static inline rocsparse_status
    rocsparse_csrgemm_scal_buffer_size_template(rocsparse_handle          handle,
                                                J                         m,
                                                J                         n,
                                                const T*                  beta,
                                                const rocsparse_mat_descr descr_D,
                                                I                         nnz_D,
                                                const I*                  csr_row_ptr_D,
                                                const J*                  csr_col_ind_D,
                                                rocsparse_mat_info        info_C,
                                                size_t*                   buffer_size)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_internal_error;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_D == nullptr || buffer_size == nullptr || beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_D == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_D != 0 && csr_col_ind_D == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix type
    if(descr_D->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // No buffer requirements for matrix scaling
    *buffer_size = 4;

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrgemm_buffer_size_template(rocsparse_handle          handle,
                                                        rocsparse_operation       trans_A,
                                                        rocsparse_operation       trans_B,
                                                        J                         m,
                                                        J                         n,
                                                        J                         k,
                                                        const T*                  alpha,
                                                        const rocsparse_mat_descr descr_A,
                                                        I                         nnz_A,
                                                        const I*                  csr_row_ptr_A,
                                                        const J*                  csr_col_ind_A,
                                                        const rocsparse_mat_descr descr_B,
                                                        I                         nnz_B,
                                                        const I*                  csr_row_ptr_B,
                                                        const J*                  csr_col_ind_B,
                                                        const T*                  beta,
                                                        const rocsparse_mat_descr descr_D,
                                                        I                         nnz_D,
                                                        const I*                  csr_row_ptr_D,
                                                        const J*                  csr_col_ind_D,
                                                        rocsparse_mat_info        info_C,
                                                        size_t*                   buffer_size)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check for valid rocsparse_mat_info
    if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrgemm_buffer_size"),
              trans_A,
              trans_B,
              m,
              n,
              k,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr_A,
              nnz_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              (const void*&)descr_B,
              nnz_B,
              (const void*&)csr_row_ptr_B,
              (const void*&)csr_col_ind_B,
              LOG_TRACE_SCALAR_VALUE(handle, beta),
              (const void*&)descr_D,
              nnz_D,
              (const void*&)csr_row_ptr_D,
              (const void*&)csr_col_ind_D,
              (const void*&)info_C,
              (const void*&)buffer_size);

    // Check operation
    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Clear csrgemm info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrgemm_info(info_C->csrgemm_info));

    // Create csrgemm info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrgemm_info(&info_C->csrgemm_info));

    // Set info parameters
    info_C->csrgemm_info->mul = (alpha != nullptr);
    info_C->csrgemm_info->add = (beta != nullptr);

    if(info_C->csrgemm_info->add)
    {
        if(nnz_D == 0)
        {
            info_C->csrgemm_info->add = false;
            if(false == info_C->csrgemm_info->mul)
            {
                *buffer_size = 4;
                return rocsparse_status_success;
            }
        }
    }

    if(m == 0)
    {
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    // Either alpha or beta can be nullptr
    if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == true)
    {
        return rocsparse_csrgemm_multadd_buffer_size_template(handle,
                                                              trans_A,
                                                              trans_B,
                                                              m,
                                                              n,
                                                              k,
                                                              alpha,
                                                              descr_A,
                                                              nnz_A,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              descr_B,
                                                              nnz_B,
                                                              csr_row_ptr_B,
                                                              csr_col_ind_B,
                                                              beta,
                                                              descr_D,
                                                              nnz_D,
                                                              csr_row_ptr_D,
                                                              csr_col_ind_D,
                                                              info_C,
                                                              buffer_size);
    }
    else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == false)
    {
        return rocsparse_csrgemm_mult_buffer_size_template(handle,
                                                           trans_A,
                                                           trans_B,
                                                           m,
                                                           n,
                                                           k,
                                                           alpha,
                                                           descr_A,
                                                           nnz_A,
                                                           csr_row_ptr_A,
                                                           csr_col_ind_A,
                                                           descr_B,
                                                           nnz_B,
                                                           csr_row_ptr_B,
                                                           csr_col_ind_B,
                                                           info_C,
                                                           buffer_size);
    }
    else if(info_C->csrgemm_info->mul == false && info_C->csrgemm_info->add == true)
    {
        return rocsparse_csrgemm_scal_buffer_size_template(
            handle, m, n, beta, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, buffer_size);
    }
    else
    {
        // alpha == nullptr && beta == nullptr
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_success;
}

template <typename I, typename J>
static inline rocsparse_status rocsparse_csrgemm_nnz_scal(rocsparse_handle          handle,
                                                          J                         m,
                                                          J                         n,
                                                          const rocsparse_mat_descr descr_D,
                                                          I                         nnz_D,
                                                          const I*                  csr_row_ptr_D,
                                                          const J*                  csr_col_ind_D,
                                                          const rocsparse_mat_descr descr_C,
                                                          I*                        csr_row_ptr_C,
                                                          I*                        nnz_C,
                                                          const rocsparse_mat_info  info_C,
                                                          void*                     temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_D == nullptr || descr_C == nullptr || nnz_C == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_D == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_D != 0 && csr_col_ind_D == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix type
    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_D->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), stream));
        }
        else
        {
            *nnz_C = 0;
        }
        if(m > 0)
        {
#define CSRGEMM_DIM 1024
            hipLaunchKernelGGL((csrgemm_set_base<CSRGEMM_DIM>),
                               dim3((m + 1) / CSRGEMM_DIM + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               m + 1,
                               csr_row_ptr_C,
                               descr_C->base);
#undef CSRGEMM_DIM
        }
        return rocsparse_status_success;
    }

    // When scaling a matrix, nnz of C will always be equal to nnz of D
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(nnz_C, &nnz_D, sizeof(I), hipMemcpyHostToDevice, stream));
    }
    else
    {
        *nnz_C = nnz_D;
    }
    // Copy row pointers
#define CSRGEMM_DIM 1024
    hipLaunchKernelGGL((csrgemm_copy<CSRGEMM_DIM>),
                       dim3(m / CSRGEMM_DIM + 1),
                       dim3(CSRGEMM_DIM),
                       0,
                       stream,
                       m + 1,
                       csr_row_ptr_D,
                       csr_row_ptr_C,
                       descr_D->base,
                       descr_C->base);
#undef CSRGEMM_DIM

    return rocsparse_status_success;
}

template <typename I, typename J>
static inline rocsparse_status rocsparse_csrgemm_nnz_calc(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          rocsparse_operation       trans_B,
                                                          J                         m,
                                                          J                         n,
                                                          J                         k,
                                                          const rocsparse_mat_descr descr_A,
                                                          I                         nnz_A,
                                                          const I*                  csr_row_ptr_A,
                                                          const J*                  csr_col_ind_A,
                                                          const rocsparse_mat_descr descr_B,
                                                          I                         nnz_B,
                                                          const I*                  csr_row_ptr_B,
                                                          const J*                  csr_col_ind_B,
                                                          const rocsparse_mat_descr descr_D,
                                                          I                         nnz_D,
                                                          const I*                  csr_row_ptr_D,
                                                          const J*                  csr_col_ind_D,
                                                          const rocsparse_mat_descr descr_C,
                                                          I*                        csr_row_ptr_C,
                                                          I*                        nnz_C,
                                                          const rocsparse_mat_info  info_C,
                                                          void*                     temp_buffer)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Index base
    rocsparse_index_base base_A
        = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
    rocsparse_index_base base_B
        = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
    rocsparse_index_base base_D = info_C->csrgemm_info->add
                                      ? ((descr_D) ? descr_D->base : rocsparse_index_base_zero)
                                      : rocsparse_index_base_zero;

    bool mul = info_C->csrgemm_info->mul;
    bool add = info_C->csrgemm_info->add;

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // rocprim buffer
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Compute number of intermediate products for each row
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
    hipLaunchKernelGGL((csrgemm_intermediate_products<CSRGEMM_DIM, CSRGEMM_SUB>),
                       dim3((m - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                       dim3(CSRGEMM_DIM),
                       0,
                       stream,
                       m,
                       csr_row_ptr_A,
                       csr_col_ind_A,
                       csr_row_ptr_B,
                       csr_row_ptr_D,
                       csr_row_ptr_C,
                       base_A,
                       mul,
                       add);
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

    // Determine maximum of all intermediate products
    RETURN_IF_HIP_ERROR(rocprim::reduce(nullptr,
                                        rocprim_size,
                                        csr_row_ptr_C,
                                        csr_row_ptr_C + m,
                                        0,
                                        m,
                                        rocprim::maximum<I>(),
                                        stream));
    rocprim_buffer = reinterpret_cast<void*>(buffer);
    RETURN_IF_HIP_ERROR(rocprim::reduce(rocprim_buffer,
                                        rocprim_size,
                                        csr_row_ptr_C,
                                        csr_row_ptr_C + m,
                                        0,
                                        m,
                                        rocprim::maximum<I>(),
                                        stream));

    I int_max;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&int_max, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToHost, stream));
    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Group offset buffer
    J* d_group_offset = reinterpret_cast<J*>(buffer);
    buffer += sizeof(J) * 256;

    // Group size buffer
    J h_group_size[CSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(J) * CSRGEMM_MAXGROUPS);

    // Permutation array
    J* d_perm = nullptr;

    // If maximum of intermediate products exceeds 32, we process the rows in groups of
    // similar sized intermediate products
    if(int_max > 32)
    {
        // Group size buffer
        J* d_group_size = reinterpret_cast<J*>(buffer);
        buffer += sizeof(J) * 256 * CSRGEMM_MAXGROUPS;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        hipLaunchKernelGGL((csrgemm_group_reduce_part1<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
                           dim3(CSRGEMM_DIM),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_C,
                           d_group_size);

        hipLaunchKernelGGL((csrgemm_group_reduce_part3<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
                           dim3(1),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           d_group_size);
#undef CSRGEMM_DIM

        // Exclusive sum to obtain group offsets
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<J>(),
                                                    stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<J>(),
                                                    stream));

        // Copy group sizes to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_group_size,
                                           d_group_size,
                                           sizeof(J) * CSRGEMM_MAXGROUPS,
                                           hipMemcpyDeviceToHost,
                                           stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // Permutation temporary arrays
        J* tmp_vals = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        J* tmp_perm = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        I* tmp_keys = reinterpret_cast<I*>(buffer);
        buffer += ((sizeof(I) * m - 1) / 256 + 1) * 256;

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_create_identity_permutation_template(handle, m, tmp_perm));

        rocprim::double_buffer<I> d_keys(csr_row_ptr_C, tmp_keys);
        rocprim::double_buffer<J> d_vals(tmp_perm, tmp_vals);

        // Sort pairs (by groups)
        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, rocprim_size, d_keys, d_vals, m, 0, 3, stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            rocprim_buffer, rocprim_size, d_keys, d_vals, m, 0, 3, stream));

        d_perm = d_vals.current();

        // Release tmp_keys buffer
        buffer -= ((sizeof(I) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(J), stream));
    }

    // Compute non-zero entries per row for each group

    // Group 0: 0 - 32 intermediate products
    if(h_group_size[0] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 4
#define CSRGEMM_HASHSIZE 32
        hipLaunchKernelGGL(
            (csrgemm_nnz_wf_per_row<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_HASHSIZE, CSRGEMM_NNZ_HASH>),
            dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[0],
            &d_group_offset[0],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            mul,
            add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 1: 33 - 64 intermediate products
    if(h_group_size[1] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 64
        hipLaunchKernelGGL(
            (csrgemm_nnz_wf_per_row<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_HASHSIZE, CSRGEMM_NNZ_HASH>),
            dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[1],
            &d_group_offset[1],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            mul,
            add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 2: 65 - 512 intermediate products
    if(h_group_size[2] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 512
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                      CSRGEMM_SUB,
                                                      CSRGEMM_HASHSIZE,
                                                      CSRGEMM_NNZ_HASH>),
                           dim3(h_group_size[2]),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           &d_group_offset[2],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           base_A,
                           base_B,
                           base_D,
                           info_C->csrgemm_info->mul,
                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 3: 513 - 1024 intermediate products
    if(h_group_size[3] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 1024
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                      CSRGEMM_SUB,
                                                      CSRGEMM_HASHSIZE,
                                                      CSRGEMM_NNZ_HASH>),
                           dim3(h_group_size[3]),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           &d_group_offset[3],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           base_A,
                           base_B,
                           base_D,
                           info_C->csrgemm_info->mul,
                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 4: 1025 - 2048 intermediate products
    if(h_group_size[4] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 2048
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                      CSRGEMM_SUB,
                                                      CSRGEMM_HASHSIZE,
                                                      CSRGEMM_NNZ_HASH>),
                           dim3(h_group_size[4]),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           &d_group_offset[4],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           base_A,
                           base_B,
                           base_D,
                           info_C->csrgemm_info->mul,
                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 5: 2049 - 4096 intermediate products
    if(h_group_size[5] > 0)
    {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 4096
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                      CSRGEMM_SUB,
                                                      CSRGEMM_HASHSIZE,
                                                      CSRGEMM_NNZ_HASH>),
                           dim3(h_group_size[5]),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           &d_group_offset[5],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           base_A,
                           base_B,
                           base_D,
                           info_C->csrgemm_info->mul,
                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 6: 4097 - 8192 intermediate products
    if(h_group_size[6] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 8192
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                      CSRGEMM_SUB,
                                                      CSRGEMM_HASHSIZE,
                                                      CSRGEMM_NNZ_HASH>),
                           dim3(h_group_size[6]),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           &d_group_offset[6],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           base_A,
                           base_B,
                           base_D,
                           info_C->csrgemm_info->mul,
                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 7: more than 8192 intermediate products
    if(h_group_size[7] > 0)
    {
        // Matrices B and D must be sorted in order to run this path
        if(descr_B->storage_mode == rocsparse_storage_mode_unsorted
           || (info_C->csrgemm_info->add ? descr_D->storage_mode == rocsparse_storage_mode_unsorted
                                         : false))
        {
            return rocsparse_status_requires_sorted_storage;
        }

#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
        I* workspace_B = nullptr;

        if(info_C->csrgemm_info->mul == true)
        {
            // Allocate additional buffer for C = alpha * A * B
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync((void**)&workspace_B, sizeof(I) * nnz_A, handle->stream));
        }

        hipLaunchKernelGGL(
            (csrgemm_nnz_block_per_row_multipass<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_CHUNKSIZE>),
            dim3(h_group_size[7]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            n,
            &d_group_offset[7],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            workspace_B,
            base_A,
            base_B,
            base_D,
            mul,
            add);

        if(info_C->csrgemm_info->mul == true)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace_B, handle->stream));
        }
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Exclusive sum to obtain row pointers of C
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                rocprim_size,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                static_cast<rocsparse_int>(descr_C->base),
                                                m + 1,
                                                rocprim::plus<I>(),
                                                stream));
    rocprim_buffer = reinterpret_cast<void*>(buffer);
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                rocprim_size,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                static_cast<rocsparse_int>(descr_C->base),
                                                m + 1,
                                                rocprim::plus<I>(),
                                                stream));

    // Store nnz of C
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(nnz_C, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToDevice, stream));

        // Adjust nnz by index base
        if(descr_C->base == rocsparse_index_base_one)
        {
            hipLaunchKernelGGL((csrgemm_index_base<1>), dim3(1), dim3(1), 0, stream, nnz_C);
        }
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz_C, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        // Adjust nnz by index base
        *nnz_C -= descr_C->base;
    }

    return rocsparse_status_success;
}

template <typename I, typename J>
static inline rocsparse_status rocsparse_csrgemm_nnz_multadd(rocsparse_handle          handle,
                                                             rocsparse_operation       trans_A,
                                                             rocsparse_operation       trans_B,
                                                             J                         m,
                                                             J                         n,
                                                             J                         k,
                                                             const rocsparse_mat_descr descr_A,
                                                             I                         nnz_A,
                                                             const I* csr_row_ptr_A,
                                                             const J* csr_col_ind_A,
                                                             const rocsparse_mat_descr descr_B,
                                                             I                         nnz_B,
                                                             const I* csr_row_ptr_B,
                                                             const J* csr_col_ind_B,
                                                             const rocsparse_mat_descr descr_D,
                                                             I                         nnz_D,
                                                             const I* csr_row_ptr_D,
                                                             const J* csr_col_ind_D,
                                                             const rocsparse_mat_descr descr_C,
                                                             I*                       csr_row_ptr_C,
                                                             I*                       nnz_C,
                                                             const rocsparse_mat_info info_C,
                                                             void*                    temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || descr_B == nullptr || descr_D == nullptr || descr_C == nullptr
       || nnz_C == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(k > 0 && csr_row_ptr_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_D == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && csr_col_ind_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_B != 0 && csr_col_ind_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_D != 0 && csr_col_ind_D == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    if(descr_B->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    if(descr_D->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Quick return if possible

    // m == 0 || n == 0 - do nothing
    if(m == 0 || n == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), stream));
        }
        else
        {
            *nnz_C = 0;
        }

        return rocsparse_status_success;
    }

    // k == 0 || nnz_A == 0 || nnz_B == 0 - scale D with beta
    if(k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_csrgemm_nnz_scal(handle,
                                          m,
                                          n,
                                          descr_D,
                                          nnz_D,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          descr_C,
                                          csr_row_ptr_C,
                                          nnz_C,
                                          info_C,
                                          temp_buffer);
    }

    if((trans_A != rocsparse_operation_none) || (trans_B != rocsparse_operation_none))
    {
        return rocsparse_status_not_implemented;
    }

    // nnz_D == 0 - compute alpha * A * B
    if(nnz_D == 0)
    {
        return rocsparse_csrgemm_nnz_mult(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          descr_A,
                                          nnz_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          descr_C,
                                          csr_row_ptr_C,
                                          nnz_C,
                                          info_C,
                                          temp_buffer);
    }

    // Perform nnz calculation
    return rocsparse_csrgemm_nnz_calc(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      descr_A,
                                      nnz_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      descr_D,
                                      nnz_D,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      descr_C,
                                      csr_row_ptr_C,
                                      nnz_C,
                                      info_C,
                                      temp_buffer);
}

template <typename I, typename J>
static inline rocsparse_status rocsparse_csrgemm_nnz_mult(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          rocsparse_operation       trans_B,
                                                          J                         m,
                                                          J                         n,
                                                          J                         k,
                                                          const rocsparse_mat_descr descr_A,
                                                          I                         nnz_A,
                                                          const I*                  csr_row_ptr_A,
                                                          const J*                  csr_col_ind_A,
                                                          const rocsparse_mat_descr descr_B,
                                                          I                         nnz_B,
                                                          const I*                  csr_row_ptr_B,
                                                          const J*                  csr_col_ind_B,
                                                          const rocsparse_mat_descr descr_C,
                                                          I*                        csr_row_ptr_C,
                                                          I*                        nnz_C,
                                                          const rocsparse_mat_info  info_C,
                                                          void*                     temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    if(m > 0 && csr_row_ptr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(k > 0 && csr_row_ptr_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid pointers
    if(descr_A == nullptr || descr_B == nullptr || descr_C == nullptr || nnz_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && csr_col_ind_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_B != 0 && csr_col_ind_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_B->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), stream));
        }
        else
        {
            *nnz_C = 0;
        }

        if(m > 0)
        {
#define CSRGEMM_DIM 1024
            hipLaunchKernelGGL((csrgemm_set_base<CSRGEMM_DIM>),
                               dim3((m + 1) / CSRGEMM_DIM + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               m + 1,
                               csr_row_ptr_C,
                               descr_C->base);
#undef CSRGEMM_DIM
        }

        return rocsparse_status_success;
    }

    if((trans_A != rocsparse_operation_none) || (trans_B != rocsparse_operation_none))
    {
        return rocsparse_status_not_implemented;
    }

    if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Perform nnz calculation
    return rocsparse_csrgemm_nnz_calc(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      descr_A,
                                      nnz_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      nullptr,
                                      (I)0,
                                      (const I*)nullptr,
                                      (const J*)nullptr,
                                      descr_C,
                                      csr_row_ptr_C,
                                      nnz_C,
                                      info_C,
                                      temp_buffer);
}

template <typename I, typename J>
rocsparse_status rocsparse_csrgemm_nnz_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                J                         m,
                                                J                         n,
                                                J                         k,
                                                const rocsparse_mat_descr descr_A,
                                                I                         nnz_A,
                                                const I*                  csr_row_ptr_A,
                                                const J*                  csr_col_ind_A,
                                                const rocsparse_mat_descr descr_B,
                                                I                         nnz_B,
                                                const I*                  csr_row_ptr_B,
                                                const J*                  csr_col_ind_B,
                                                const rocsparse_mat_descr descr_D,
                                                I                         nnz_D,
                                                const I*                  csr_row_ptr_D,
                                                const J*                  csr_col_ind_D,
                                                const rocsparse_mat_descr descr_C,
                                                I*                        csr_row_ptr_C,
                                                I*                        nnz_C,
                                                const rocsparse_mat_info  info_C,
                                                void*                     temp_buffer)
{

    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check for valid rocsparse_mat_info
    if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csrgemm_nnz",
              trans_A,
              trans_B,
              m,
              n,
              k,
              (const void*&)descr_A,
              nnz_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              (const void*&)descr_B,
              nnz_B,
              (const void*&)csr_row_ptr_B,
              (const void*&)csr_col_ind_B,
              (const void*&)descr_D,
              nnz_D,
              (const void*&)csr_row_ptr_D,
              (const void*&)csr_col_ind_D,
              (const void*&)descr_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)nnz_C,
              (const void*&)info_C,
              (const void*&)temp_buffer);

    // Check operation
    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid rocsparse_csrgemm_info
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // quick return
    if(m == 0 || n == 0
       || (info_C->csrgemm_info->mul == false && info_C->csrgemm_info->add == false))
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), handle->stream));
        }
        else
        {
            *nnz_C = 0;
        }

        if(m > 0)
        {

#define CSRGEMM_DIM 1024
            hipLaunchKernelGGL((csrgemm_set_base<CSRGEMM_DIM>),
                               dim3((m + 1) / CSRGEMM_DIM + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               handle->stream,
                               m + 1,
                               csr_row_ptr_C,
                               descr_C->base);
#undef CSRGEMM_DIM
        }
        return rocsparse_status_success;
    }

    // Either mult, add or multadd need to be performed
    if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == true)
    {
        // C = alpha * A * B + beta * D
        return rocsparse_csrgemm_nnz_multadd(handle,
                                             trans_A,
                                             trans_B,
                                             m,
                                             n,
                                             k,
                                             descr_A,
                                             nnz_A,
                                             csr_row_ptr_A,
                                             csr_col_ind_A,
                                             descr_B,
                                             nnz_B,
                                             csr_row_ptr_B,
                                             csr_col_ind_B,
                                             descr_D,
                                             nnz_D,
                                             csr_row_ptr_D,
                                             csr_col_ind_D,
                                             descr_C,
                                             csr_row_ptr_C,
                                             nnz_C,
                                             info_C,
                                             temp_buffer);
    }
    else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == false)
    {
        // C = alpha * A * B
        return rocsparse_csrgemm_nnz_mult(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          descr_A,
                                          nnz_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          descr_C,
                                          csr_row_ptr_C,
                                          nnz_C,
                                          info_C,
                                          temp_buffer);
    }
    else
    {
        assert(info_C->csrgemm_info->mul == false && info_C->csrgemm_info->add == true);
        // C = beta * D
        return rocsparse_csrgemm_nnz_scal(handle,
                                          m,
                                          n,
                                          descr_D,
                                          nnz_D,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          descr_C,
                                          csr_row_ptr_C,
                                          nnz_C,
                                          info_C,
                                          temp_buffer);
    }
}

#define INSTANTIATE(ITYPE, JTYPE)                                           \
    template rocsparse_status rocsparse_csrgemm_nnz_template<ITYPE, JTYPE>( \
        rocsparse_handle          handle,                                   \
        rocsparse_operation       trans_A,                                  \
        rocsparse_operation       trans_B,                                  \
        JTYPE                     m,                                        \
        JTYPE                     n,                                        \
        JTYPE                     k,                                        \
        const rocsparse_mat_descr descr_A,                                  \
        ITYPE                     nnz_A,                                    \
        const ITYPE*              csr_row_ptr_A,                            \
        const JTYPE*              csr_col_ind_A,                            \
        const rocsparse_mat_descr descr_B,                                  \
        ITYPE                     nnz_B,                                    \
        const ITYPE*              csr_row_ptr_B,                            \
        const JTYPE*              csr_col_ind_B,                            \
        const rocsparse_mat_descr descr_D,                                  \
        ITYPE                     nnz_D,                                    \
        const ITYPE*              csr_row_ptr_D,                            \
        const JTYPE*              csr_col_ind_D,                            \
        const rocsparse_mat_descr descr_C,                                  \
        ITYPE*                    csr_row_ptr_C,                            \
        ITYPE*                    nnz_C,                                    \
        const rocsparse_mat_info  info_C,                                   \
        void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                   \
    template rocsparse_status rocsparse_csrgemm_buffer_size_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                                  \
        rocsparse_operation       trans_A,                                                 \
        rocsparse_operation       trans_B,                                                 \
        JTYPE                     m,                                                       \
        JTYPE                     n,                                                       \
        JTYPE                     k,                                                       \
        const TTYPE*              alpha,                                                   \
        const rocsparse_mat_descr descr_A,                                                 \
        ITYPE                     nnz_A,                                                   \
        const ITYPE*              csr_row_ptr_A,                                           \
        const JTYPE*              csr_col_ind_A,                                           \
        const rocsparse_mat_descr descr_B,                                                 \
        ITYPE                     nnz_B,                                                   \
        const ITYPE*              csr_row_ptr_B,                                           \
        const JTYPE*              csr_col_ind_B,                                           \
        const TTYPE*              beta,                                                    \
        const rocsparse_mat_descr descr_D,                                                 \
        ITYPE                     nnz_D,                                                   \
        const ITYPE*              csr_row_ptr_D,                                           \
        const JTYPE*              csr_col_ind_D,                                           \
        rocsparse_mat_info        info_C,                                                  \
        size_t*                   buffer_size);

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
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

//
// rocsparse_xcsrgemm_buffer_size
//
#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_operation       trans_A,       \
                                     rocsparse_operation       trans_B,       \
                                     rocsparse_int             m,             \
                                     rocsparse_int             n,             \
                                     rocsparse_int             k,             \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr_A,       \
                                     rocsparse_int             nnz_A,         \
                                     const rocsparse_int*      csr_row_ptr_A, \
                                     const rocsparse_int*      csr_col_ind_A, \
                                     const rocsparse_mat_descr descr_B,       \
                                     rocsparse_int             nnz_B,         \
                                     const rocsparse_int*      csr_row_ptr_B, \
                                     const rocsparse_int*      csr_col_ind_B, \
                                     const TYPE*               beta,          \
                                     const rocsparse_mat_descr descr_D,       \
                                     rocsparse_int             nnz_D,         \
                                     const rocsparse_int*      csr_row_ptr_D, \
                                     const rocsparse_int*      csr_col_ind_D, \
                                     rocsparse_mat_info        info_C,        \
                                     size_t*                   buffer_size)   \
    {                                                                         \
        return rocsparse_csrgemm_buffer_size_template(handle,                 \
                                                      trans_A,                \
                                                      trans_B,                \
                                                      m,                      \
                                                      n,                      \
                                                      k,                      \
                                                      alpha,                  \
                                                      descr_A,                \
                                                      nnz_A,                  \
                                                      csr_row_ptr_A,          \
                                                      csr_col_ind_A,          \
                                                      descr_B,                \
                                                      nnz_B,                  \
                                                      csr_row_ptr_B,          \
                                                      csr_col_ind_B,          \
                                                      beta,                   \
                                                      descr_D,                \
                                                      nnz_D,                  \
                                                      csr_row_ptr_D,          \
                                                      csr_col_ind_D,          \
                                                      info_C,                 \
                                                      buffer_size);           \
    }

C_IMPL(rocsparse_scsrgemm_buffer_size, float);
C_IMPL(rocsparse_dcsrgemm_buffer_size, double);
C_IMPL(rocsparse_ccsrgemm_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrgemm_buffer_size, rocsparse_double_complex);

#undef C_IMPL

//
// rocsparse_xcsrgemm_nnz
//
extern "C" rocsparse_status rocsparse_csrgemm_nnz(rocsparse_handle          handle,
                                                  rocsparse_operation       trans_A,
                                                  rocsparse_operation       trans_B,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  rocsparse_int             k,
                                                  const rocsparse_mat_descr descr_A,
                                                  rocsparse_int             nnz_A,
                                                  const rocsparse_int*      csr_row_ptr_A,
                                                  const rocsparse_int*      csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  rocsparse_int             nnz_B,
                                                  const rocsparse_int*      csr_row_ptr_B,
                                                  const rocsparse_int*      csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_D,
                                                  rocsparse_int             nnz_D,
                                                  const rocsparse_int*      csr_row_ptr_D,
                                                  const rocsparse_int*      csr_col_ind_D,
                                                  const rocsparse_mat_descr descr_C,
                                                  rocsparse_int*            csr_row_ptr_C,
                                                  rocsparse_int*            nnz_C,
                                                  const rocsparse_mat_info  info_C,
                                                  void*                     temp_buffer)
{
    return rocsparse_csrgemm_nnz_template(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          descr_A,
                                          nnz_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          descr_D,
                                          nnz_D,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          descr_C,
                                          csr_row_ptr_C,
                                          nnz_C,
                                          info_C,
                                          temp_buffer);
}
