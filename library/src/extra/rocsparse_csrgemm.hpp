/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef ROCSPARSE_CSRGEMM_HPP
#define ROCSPARSE_CSRGEMM_HPP

#include "csrgemm_device.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim_hip.hpp>

#define CSRGEMM_MAXGROUPS 8

template <typename T>
rocsparse_status rocsparse_csrgemm_mult_buffer_size_template(rocsparse_handle          handle,
                                                             rocsparse_operation       trans_A,
                                                             rocsparse_operation       trans_B,
                                                             rocsparse_int             m,
                                                             rocsparse_int             n,
                                                             rocsparse_int             k,
                                                             const T*                  alpha,
                                                             const rocsparse_mat_descr descr_A,
                                                             rocsparse_int             nnz_A,
                                                             const rocsparse_int* csr_row_ptr_A,
                                                             const rocsparse_int* csr_col_ind_A,
                                                             const rocsparse_mat_descr descr_B,
                                                             rocsparse_int             nnz_B,
                                                             const rocsparse_int* csr_row_ptr_B,
                                                             const rocsparse_int* csr_col_ind_B,
                                                             rocsparse_mat_info   info,
                                                             size_t*              buffer_size)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info->csrgemm_info == nullptr)
    {
        return rocsparse_status_internal_error;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || csr_row_ptr_A == nullptr || csr_col_ind_A == nullptr
       || descr_B == nullptr || csr_row_ptr_B == nullptr || csr_col_ind_B == nullptr
       || buffer_size == nullptr || alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr_A->base != rocsparse_index_base_zero && descr_A->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr_B->base != rocsparse_index_base_zero && descr_B->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
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

    // Stream
    hipStream_t stream = handle->stream;

    // rocprim buffer
    size_t rocprim_size;
    size_t rocprim_max = 0;

    // rocprim::reduce
    RETURN_IF_HIP_ERROR(rocprim::reduce(nullptr,
                                        rocprim_size,
                                        csr_row_ptr_A,
                                        &nnz_A,
                                        0,
                                        m,
                                        rocprim::maximum<rocsparse_int>(),
                                        stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    // rocprim exclusive scan
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                rocprim_size,
                                                csr_row_ptr_A,
                                                &nnz_A,
                                                0,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    // rocprim::radix_sort_pairs
    rocprim::double_buffer<rocsparse_int> buf(&nnz_A, &nnz_B);
    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, rocprim_size, buf, buf, m, 0, 3, stream));
    rocprim_max = std::max(rocprim_max, rocprim_size);

    *buffer_size = ((rocprim_max - 1) / 256 + 1) * 256;

    // Group arrays
    *buffer_size += sizeof(int) * 256 * CSRGEMM_MAXGROUPS;
    *buffer_size += sizeof(int) * 256;
    *buffer_size += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    // Permutation arrays
    *buffer_size += ((sizeof(int) * m - 1) / 256 + 1) * 256;
    *buffer_size += ((sizeof(int) * m - 1) / 256 + 1) * 256;
    *buffer_size += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrgemm_buffer_size_template(rocsparse_handle          handle,
                                                        rocsparse_operation       trans_A,
                                                        rocsparse_operation       trans_B,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        rocsparse_int             k,
                                                        const T*                  alpha,
                                                        const rocsparse_mat_descr descr_A,
                                                        rocsparse_int             nnz_A,
                                                        const rocsparse_int*      csr_row_ptr_A,
                                                        const rocsparse_int*      csr_col_ind_A,
                                                        const rocsparse_mat_descr descr_B,
                                                        rocsparse_int             nnz_B,
                                                        const rocsparse_int*      csr_row_ptr_B,
                                                        const rocsparse_int*      csr_col_ind_B,
                                                        const T*                  beta,
                                                        const rocsparse_mat_descr descr_D,
                                                        rocsparse_int             nnz_D,
                                                        const rocsparse_int*      csr_row_ptr_D,
                                                        const rocsparse_int*      csr_col_ind_D,
                                                        rocsparse_mat_info        info,
                                                        size_t*                   buffer_size)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgemm_buffer_size"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  *alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  *beta,
                  (const void*&)descr_D,
                  nnz_D,
                  (const void*&)csr_row_ptr_D,
                  (const void*&)csr_col_ind_D,
                  (const void*&)info,
                  (const void*&)buffer_size);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgemm_buffer_size"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  (const void*&)alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  (const void*&)beta,
                  (const void*&)descr_D,
                  nnz_D,
                  (const void*&)csr_row_ptr_D,
                  (const void*&)csr_col_ind_D,
                  (const void*&)info,
                  (const void*&)buffer_size);
    }

    // Check for valid rocsparse_mat_info
    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Clear csrgemm info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrgemm_info(info->csrgemm_info));

    // Create csrgemm info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrgemm_info(&info->csrgemm_info));

    // Set info parameters
    info->csrgemm_info->mul = (alpha != nullptr);
    info->csrgemm_info->add = (beta != nullptr);

    // Either alpha or beta can be nullptr
    if(alpha != nullptr && beta != nullptr)
    {
        // alpha != nullptr && beta != nullptr
        // TODO
        // rocsparse_csrgemm_multadd_template(...)
        return rocsparse_status_not_implemented;
    }
    else if(alpha != nullptr && beta == nullptr)
    {
        // alpha != nullptr && beta == nullptr
        return rocsparse_csrgemm_mult_buffer_size_template<T>(handle,
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
                                                              info,
                                                              buffer_size);
    }
    else if(alpha == nullptr && beta != nullptr)
    {
        // alpha == nullptr && beta != nullptr

        // TODO
        // RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrgeam_buffer_size_template<T>(...);
        return rocsparse_status_not_implemented;
    }
    else
    {
        // alpha == nullptr && beta == nullptr
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrgemm_calc_template(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 rocsparse_int             k,
                                                 const T*                  alpha,
                                                 const rocsparse_mat_descr descr_A,
                                                 rocsparse_int             nnz_A,
                                                 const T*                  csr_val_A,
                                                 const rocsparse_int*      csr_row_ptr_A,
                                                 const rocsparse_int*      csr_col_ind_A,
                                                 const rocsparse_mat_descr descr_B,
                                                 rocsparse_int             nnz_B,
                                                 const T*                  csr_val_B,
                                                 const rocsparse_int*      csr_row_ptr_B,
                                                 const rocsparse_int*      csr_col_ind_B,
                                                 const T*                  beta,
                                                 const rocsparse_mat_descr descr_D,
                                                 rocsparse_int             nnz_D,
                                                 const T*                  csr_val_D,
                                                 const rocsparse_int*      csr_row_ptr_D,
                                                 const rocsparse_int*      csr_col_ind_D,
                                                 const rocsparse_mat_descr descr_C,
                                                 T*                        csr_val_C,
                                                 const rocsparse_int*      csr_row_ptr_C,
                                                 rocsparse_int*            csr_col_ind_C,
                                                 const rocsparse_mat_info  info,
                                                 void*                     temp_buffer)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // rocprim buffer
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Determine maximum non-zero entries per row of all rows
    rocsparse_int* workspace = reinterpret_cast<rocsparse_int*>(buffer);

#define CSRGEMM_DIM 256
    hipLaunchKernelGGL((csrgemm_max_row_nnz_part1<CSRGEMM_DIM>),
                       dim3(CSRGEMM_DIM),
                       dim3(CSRGEMM_DIM),
                       0,
                       stream,
                       m,
                       csr_row_ptr_C,
                       workspace);

    hipLaunchKernelGGL(
        (csrgemm_max_row_nnz_part2<CSRGEMM_DIM>), dim3(1), dim3(CSRGEMM_DIM), 0, stream, workspace);
#undef CSRGEMM_DIM

    rocsparse_int nnz_max;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&nnz_max, workspace, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

    // Group offset buffer
    rocsparse_int* d_group_offset = reinterpret_cast<rocsparse_int*>(buffer);
    buffer += sizeof(rocsparse_int) * 256;

    // Group size buffer
    rocsparse_int h_group_size[CSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(rocsparse_int) * CSRGEMM_MAXGROUPS);

    // Permutation array
    rocsparse_int* d_perm = nullptr;

    // If maximum of row nnz exceeds 16, we process the rows in groups of
    // similar sized row nnz
    if(nnz_max > 16)
    {
        // Group size buffer
        rocsparse_int* d_group_size = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += sizeof(rocsparse_int) * 256 * CSRGEMM_MAXGROUPS;

        // Permutation temporary arrays
        rocsparse_int* tmp_vals = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        rocsparse_int* tmp_keys = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        rocsparse_int* tmp_groups = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        hipLaunchKernelGGL((csrgemm_group_reduce_part2<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
                           dim3(CSRGEMM_DIM),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_C,
                           d_group_size,
                           tmp_groups);

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
                                                    rocprim::plus<rocsparse_int>(),
                                                    stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<rocsparse_int>(),
                                                    stream));

        // Copy group sizes to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_group_size,
                                           d_group_size,
                                           sizeof(rocsparse_int) * CSRGEMM_MAXGROUPS,
                                           hipMemcpyDeviceToHost,
                                           stream));

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, m, tmp_perm));

        rocprim::double_buffer<rocsparse_int> d_keys(tmp_groups, tmp_keys);
        rocprim::double_buffer<rocsparse_int> d_vals(tmp_perm, tmp_vals);

        // Sort pairs (by groups)
        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, rocprim_size, d_keys, d_vals, m, 0, 3, stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            rocprim_buffer, rocprim_size, d_keys, d_vals, m, 0, 3, stream));

        d_perm = d_vals.current();

        // Release tmp_groups buffer
        buffer -= ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        // Release tmp_keys buffer
        buffer -= ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(rocsparse_int), stream));
    }

    printf("Group sizes:\n");
    printf("\t   0 -   16: %d\n", h_group_size[0]);
    printf("\t  17 -   32: %d\n", h_group_size[1]);
    printf("\t  33 -  256: %d\n", h_group_size[2]);
    printf("\t 257 -  512: %d\n", h_group_size[3]);
    printf("\t 513 - 1024: %d\n", h_group_size[4]);
    printf("\t1025 - 2048: %d\n", h_group_size[5]);
    printf("\t2049 - 4096: %d\n", h_group_size[6]);
    printf("\t4097 -  inf: %d\n", h_group_size[7]);

    // Compute columns and accumulate values for each group

    // Group 0: 0 - 16 non-zeros per row
    if(h_group_size[0] > 0)
    {
    }

    // Group 1: 17 - 32 non-zeros per row
    if(h_group_size[1] > 0)
    {
    }

    // Group 2: 33 - 256 non-zeros per row
    if(h_group_size[2] > 0)
    {
    }

    // Group 3: 257 - 512 non-zeros per row
    if(h_group_size[3] > 0)
    {
    }

    // Group 4: 513 - 1024 non-zeros per row
    if(h_group_size[4] > 0)
    {
    }

    // Group 5: 1025 - 2048 non-zeros per row
    if(h_group_size[5] > 0)
    {
    }

    // Group 6: 2049 - 4096 non-zeros per row
    if(h_group_size[6] > 0)
    {
    }

    // Group 7: more than 4096 non-zeros per row
    if(h_group_size[7] > 0)
    {
        printf("\n# max nnz > 4096: %d ; exiting\n", h_group_size[7]);
        return rocsparse_status_not_implemented;
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrgemm_mult_template(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 rocsparse_int             k,
                                                 const T*                  alpha,
                                                 const rocsparse_mat_descr descr_A,
                                                 rocsparse_int             nnz_A,
                                                 const T*                  csr_val_A,
                                                 const rocsparse_int*      csr_row_ptr_A,
                                                 const rocsparse_int*      csr_col_ind_A,
                                                 const rocsparse_mat_descr descr_B,
                                                 rocsparse_int             nnz_B,
                                                 const T*                  csr_val_B,
                                                 const rocsparse_int*      csr_row_ptr_B,
                                                 const rocsparse_int*      csr_col_ind_B,
                                                 const T*                  beta,
                                                 const rocsparse_mat_descr descr_D,
                                                 rocsparse_int             nnz_D,
                                                 const T*                  csr_val_D,
                                                 const rocsparse_int*      csr_row_ptr_D,
                                                 const rocsparse_int*      csr_col_ind_D,
                                                 const rocsparse_mat_descr descr_C,
                                                 T*                        csr_val_C,
                                                 const rocsparse_int*      csr_row_ptr_C,
                                                 rocsparse_int*            csr_col_ind_C,
                                                 const rocsparse_mat_info  info,
                                                 void*                     temp_buffer)
{
    // Check for valid info structure
    if(info->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || csr_val_A == nullptr || csr_row_ptr_A == nullptr
       || csr_col_ind_A == nullptr || descr_B == nullptr || csr_val_B == nullptr
       || csr_row_ptr_B == nullptr || csr_col_ind_B == nullptr || descr_C == nullptr
       || csr_val_C == nullptr || csr_row_ptr_C == nullptr || csr_col_ind_C == nullptr
       || temp_buffer == nullptr || alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr_A->base != rocsparse_index_base_zero && descr_A->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr_B->base != rocsparse_index_base_zero && descr_B->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr_C->base != rocsparse_index_base_zero && descr_C->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
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

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_status_success;
    }

    // Perform gemm calculation
    return rocsparse_csrgemm_calc_template<T>(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_val_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_val_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              beta,
                                              descr_D,
                                              nnz_D,
                                              csr_val_D,
                                              csr_row_ptr_D,
                                              csr_col_ind_D,
                                              descr_C,
                                              csr_val_C,
                                              csr_row_ptr_C,
                                              csr_col_ind_C,
                                              info,
                                              temp_buffer);
}

template <typename T>
rocsparse_status rocsparse_csrgemm_template(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            rocsparse_int             k,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const T*                  csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const T*                  csr_val_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_D,
                                            rocsparse_int             nnz_D,
                                            const T*                  csr_val_D,
                                            const rocsparse_int*      csr_row_ptr_D,
                                            const rocsparse_int*      csr_col_ind_D,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        csr_val_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C,
                                            const rocsparse_mat_info  info,
                                            void*                     temp_buffer)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgemm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  *alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_val_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_val_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  *beta,
                  (const void*&)descr_D,
                  nnz_D,
                  (const void*&)csr_val_D,
                  (const void*&)csr_row_ptr_D,
                  (const void*&)csr_col_ind_D,
                  (const void*&)descr_C,
                  (const void*&)csr_val_C,
                  (const void*&)csr_row_ptr_C,
                  (const void*&)csr_col_ind_C,
                  (const void*&)info,
                  (const void*&)temp_buffer);

        log_bench(handle,
                  "./rocsparse-bench -f csrgemm -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx>"); // TODO alpha
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrgemm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  (const void*&)alpha,
                  (const void*&)descr_A,
                  nnz_A,
                  (const void*&)csr_val_A,
                  (const void*&)csr_row_ptr_A,
                  (const void*&)csr_col_ind_A,
                  (const void*&)descr_B,
                  nnz_B,
                  (const void*&)csr_val_B,
                  (const void*&)csr_row_ptr_B,
                  (const void*&)csr_col_ind_B,
                  (const void*&)beta,
                  (const void*&)descr_D,
                  nnz_D,
                  (const void*&)csr_val_D,
                  (const void*&)csr_row_ptr_D,
                  (const void*&)csr_col_ind_D,
                  (const void*&)descr_C,
                  (const void*&)csr_val_C,
                  (const void*&)csr_row_ptr_C,
                  (const void*&)csr_col_ind_C,
                  (const void*&)info,
                  (const void*&)temp_buffer);
    }

    // Check for valid rocsparse_mat_info
    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for valid rocsparse_csrgemm_info
    if(info->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Either mult, add or multadd need to be performed
    if(info->csrgemm_info->mul == true && info->csrgemm_info->add == true)
    {
        // C = alpha * A * B + beta * D
        // TODO
        return rocsparse_status_not_implemented;
    }
    else if(info->csrgemm_info->mul == true && info->csrgemm_info->add == false)
    {
        // C = alpha * A * B
        return rocsparse_csrgemm_mult_template<T>(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_val_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_val_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_val_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  descr_C,
                                                  csr_val_C,
                                                  csr_row_ptr_C,
                                                  csr_col_ind_C,
                                                  info,
                                                  temp_buffer);
    }
    else if(info->csrgemm_info->mul == false && info->csrgemm_info->add == true)
    {
        // C = beta * D
        // TODO
        return rocsparse_status_not_implemented;
    }
    else
    {
        // C = 0
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRGEMM_HPP
