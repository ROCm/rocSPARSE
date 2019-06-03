/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "csrgemm_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_csrgemm_buffer_size_template(rocsparse_handle handle,
                                                        rocsparse_operation trans_A,
                                                        rocsparse_operation trans_B,
                                                        rocsparse_int m,
                                                        rocsparse_int n,
                                                        rocsparse_int k,
                                                        const T* alpha,
                                                        const rocsparse_mat_descr descr_A,
                                                        rocsparse_int nnz_A,
                                                        const rocsparse_int* csr_row_ptr_A,
                                                        const rocsparse_int* csr_col_ind_A,
                                                        const rocsparse_mat_descr descr_B,
                                                        rocsparse_int nnz_B,
                                                        const rocsparse_int* csr_row_ptr_B,
                                                        const rocsparse_int* csr_col_ind_B,
                                                        const T* beta,
                                                        const rocsparse_mat_descr descr_D,
                                                        rocsparse_int nnz_D,
                                                        const rocsparse_int* csr_row_ptr_D,
                                                        const rocsparse_int* csr_col_ind_D,
                                                        rocsparse_mat_info info,
                                                        size_t* buffer_size)
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

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Either alpha or beta can be nullptr
    if(alpha == nullptr && beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;

        return rocsparse_status_success;
    }
    if(alpha == nullptr)
    {
        if(nnz_D == 0)
        {
            // Do not return 0 as buffer size
            *buffer_size = 4;

            return rocsparse_status_success;
        }
    }
    if(beta == nullptr)
    {
        if(k == 0 || nnz_A == 0 || nnz_B == 0)
        {
            // Do not return 0 as buffer size
            *buffer_size = 4;

            return rocsparse_status_success;
        }
    }
    if((nnz_A == 0 || nnz_B == 0) && nnz_D == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;

        return rocsparse_status_success;
    }

    // If alpha != nullptr, A and B must be valid
    if(alpha != nullptr)
    {
        // Check valid pointers
        if(descr_A == nullptr || csr_row_ptr_A == nullptr || csr_col_ind_A == nullptr ||
           descr_B == nullptr || csr_row_ptr_B == nullptr || csr_col_ind_B == nullptr)
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
    }

    // If beta != nullptr, D must be valid
    if(beta != nullptr)
    {
        // Check valid pointers
        if(descr_D == nullptr || csr_row_ptr_D == nullptr || csr_col_ind_D == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }

        // Check index base
        if(descr_D->base != rocsparse_index_base_zero && descr_D->base != rocsparse_index_base_one)
        {
            return rocsparse_status_invalid_value;
        }

        // Check matrix type
        if(descr_D->type != rocsparse_matrix_type_general)
        {
            return rocsparse_status_not_implemented;
        }
    }

    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Clear csrgemm info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrgemm_info(info->csrgemm_info));

    // Create csrgemm info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrgemm_info(&info->csrgemm_info));

    // Multiplicative part
    info->csrgemm_info->mul = (alpha != nullptr);

    // Additive part
    info->csrgemm_info->add = (beta != nullptr);












*buffer_size = 4;





    return rocsparse_status_success;
}


template <typename T>
rocsparse_status rocsparse_csrgemm_template(rocsparse_handle handle,
                                            rocsparse_operation trans_A,
                                            rocsparse_operation trans_B,
                                            rocsparse_int m,
                                            rocsparse_int n,
                                            rocsparse_int k,
                                            const T* alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int nnz_A,
                                            const T* csr_val_A,
                                            const rocsparse_int* csr_row_ptr_A,
                                            const rocsparse_int* csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int nnz_B,
                                            const T* csr_val_B,
                                            const rocsparse_int* csr_row_ptr_B,
                                            const rocsparse_int* csr_col_ind_B,
                                            const T* beta,
                                            const rocsparse_mat_descr descr_D,
                                            rocsparse_int nnz_D,
                                            const T* csr_val_D,
                                            const rocsparse_int* csr_row_ptr_D,
                                            const rocsparse_int* csr_col_ind_D,
                                            const rocsparse_mat_descr descr_C,
                                            T* csr_val_C,
                                            const rocsparse_int* csr_row_ptr_C,
                                            rocsparse_int* csr_col_ind_C,
                                            const rocsparse_mat_info info,
                                            void* temp_buffer)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
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

        log_bench(handle, "./rocsparse-bench -f csrgemm -r", replaceX<T>("X"), "--mtx <matrix.mtx>"); // TODO alpha
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

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Either alpha or beta can be nullptr
    if(alpha == nullptr && beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }
    if(alpha == nullptr)
    {
        if(nnz_D == 0)
        {
            return rocsparse_status_success;
        }
    }
    if(beta == nullptr)
    {
        if(k == 0 || nnz_A == 0 || nnz_B == 0)
        {
            return rocsparse_status_success;
        }
    }
    if((nnz_A == 0 || nnz_B == 0) && nnz_D == 0)
    {
        return rocsparse_status_success;
    }

    // If alpha != nullptr, A and B must be valid
    if(alpha != nullptr)
    {
        // Check valid pointers
        if(descr_A == nullptr || csr_val_A == nullptr || csr_row_ptr_A == nullptr || csr_col_ind_A == nullptr ||
           descr_B == nullptr || csr_val_B == nullptr || csr_row_ptr_B == nullptr || csr_col_ind_B == nullptr)
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
    }

    // If beta != nullptr, D must be valid
    if(beta != nullptr)
    {
        // Check valid pointers
        if(descr_D == nullptr || csr_val_D == nullptr || csr_row_ptr_D == nullptr || csr_col_ind_D == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }

        // Check index base
        if(descr_D->base != rocsparse_index_base_zero && descr_D->base != rocsparse_index_base_one)
        {
            return rocsparse_status_invalid_value;
        }

        // Check matrix type
        if(descr_D->type != rocsparse_matrix_type_general)
        {
            return rocsparse_status_not_implemented;
        }
    }

    if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_val_C == nullptr || csr_row_ptr_C == nullptr || csr_col_ind_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSRGEMM_HPP
