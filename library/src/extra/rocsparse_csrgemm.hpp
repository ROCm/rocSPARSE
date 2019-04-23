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
rocsparse_status rocsparse_csrgemm_template(rocsparse_handle handle,
                                            rocsparse_operation trans_A,
                                            rocsparse_operation trans_B,
                                            rocsparse_int m,
                                            rocsparse_int n,
                                            rocsparse_int k,
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
                                            const rocsparse_mat_descr descr_C,
                                            T* csr_val_C,
                                            const rocsparse_int* csr_row_ptr_C,
                                            rocsparse_int* csr_col_ind_C,
                                            void* temp_buffer)
{
    // Check for valid handle and matrix descriptors
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(descr_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(descr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrgemm"),
              trans_A,
              trans_B,
              m,
              n,
              k,
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
              (const void*&)descr_C,
              (const void*&)csr_val_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)csr_col_ind_C,
              (const void*&)temp_buffer);

    log_bench(handle, "./rocsparse-bench -f csrgemm -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

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

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(k < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz_A < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_val_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_val_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_not_implemented;
}

#endif // ROCSPARSE_CSRGEMM_HPP
