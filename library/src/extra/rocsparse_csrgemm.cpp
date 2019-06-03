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

#include "definitions.h"
#include "rocsparse.h"
#include "rocsparse_csrgemm.hpp"
#include "handle.h"
#include "utility.h"

extern "C" rocsparse_status rocsparse_csrgemm_nnz(rocsparse_handle handle,
                                                  rocsparse_operation trans_A,
                                                  rocsparse_operation trans_B,
                                                  rocsparse_int m,
                                                  rocsparse_int n,
                                                  rocsparse_int k,
                                                  const rocsparse_mat_descr descr_A,
                                                  rocsparse_int nnz_A,
                                                  const rocsparse_int* csr_row_ptr_A,
                                                  const rocsparse_int* csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  rocsparse_int nnz_B,
                                                  const rocsparse_int* csr_row_ptr_B,
                                                  const rocsparse_int* csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_D,
                                                  rocsparse_int nnz_D,
                                                  const rocsparse_int* csr_row_ptr_D,
                                                  const rocsparse_int* csr_col_ind_D,
                                                  const rocsparse_mat_descr descr_C,
                                                  rocsparse_int* csr_row_ptr_C,
                                                  rocsparse_int* nnz_C,
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
              (const void*&)info,
              (const void*&)temp_buffer);

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check if buffer size function has been called
    if(info->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 ||
       n == 0 ||
       (info->csrgemm_info->mul != true && nnz_D == 0) ||
       (info->csrgemm_info->add != true && (k == 0 || nnz_A == 0 || nnz_B == 0)) ||
       (nnz_D == 0 && (nnz_A == 0 || nnz_B == 0)))
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(rocsparse_int), stream));
        }
        else
        {
            *nnz_C = 0;
        }

        return rocsparse_status_success;
    }

    // If mul == true, A and B must be valid
    if(info->csrgemm_info->mul == true)
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

    // If add == true, D must be valid
    if(info->csrgemm_info->add == true)
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

    if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }



    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrgemm_buffer_size(rocsparse_handle handle,
                                                           rocsparse_operation trans_A,
                                                           rocsparse_operation trans_B,
                                                           rocsparse_int m,
                                                           rocsparse_int n,
                                                           rocsparse_int k,
                                                           const float* alpha,
                                                           const rocsparse_mat_descr descr_A,
                                                           rocsparse_int nnz_A,
                                                           const rocsparse_int* csr_row_ptr_A,
                                                           const rocsparse_int* csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           rocsparse_int nnz_B,
                                                           const rocsparse_int* csr_row_ptr_B,
                                                           const rocsparse_int* csr_col_ind_B,
                                                           const float* beta,
                                                           const rocsparse_mat_descr descr_D,
                                                           rocsparse_int nnz_D,
                                                           const rocsparse_int* csr_row_ptr_D,
                                                           const rocsparse_int* csr_col_ind_D,
                                                           rocsparse_mat_info info,
                                                           size_t* buffer_size)
{
    return rocsparse_csrgemm_buffer_size_template<float>(handle,
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
                                                         info,
                                                         buffer_size);
}

extern "C" rocsparse_status rocsparse_dcsrgemm_buffer_size(rocsparse_handle handle,
                                                           rocsparse_operation trans_A,
                                                           rocsparse_operation trans_B,
                                                           rocsparse_int m,
                                                           rocsparse_int n,
                                                           rocsparse_int k,
                                                           const double* alpha,
                                                           const rocsparse_mat_descr descr_A,
                                                           rocsparse_int nnz_A,
                                                           const rocsparse_int* csr_row_ptr_A,
                                                           const rocsparse_int* csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           rocsparse_int nnz_B,
                                                           const rocsparse_int* csr_row_ptr_B,
                                                           const rocsparse_int* csr_col_ind_B,
                                                           const double* beta,
                                                           const rocsparse_mat_descr descr_D,
                                                           rocsparse_int nnz_D,
                                                           const rocsparse_int* csr_row_ptr_D,
                                                           const rocsparse_int* csr_col_ind_D,
                                                           rocsparse_mat_info info,
                                                           size_t* buffer_size)
{
    return rocsparse_csrgemm_buffer_size_template<double>(handle,
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
                                                          info,
                                                          buffer_size);
}

extern "C" rocsparse_status rocsparse_scsrgemm(rocsparse_handle handle,
                                               rocsparse_operation trans_A,
                                               rocsparse_operation trans_B,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int k,
                                               const float* alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int nnz_A,
                                               const float* csr_val_A,
                                               const rocsparse_int* csr_row_ptr_A,
                                               const rocsparse_int* csr_col_ind_A,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int nnz_B,
                                               const float* csr_val_B,
                                               const rocsparse_int* csr_row_ptr_B,
                                               const rocsparse_int* csr_col_ind_B,
                                               const float* beta,
                                               const rocsparse_mat_descr descr_D,
                                               rocsparse_int nnz_D,
                                               const float* csr_val_D,
                                               const rocsparse_int* csr_row_ptr_D,
                                               const rocsparse_int* csr_col_ind_D,
                                               const rocsparse_mat_descr descr_C,
                                               float* csr_val_C,
                                               const rocsparse_int* csr_row_ptr_C,
                                               rocsparse_int* csr_col_ind_C,
                                               const rocsparse_mat_info info,
                                               void* temp_buffer)
{
    return rocsparse_csrgemm_template<float>(handle,
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

extern "C" rocsparse_status rocsparse_dcsrgemm(rocsparse_handle handle,
                                               rocsparse_operation trans_A,
                                               rocsparse_operation trans_B,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int k,
                                               const double* alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int nnz_A,
                                               const double* csr_val_A,
                                               const rocsparse_int* csr_row_ptr_A,
                                               const rocsparse_int* csr_col_ind_A,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int nnz_B,
                                               const double* csr_val_B,
                                               const rocsparse_int* csr_row_ptr_B,
                                               const rocsparse_int* csr_col_ind_B,
                                               const double* beta,
                                               const rocsparse_mat_descr descr_D,
                                               rocsparse_int nnz_D,
                                               const double* csr_val_D,
                                               const rocsparse_int* csr_row_ptr_D,
                                               const rocsparse_int* csr_col_ind_D,
                                               const rocsparse_mat_descr descr_C,
                                               double* csr_val_C,
                                               const rocsparse_int* csr_row_ptr_C,
                                               rocsparse_int* csr_col_ind_C,
                                               const rocsparse_mat_info info,
                                               void* temp_buffer)
{
    return rocsparse_csrgemm_template<double>(handle,
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
