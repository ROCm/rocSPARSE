/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "rocsparse_csrgeam.hpp"
#include "csrgeam_device.h"
#include "definitions.h"
#include "rocsparse.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_csrgeam_nnz(rocsparse_handle          handle,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr descr_A,
                                                  rocsparse_int             nnz_A,
                                                  const rocsparse_int*      csr_row_ptr_A,
                                                  const rocsparse_int*      csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  rocsparse_int             nnz_B,
                                                  const rocsparse_int*      csr_row_ptr_B,
                                                  const rocsparse_int*      csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_C,
                                                  rocsparse_int*            csr_row_ptr_C,
                                                  rocsparse_int*            nnz_C)
{
    // Check for valid handle and descriptors
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
              "rocsparse_csrgeam_nnz",
              m,
              n,
              (const void*&)descr_A,
              nnz_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              (const void*&)descr_B,
              nnz_B,
              (const void*&)csr_row_ptr_B,
              (const void*&)csr_col_ind_B,
              (const void*&)descr_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)nnz_C);

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

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid nnz_C pointer
    if(nnz_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz_A == 0 || nnz_B == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            *nnz_C = 0;
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(rocsparse_int)));
        }

        return rocsparse_status_success;
    }

    // Check valid pointers
    if(csr_row_ptr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind_A == nullptr)
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
    else if(csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define CSRGEAM_DIM 256
    hipLaunchKernelGGL((csrgeam_nnz_multipass<CSRGEAM_DIM, 64>),
                       dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                       dim3(CSRGEAM_DIM),
                       0,
                       stream,
                       m,
                       n,
                       csr_row_ptr_A,
                       csr_col_ind_A,
                       csr_row_ptr_B,
                       csr_col_ind_B,
                       csr_row_ptr_C,
                       descr_A->base,
                       descr_B->base);
#undef CSRGEAM_DIM

    // Exclusive sum to obtain row pointers of C
    size_t rocprim_size;
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                rocprim_size,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                descr_C->base,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));

    bool  rocprim_alloc;
    void* rocprim_buffer;

    if(handle->buffer_size >= rocprim_size)
    {
        rocprim_buffer = handle->buffer;
        rocprim_alloc  = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc(&rocprim_buffer, rocprim_size));
        rocprim_alloc = true;
    }

    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                rocprim_size,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                descr_C->base,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));

    if(rocprim_alloc == true)
    {
        RETURN_IF_HIP_ERROR(hipFree(rocprim_buffer));
    }

    // Extract the number of non-zero elements of C
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        // Blocking mode
        RETURN_IF_HIP_ERROR(
            hipMemcpy(nnz_C, csr_row_ptr_C + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // Adjust index base of nnz_C
        *nnz_C -= descr_C->base;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz_C, csr_row_ptr_C + m, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));

        // Adjust index base of nnz_C
        if(descr_C->base == rocsparse_index_base_one)
        {
            hipLaunchKernelGGL((csrgeam_index_base), dim3(1), dim3(1), 0, stream, nnz_C);
        }
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsrgeam(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const float*              alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int             nnz_A,
                                               const float*              csr_val_A,
                                               const rocsparse_int*      csr_row_ptr_A,
                                               const rocsparse_int*      csr_col_ind_A,
                                               const float*              beta,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int             nnz_B,
                                               const float*              csr_val_B,
                                               const rocsparse_int*      csr_row_ptr_B,
                                               const rocsparse_int*      csr_col_ind_B,
                                               const rocsparse_mat_descr descr_C,
                                               float*                    csr_val_C,
                                               const rocsparse_int*      csr_row_ptr_C,
                                               rocsparse_int*            csr_col_ind_C)
{
    return rocsparse_csrgeam_template(handle,
                                      m,
                                      n,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      beta,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C);
}

extern "C" rocsparse_status rocsparse_dcsrgeam(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const double*             alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int             nnz_A,
                                               const double*             csr_val_A,
                                               const rocsparse_int*      csr_row_ptr_A,
                                               const rocsparse_int*      csr_col_ind_A,
                                               const double*             beta,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int             nnz_B,
                                               const double*             csr_val_B,
                                               const rocsparse_int*      csr_row_ptr_B,
                                               const rocsparse_int*      csr_col_ind_B,
                                               const rocsparse_mat_descr descr_C,
                                               double*                   csr_val_C,
                                               const rocsparse_int*      csr_row_ptr_C,
                                               rocsparse_int*            csr_col_ind_C)
{
    return rocsparse_csrgeam_template(handle,
                                      m,
                                      n,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      beta,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C);
}

extern "C" rocsparse_status rocsparse_ccsrgeam(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               rocsparse_int                  n,
                                               const rocsparse_float_complex* alpha,
                                               const rocsparse_mat_descr      descr_A,
                                               rocsparse_int                  nnz_A,
                                               const rocsparse_float_complex* csr_val_A,
                                               const rocsparse_int*           csr_row_ptr_A,
                                               const rocsparse_int*           csr_col_ind_A,
                                               const rocsparse_float_complex* beta,
                                               const rocsparse_mat_descr      descr_B,
                                               rocsparse_int                  nnz_B,
                                               const rocsparse_float_complex* csr_val_B,
                                               const rocsparse_int*           csr_row_ptr_B,
                                               const rocsparse_int*           csr_col_ind_B,
                                               const rocsparse_mat_descr      descr_C,
                                               rocsparse_float_complex*       csr_val_C,
                                               const rocsparse_int*           csr_row_ptr_C,
                                               rocsparse_int*                 csr_col_ind_C)
{
    return rocsparse_csrgeam_template(handle,
                                      m,
                                      n,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      beta,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C);
}

extern "C" rocsparse_status rocsparse_zcsrgeam(rocsparse_handle                handle,
                                               rocsparse_int                   m,
                                               rocsparse_int                   n,
                                               const rocsparse_double_complex* alpha,
                                               const rocsparse_mat_descr       descr_A,
                                               rocsparse_int                   nnz_A,
                                               const rocsparse_double_complex* csr_val_A,
                                               const rocsparse_int*            csr_row_ptr_A,
                                               const rocsparse_int*            csr_col_ind_A,
                                               const rocsparse_double_complex* beta,
                                               const rocsparse_mat_descr       descr_B,
                                               rocsparse_int                   nnz_B,
                                               const rocsparse_double_complex* csr_val_B,
                                               const rocsparse_int*            csr_row_ptr_B,
                                               const rocsparse_int*            csr_col_ind_B,
                                               const rocsparse_mat_descr       descr_C,
                                               rocsparse_double_complex*       csr_val_C,
                                               const rocsparse_int*            csr_row_ptr_C,
                                               rocsparse_int*                  csr_col_ind_C)
{
    return rocsparse_csrgeam_template(handle,
                                      m,
                                      n,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      beta,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C);
}
