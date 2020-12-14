/*! \file */
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

#include "rocsparse_hyb2csr.hpp"
#include "definitions.h"
#include "utility.h"

#include "hyb2csr_device.h"
#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_hyb2csr_template(rocsparse_handle          handle,
                                            const rocsparse_mat_descr descr,
                                            const rocsparse_hyb_mat   hyb,
                                            T*                        csr_val,
                                            rocsparse_int*            csr_row_ptr,
                                            rocsparse_int*            csr_col_ind,
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
    else if(hyb == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xhyb2csr"),
              (const void*&)descr,
              (const void*&)hyb,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)temp_buffer);

    log_bench(handle, "./rocsparse-bench -f hyb2csr -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes in HYB structure
    if(hyb->m < 0 || hyb->n < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(hyb->m == 0 || hyb->n == 0 || (hyb->ell_nnz == 0 && hyb->coo_nnz == 0))
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Temporary storage buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // COO row pointer buffer
    rocsparse_int* workspace = reinterpret_cast<rocsparse_int*>(ptr);

    // Get row offset pointers from COO part
    if(hyb->coo_nnz > 0)
    {
        // Shift ptr by workspace size
        ptr += sizeof(rocsparse_int) * (hyb->m / 256 + 1) * 256;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_coo2csr(
            handle, hyb->coo_row_ind, hyb->coo_nnz, hyb->m, workspace, descr->base));
    }

    // Compute row pointers
#define HYB2CSR_DIM 256
    dim3 hyb2csr_blocks((hyb->m - 1) / HYB2CSR_DIM + 1);
    dim3 hyb2csr_threads(HYB2CSR_DIM);

    hipLaunchKernelGGL((hyb2csr_nnz_kernel<HYB2CSR_DIM>),
                       hyb2csr_blocks,
                       hyb2csr_threads,
                       0,
                       stream,
                       hyb->m,
                       hyb->n,
                       hyb->ell_nnz,
                       hyb->ell_width,
                       hyb->ell_col_ind,
                       hyb->coo_nnz,
                       workspace,
                       csr_row_ptr,
                       descr->base);

    // Exclusive sum to obtain csr_row_ptr array
    size_t rocprim_size;
    void*  rocprim_buffer = reinterpret_cast<void*>(ptr);

    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                rocprim_size,
                                                csr_row_ptr,
                                                csr_row_ptr,
                                                descr->base,
                                                hyb->m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                rocprim_size,
                                                csr_row_ptr,
                                                csr_row_ptr,
                                                descr->base,
                                                hyb->m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));

    // Fill columns and values
    hipLaunchKernelGGL((hyb2csr_fill_kernel<HYB2CSR_DIM>),
                       hyb2csr_blocks,
                       hyb2csr_threads,
                       0,
                       stream,
                       hyb->m,
                       hyb->n,
                       hyb->ell_nnz,
                       hyb->ell_width,
                       hyb->ell_col_ind,
                       reinterpret_cast<T*>(hyb->ell_val),
                       hyb->coo_nnz,
                       workspace,
                       hyb->coo_col_ind,
                       reinterpret_cast<T*>(hyb->coo_val),
                       csr_row_ptr,
                       csr_col_ind,
                       csr_val,
                       descr->base);
#undef HYB2CSR_DIM

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_hyb2csr_buffer_size(rocsparse_handle          handle,
                                                          const rocsparse_mat_descr descr,
                                                          const rocsparse_hyb_mat   hyb,
                                                          const rocsparse_int*      csr_row_ptr,
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
    else if(hyb == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              "rocsparse_hyb2csr_buffer_size",
              (const void*&)descr,
              (const void*&)hyb,
              (const void*&)csr_row_ptr,
              (const void*&)buffer_size);

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes in HYB structure
    if(hyb->m < 0 || hyb->n < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check invalid buffer size pointer
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(hyb->m == 0 || hyb->n == 0 || (hyb->ell_nnz == 0 && hyb->coo_nnz == 0))
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Initialize buffer size
    *buffer_size = 0;

    // COO part requires conversion buffer
    if(hyb->coo_nnz > 0)
    {
        *buffer_size += sizeof(rocsparse_int) * (hyb->m / 256 + 1) * 256;
    }

    // Exclusive scan
    size_t         rocprim_size;
    rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(buffer_size);

    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                rocprim_size,
                                                ptr,
                                                ptr,
                                                descr->base,
                                                hyb->m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));

    // rocprim buffer
    *buffer_size += rocprim_size;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_shyb2csr(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               float*                    csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind,
                                               void*                     temp_buffer)
{
    return rocsparse_hyb2csr_template(
        handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

extern "C" rocsparse_status rocsparse_dhyb2csr(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               double*                   csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind,
                                               void*                     temp_buffer)
{
    return rocsparse_hyb2csr_template(
        handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

extern "C" rocsparse_status rocsparse_chyb2csr(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               rocsparse_float_complex*  csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind,
                                               void*                     temp_buffer)
{
    return rocsparse_hyb2csr_template(
        handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

extern "C" rocsparse_status rocsparse_zhyb2csr(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               rocsparse_double_complex* csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind,
                                               void*                     temp_buffer)
{
    return rocsparse_hyb2csr_template(
        handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}
