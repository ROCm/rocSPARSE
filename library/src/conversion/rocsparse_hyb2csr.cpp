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
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime_api.h>
#include <rocprim/rocprim.hpp>

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
