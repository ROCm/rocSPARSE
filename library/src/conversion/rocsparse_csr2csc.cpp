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
#include "rocsparse.h"

#include "rocsparse_csr2csc.hpp"

#include <hip/hip_runtime_api.h>
#include <rocprim/rocprim.hpp>

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_csr2csc_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_action     copy_values,
                                                          size_t*              buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csr2csc_buffer_size",
              m,
              n,
              nnz,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              copy_values,
              (const void*&)buffer_size);

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check buffer size argument
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
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
    else if(csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    hipStream_t stream = handle->stream;

    // Determine rocprim buffer size
    rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(buffer_size);

    rocprim::double_buffer<rocsparse_int> dummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, *buffer_size, dummy, dummy, nnz, 0, 32, stream));

    *buffer_size = ((*buffer_size - 1) / 256 + 1) * 256;

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // Do not return 0 as size
    if(*buffer_size == 0)
    {
        *buffer_size = 4;
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsr2csc(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const float*         csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               float*               csc_val,
                                               rocsparse_int*       csc_row_ind,
                                               rocsparse_int*       csc_col_ptr,
                                               rocsparse_action     copy_values,
                                               rocsparse_index_base idx_base,
                                               void*                temp_buffer)
{
    return rocsparse_csr2csc_template<float>(handle,
                                             m,
                                             n,
                                             nnz,
                                             csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             csc_val,
                                             csc_row_ind,
                                             csc_col_ptr,
                                             copy_values,
                                             idx_base,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsr2csc(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const double*        csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               double*              csc_val,
                                               rocsparse_int*       csc_row_ind,
                                               rocsparse_int*       csc_col_ptr,
                                               rocsparse_action     copy_values,
                                               rocsparse_index_base idx_base,
                                               void*                temp_buffer)
{
    return rocsparse_csr2csc_template<double>(handle,
                                              m,
                                              n,
                                              nnz,
                                              csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              csc_val,
                                              csc_row_ind,
                                              csc_col_ptr,
                                              copy_values,
                                              idx_base,
                                              temp_buffer);
}
