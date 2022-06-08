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

#include "definitions.h"
#include "utility.h"

#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_bsrsm_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_X,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nrhs,
                                                      rocsparse_int             nnzb,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             block_dim,
                                                      rocsparse_mat_info        info,
                                                      size_t*                   buffer_size)
{
    // Check for valid handle, matrix descriptor and info
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrsm_buffer_size"),
              dir,
              trans_A,
              trans_X,
              mb,
              nrhs,
              nnzb,
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              (const void*&)info,
              (const void*&)buffer_size);

    if(rocsparse_enum_utils::is_invalid(dir) || rocsparse_enum_utils::is_invalid(trans_A)
       || rocsparse_enum_utils::is_invalid(trans_X))
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
    if(mb < 0 || nrhs < 0 || nnzb < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid buffer_size pointer
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(mb == 0 || nrhs == 0)
    {
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((bsr_val == nullptr && bsr_col_ind != nullptr)
       || (bsr_val != nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnzb != 0 && (bsr_val == nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // rocsparse_int max_nnz
    *buffer_size = 256;

    // 16 columns per block seem to work very well
    unsigned int ncol = 16;

    int narrays = (nrhs - 1) / ncol + 1;

    // int done_array
    *buffer_size += sizeof(int) * ((size_t(mb) * narrays - 1) / 256 + 1) * 256;

    // rocsparse_int workspace
    *buffer_size += sizeof(rocsparse_int) * ((mb - 1) / 256 + 1) * 256;

    // int workspace2
    *buffer_size += sizeof(int) * ((mb - 1) / 256 + 1) * 256;

    size_t         rocprim_size;
    rocsparse_int* ptr  = reinterpret_cast<rocsparse_int*>(buffer_size);
    int*           ptr2 = reinterpret_cast<int*>(buffer_size);

    rocprim::double_buffer<rocsparse_int> dummy(ptr, ptr);
    rocprim::double_buffer<int>           dummy2(ptr2, ptr2);

    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, rocprim_size, dummy2, dummy, mb, 0, 32, stream));

    // rocprim buffer
    *buffer_size += rocprim_size;

    // Additional buffer to store transpose of B, if trans_X == rocsparse_operation_none
    if(trans_X == rocsparse_operation_none)
    {
        *buffer_size += sizeof(T) * ((size_t(mb) * block_dim * nrhs - 1) / 256 + 1) * 256;
    }

    // Additional buffer to store transpose A, if transA == rocsparse_operation_transpose
    if(trans_A == rocsparse_operation_transpose)
    {
        size_t transpose_size;

        // Determine rocprim buffer size
        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, transpose_size, dummy, dummy, nnzb, 0, 32, stream));

        // rocPRIM does not support in-place sorting, so we need an additional buffer
        transpose_size += sizeof(rocsparse_int) * ((nnzb - 1) / 256 + 1) * 256;
        transpose_size += sizeof(T) * ((size_t(nnzb) * block_dim * block_dim - 1) / 256 + 1) * 256;

        *buffer_size += transpose_size;
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_operation       trans_A,     \
                                     rocsparse_operation       trans_X,     \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             nrhs,        \
                                     rocsparse_int             nnzb,        \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     rocsparse_mat_info        info,        \
                                     size_t*                   buffer_size) \
    {                                                                       \
        return rocsparse_bsrsm_buffer_size_template(handle,                 \
                                                    dir,                    \
                                                    trans_A,                \
                                                    trans_X,                \
                                                    mb,                     \
                                                    nrhs,                   \
                                                    nnzb,                   \
                                                    descr,                  \
                                                    bsr_val,                \
                                                    bsr_row_ptr,            \
                                                    bsr_col_ind,            \
                                                    block_dim,              \
                                                    info,                   \
                                                    buffer_size);           \
    }

C_IMPL(rocsparse_sbsrsm_buffer_size, float);
C_IMPL(rocsparse_dbsrsm_buffer_size, double);
C_IMPL(rocsparse_cbsrsm_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsm_buffer_size, rocsparse_double_complex);

#undef C_IMPL
