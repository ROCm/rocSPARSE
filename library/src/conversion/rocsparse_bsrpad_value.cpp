/*! \file */
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include "rocsparse_bsrpad_value.hpp"
#include "definitions.h"
#include "utility.h"

#include "bsrpad_value_device.h"

template <typename T>
rocsparse_status rocsparse_bsrpad_value_template(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             mb,
                                                 rocsparse_int             nnzb,
                                                 rocsparse_int             block_dim,
                                                 T                         value,
                                                 const rocsparse_mat_descr bsr_descr,
                                                 T*                        bsr_val,
                                                 const rocsparse_int*      bsr_row_ptr,
                                                 const rocsparse_int*      bsr_col_ind)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check matrix descriptors
    if(bsr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrpad_value"),
              m,
              mb,
              nnzb,
              block_dim,
              value,
              bsr_descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind);

    log_bench(
        handle, "./rocsparse-bench -f bsrpad_value -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check sizes
    if(m < 0 || mb < 0 || nnzb < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    if((mb * block_dim < m) || (m <= (mb - 1) * block_dim))
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nnzb == 0)
    {
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

    if(bsr_val == nullptr && bsr_col_ind == nullptr && nnzb != 0)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(mb * block_dim == m)
    {
        return rocsparse_status_success;
    }

    constexpr rocsparse_int block_size = 1024;

    rocsparse_int grid_size = (block_dim + block_size - 1) / block_size;

    // Check matrix sorting mode
    if(bsr_descr->storage_mode == rocsparse_storage_mode_sorted)
    {
        hipLaunchKernelGGL((bsrpad_value_kernel_sorted<block_size>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           handle->stream,
                           m,
                           mb,
                           block_dim,
                           value,
                           bsr_descr->base,
                           bsr_val,
                           bsr_row_ptr,
                           bsr_col_ind);
    }
    else
    {
        hipLaunchKernelGGL((bsrpad_value_kernel_unsorted<block_size>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           handle->stream,
                           m,
                           mb,
                           block_dim,
                           value,
                           bsr_descr->base,
                           bsr_val,
                           bsr_row_ptr,
                           bsr_col_ind);
    }

    return rocsparse_status_success;
}

//
// C INTERFACE
//
#define C_IMPL(NAME, TYPE)                                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                            \
                                     rocsparse_int             m,                                 \
                                     rocsparse_int             mb,                                \
                                     rocsparse_int             nnzb,                              \
                                     rocsparse_int             block_dim,                         \
                                     TYPE                      value,                             \
                                     const rocsparse_mat_descr bsr_descr,                         \
                                     TYPE*                     bsr_val,                           \
                                     const rocsparse_int*      bsr_row_ptr,                       \
                                     const rocsparse_int*      bsr_col_ind)                       \
                                                                                                  \
    {                                                                                             \
        return rocsparse_bsrpad_value_template<TYPE>(                                             \
            handle, m, mb, nnzb, block_dim, value, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind); \
    }

C_IMPL(rocsparse_sbsrpad_value, float);
C_IMPL(rocsparse_dbsrpad_value, double);
C_IMPL(rocsparse_cbsrpad_value, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrpad_value, rocsparse_double_complex);
#undef C_IMPL
