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

#pragma once
#ifndef ROCSPARSE_DENSE2COO_HPP
#define ROCSPARSE_DENSE2COO_HPP

#include "definitions.h"
#include "utility.h"

#include "rocsparse_dense2csx_impl.hpp"

#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_dense2coo_template(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              const rocsparse_mat_descr descr,
                                              const T*                  A,
                                              rocsparse_int             ld,
                                              const rocsparse_int*      nnz_per_rows,
                                              T*                        coo_val,
                                              rocsparse_int*            coo_row_ind,
                                              rocsparse_int*            coo_col_ind)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xdense2coo"),
              m,
              n,
              descr,
              (const void*&)A,
              ld,
              (const void*&)nnz_per_rows,
              (void*&)coo_val,
              (void*&)coo_row_ind,
              (void*&)coo_col_ind);

    log_bench(handle, "./rocsparse-bench -f dense2coo -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check sizes
    if(m < 0 || n < 0 || ld < m)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(A == nullptr || nnz_per_rows == nullptr || coo_val == nullptr || coo_row_ind == nullptr
       || coo_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    rocsparse_int* row_ptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&row_ptr, (m + 1) * sizeof(rocsparse_int)));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dense2csx_impl<rocsparse_direction_row>(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, row_ptr, coo_col_ind));

    rocsparse_int start;
    rocsparse_int end;
    RETURN_IF_HIP_ERROR(
        hipMemcpy(&start, &row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(hipMemcpy(&end, &row_ptr[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost));

    rocsparse_int nnz = end - start;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2coo(handle, row_ptr, nnz, m, coo_row_ind, descr->base));

    RETURN_IF_HIP_ERROR(hipFree(row_ptr));

    return rocsparse_status_success;
}

#endif // ROCSPARSE_DENSE2COO_HPP
