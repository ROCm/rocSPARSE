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
#ifndef ROCSPARSE_COO2DENSE_HPP
#define ROCSPARSE_COO2DENSE_HPP

#include "utility.h"

#include "definitions.h"

#include "coo2dense_device.h"

#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_coo2dense_template(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const T*                  coo_val,
                                              const rocsparse_int*      coo_row_ind,
                                              const rocsparse_int*      coo_col_ind,
                                              T*                        A,
                                              rocsparse_int             lda)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcoo2dense"),
              m,
              n,
              nnz,
              descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (void*&)A,
              lda);

    log_bench(handle, "./rocsparse-bench -f coo2dense -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0 || lda < m)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {

        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(coo_val == nullptr || coo_row_ind == nullptr || coo_col_ind == nullptr || A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define COO2DENSE_DIM 512
    dim3 blocks((nnz - 1) / COO2DENSE_DIM + 1);
    dim3 threads(COO2DENSE_DIM);

    hipLaunchKernelGGL((coo2dense_kernel<COO2DENSE_DIM, T>),
                       blocks,
                       threads,
                       0,
                       stream,
                       m,
                       n,
                       nnz,
                       lda,
                       descr->base,
                       coo_val,
                       coo_row_ind,
                       coo_col_ind,
                       A);
#undef COO2DENSE_DIM

    return rocsparse_status_success;
}

#endif // ROCSPARSE_COO2DENSE_HPP
