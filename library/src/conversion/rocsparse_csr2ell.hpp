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
#ifndef ROCSPARSE_CSR2ELL_HPP
#define ROCSPARSE_CSR2ELL_HPP

#include "csr2ell_device.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_csr2ell_template(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            const rocsparse_mat_descr csr_descr,
                                            const T*                  csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            const rocsparse_mat_descr ell_descr,
                                            rocsparse_int             ell_width,
                                            T*                        ell_val,
                                            rocsparse_int*            ell_col_ind)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(csr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(ell_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2ell"),
              m,
              (const void*&)csr_descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)ell_descr,
              ell_width,
              (const void*&)ell_val,
              (const void*&)ell_col_ind);

    log_bench(handle, "./rocsparse-bench -f csr2ell -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check index base
    if(csr_descr->base != rocsparse_index_base_zero && csr_descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(ell_descr->base != rocsparse_index_base_zero && ell_descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    // Check matrix type
    if(csr_descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    if(ell_descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || ell_width < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || ell_width == 0)
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
    else if(ell_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define CSR2ELL_DIM 512
    dim3 csr2ell_blocks((m - 1) / CSR2ELL_DIM + 1);
    dim3 csr2ell_threads(CSR2ELL_DIM);

    hipLaunchKernelGGL((csr2ell_kernel<T>),
                       csr2ell_blocks,
                       csr2ell_threads,
                       0,
                       stream,
                       m,
                       csr_val,
                       csr_row_ptr,
                       csr_col_ind,
                       csr_descr->base,
                       ell_width,
                       ell_col_ind,
                       ell_val,
                       ell_descr->base);
#undef CSR2ELL_DIM
    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSR2ELL_HPP
