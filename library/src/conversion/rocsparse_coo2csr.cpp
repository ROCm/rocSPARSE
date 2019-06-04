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

#include "coo2csr_device.h"
#include "handle.h"
#include "utility.h"

#include <hip/hip_runtime.h>

extern "C" rocsparse_status rocsparse_coo2csr(rocsparse_handle     handle,
                                              const rocsparse_int* coo_row_ind,
                                              rocsparse_int        nnz,
                                              rocsparse_int        m,
                                              rocsparse_int*       csr_row_ptr,
                                              rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              "rocsparse_coo2csr",
              (const void*&)coo_row_ind,
              nnz,
              m,
              (const void*&)csr_row_ptr,
              idx_base);

    log_bench(handle, "./rocsparse-bench -f coo2csr", "--mtx <matrix.mtx>");

    // Check sizes
    if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(nnz == 0 || m == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(coo_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define COO2CSR_DIM 512
    dim3 coo2csr_blocks((m - 1) / COO2CSR_DIM + 1);
    dim3 coo2csr_threads(COO2CSR_DIM);

    hipLaunchKernelGGL((coo2csr_kernel),
                       coo2csr_blocks,
                       coo2csr_threads,
                       0,
                       stream,
                       m,
                       nnz,
                       coo_row_ind,
                       csr_row_ptr,
                       idx_base);
#undef COO2CSR_DIM
    return rocsparse_status_success;
}
