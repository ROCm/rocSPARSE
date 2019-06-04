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

#include "csr2coo_device.h"
#include "handle.h"
#include "utility.h"

#include <hip/hip_runtime.h>

extern "C" rocsparse_status rocsparse_csr2coo(rocsparse_handle     handle,
                                              const rocsparse_int* csr_row_ptr,
                                              rocsparse_int        nnz,
                                              rocsparse_int        m,
                                              rocsparse_int*       coo_row_ind,
                                              rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging TODO bench logging
    log_trace(handle,
              "rocsparse_csr2coo",
              (const void*&)csr_row_ptr,
              nnz,
              m,
              (const void*&)coo_row_ind,
              idx_base);

    log_bench(handle, "./rocsparse-bench -f csr2coo ", "--mtx <matrix.mtx>");

    // Check sizes
    if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(coo_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(nnz == 0 || m == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define CSR2COO_DIM 512
    rocsparse_int nnz_per_row = nnz / m;

    dim3 csr2coo_blocks((m - 1) / CSR2COO_DIM + 1);
    dim3 csr2coo_threads(CSR2COO_DIM);

    if(handle->wavefront_size == 32)
    {
        if(nnz_per_row < 4)
        {
            hipLaunchKernelGGL((csr2coo_kernel<2>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
        else if(nnz_per_row < 8)
        {
            hipLaunchKernelGGL((csr2coo_kernel<4>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
        else if(nnz_per_row < 16)
        {
            hipLaunchKernelGGL((csr2coo_kernel<8>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
        else if(nnz_per_row < 32)
        {
            hipLaunchKernelGGL((csr2coo_kernel<16>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
        else
        {
            hipLaunchKernelGGL((csr2coo_kernel<32>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
    }
    else if(handle->wavefront_size == 64)
    {
        if(nnz_per_row < 4)
        {
            hipLaunchKernelGGL((csr2coo_kernel<2>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
        else if(nnz_per_row < 8)
        {
            hipLaunchKernelGGL((csr2coo_kernel<4>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
        else if(nnz_per_row < 16)
        {
            hipLaunchKernelGGL((csr2coo_kernel<8>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
        else if(nnz_per_row < 32)
        {
            hipLaunchKernelGGL((csr2coo_kernel<16>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
        else if(nnz_per_row < 64)
        {
            hipLaunchKernelGGL((csr2coo_kernel<32>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
        else
        {
            hipLaunchKernelGGL((csr2coo_kernel<64>),
                               csr2coo_blocks,
                               csr2coo_threads,
                               0,
                               stream,
                               m,
                               csr_row_ptr,
                               coo_row_ind,
                               idx_base);
        }
    }
    else
    {
        return rocsparse_status_arch_mismatch;
    }
#undef CSR2COO_DIM
    return rocsparse_status_success;
}
