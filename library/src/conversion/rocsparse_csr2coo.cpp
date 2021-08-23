/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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
#include "utility.h"

#include "rocsparse_csr2coo.hpp"

#include "csr2coo_device.h"

template <typename I, typename J>
rocsparse_status rocsparse_csr2coo_template(rocsparse_handle     handle,
                                            const I*             csr_row_ptr,
                                            I                    nnz,
                                            J                    m,
                                            J*                   coo_row_ind,
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

    // Check index base
    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(nnz < 0 || m < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(nnz == 0 || m == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || coo_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define CSR2COO_DIM 512
    I nnz_per_row = nnz / m;

    dim3 csr2coo_blocks((m - 1) / CSR2COO_DIM + 1);
    dim3 csr2coo_threads(CSR2COO_DIM);

    if(handle->wavefront_size == 32)
    {
        if(nnz_per_row < 4)
        {
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 2>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 4>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 8>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 16>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 32>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 2>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 4>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 8>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 16>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 32>),
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
            hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 64>),
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

#define INSTANTIATE(ITYPE, JTYPE)                                       \
    template rocsparse_status rocsparse_csr2coo_template<ITYPE, JTYPE>( \
        rocsparse_handle     handle,                                    \
        const ITYPE*         csr_row_ptr,                               \
        ITYPE                nnz,                                       \
        JTYPE                m,                                         \
        JTYPE*               coo_row_ind,                               \
        rocsparse_index_base idx_base);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status rocsparse_csr2coo(rocsparse_handle     handle,
                                              const rocsparse_int* csr_row_ptr,
                                              rocsparse_int        nnz,
                                              rocsparse_int        m,
                                              rocsparse_int*       coo_row_ind,
                                              rocsparse_index_base idx_base)
{
    return rocsparse_csr2coo_template(handle, csr_row_ptr, nnz, m, coo_row_ind, idx_base);
}
