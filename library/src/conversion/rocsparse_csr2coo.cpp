/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
rocsparse_status rocsparse_csr2coo_core(rocsparse_handle     handle,
                                        const I*             csr_row_ptr_begin,
                                        const I*             csr_row_ptr_end,
                                        I                    nnz,
                                        J                    m,
                                        J*                   coo_row_ind,
                                        rocsparse_index_base idx_base)
{

    // Stream
    hipStream_t stream = handle->stream;

    I nnz_per_row = nnz / m;

#define CSR2COO_DIM 256
    if(nnz_per_row < 4)
    {
        hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 2>),
                           dim3(((int64_t)2 * m - 1) / CSR2COO_DIM + 1),
                           dim3(CSR2COO_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_begin,
                           csr_row_ptr_end,
                           coo_row_ind,
                           idx_base);
    }
    else if(nnz_per_row < 8)
    {
        hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 4>),
                           dim3(((int64_t)4 * m - 1) / CSR2COO_DIM + 1),
                           dim3(CSR2COO_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_begin,
                           csr_row_ptr_end,
                           coo_row_ind,
                           idx_base);
    }
    else if(nnz_per_row < 16)
    {
        hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 8>),
                           dim3(((int64_t)8 * m - 1) / CSR2COO_DIM + 1),
                           dim3(CSR2COO_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_begin,
                           csr_row_ptr_end,
                           coo_row_ind,
                           idx_base);
    }
    else if(nnz_per_row < 32)
    {
        hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 16>),
                           dim3(((int64_t)16 * m - 1) / CSR2COO_DIM + 1),
                           dim3(CSR2COO_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_begin,
                           csr_row_ptr_end,
                           coo_row_ind,
                           idx_base);
    }
    else if(nnz_per_row < 64)
    {
        hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 32>),
                           dim3(((int64_t)32 * m - 1) / CSR2COO_DIM + 1),
                           dim3(CSR2COO_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_begin,
                           csr_row_ptr_end,
                           coo_row_ind,
                           idx_base);
    }
    else if(nnz_per_row < 128)
    {
        hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 64>),
                           dim3(((int64_t)64 * m - 1) / CSR2COO_DIM + 1),
                           dim3(CSR2COO_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_begin,
                           csr_row_ptr_end,
                           coo_row_ind,
                           idx_base);
    }
    else if(handle->wavefront_size == 64)
    {
        hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 128>),
                           dim3(((int64_t)128 * m - 1) / CSR2COO_DIM + 1),
                           dim3(CSR2COO_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_begin,
                           csr_row_ptr_end,
                           coo_row_ind,
                           idx_base);
    }
    else
    {
        hipLaunchKernelGGL((csr2coo_kernel<CSR2COO_DIM, 256>),
                           dim3(((int64_t)256 * m - 1) / CSR2COO_DIM + 1),
                           dim3(CSR2COO_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_begin,
                           csr_row_ptr_end,
                           coo_row_ind,
                           idx_base);
    }
#undef CSR2COO_DIM
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse_csr2coo_template(rocsparse_handle     handle,
                                            const I*             csr_row_ptr,
                                            I                    nnz,
                                            J                    m,
                                            J*                   coo_row_ind,
                                            rocsparse_index_base idx_base)
{
    // Quick return if possible
    if(nnz == 0 || m == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_csr2coo_core(
        handle, csr_row_ptr, csr_row_ptr + 1, nnz, m, coo_row_ind, idx_base);
}

template <typename I, typename J>
rocsparse_status rocsparse_csr2coo_impl(rocsparse_handle     handle,
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

    // Check pointer arguments
    if((m > 0 && csr_row_ptr == nullptr) || (nnz > 0 && coo_row_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_csr2coo_template(handle, csr_row_ptr, nnz, m, coo_row_ind, idx_base);
}

#define INSTANTIATE(ITYPE, JTYPE)                                                                    \
    template rocsparse_status rocsparse_csr2coo_template<ITYPE, JTYPE>(                              \
        rocsparse_handle     handle,                                                                 \
        const ITYPE*         csr_row_ptr,                                                            \
        ITYPE                nnz,                                                                    \
        JTYPE                m,                                                                      \
        JTYPE*               coo_row_ind,                                                            \
        rocsparse_index_base idx_base);                                                              \
    template rocsparse_status rocsparse_csr2coo_impl<ITYPE, JTYPE>(rocsparse_handle     handle,      \
                                                                   const ITYPE*         csr_row_ptr, \
                                                                   ITYPE                nnz,         \
                                                                   JTYPE                m,           \
                                                                   JTYPE*               coo_row_ind, \
                                                                   rocsparse_index_base idx_base)

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
    return rocsparse_csr2coo_impl(handle, csr_row_ptr, nnz, m, coo_row_ind, idx_base);
}
