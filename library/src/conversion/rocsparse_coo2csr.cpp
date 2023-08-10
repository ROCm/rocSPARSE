/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"
#include "utility.h"

#include "internal/conversion/rocsparse_coo2csr.h"
#include "rocsparse_coo2csr.hpp"

#include "coo2csr_device.h"

template <typename I, typename J>
rocsparse_status rocsparse_coo2csr_core(rocsparse_handle     handle,
                                        const J*             coo_row_ind,
                                        I                    nnz,
                                        J                    m,
                                        I*                   csr_row_ptr,
                                        rocsparse_index_base idx_base)
{
    // Stream
    hipStream_t stream = handle->stream;

#define COO2CSR_DIM 512
    dim3 coo2csr_blocks((m - 1) / COO2CSR_DIM + 1);
    dim3 coo2csr_threads(COO2CSR_DIM);

    hipLaunchKernelGGL((coo2csr_kernel<COO2CSR_DIM>),
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

template <typename I, typename J>
rocsparse_status rocsparse_coo2csr_template(rocsparse_handle     handle,
                                            const J*             coo_row_ind,
                                            I                    nnz,
                                            J                    m,
                                            I*                   csr_row_ptr,
                                            rocsparse_index_base idx_base)
{
    if(m == 0)
    {
        if(csr_row_ptr != nullptr)
        {
            hipLaunchKernelGGL((set_array_to_value<1>),
                               dim3(1),
                               dim3(1),
                               0,
                               handle->stream,
                               m + 1,
                               csr_row_ptr,
                               static_cast<I>(idx_base));
        }
        return rocsparse_status_success;
    }
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_coo2csr_core(handle, coo_row_ind, nnz, m, csr_row_ptr, idx_base));
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse_coo2csr_impl(rocsparse_handle     handle,
                                        const J*             coo_row_ind,
                                        I                    nnz,
                                        J                    m,
                                        I*                   csr_row_ptr,
                                        rocsparse_index_base idx_base)
{
    // Logging
    log_trace(handle,
              "rocsparse_coo2csr",
              (const void*&)coo_row_ind,
              nnz,
              m,
              (const void*&)csr_row_ptr,
              idx_base);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ARRAY(1, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_SIZE(2, nnz);
    ROCSPARSE_CHECKARG_SIZE(3, m);
    ROCSPARSE_CHECKARG_ARRAY(4, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ENUM(5, idx_base);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_coo2csr_template(handle, coo_row_ind, nnz, m, csr_row_ptr, idx_base));
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE)                                                                    \
    template rocsparse_status rocsparse_coo2csr_core<ITYPE, JTYPE>(rocsparse_handle     handle,      \
                                                                   const JTYPE*         coo_row_ind, \
                                                                   ITYPE                nnz,         \
                                                                   JTYPE                m,           \
                                                                   ITYPE*               csr_row_ptr, \
                                                                   rocsparse_index_base idx_base);   \
    template rocsparse_status rocsparse_coo2csr_impl<ITYPE, JTYPE>(rocsparse_handle     handle,      \
                                                                   const JTYPE*         coo_row_ind, \
                                                                   ITYPE                nnz,         \
                                                                   JTYPE                m,           \
                                                                   ITYPE*               csr_row_ptr, \
                                                                   rocsparse_index_base idx_base);   \
    template rocsparse_status rocsparse_coo2csr_template<ITYPE, JTYPE>(                              \
        rocsparse_handle     handle,                                                                 \
        const JTYPE*         coo_row_ind,                                                            \
        ITYPE                nnz,                                                                    \
        JTYPE                m,                                                                      \
        ITYPE*               csr_row_ptr,                                                            \
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

extern "C" rocsparse_status rocsparse_coo2csr(rocsparse_handle     handle,
                                              const rocsparse_int* coo_row_ind,
                                              rocsparse_int        nnz,
                                              rocsparse_int        m,
                                              rocsparse_int*       csr_row_ptr,
                                              rocsparse_index_base idx_base)
try
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_coo2csr_impl(handle, coo_row_ind, nnz, m, csr_row_ptr, idx_base));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
