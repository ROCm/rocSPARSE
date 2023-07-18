/*! \file */
/* ************************************************************************
* Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_dense2coo.h"
#include "rocsparse_csr2coo.hpp"
#include "rocsparse_dense2coo.hpp"

#include "rocsparse_dense2csx_impl.hpp"

#include <rocprim/rocprim.hpp>

template <typename I, typename T>
rocsparse_status rocsparse_dense2coo_template(rocsparse_handle          handle,
                                              rocsparse_order           order,
                                              I                         m,
                                              I                         n,
                                              const rocsparse_mat_descr descr,
                                              const T*                  A,
                                              I                         ld,
                                              const I*                  nnz_per_rows,
                                              T*                        coo_val,
                                              I*                        coo_row_ind,
                                              I*                        coo_col_ind)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xdense2coo"),
              order,
              m,
              n,
              descr,
              (const void*&)A,
              ld,
              (const void*&)nnz_per_rows,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind);

    log_bench(handle, "./rocsparse-bench -f dense2coo -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check order
    if(rocsparse_enum_utils::is_invalid(order))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_requires_sorted_storage;
    }

    // Check sizes
    if(m < 0 || n < 0 || ld < (order == rocsparse_order_column ? m : n))
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(A == nullptr || nnz_per_rows == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value, row, and column arrays must all be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    I* row_ptr;
    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&row_ptr, sizeof(I) * (m + 1), handle->stream));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dense2csx_impl<rocsparse_direction_row>(
        handle, order, m, n, descr, A, ld, nnz_per_rows, coo_val, row_ptr, coo_col_ind));

    I start;
    I end;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&start, &row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&end, &row_ptr[m], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    I nnz = end - start;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csr2coo_template(handle, row_ptr, nnz, m, coo_row_ind, descr->base));

    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(row_ptr, handle->stream));

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                         \
    template rocsparse_status rocsparse_dense2coo_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                 \
        rocsparse_order           order,                                  \
        ITYPE                     m,                                      \
        ITYPE                     n,                                      \
        const rocsparse_mat_descr descr,                                  \
        const TTYPE*              A,                                      \
        ITYPE                     ld,                                     \
        const ITYPE*              nnz_per_rows,                           \
        TTYPE*                    coo_val,                                \
        ITYPE*                    coo_row_ind,                            \
        ITYPE*                    coo_col_ind);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status rocsparse_sdense2coo(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              A,
                                                 rocsparse_int             ld,
                                                 const rocsparse_int*      nnz_per_rows,
                                                 float*                    coo_val,
                                                 rocsparse_int*            coo_row_ind,
                                                 rocsparse_int*            coo_col_ind)
try
{
    return rocsparse_dense2coo_template(handle,
                                        rocsparse_order_column,
                                        m,
                                        n,
                                        descr,
                                        A,
                                        ld,
                                        nnz_per_rows,
                                        coo_val,
                                        coo_row_ind,
                                        coo_col_ind);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_ddense2coo(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             A,
                                                 rocsparse_int             ld,
                                                 const rocsparse_int*      nnz_per_rows,
                                                 double*                   coo_val,
                                                 rocsparse_int*            coo_row_ind,
                                                 rocsparse_int*            coo_col_ind)
try
{
    return rocsparse_dense2coo_template(handle,
                                        rocsparse_order_column,
                                        m,
                                        n,
                                        descr,
                                        A,
                                        ld,
                                        nnz_per_rows,
                                        coo_val,
                                        coo_row_ind,
                                        coo_col_ind);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_cdense2coo(rocsparse_handle               handle,
                                                 rocsparse_int                  m,
                                                 rocsparse_int                  n,
                                                 const rocsparse_mat_descr      descr,
                                                 const rocsparse_float_complex* A,
                                                 rocsparse_int                  ld,
                                                 const rocsparse_int*           nnz_per_rows,
                                                 rocsparse_float_complex*       coo_val,
                                                 rocsparse_int*                 coo_row_ind,
                                                 rocsparse_int*                 coo_col_ind)
try
{
    return rocsparse_dense2coo_template(handle,
                                        rocsparse_order_column,
                                        m,
                                        n,
                                        descr,
                                        A,
                                        ld,
                                        nnz_per_rows,
                                        coo_val,
                                        coo_row_ind,
                                        coo_col_ind);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

extern "C" rocsparse_status rocsparse_zdense2coo(rocsparse_handle                handle,
                                                 rocsparse_int                   m,
                                                 rocsparse_int                   n,
                                                 const rocsparse_mat_descr       descr,
                                                 const rocsparse_double_complex* A,
                                                 rocsparse_int                   ld,
                                                 const rocsparse_int*            nnz_per_rows,
                                                 rocsparse_double_complex*       coo_val,
                                                 rocsparse_int*                  coo_row_ind,
                                                 rocsparse_int*                  coo_col_ind)
try
{
    return rocsparse_dense2coo_template(handle,
                                        rocsparse_order_column,
                                        m,
                                        n,
                                        descr,
                                        A,
                                        ld,
                                        nnz_per_rows,
                                        coo_val,
                                        coo_row_ind,
                                        coo_col_ind);
}
catch(...)
{
    return exception_to_rocsparse_status();
}
