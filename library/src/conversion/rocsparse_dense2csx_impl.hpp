/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_DENSE2CSX_IMPL_HPP
#define ROCSPARSE_DENSE2CSX_IMPL_HPP

#include "utility.h"

#include "definitions.h"
#include "rocsparse_dense2csx.hpp"
#include <rocprim/rocprim.hpp>

template <rocsparse_direction DIRA, typename I, typename J, typename T>
rocsparse_status rocsparse_dense2csx_impl(rocsparse_handle          handle,
                                          rocsparse_order           order,
                                          J                         m,
                                          J                         n,
                                          const rocsparse_mat_descr descr_A,
                                          const T*                  A,
                                          I                         lda,
                                          const I*                  nnz_per_row_column,
                                          T*                        csx_val_A,
                                          I*                        csx_row_col_ptr_A,
                                          J*                        csx_col_row_ind_A)
{
    static constexpr bool is_row_oriented = (rocsparse_direction_row == DIRA);
    //
    // Checks for valid handle
    //
    if(nullptr == handle)
    {
        return rocsparse_status_invalid_handle;
    }

    if(nullptr == descr_A)
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Loggings
    //
    log_trace(handle,
              is_row_oriented ? "rocsparse_dense2csr" : "rocsparse_dense2csc",
              order,
              m,
              n,
              descr_A,
              (const void*&)A,
              lda,
              (const void*&)nnz_per_row_column,
              (const void*&)csx_val_A,
              (const void*&)csx_row_col_ptr_A,
              (const void*&)csx_col_row_ind_A);

    log_bench(handle,
              "./rocsparse-bench",
              "-f",
              is_row_oriented ? "dense2csr" : "dense2csc",
              "-m",
              m,
              "-n",
              n,
              "--denseld",
              lda,
              "--indexbaseA",
              descr_A->base);

    // Check order
    if(rocsparse_enum_utils::is_invalid(order))
    {
        return rocsparse_status_invalid_value;
    }

    //
    // Check sizes
    //
    if((m < 0) || (n < 0) || (lda < (order == rocsparse_order_column ? m : n)))
    {
        return rocsparse_status_invalid_size;
    }

    //
    // Quick return if possible, before checking for invalid pointers.
    //
    if(!m || !n)
    {
        return rocsparse_status_success;
    }

    //
    // Check invalid pointers.
    //
    if(nullptr == nnz_per_row_column || nullptr == A || nullptr == csx_row_col_ptr_A)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column/row indices arrays must both be null (zero matrix) or both not null
    if((csx_col_row_ind_A == nullptr && csx_val_A != nullptr)
       || (csx_col_row_ind_A != nullptr && csx_val_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Check the description type of the matrix.
    //
    if(rocsparse_matrix_type_general != descr_A->type)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr_A->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    //
    // Compute csx_row_col_ptr_A with the right index base.
    //
    {
        J dimdir = is_row_oriented ? m : n;

        I first_value = descr_A->base;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            csx_row_col_ptr_A, &first_value, sizeof(I), hipMemcpyHostToDevice, handle->stream));

        RETURN_IF_HIP_ERROR(hipMemcpy(csx_row_col_ptr_A + 1,
                                      nnz_per_row_column,
                                      sizeof(I) * dimdir,
                                      hipMemcpyDeviceToDevice));

        size_t temp_storage_bytes = 0;
        // Obtain rocprim buffer size
        RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                    temp_storage_bytes,
                                                    csx_row_col_ptr_A,
                                                    csx_row_col_ptr_A,
                                                    dimdir + 1,
                                                    rocprim::plus<I>(),
                                                    handle->stream));

        // Get rocprim buffer
        bool  d_temp_alloc;
        void* d_temp_storage;

        // Device buffer should be sufficient for rocprim in most cases
        if(handle->buffer_size >= temp_storage_bytes)
        {
            d_temp_storage = handle->buffer;
            d_temp_alloc   = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMalloc(&d_temp_storage, temp_storage_bytes));
            d_temp_alloc = true;
        }

        // Perform actual inclusive sum
        RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(d_temp_storage,
                                                    temp_storage_bytes,
                                                    csx_row_col_ptr_A,
                                                    csx_row_col_ptr_A,
                                                    dimdir + 1,
                                                    rocprim::plus<I>(),
                                                    handle->stream));
        // Free rocprim buffer, if allocated
        if(d_temp_alloc == true)
        {
            RETURN_IF_HIP_ERROR(hipFree(d_temp_storage));
        }
    }

    if(csx_col_row_ind_A == nullptr && csx_val_A == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        if(is_row_oriented)
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(
                &end, &csx_row_col_ptr_A[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(hipMemcpy(
                &start, &csx_row_col_ptr_A[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(
                &end, &csx_row_col_ptr_A[n], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(hipMemcpy(
                &start, &csx_row_col_ptr_A[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        }

        rocsparse_int nnz = (end - start);

        if(nnz != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    //
    // Compute csx_val_A csx_col_row_ind_A with right index base and update the 0-based csx_row_col_ptr_A if necessary.
    //
    return rocsparse_dense2csx_template<DIRA>(
        handle, order, m, n, descr_A, A, lda, csx_val_A, csx_row_col_ptr_A, csx_col_row_ind_A);
}

#endif // ROCSPARSE_DENSE2CSX_IMPL_HPP
