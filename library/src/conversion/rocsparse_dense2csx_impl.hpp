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
#include "utility.h"

#include "definitions.h"
#include "rocsparse_dense2csx.hpp"
#include <rocprim/rocprim.hpp>

template <rocsparse_direction DIRA, typename T>
rocsparse_status rocsparse_dense2csx_impl(rocsparse_handle          handle,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          const rocsparse_mat_descr descrA,
                                          const T*                  A,
                                          rocsparse_int             lda,
                                          const rocsparse_int*      nnzPerRowColumn,
                                          T*                        csxValA,
                                          rocsparse_int*            csxRowColPtrA,
                                          rocsparse_int*            csxColRowIndA)
{
    static constexpr bool is_row_oriented = (rocsparse_direction_row == DIRA);
    //
    // Checks for valid handle
    //
    if(nullptr == handle)
    {
        return rocsparse_status_invalid_handle;
    }

    //
    // Loggings
    //
    log_trace(handle,
              is_row_oriented ? "rocsparse_dense2csr" : "rocsparse_dense2csc",
              m,
              n,
              descrA,
              (const void*&)A,
              lda,
              (const void*&)nnzPerRowColumn,
              (const void*&)csxValA,
              (const void*&)csxRowColPtrA,
              (const void*&)csxColRowIndA);

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
              descrA->base);

    //
    // Check sizes
    //
    if((m < 0) || (n < 0) || (lda < m))
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
    if(nullptr == descrA || nullptr == nnzPerRowColumn || nullptr == A || nullptr == csxRowColPtrA
       || nullptr == csxColRowIndA || nullptr == csxValA)
    {
        return rocsparse_status_invalid_pointer;
    }

    //
    // Check the description type of the matrix.
    //
    if(rocsparse_matrix_type_general != descrA->type)
    {
        return rocsparse_status_not_implemented;
    }

    //
    // Compute csxRowColPtrA with the right index base.
    //
    {
        rocsparse_int dimdir = is_row_oriented ? m : n;

        rocsparse_int first_value = descrA->base;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(csxRowColPtrA,
                                           &first_value,
                                           sizeof(rocsparse_int),
                                           hipMemcpyHostToDevice,
                                           handle->stream));

        RETURN_IF_HIP_ERROR(hipMemcpy(csxRowColPtrA + 1,
                                      nnzPerRowColumn,
                                      sizeof(rocsparse_int) * dimdir,
                                      hipMemcpyDeviceToDevice));

        size_t temp_storage_bytes = 0;
        // Obtain rocprim buffer size
        RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                    temp_storage_bytes,
                                                    csxRowColPtrA,
                                                    csxRowColPtrA,
                                                    dimdir + 1,
                                                    rocprim::plus<rocsparse_int>(),
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
                                                    csxRowColPtrA,
                                                    csxRowColPtrA,
                                                    dimdir + 1,
                                                    rocprim::plus<rocsparse_int>(),
                                                    handle->stream));
        // Free rocprim buffer, if allocated
        if(d_temp_alloc == true)
        {
            RETURN_IF_HIP_ERROR(hipFree(d_temp_storage));
        }
    }

    //
    // Compute csxValA csxColRowIndA with right index base and update the 0-based csxRowColPtrA if necessary.
    //
    return rocsparse_dense2csx_template<DIRA>(
        handle, m, n, descrA, A, lda, csxValA, csxRowColPtrA, csxColRowIndA);
}
