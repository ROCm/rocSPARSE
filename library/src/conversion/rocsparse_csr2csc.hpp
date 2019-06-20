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
#ifndef ROCSPARSE_CSR2CSC_HPP
#define ROCSPARSE_CSR2CSC_HPP

#include "csr2csc_device.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_csr2csc_template(rocsparse_handle     handle,
                                            rocsparse_int        m,
                                            rocsparse_int        n,
                                            rocsparse_int        nnz,
                                            const T*             csr_val,
                                            const rocsparse_int* csr_row_ptr,
                                            const rocsparse_int* csr_col_ind,
                                            T*                   csc_val,
                                            rocsparse_int*       csc_row_ind,
                                            rocsparse_int*       csc_col_ptr,
                                            rocsparse_action     copy_values,
                                            rocsparse_index_base idx_base,
                                            void*                temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2csc"),
              m,
              n,
              nnz,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)csc_val,
              (const void*&)csc_row_ind,
              (const void*&)csc_col_ptr,
              copy_values,
              idx_base,
              (const void*&)temp_buffer);

    log_bench(handle, "./rocsparse-bench -f csr2csc -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check index base
    if(idx_base != rocsparse_index_base_zero && idx_base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr && rocsparse_action_numeric)
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
    else if(csc_val == nullptr && copy_values == rocsparse_action_numeric)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csc_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csc_col_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    unsigned int startbit = 0;
    unsigned int endbit   = rocsparse_clz(n);

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // work1 buffer
    rocsparse_int* tmp_work1 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // work2 buffer
    rocsparse_int* tmp_work2 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // perm buffer
    rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // rocprim buffer
    void* tmp_rocprim = reinterpret_cast<void*>(ptr);

    // Load CSR column indices into work1 buffer
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        tmp_work1, csr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice, stream));

    if(copy_values == rocsparse_action_symbolic)
    {
        // action symbolic

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_csr2coo(handle, csr_row_ptr, nnz, m, csc_row_ind, idx_base));
        // Stable sort COO by columns
        rocprim::double_buffer<rocsparse_int> keys(tmp_work1, tmp_perm);
        rocprim::double_buffer<rocsparse_int> vals(csc_row_ind, tmp_work2);

        size_t size = 0;

        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, size, keys, vals, nnz, startbit, endbit, stream));
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            tmp_rocprim, size, keys, vals, nnz, startbit, endbit, stream));

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_coo2csr(handle, keys.current(), nnz, n, csc_col_ptr, idx_base));

        // Copy csc_row_ind if not current
        if(vals.current() != csc_row_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(csc_row_ind,
                                               vals.current(),
                                               sizeof(rocsparse_int) * nnz,
                                               hipMemcpyDeviceToDevice,
                                               stream));
        }
    }
    else
    {
        // action numeric

        // Create identitiy permutation
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, nnz, tmp_perm));

        // Stable sort COO by columns
        rocprim::double_buffer<rocsparse_int> keys(tmp_work1, csc_row_ind);
        rocprim::double_buffer<rocsparse_int> vals(tmp_perm, tmp_work2);

        size_t size = 0;

        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, size, keys, vals, nnz, startbit, endbit, stream));
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            tmp_rocprim, size, keys, vals, nnz, startbit, endbit, stream));

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_coo2csr(handle, keys.current(), nnz, n, csc_col_ptr, idx_base));

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_csr2coo(handle, csr_row_ptr, nnz, m, tmp_work1, idx_base));

// Permute row indices and values
#define CSR2CSC_DIM 512
        dim3 csr2csc_blocks((nnz - 1) / CSR2CSC_DIM + 1);
        dim3 csr2csc_threads(CSR2CSC_DIM);

        hipLaunchKernelGGL((csr2csc_permute_kernel<T>),
                           csr2csc_blocks,
                           csr2csc_threads,
                           0,
                           stream,
                           nnz,
                           tmp_work1,
                           csr_val,
                           vals.current(),
                           csc_row_ind,
                           csc_val);
#undef CSR2CSC_DIM
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSR2CSC_HPP
