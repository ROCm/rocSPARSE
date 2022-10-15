/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csr2csc.hpp"
#include "common.h"
#include "definitions.h"
#include "utility.h"

#include "csr2csc_device.h"
#include "rocsparse_coo2csr.hpp"
#include "rocsparse_csr2coo.hpp"
#include "rocsparse_identity.hpp"
#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_csr2csc_core(rocsparse_handle     handle,
                                        rocsparse_int        m,
                                        rocsparse_int        n,
                                        rocsparse_int        nnz,
                                        const T*             csr_val,
                                        const rocsparse_int* csr_row_ptr_begin,
                                        const rocsparse_int* csr_row_ptr_end,
                                        const rocsparse_int* csr_col_ind,
                                        T*                   csc_val,
                                        rocsparse_int*       csc_row_ind,
                                        rocsparse_int*       csc_col_ptr,
                                        rocsparse_action     copy_values,
                                        rocsparse_index_base idx_base,
                                        void*                temp_buffer)
{
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
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2coo_core(
            handle, csr_row_ptr_begin, csr_row_ptr_end, nnz, m, csc_row_ind, idx_base));
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
            rocsparse_coo2csr_core(handle, keys.current(), nnz, n, csc_col_ptr, idx_base));

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
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_create_identity_permutation_core(handle, nnz, tmp_perm));

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
            rocsparse_coo2csr_core(handle, keys.current(), nnz, n, csc_col_ptr, idx_base));

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2coo_core(
            handle, csr_row_ptr_begin, csr_row_ptr_end, nnz, m, tmp_work1, idx_base));

// Permute row indices and values
#define CSR2CSC_DIM 512
        dim3 csr2csc_blocks((nnz - 1) / CSR2CSC_DIM + 1);
        dim3 csr2csc_threads(CSR2CSC_DIM);
        hipLaunchKernelGGL((csr2csc_permute_kernel<CSR2CSC_DIM>),
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

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    if(nnz == 0)
    {
        hipLaunchKernelGGL((set_array_to_value<256>),
                           dim3(n / 256 + 1),
                           dim3(256),
                           0,
                           handle->stream,
                           (n + 1),
                           csc_col_ptr,
                           static_cast<rocsparse_int>(idx_base));

        return rocsparse_status_success;
    }

    return rocsparse_csr2csc_core(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_row_ptr + 1,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
}

template rocsparse_status
    rocsparse_csr2csc_template<rocsparse_int>(rocsparse_handle     handle,
                                              rocsparse_int        m,
                                              rocsparse_int        n,
                                              rocsparse_int        nnz,
                                              const rocsparse_int* csr_val,
                                              const rocsparse_int* csr_row_ptr,
                                              const rocsparse_int* csr_col_ind,
                                              rocsparse_int*       csc_val,
                                              rocsparse_int*       csc_row_ind,
                                              rocsparse_int*       csc_col_ptr,
                                              rocsparse_action     copy_values,
                                              rocsparse_index_base idx_base,
                                              void*                temp_buffer);

template <typename T>
rocsparse_status rocsparse_csr2csc_impl(rocsparse_handle     handle,
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

    // Check action
    if(rocsparse_enum_utils::is_invalid(copy_values))
    {
        return rocsparse_status_invalid_value;
    }

    // Check index base
    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if((m > 0 && csr_row_ptr == nullptr) || (n > 0 && csc_col_ptr == nullptr)
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(copy_values == rocsparse_action_numeric)
    {
        // value arrays and column indices arrays must both be null (zero matrix) or both not null
        if((csr_val == nullptr && csr_col_ind != nullptr)
           || (csr_val != nullptr && csr_col_ind == nullptr))
        {
            return rocsparse_status_invalid_pointer;
        }

        if((csc_val == nullptr && csc_row_ind != nullptr)
           || (csc_val != nullptr && csc_row_ind == nullptr))
        {
            return rocsparse_status_invalid_pointer;
        }

        if(nnz != 0 && (csr_val == nullptr && csr_col_ind == nullptr))
        {
            return rocsparse_status_invalid_pointer;
        }

        if(nnz != 0 && (csc_val == nullptr && csc_row_ind == nullptr))
        {
            return rocsparse_status_invalid_pointer;
        }
    }
    else
    {
        // if copying symbolically, then column/row indices arrays can only be null if the zero matrix
        if(nnz != 0 && (csr_col_ind == nullptr || csc_row_ind == nullptr))
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    return rocsparse_csr2csc_template(handle,
                                      m,
                                      n,
                                      nnz,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      csc_val,
                                      csc_row_ind,
                                      csc_col_ptr,
                                      copy_values,
                                      idx_base,
                                      temp_buffer);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
rocsparse_status rocsparse_csr2csc_buffer_size_core(rocsparse_handle     handle,
                                                    rocsparse_int        m,
                                                    rocsparse_int        n,
                                                    rocsparse_int        nnz,
                                                    const rocsparse_int* csr_row_ptr_begin,
                                                    const rocsparse_int* csr_row_ptr_end,
                                                    const rocsparse_int* csr_col_ind,
                                                    rocsparse_action     copy_values,
                                                    size_t*              buffer_size)
{

    hipStream_t stream = handle->stream;

    // Determine rocprim buffer size
    rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(buffer_size);

    rocprim::double_buffer<rocsparse_int> dummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, *buffer_size, dummy, dummy, nnz, 0, 32, stream));

    *buffer_size = ((*buffer_size - 1) / 256 + 1) * 256;

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // Do not return 0 as size
    if(*buffer_size == 0)
    {
        *buffer_size = 4;
    }
    return rocsparse_status_success;
}

rocsparse_status rocsparse_csr2csc_buffer_size_template(rocsparse_handle     handle,
                                                        rocsparse_int        m,
                                                        rocsparse_int        n,
                                                        rocsparse_int        nnz,
                                                        const rocsparse_int* csr_row_ptr,
                                                        const rocsparse_int* csr_col_ind,
                                                        rocsparse_action     copy_values,
                                                        size_t*              buffer_size)
{
    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    return rocsparse_csr2csc_buffer_size_core(
        handle, m, n, nnz, csr_row_ptr, csr_row_ptr + 1, csr_col_ind, copy_values, buffer_size);
}

extern "C" rocsparse_status rocsparse_csr2csc_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_action     copy_values,
                                                          size_t*              buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csr2csc_buffer_size",
              m,
              n,
              nnz,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              copy_values,
              (const void*&)buffer_size);

    // Check action
    if(rocsparse_enum_utils::is_invalid(copy_values))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check buffer size argument
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check pointer arguments
    if(m > 0 && csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(nnz > 0 && csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_csr2csc_buffer_size_template(
        handle, m, n, nnz, csr_row_ptr, csr_col_ind, copy_values, buffer_size);

    hipStream_t stream = handle->stream;

    // Determine rocprim buffer size
    rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(buffer_size);

    rocprim::double_buffer<rocsparse_int> dummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, *buffer_size, dummy, dummy, nnz, 0, 32, stream));

    *buffer_size = ((*buffer_size - 1) / 256 + 1) * 256;

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // Do not return 0 as size
    if(*buffer_size == 0)
    {
        *buffer_size = 4;
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsr2csc(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const float*         csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               float*               csc_val,
                                               rocsparse_int*       csc_row_ind,
                                               rocsparse_int*       csc_col_ptr,
                                               rocsparse_action     copy_values,
                                               rocsparse_index_base idx_base,
                                               void*                temp_buffer)
{
    return rocsparse_csr2csc_impl(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsr2csc(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const double*        csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               double*              csc_val,
                                               rocsparse_int*       csc_row_ind,
                                               rocsparse_int*       csc_col_ptr,
                                               rocsparse_action     copy_values,
                                               rocsparse_index_base idx_base,
                                               void*                temp_buffer)
{
    return rocsparse_csr2csc_impl(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
}

extern "C" rocsparse_status rocsparse_ccsr2csc(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               rocsparse_int                  n,
                                               rocsparse_int                  nnz,
                                               const rocsparse_float_complex* csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               const rocsparse_int*           csr_col_ind,
                                               rocsparse_float_complex*       csc_val,
                                               rocsparse_int*                 csc_row_ind,
                                               rocsparse_int*                 csc_col_ptr,
                                               rocsparse_action               copy_values,
                                               rocsparse_index_base           idx_base,
                                               void*                          temp_buffer)
{
    return rocsparse_csr2csc_impl(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
}

extern "C" rocsparse_status rocsparse_zcsr2csc(rocsparse_handle                handle,
                                               rocsparse_int                   m,
                                               rocsparse_int                   n,
                                               rocsparse_int                   nnz,
                                               const rocsparse_double_complex* csr_val,
                                               const rocsparse_int*            csr_row_ptr,
                                               const rocsparse_int*            csr_col_ind,
                                               rocsparse_double_complex*       csc_val,
                                               rocsparse_int*                  csc_row_ind,
                                               rocsparse_int*                  csc_col_ptr,
                                               rocsparse_action                copy_values,
                                               rocsparse_index_base            idx_base,
                                               void*                           temp_buffer)
{
    return rocsparse_csr2csc_impl(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
}
