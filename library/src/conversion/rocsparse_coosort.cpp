/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_coosort.h"
#include "internal/conversion/rocsparse_inverse_permutation.h"

#include "utility.h"

#include "control.h"
#include "coosort_device.h"
#include "rocsparse_coosort.hpp"
#include "rocsparse_identity.hpp"
#include <rocprim/rocprim.hpp>

template <typename J>
rocsparse_status rocsparse::coosort_buffer_size_template(rocsparse_handle handle,
                                                         J                m,
                                                         J                n,
                                                         J                nnz,
                                                         const J*         coo_row_ind,
                                                         const J*         coo_col_ind,
                                                         size_t*          buffer_size)
{

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_buffer_size",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size);

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    J* ptr = reinterpret_cast<J*>(buffer_size);

    // Determine max buffer size
    size_t size;
    *buffer_size = 0;
    rocprim::double_buffer<J> dummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(rocprim::run_length_encode(nullptr, size, ptr, nnz, ptr, ptr, ptr, stream));
    *buffer_size = rocsparse::max(size, *buffer_size);
    RETURN_IF_HIP_ERROR(
        rocprim::exclusive_scan(nullptr, size, ptr, ptr, 0, m + 1, rocprim::plus<J>(), stream));
    *buffer_size = rocsparse::max(size, *buffer_size);
    RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(nullptr, size, dummy, dummy, nnz, 0, 32, stream));
    *buffer_size = rocsparse::max(size, *buffer_size);
    rocprim::double_buffer<J> rpdummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(
        nullptr, size, rpdummy, rpdummy, nnz, m, ptr, ptr + 1, 0, 32, stream));
    *buffer_size = rocsparse::max(size, *buffer_size);
    *buffer_size = ((*buffer_size - 1) / 256 + 1) * 256;

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays

    // rows buffer
    *buffer_size += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;
    // columns buffer
    *buffer_size += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;
    // perm buffer
    *buffer_size += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;
    // segment buffer
    *buffer_size += ((sizeof(J) * rocsparse::max(m, n)) / 256 + 1) * 256;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_coosort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* coo_row_ind,
                                                          const rocsparse_int* coo_col_ind,
                                                          size_t*              buffer_size)
try
{
    return rocsparse::coosort_buffer_size_template(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, buffer_size);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

namespace rocsparse
{
    template <typename J>
    static rocsparse_status coosort_by_row_quickreturn(rocsparse_handle handle,
                                                       J                m,
                                                       J                n,
                                                       J                nnz,
                                                       J*               coo_row_ind,
                                                       J*               coo_col_ind,
                                                       J*               perm,
                                                       void*            temp_buffer)
    {

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename J>
    static rocsparse_status coosort_by_row_checkarg(rocsparse_handle handle,
                                                    J                m,
                                                    J                n,
                                                    J                nnz,
                                                    J*               coo_row_ind,
                                                    J*               coo_col_ind,
                                                    J*               perm,
                                                    void*            temp_buffer)
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(3, nnz);
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, temp_buffer);

        const rocsparse_status status = rocsparse::coosort_by_row_quickreturn(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
}

template <typename J>
rocsparse_status rocsparse::coosort_by_row_template(rocsparse_handle handle,
                                                    J                m,
                                                    J                n,
                                                    J                nnz,
                                                    J*               coo_row_ind,
                                                    J*               coo_col_ind,
                                                    J*               perm,
                                                    void*            temp_buffer)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_by_row",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

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
    if(coo_row_ind == nullptr || coo_col_ind == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(m);

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // Permutation vector given
    J* work1 = reinterpret_cast<J*>(ptr);
    ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

    J* work2 = reinterpret_cast<J*>(ptr);
    ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

    J* work3 = reinterpret_cast<J*>(ptr);
    ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

    J* work4 = reinterpret_cast<J*>(ptr);
    ptr += ((sizeof(J) * rocsparse::max(m, n)) / 256 + 1) * 256;

    // Temporary rocprim buffer
    size_t size        = 0;
    void*  tmp_rocprim = reinterpret_cast<void*>(ptr);

    if(perm != nullptr)
    {
        // Create identitiy permutation to keep track of reorderings
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::create_identity_permutation_template(handle, nnz, work1));

        // Sort by rows and store permutation
        rocprim::double_buffer<J> keys(coo_row_ind, work3);
        rocprim::double_buffer<J> vals(work1, work2);

        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, size, keys, vals, nnz, startbit, endbit, stream));
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            tmp_rocprim, size, keys, vals, nnz, startbit, endbit, stream));

        J* output  = keys.current();
        J* mapping = vals.current();
        J* alt_map = vals.alternate();

        // Copy sorted rows, if stored in buffer
        if(output != coo_row_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                coo_row_ind, output, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }

        // Obtain segments for segmented sort by columns
        RETURN_IF_HIP_ERROR(rocprim::run_length_encode(
            nullptr, size, coo_row_ind, nnz, work3 + 1, work4, work3, stream));
        RETURN_IF_HIP_ERROR(rocprim::run_length_encode(
            tmp_rocprim, size, coo_row_ind, nnz, work3 + 1, work4, work3, stream));

        J nsegm;
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(&nsegm, work3, sizeof(J), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
            nullptr, size, work4, work4, 0, nsegm + 1, rocprim::plus<J>(), stream));
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
            tmp_rocprim, size, work4, work4, 0, nsegm + 1, rocprim::plus<J>(), stream));

// Reorder columns
#define COOSORT_DIM 512
        dim3 coosort_blocks((nnz - 1) / COOSORT_DIM + 1);
        dim3 coosort_threads(COOSORT_DIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coosort_permute_kernel<COOSORT_DIM>),
                                           coosort_blocks,
                                           coosort_threads,
                                           0,
                                           stream,
                                           nnz,
                                           coo_col_ind,
                                           mapping,
                                           work3);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coosort_permute_kernel<COOSORT_DIM>),
                                           coosort_blocks,
                                           coosort_threads,
                                           0,
                                           stream,
                                           nnz,
                                           perm,
                                           mapping,
                                           alt_map);
#undef COOSORT_DIM

        // Sort columns per row
        endbit = rocsparse::clz(n);

        rocprim::double_buffer<J> keys2(work3, coo_col_ind);
        rocprim::double_buffer<J> vals2(alt_map, perm);

        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(
            nullptr, size, keys2, vals2, nnz, nsegm, work4, work4 + 1, startbit, endbit, stream));

        J avg_row_nnz = nnz / nsegm;

        if(avg_row_nnz < 64)
        {
            using config = rocprim::segmented_radix_sort_config<6, 5, 64, 1>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(tmp_rocprim,
                                                                            size,
                                                                            keys2,
                                                                            vals2,
                                                                            nnz,
                                                                            nsegm,
                                                                            work4,
                                                                            work4 + 1,
                                                                            startbit,
                                                                            endbit,
                                                                            stream));
        }
        else if(avg_row_nnz < 128)
        {
            using config = rocprim::segmented_radix_sort_config<6, 5, 64, 2>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(tmp_rocprim,
                                                                            size,
                                                                            keys2,
                                                                            vals2,
                                                                            nnz,
                                                                            nsegm,
                                                                            work4,
                                                                            work4 + 1,
                                                                            startbit,
                                                                            endbit,
                                                                            stream));
        }
        else if(avg_row_nnz < 256)
        {
            using config = rocprim::segmented_radix_sort_config<6, 5, 64, 4>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(tmp_rocprim,
                                                                            size,
                                                                            keys2,
                                                                            vals2,
                                                                            nnz,
                                                                            nsegm,
                                                                            work4,
                                                                            work4 + 1,
                                                                            startbit,
                                                                            endbit,
                                                                            stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(tmp_rocprim,
                                                                    size,
                                                                    keys2,
                                                                    vals2,
                                                                    nnz,
                                                                    nsegm,
                                                                    work4,
                                                                    work4 + 1,
                                                                    startbit,
                                                                    endbit,
                                                                    stream));
        }

        output  = keys2.current();
        mapping = vals2.current();

        // Copy sorted columns, if stored in buffer
        if(output != coo_col_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                coo_col_ind, output, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }

        // Copy reordered permutation, if stored in buffer
        if(mapping != perm)
        {
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(perm, mapping, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }
    }
    else
    {
        // No permutation vector given

        // Sort by rows and permute columns
        rocprim::double_buffer<J> keys(coo_row_ind, work3);
        rocprim::double_buffer<J> vals(coo_col_ind, work2);

        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, size, keys, vals, nnz, startbit, endbit, stream));
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            tmp_rocprim, size, keys, vals, nnz, startbit, endbit, stream));
        J* output = keys.current();

        // Copy sorted rows, if stored in buffer
        if(output != coo_row_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                coo_row_ind, output, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }

        // Obtain segments for segmented sort by columns
        RETURN_IF_HIP_ERROR(rocprim::run_length_encode(
            nullptr, size, coo_row_ind, nnz, work3 + 1, work4, work3, stream));
        RETURN_IF_HIP_ERROR(rocprim::run_length_encode(
            tmp_rocprim, size, coo_row_ind, nnz, work3 + 1, work4, work3, stream));

        J nsegm;
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(&nsegm, work3, sizeof(J), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
            nullptr, size, work4, work4, 0, nsegm + 1, rocprim::plus<J>(), stream));
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
            tmp_rocprim, size, work4, work4, 0, nsegm + 1, rocprim::plus<J>(), stream));

        // Sort columns per row
        endbit = rocsparse::clz(n);

        J avg_row_nnz = nnz / nsegm;

        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(
            nullptr, size, vals, nnz, nsegm, work4, work4 + 1, startbit, endbit, stream));

        if(avg_row_nnz < 64)
        {
            using config = rocprim::segmented_radix_sort_config<6, 5, 64, 1>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                tmp_rocprim, size, vals, nnz, nsegm, work4, work4 + 1, startbit, endbit, stream));
        }
        else if(avg_row_nnz < 128)
        {
            using config = rocprim::segmented_radix_sort_config<6, 5, 64, 2>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                tmp_rocprim, size, vals, nnz, nsegm, work4, work4 + 1, startbit, endbit, stream));
        }
        else if(avg_row_nnz < 256)
        {
            using config = rocprim::segmented_radix_sort_config<6, 5, 64, 4>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                tmp_rocprim, size, vals, nnz, nsegm, work4, work4 + 1, startbit, endbit, stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(
                tmp_rocprim, size, vals, nnz, nsegm, work4, work4 + 1, startbit, endbit, stream));
        }
        output = vals.current();

        // Copy sorted columns, if stored in buffer
        if(output != coo_col_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                coo_col_ind, output, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_coosort_by_row(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     rocsparse_int    n,
                                                     rocsparse_int    nnz,
                                                     rocsparse_int*   coo_row_ind,
                                                     rocsparse_int*   coo_col_ind,
                                                     rocsparse_int*   perm,
                                                     void*            temp_buffer)
try
{

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_by_row",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    const rocsparse_status status = rocsparse::coosort_by_row_checkarg(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_by_row_template(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer));

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

namespace rocsparse
{
    template <typename J>
    static rocsparse_status coosort_by_column_quickreturn(rocsparse_handle handle,
                                                          J                m,
                                                          J                n,
                                                          J                nnz,
                                                          J*               coo_row_ind,
                                                          J*               coo_col_ind,
                                                          J*               perm,
                                                          void*            temp_buffer)
    {
        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename J>
    static rocsparse_status coosort_by_column_checkarg(rocsparse_handle handle,
                                                       J                m,
                                                       J                n,
                                                       J                nnz,
                                                       J*               coo_row_ind,
                                                       J*               coo_col_ind,
                                                       J*               perm,
                                                       void*            temp_buffer)
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(3, nnz);
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, temp_buffer);

        const rocsparse_status status = rocsparse::coosort_by_column_quickreturn(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }
}

template <typename J>
rocsparse_status rocsparse::coosort_by_column_template(rocsparse_handle handle,
                                                       J                m,
                                                       J                n,
                                                       J                nnz,
                                                       J*               coo_row_ind,
                                                       J*               coo_col_ind,
                                                       J*               perm,
                                                       void*            temp_buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_by_row_template(
        handle, n, m, nnz, coo_col_ind, coo_row_ind, perm, temp_buffer));
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_coosort_by_column(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        rocsparse_int    n,
                                                        rocsparse_int    nnz,
                                                        rocsparse_int*   coo_row_ind,
                                                        rocsparse_int*   coo_col_ind,
                                                        rocsparse_int*   perm,
                                                        void*            temp_buffer)
try
{

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_by_column",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    const rocsparse_status status = rocsparse::coosort_by_column_checkarg(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_by_column_template(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer));

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

#define INSTANTIATE(J)                                                                            \
    template rocsparse_status rocsparse::coosort_buffer_size_template<J>(rocsparse_handle handle, \
                                                                         J                m,      \
                                                                         J                n,      \
                                                                         J                nnz,    \
                                                                         const J* coo_row_ind,    \
                                                                         const J* coo_col_ind,    \
                                                                         size_t*  buffer_size);    \
    template rocsparse_status rocsparse::coosort_by_row_template<J>(rocsparse_handle handle,      \
                                                                    J                m,           \
                                                                    J                n,           \
                                                                    J                nnz,         \
                                                                    J * coo_row_ind,              \
                                                                    J * coo_col_ind,              \
                                                                    J * perm,                     \
                                                                    void* temp_buffer);           \
                                                                                                  \
    template rocsparse_status rocsparse::coosort_by_column_template<J>(rocsparse_handle handle,   \
                                                                       J                m,        \
                                                                       J                n,        \
                                                                       J                nnz,      \
                                                                       J * coo_row_ind,           \
                                                                       J * coo_col_ind,           \
                                                                       J * perm,                  \
                                                                       void* temp_buffer)

INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
#undef INSTANTIATE
