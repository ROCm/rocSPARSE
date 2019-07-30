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

#include "coosort_device.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

extern "C" rocsparse_status rocsparse_coosort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* coo_row_ind,
                                                          const rocsparse_int* coo_col_ind,
                                                          size_t*              buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              "rocsparse_coosort_buffer_size",
              m,
              n,
              nnz,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)buffer_size);

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(coo_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(coo_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(buffer_size);

    // Determine max buffer size
    size_t size;
    *buffer_size = 0;
    rocprim::double_buffer<rocsparse_int> dummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(rocprim::run_length_encode(nullptr, size, ptr, nnz, ptr, ptr, ptr, stream));
    *buffer_size = std::max(size, *buffer_size);
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
        nullptr, size, ptr, ptr, 0, m + 1, rocprim::plus<rocsparse_int>(), stream));
    *buffer_size = std::max(size, *buffer_size);
    RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(nullptr, size, dummy, dummy, nnz, 0, 32, stream));
    *buffer_size = std::max(size, *buffer_size);
    rocprim::double_buffer<rocsparse_int> rpdummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(
        nullptr, size, rpdummy, rpdummy, nnz, m, ptr, ptr + 1, 0, 32, stream));
    *buffer_size = std::max(size, *buffer_size);
    *buffer_size = ((*buffer_size - 1) / 256 + 1) * 256;

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays

    // rows buffer
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    // columns buffer
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    // perm buffer
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    // segment buffer
    *buffer_size += sizeof(rocsparse_int) * (std::max(m, n) / 256 + 1) * 256;

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
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              "rocsparse_coosort_by_row",
              m,
              n,
              nnz,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)perm,
              (const void*&)temp_buffer);

    log_bench(handle, "./rocsparse-bench -f coosort", "--mtx <matrix.mtx>");

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(coo_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(coo_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    unsigned int startbit = 0;
    unsigned int endbit   = rocsparse_clz(m);

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // Permutation vector given
    rocsparse_int* work1 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    rocsparse_int* work2 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    rocsparse_int* work3 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    rocsparse_int* work4 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * (std::max(m, n) / 256 + 1) * 256;

    // Temporary rocprim buffer
    size_t size        = 0;
    void*  tmp_rocprim = reinterpret_cast<void*>(ptr);

    if(perm != nullptr)
    {
        // Create identitiy permutation to keep track of reorderings
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, nnz, work1));

        // Sort by rows and store permutation
        rocprim::double_buffer<rocsparse_int> keys(coo_row_ind, work3);
        rocprim::double_buffer<rocsparse_int> vals(work1, work2);

        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, size, keys, vals, nnz, startbit, endbit, stream));
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            tmp_rocprim, size, keys, vals, nnz, startbit, endbit, stream));

        rocsparse_int* output  = keys.current();
        rocsparse_int* mapping = vals.current();
        rocsparse_int* alt_map = vals.alternate();

        // Copy sorted rows, if stored in buffer
        if(output != coo_row_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                coo_row_ind, output, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice, stream));
        }

        // Obtain segments for segmented sort by columns
        RETURN_IF_HIP_ERROR(rocprim::run_length_encode(
            nullptr, size, coo_row_ind, nnz, work3 + 1, work4, work3, stream));
        RETURN_IF_HIP_ERROR(rocprim::run_length_encode(
            tmp_rocprim, size, coo_row_ind, nnz, work3 + 1, work4, work3, stream));

        rocsparse_int nsegm;
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(&nsegm, work3, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
            nullptr, size, work4, work4, 0, nsegm + 1, rocprim::plus<rocsparse_int>(), stream));
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
            tmp_rocprim, size, work4, work4, 0, nsegm + 1, rocprim::plus<rocsparse_int>(), stream));

// Reorder columns
#define COOSORT_DIM 512
        dim3 coosort_blocks((nnz - 1) / COOSORT_DIM + 1);
        dim3 coosort_threads(COOSORT_DIM);
#undef COOSORT_DIM

        hipLaunchKernelGGL((coosort_permute_kernel),
                           coosort_blocks,
                           coosort_threads,
                           0,
                           stream,
                           nnz,
                           coo_col_ind,
                           mapping,
                           work3);

        hipLaunchKernelGGL((coosort_permute_kernel),
                           coosort_blocks,
                           coosort_threads,
                           0,
                           stream,
                           nnz,
                           perm,
                           mapping,
                           alt_map);

        // Sort columns per row
        endbit = rocsparse_clz(n);

        rocprim::double_buffer<rocsparse_int> keys2(work3, coo_col_ind);
        rocprim::double_buffer<rocsparse_int> vals2(alt_map, perm);

        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(
            nullptr, size, keys2, vals2, nnz, nsegm, work4, work4 + 1, startbit, endbit, stream));

        rocsparse_int avg_row_nnz = nnz / nsegm;

        if(avg_row_nnz < 64)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1>>;
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
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 2>>;
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
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 4>>;
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
                coo_col_ind, output, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice, stream));
        }

        // Copy reordered permutation, if stored in buffer
        if(mapping != perm)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                perm, mapping, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice, stream));
        }
    }
    else
    {
        // No permutation vector given

        // Sort by rows and permute columns
        rocprim::double_buffer<rocsparse_int> keys(coo_row_ind, work3);
        rocprim::double_buffer<rocsparse_int> vals(coo_col_ind, work2);

        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, size, keys, vals, nnz, startbit, endbit, stream));
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            tmp_rocprim, size, keys, vals, nnz, startbit, endbit, stream));
        rocsparse_int* output = keys.current();

        // Copy sorted rows, if stored in buffer
        if(output != coo_row_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                coo_row_ind, output, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice, stream));
        }

        // Obtain segments for segmented sort by columns
        RETURN_IF_HIP_ERROR(rocprim::run_length_encode(
            nullptr, size, coo_row_ind, nnz, work3 + 1, work4, work3, stream));
        RETURN_IF_HIP_ERROR(rocprim::run_length_encode(
            tmp_rocprim, size, coo_row_ind, nnz, work3 + 1, work4, work3, stream));

        rocsparse_int nsegm;
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(&nsegm, work3, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
            nullptr, size, work4, work4, 0, nsegm + 1, rocprim::plus<rocsparse_int>(), stream));
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(
            tmp_rocprim, size, work4, work4, 0, nsegm + 1, rocprim::plus<rocsparse_int>(), stream));

        // Sort columns per row
        endbit = rocsparse_clz(n);

        rocsparse_int avg_row_nnz = nnz / nsegm;

        if(avg_row_nnz < 64)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                tmp_rocprim, size, vals, nnz, nsegm, work4, work4 + 1, startbit, endbit, stream));
        }
        else if(avg_row_nnz < 128)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 2>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                tmp_rocprim, size, vals, nnz, nsegm, work4, work4 + 1, startbit, endbit, stream));
        }
        else if(avg_row_nnz < 256)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 4>>;
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
                coo_col_ind, output, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice, stream));
        }
    }

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
{
    return rocsparse_coosort_by_row(handle, n, m, nnz, coo_col_ind, coo_row_ind, perm, temp_buffer);
}
