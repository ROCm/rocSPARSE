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

#include "csrsort_device.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

extern "C" rocsparse_status rocsparse_csrsort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          size_t*              buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csrsort_buffer_size",
              m,
              n,
              nnz,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
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
    if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
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

    rocsparse_int*                        ptr = reinterpret_cast<rocsparse_int*>(buffer_size);
    rocprim::double_buffer<rocsparse_int> dummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(
        nullptr, *buffer_size, dummy, dummy, nnz, m, buffer_size, buffer_size, 0, 32, stream));
    *buffer_size = ((*buffer_size - 1) / 256 + 1) * 256;

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays

    // columns buffer
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    // perm buffer
    *buffer_size += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;
    // segm buffer
    *buffer_size += sizeof(rocsparse_int) * (m / 256 + 1) * 256;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_csrsort(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const rocsparse_int*      csr_row_ptr,
                                              rocsparse_int*            csr_col_ind,
                                              rocsparse_int*            perm,
                                              void*                     temp_buffer)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csrsort",
              m,
              n,
              nnz,
              (const void*&)descr,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)perm,
              (const void*&)temp_buffer);

    log_bench(handle, "./rocsparse-bench -f csrsort", "--mtx <matrix.mtx>");

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
    if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
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
    unsigned int endbit   = rocsparse_clz(n);
    size_t       size;

    if(perm != nullptr)
    {
        // Sort pairs, if permutation vector is present
        rocprim::double_buffer<rocsparse_int> dummy(csr_col_ind, perm);

        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(nullptr,
                                                                size,
                                                                dummy,
                                                                dummy,
                                                                nnz,
                                                                m,
                                                                csr_row_ptr,
                                                                csr_row_ptr + 1,
                                                                startbit,
                                                                endbit,
                                                                stream));
    }
    else
    {
        // Sort keys, if no permutation vector is present
        rocprim::double_buffer<rocsparse_int> dummy(csr_col_ind, csr_col_ind);

        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(
            nullptr, size, dummy, nnz, m, csr_row_ptr, csr_row_ptr + 1, startbit, endbit, stream));
    }

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // columns buffer
    rocsparse_int* tmp_cols = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // perm buffer
    rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // segm buffer
    rocsparse_int* tmp_segm = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // Index base one requires shift of offset positions
    if(descr->base == rocsparse_index_base_one)
    {
#define CSRSORT_DIM 512
        dim3 csrsort_blocks(m / CSRSORT_DIM + 1);
        dim3 csrsort_threads(CSRSORT_DIM);

        hipLaunchKernelGGL((csrsort_shift_kernel),
                           csrsort_blocks,
                           csrsort_threads,
                           0,
                           stream,
                           m + 1,
                           csr_row_ptr,
                           tmp_segm);
#undef CSRSORT_DIM
    }

    // rocprim buffer
    void* tmp_rocprim = reinterpret_cast<void*>(ptr);

    // Switch between offsets
    const rocsparse_int* offsets = descr->base == rocsparse_index_base_one ? tmp_segm : csr_row_ptr;

    // Sort by columns and obtain permutation vector

    if(perm != nullptr)
    {
        // Sort by pairs, if permutation vector is present
        rocprim::double_buffer<rocsparse_int> keys(csr_col_ind, tmp_cols);
        rocprim::double_buffer<rocsparse_int> vals(perm, tmp_perm);

        // Determine blocksize and items per thread depending on average nnz per row
        rocsparse_int avg_row_nnz = nnz / m;

        if(avg_row_nnz < 64)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(tmp_rocprim,
                                                                            size,
                                                                            keys,
                                                                            vals,
                                                                            nnz,
                                                                            m,
                                                                            offsets,
                                                                            offsets + 1,
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
                                                                            keys,
                                                                            vals,
                                                                            nnz,
                                                                            m,
                                                                            offsets,
                                                                            offsets + 1,
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
                                                                            keys,
                                                                            vals,
                                                                            nnz,
                                                                            m,
                                                                            offsets,
                                                                            offsets + 1,
                                                                            startbit,
                                                                            endbit,
                                                                            stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(tmp_rocprim,
                                                                    size,
                                                                    keys,
                                                                    vals,
                                                                    nnz,
                                                                    m,
                                                                    offsets,
                                                                    offsets + 1,
                                                                    startbit,
                                                                    endbit,
                                                                    stream));
        }
        if(keys.current() != csr_col_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(csr_col_ind,
                                               keys.current(),
                                               sizeof(rocsparse_int) * nnz,
                                               hipMemcpyDeviceToDevice,
                                               stream));
        }
        if(vals.current() != perm)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(perm,
                                               vals.current(),
                                               sizeof(rocsparse_int) * nnz,
                                               hipMemcpyDeviceToDevice,
                                               stream));
        }
    }
    else
    {
        // Sort by keys, if no permutation vector is present
        rocprim::double_buffer<rocsparse_int> keys(csr_col_ind, tmp_cols);

        // Determine blocksize and items per thread depending on average nnz per row
        rocsparse_int avg_row_nnz = nnz / m;

        if(avg_row_nnz < 64)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                tmp_rocprim, size, keys, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        }
        else if(avg_row_nnz < 128)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 2>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                tmp_rocprim, size, keys, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        }
        else if(avg_row_nnz < 256)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 4>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                tmp_rocprim, size, keys, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(
                tmp_rocprim, size, keys, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        }
        if(keys.current() != csr_col_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(csr_col_ind,
                                               keys.current(),
                                               sizeof(rocsparse_int) * nnz,
                                               hipMemcpyDeviceToDevice,
                                               stream));
        }
    }
    return rocsparse_status_success;
}
