/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"
#include "csrsort_device.h"

#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_HCC__)
#include <rocprim/rocprim.hpp>
#elif defined(__HIP_PLATFORM_NVCC__)
#include <hipcub/hipcub.hpp>
#endif

extern "C" rocsparse_status rocsparse_csrsort_buffer_size(rocsparse_handle handle,
                                                          rocsparse_int m,
                                                          rocsparse_int n,
                                                          rocsparse_int nnz,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          size_t* buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging TODO bench logging
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
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(buffer_size);
#if defined(__HIP_PLATFORM_HCC__)
    rocprim::double_buffer<rocsparse_int> dummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(
        nullptr, *buffer_size, dummy, dummy, nnz, m, buffer_size, buffer_size, 0, 32, stream));
#elif defined(__HIP_PLATFORM_NVCC__)
    hipcub::DoubleBuffer<rocsparse_int> dummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(hipcub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, *buffer_size, dummy, dummy, nnz, m, buffer_size, buffer_size, 0, 32, stream));
#endif

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays

    // columns buffer
    *buffer_size += sizeof(rocsparse_int) * nnz;
    // perm buffer
    *buffer_size += sizeof(rocsparse_int) * nnz;
    // segm buffer
    *buffer_size += sizeof(rocsparse_int) * (m + 1);

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_csrsort(rocsparse_handle handle,
                                              rocsparse_int m,
                                              rocsparse_int n,
                                              rocsparse_int nnz,
                                              const rocsparse_mat_descr descr,
                                              const rocsparse_int* csr_row_ptr,
                                              rocsparse_int* csr_col_ind,
                                              rocsparse_int* perm,
                                              void* temp_buffer)
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

    // Logging TODO bench logging
    log_trace(handle,
              "rocsparse_csrsort_buffer_size",
              m,
              n,
              nnz,
              (const void*&)descr,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)perm,
              (const void*&)temp_buffer);

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
    size_t size;

    if(perm != nullptr)
    {
// Sort pairs, if permutation vector is present
#if defined(__HIP_PLATFORM_HCC__)
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
#elif defined(__HIP_PLATFORM_NVCC__)
        hipcub::DoubleBuffer<rocsparse_int> dummy(csr_col_ind, perm);

        RETURN_IF_HIP_ERROR(hipcub::DeviceSegmentedRadixSort::SortPairs(nullptr,
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
#endif
    }
    else
    {
// Sort keys, if no permutation vector is present
#if defined(__HIP_PLATFORM_HCC__)
        rocprim::double_buffer<rocsparse_int> dummy(csr_col_ind, csr_col_ind);

        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(
            nullptr, size, dummy, nnz, m, csr_row_ptr, csr_row_ptr + 1, startbit, endbit, stream));
#elif defined(__HIP_PLATFORM_NVCC__)
        hipcub::DoubleBuffer<rocsparse_int> dummy(csr_col_ind, csr_col_ind);

        ETURN_IF_HIP_ERROR(hipcub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, size, dummy, nnz, m, csr_row_ptr, csr_row_ptr + 1, startbit, endbit, stream));
#endif
    }

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    ptr += size;

    // columns buffer
    rocsparse_int* tmp_cols = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * nnz;

    // perm buffer
    rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * nnz;

    // segm buffer
    rocsparse_int* tmp_segm = nullptr;

    // Index base one requires shift of offset positions
    if(descr->base == rocsparse_index_base_one)
    {
        tmp_segm = reinterpret_cast<rocsparse_int*>(ptr);

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

    // Switch between offsets
    const rocsparse_int* offsets = tmp_segm ? tmp_segm : csr_row_ptr;

    // Sort by columns and obtain permutation vector

    if(perm != nullptr)
    {
// Sort by pairs, if permutation vector is present
#if defined(__HIP_PLATFORM_HCC__)
        rocprim::double_buffer<rocsparse_int> keys(csr_col_ind, tmp_cols);
        rocprim::double_buffer<rocsparse_int> vals(perm, tmp_perm);

        // Determine blocksize and items per thread depending on average nnz per row
        rocsparse_int avg_row_nnz = nnz / m;

        if(avg_row_nnz < 64)
        {
            using config =
                rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(temp_buffer,
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
            using config =
                rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 2>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(temp_buffer,
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
            using config =
                rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 4>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(temp_buffer,
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
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(temp_buffer,
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
            RETURN_IF_HIP_ERROR(hipMemcpy(
                csr_col_ind, keys.current(), sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice));
        }
        if(vals.current() != perm)
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(
                perm, vals.current(), sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice));
        }
#elif defined(__HIP_PLATFORM_NVCC__)
        hipcub::DoubleBuffer<rocsparse_int> keys(csr_col_ind, tmp_cols);
        hipcub::DoubleBuffer<rocsparse_int> vals(perm, tmp_perm);

        RETURN_IF_HIP_ERROR(hipcub::DeviceSegmentedRadixSort::SortPairs(
            temp_buffer, size, keys, vals, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        if(keys.Current() != csr_col_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(
                csr_col_ind, keys.Current(), sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice));
        }
        if(vals.Current() != perm)
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(
                perm, vals.Current(), sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice));
        }
#endif
    }
    else
    {
// Sort by keys, if no permutation vector is present
#if defined(__HIP_PLATFORM_HCC__)
        rocprim::double_buffer<rocsparse_int> keys(csr_col_ind, tmp_cols);

        // Determine blocksize and items per thread depending on average nnz per row
        rocsparse_int avg_row_nnz = nnz / m;

        if(avg_row_nnz < 64)
        {
            using config =
                rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                temp_buffer, size, keys, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        }
        else if(avg_row_nnz < 128)
        {
            using config =
                rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 2>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                temp_buffer, size, keys, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        }
        else if(avg_row_nnz < 256)
        {
            using config =
                rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 4>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(
                temp_buffer, size, keys, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(
                temp_buffer, size, keys, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        }
        if(keys.current() != csr_col_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(
                csr_col_ind, keys.current(), sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice));
        }
#elif defined(__HIP_PLATFORM_NVCC__)
        hipcub::DoubleBuffer<rocsparse_int> keys(csr_col_ind, tmp_cols);

        RETURN_IF_HIP_ERROR(hipcub::DeviceSegmentedRadixSort::SortKeys(
            temp_buffer, size, keys, nnz, m, offsets, offsets + 1, startbit, endbit, stream));
        if(keys.Current() != csr_col_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(
                csr_col_ind, keys.Current(), sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice));
        }
#endif
    }
    return rocsparse_status_success;
}
