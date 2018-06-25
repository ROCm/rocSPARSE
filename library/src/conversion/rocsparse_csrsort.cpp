/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"

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

    rocsparse_int* null_ptr = nullptr;

#if defined(__HIP_PLATFORM_HCC__)
    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(nullptr,
                                                            *buffer_size,
                                                            null_ptr,
                                                            null_ptr,
                                                            null_ptr,
                                                            null_ptr,
                                                            nnz,
                                                            m,
                                                            null_ptr,
                                                            null_ptr,
                                                            0,
                                                            32,
                                                            stream));
#elif defined(__HIP_PLATFORM_NVCC__)
    RETURN_IF_HIP_ERROR(hipcub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                                                    *buffer_size,
                                                                    null_ptr,
                                                                    null_ptr,
                                                                    null_ptr,
                                                                    null_ptr,
                                                                    nnz,
                                                                    m,
                                                                    null_ptr,
                                                                    null_ptr,
                                                                    0,
                                                                    32,
                                                                    stream));
#endif

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays

    // columns buffer
    *buffer_size += sizeof(rocsparse_int) * nnz;
    // perm buffer
    *buffer_size += sizeof(rocsparse_int) * nnz;

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
    else if(perm == nullptr)
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

#if defined(__HIP_PLATFORM_HCC__)
    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(nullptr,
                                                            size,
                                                            csr_col_ind,
                                                            csr_col_ind,
                                                            perm,
                                                            perm,
                                                            nnz,
                                                            m,
                                                            csr_row_ptr,
                                                            csr_row_ptr + 1,
                                                            startbit,
                                                            endbit,
                                                            stream));

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    ptr += size;

    // columns buffer
    rocsparse_int* tmp_cols = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * nnz;

    // perm buffer
    rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(ptr);

    // Sort by columns and obtain permutation vector

    // Determine blocksize and items per thread depending on average nnz per row
    rocsparse_int avg_row_nnz = nnz / m;

    if(avg_row_nnz < 64)
    {
        using config = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1>>;
        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(temp_buffer,
                                                                        size,
                                                                        csr_col_ind,
                                                                        tmp_cols,
                                                                        perm,
                                                                        tmp_perm,
                                                                        nnz,
                                                                        m,
                                                                        csr_row_ptr,
                                                                        csr_row_ptr + 1,
                                                                        startbit,
                                                                        endbit,
                                                                        stream));
    }
    else if(avg_row_nnz < 128)
    {
        using config = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 2>>;
        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(temp_buffer,
                                                                        size,
                                                                        csr_col_ind,
                                                                        tmp_cols,
                                                                        perm,
                                                                        tmp_perm,
                                                                        nnz,
                                                                        m,
                                                                        csr_row_ptr,
                                                                        csr_row_ptr + 1,
                                                                        startbit,
                                                                        endbit,
                                                                        stream));
    }
    else if(avg_row_nnz < 256)
    {
        using config = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 4>>;
        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(temp_buffer,
                                                                        size,
                                                                        csr_col_ind,
                                                                        tmp_cols,
                                                                        perm,
                                                                        tmp_perm,
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
        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(temp_buffer,
                                                                size,
                                                                csr_col_ind,
                                                                tmp_cols,
                                                                perm,
                                                                tmp_perm,
                                                                nnz,
                                                                m,
                                                                csr_row_ptr,
                                                                csr_row_ptr + 1,
                                                                startbit,
                                                                endbit,
                                                                stream));
    }
#elif defined(__HIP_PLATFORM_NVCC__)
    RETURN_IF_HIP_ERROR(hipcub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                                                    size,
                                                                    csr_col_ind,
                                                                    csr_col_ind,
                                                                    perm,
                                                                    perm,
                                                                    nnz,
                                                                    m,
                                                                    csr_row_ptr,
                                                                    csr_row_ptr + 1,
                                                                    startbit,
                                                                    endbit,
                                                                    stream));

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    ptr += size;

    // columns buffer
    rocsparse_int* tmp_cols = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * nnz;

    // perm buffer
    rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(ptr);

    // Sort by columns and obtain permutation vector
    RETURN_IF_HIP_ERROR(hipcub::DeviceSegmentedRadixSort::SortPairs(temp_buffer,
                                                                    size,
                                                                    csr_col_ind,
                                                                    tmp_cols,
                                                                    perm,
                                                                    tmp_perm,
                                                                    nnz,
                                                                    m,
                                                                    csr_row_ptr,
                                                                    csr_row_ptr + 1,
                                                                    startbit,
                                                                    endbit,
                                                                    stream));
#endif

    // Extract results from buffer
    RETURN_IF_HIP_ERROR(
        hipMemcpy(csr_col_ind, tmp_cols, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice));
    RETURN_IF_HIP_ERROR(
        hipMemcpy(perm, tmp_perm, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice));

    return rocsparse_status_success;
}
