/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
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
    log_trace(handle, "rocsparse_csrsort_buffer_size",
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

// TODO config required for buffer??
#if defined(__HIP_PLATFORM_HCC__)
    using config = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1> >;

    rocprim::segmented_radix_sort_pairs<config>(nullptr, *buffer_size, null_ptr, null_ptr, null_ptr, null_ptr, nnz, m, null_ptr, null_ptr, 0, 32, stream);
#elif defined(__HIP_PLATFORM_NVCC__)
    hipcub::DeviceSegmentedRadixSort::SortPairs(nullptr, *buffer_size, null_ptr, null_ptr, null_ptr, null_ptr, nnz, m, null_ptr, null_ptr, 0, 32, stream);
#endif

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
    log_trace(handle, "rocsparse_csrsort_buffer_size",
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
    unsigned int endbit = 32 - __builtin_clz(n);
    size_t size;

#if defined(__HIP_PLATFORM_HCC__)
    using config = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1> >;

    rocprim::segmented_radix_sort_pairs<config>(nullptr, size, csr_col_ind, csr_col_ind, perm, perm, nnz, m, csr_row_ptr, csr_row_ptr + 1, startbit, endbit, stream);
    rocprim::segmented_radix_sort_pairs<config>(temp_buffer, size, csr_col_ind, csr_col_ind, perm, perm, nnz, m, csr_row_ptr, csr_row_ptr + 1, startbit, endbit, stream);
#elif defined(__HIP_PLATFORM_NVCC__)
    hipcub::DeviceSegmentedRadixSort::SortPairs(nullptr, size, csr_col_ind, csr_col_ind, perm, perm, nnz, m, csr_row_ptr, csr_row_ptr + 1, startbit, endbit, stream);
    hipcub::DeviceSegmentedRadixSort::SortPairs(temp_buffer, size, csr_col_ind, csr_col_ind, perm, perm, nnz, m, csr_row_ptr, csr_row_ptr + 1, startbit, endbit, stream);
#endif

    return rocsparse_status_success;
}
