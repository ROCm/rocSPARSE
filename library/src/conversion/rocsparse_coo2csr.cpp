/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "coo2csr_device.h"

#include <hip/hip_runtime.h>

extern "C" rocsparse_status rocsparse_coo2csr(rocsparse_handle handle,
                                              const rocsparse_int* coo_row_ind,
                                              rocsparse_int nnz,
                                              rocsparse_int m,
                                              rocsparse_int* csr_row_ptr,
                                              rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              "rocsparse_coo2csr",
              (const void*&)coo_row_ind,
              nnz,
              m,
              (const void*&)csr_row_ptr,
              idx_base);

    log_bench(handle,
              "./rocsparse-bench -f coo2csr",
              "--mtx <matrix.mtx>");

    // Check sizes
    if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(coo_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(nnz == 0 || m == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define COO2CSR_DIM 512
    dim3 coo2csr_blocks((m - 1) / COO2CSR_DIM + 1);
    dim3 coo2csr_threads(COO2CSR_DIM);

    hipLaunchKernelGGL((coo2csr_kernel),
                       coo2csr_blocks,
                       coo2csr_threads,
                       0,
                       stream,
                       m,
                       nnz,
                       coo_row_ind,
                       csr_row_ptr,
                       idx_base);
#undef COO2CSR_DIM
    return rocsparse_status_success;
}
