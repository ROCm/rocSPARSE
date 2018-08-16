/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "identity_device.h"

#include <hip/hip_runtime.h>

extern "C" rocsparse_status
rocsparse_create_identity_permutation(rocsparse_handle handle, rocsparse_int n, rocsparse_int* p)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle, "rocsparse_create_identity_permutation", n, (const void*&)p);

    log_bench(handle,
              "./rocsparse-bench -f identity",
              "-n",
              n);

    // Check sizes
    if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(p == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(n == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define IDENTITY_DIM 512
    dim3 identity_blocks((n - 1) / IDENTITY_DIM + 1);
    dim3 identity_threads(IDENTITY_DIM);

    hipLaunchKernelGGL((identity_kernel), identity_blocks, identity_threads, 0, stream, n, p);
#undef IDENTITY_DIM
    return rocsparse_status_success;
}
