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

#include "handle.h"
#include "identity_device.h"
#include "utility.h"

#include <hip/hip_runtime.h>

extern "C" rocsparse_status rocsparse_create_identity_permutation(rocsparse_handle handle,
                                                                  rocsparse_int    n,
                                                                  rocsparse_int*   p)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle, "rocsparse_create_identity_permutation", n, (const void*&)p);

    log_bench(handle, "./rocsparse-bench -f identity", "-n", n);

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
