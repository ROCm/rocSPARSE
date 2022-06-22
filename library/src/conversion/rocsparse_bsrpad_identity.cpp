/*! \file */
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include "rocsparse_bsrpad_identity.hpp"
#include "definitions.h"
#include "utility.h"

#include "bsrpad_identity_device.h"

template <typename T>
rocsparse_status rocsparse_bsrpad_identity_template(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    rocsparse_int             n,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nb,
                                                    rocsparse_int             block_dim,
                                                    const rocsparse_mat_descr bsr_descr,
                                                    T*                        bsr_val,
                                                    const rocsparse_int*      bsr_row_ptr,
                                                    const rocsparse_int*      bsr_col_ind)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check matrix descriptors
    if(bsr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrpad_identity"),
              m,
              n,
              mb,
              nb,
              block_dim,
              bsr_descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind);

    log_bench(
        handle, "./rocsparse-bench -f bsrpad_identity -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check matrix sorting mode
    if(bsr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || mb < 0 || nb < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    if(mb * block_dim < m || nb * block_dim < n)
    {
        return rocsparse_status_invalid_size;
    }

    if(m != n || mb != nb)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0 || block_dim == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((bsr_val == nullptr && bsr_col_ind != nullptr)
       || (bsr_val != nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    rocsparse_int start = 0;
    rocsparse_int end   = 0;

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    rocsparse_int nnzb = (end - start);

    if(bsr_val == nullptr && bsr_col_ind == nullptr)
    {
        if(nnzb != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Quick return if possible
    if(mb * block_dim == m && nb * block_dim == n)
    {
        return rocsparse_status_success;
    }

    rocsparse_int remaining_blocks = mb - (m / block_dim);

    constexpr rocsparse_int block_size = 32;

    rocsparse_int grid_size = (remaining_blocks + block_size - 1) / block_size;

    hipLaunchKernelGGL((bsrpad_identity_kernel<block_size>),
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       handle->stream,
                       m,
                       n,
                       mb,
                       nb,
                       block_dim,
                       bsr_descr->base,
                       bsr_val,
                       bsr_row_ptr,
                       bsr_col_ind);

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_sbsrpad_identity(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             mb,
                                                       rocsparse_int             nb,
                                                       rocsparse_int             block_dim,
                                                       const rocsparse_mat_descr bsr_descr,
                                                       float*                    bsr_val,
                                                       rocsparse_int*            bsr_row_ptr,
                                                       rocsparse_int*            bsr_col_ind)
{
    return rocsparse_bsrpad_identity_template(
        handle, m, n, mb, nb, block_dim, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind);
}

extern "C" rocsparse_status rocsparse_dbsrpad_identity(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             mb,
                                                       rocsparse_int             nb,
                                                       rocsparse_int             block_dim,
                                                       const rocsparse_mat_descr bsr_descr,
                                                       double*                   bsr_val,
                                                       rocsparse_int*            bsr_row_ptr,
                                                       rocsparse_int*            bsr_col_ind)
{
    return rocsparse_bsrpad_identity_template(
        handle, m, n, mb, nb, block_dim, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind);
}

extern "C" rocsparse_status rocsparse_cbsrpad_identity(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             mb,
                                                       rocsparse_int             nb,
                                                       rocsparse_int             block_dim,
                                                       const rocsparse_mat_descr bsr_descr,
                                                       rocsparse_float_complex*  bsr_val,
                                                       rocsparse_int*            bsr_row_ptr,
                                                       rocsparse_int*            bsr_col_ind)
{
    return rocsparse_bsrpad_identity_template(
        handle, m, n, mb, nb, block_dim, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind);
}

extern "C" rocsparse_status rocsparse_zbsrpad_identity(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             mb,
                                                       rocsparse_int             nb,
                                                       rocsparse_int             block_dim,
                                                       const rocsparse_mat_descr bsr_descr,
                                                       rocsparse_double_complex* bsr_val,
                                                       rocsparse_int*            bsr_row_ptr,
                                                       rocsparse_int*            bsr_col_ind)
{
    return rocsparse_bsrpad_identity_template(
        handle, m, n, mb, nb, block_dim, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind);
}
