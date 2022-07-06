/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csr2ell.hpp"
#include "definitions.h"
#include "utility.h"

#include "csr2ell_device.h"

template <typename T>
rocsparse_status rocsparse_csr2ell_template(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            const rocsparse_mat_descr csr_descr,
                                            const T*                  csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            const rocsparse_mat_descr ell_descr,
                                            rocsparse_int             ell_width,
                                            T*                        ell_val,
                                            rocsparse_int*            ell_col_ind)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(csr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(ell_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2ell"),
              m,
              (const void*&)csr_descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)ell_descr,
              ell_width,
              (const void*&)ell_val,
              (const void*&)ell_col_ind);

    log_bench(handle, "./rocsparse-bench -f csr2ell -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check matrix type
    if(csr_descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(ell_descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(csr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }
    if(ell_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || ell_width < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || ell_width == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(ell_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define CSR2ELL_DIM 512
    dim3 csr2ell_blocks((m - 1) / CSR2ELL_DIM + 1);
    dim3 csr2ell_threads(CSR2ELL_DIM);

    hipLaunchKernelGGL((csr2ell_kernel<CSR2ELL_DIM>),
                       csr2ell_blocks,
                       csr2ell_threads,
                       0,
                       stream,
                       m,
                       csr_val,
                       csr_row_ptr,
                       csr_col_ind,
                       csr_descr->base,
                       ell_width,
                       ell_col_ind,
                       ell_val,
                       ell_descr->base);
#undef CSR2ELL_DIM
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_csr2ell_width(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    const rocsparse_mat_descr csr_descr,
                                                    const rocsparse_int*      csr_row_ptr,
                                                    const rocsparse_mat_descr ell_descr,
                                                    rocsparse_int*            ell_width)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(csr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(ell_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csr2ell_width",
              m,
              (const void*&)csr_descr,
              (const void*&)csr_row_ptr,
              (const void*&)ell_descr,
              (const void*&)ell_width);

    // Check matrix type
    if(csr_descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(ell_descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(csr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }
    if(ell_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check ell_width pointer
    if(ell_width == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(ell_width, 0, sizeof(rocsparse_int), stream));
        }
        else
        {
            *ell_width = 0;
        }
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Determine ELL width

#define CSR2ELL_DIM 256
    // Workspace size
    rocsparse_int nblocks = CSR2ELL_DIM;

    // Get workspace from handle device buffer
    rocsparse_int* workspace = reinterpret_cast<rocsparse_int*>(handle->buffer);

    dim3 csr2ell_blocks(nblocks);
    dim3 csr2ell_threads(CSR2ELL_DIM);

    // Compute maximum nnz per row
    hipLaunchKernelGGL((ell_width_kernel_part1<CSR2ELL_DIM>),
                       csr2ell_blocks,
                       csr2ell_threads,
                       0,
                       stream,
                       m,
                       csr_row_ptr,
                       workspace);

    hipLaunchKernelGGL((ell_width_kernel_part2<CSR2ELL_DIM>),
                       dim3(1),
                       csr2ell_threads,
                       0,
                       stream,
                       nblocks,
                       workspace);

    // Copy ELL width back to host, if handle says so
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            ell_width, workspace, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            ell_width, workspace, sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsr2ell(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               const rocsparse_mat_descr csr_descr,
                                               const float*              csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int             ell_width,
                                               float*                    ell_val,
                                               rocsparse_int*            ell_col_ind)
{
    return rocsparse_csr2ell_template(handle,
                                      m,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      ell_descr,
                                      ell_width,
                                      ell_val,
                                      ell_col_ind);
}

extern "C" rocsparse_status rocsparse_dcsr2ell(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               const rocsparse_mat_descr csr_descr,
                                               const double*             csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int             ell_width,
                                               double*                   ell_val,
                                               rocsparse_int*            ell_col_ind)
{
    return rocsparse_csr2ell_template(handle,
                                      m,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      ell_descr,
                                      ell_width,
                                      ell_val,
                                      ell_col_ind);
}

extern "C" rocsparse_status rocsparse_ccsr2ell(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               const rocsparse_mat_descr      csr_descr,
                                               const rocsparse_float_complex* csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               const rocsparse_int*           csr_col_ind,
                                               const rocsparse_mat_descr      ell_descr,
                                               rocsparse_int                  ell_width,
                                               rocsparse_float_complex*       ell_val,
                                               rocsparse_int*                 ell_col_ind)
{
    return rocsparse_csr2ell_template(handle,
                                      m,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      ell_descr,
                                      ell_width,
                                      ell_val,
                                      ell_col_ind);
}

extern "C" rocsparse_status rocsparse_zcsr2ell(rocsparse_handle                handle,
                                               rocsparse_int                   m,
                                               const rocsparse_mat_descr       csr_descr,
                                               const rocsparse_double_complex* csr_val,
                                               const rocsparse_int*            csr_row_ptr,
                                               const rocsparse_int*            csr_col_ind,
                                               const rocsparse_mat_descr       ell_descr,
                                               rocsparse_int                   ell_width,
                                               rocsparse_double_complex*       ell_val,
                                               rocsparse_int*                  ell_col_ind)
{
    return rocsparse_csr2ell_template(handle,
                                      m,
                                      csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      ell_descr,
                                      ell_width,
                                      ell_val,
                                      ell_col_ind);
}
