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
#include "rocsparse_csr2ell.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_csr2ell_width(rocsparse_handle handle,
                                                    rocsparse_int m,
                                                    const rocsparse_mat_descr csr_descr,
                                                    const rocsparse_int* csr_row_ptr,
                                                    const rocsparse_mat_descr ell_descr,
                                                    rocsparse_int* ell_width)
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

    // Check index base
    if(csr_descr->base != rocsparse_index_base_zero && csr_descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(ell_descr->base != rocsparse_index_base_zero && ell_descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    // Check matrix type
    if(csr_descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    if(ell_descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
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
    else if(ell_width == nullptr)
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
        RETURN_IF_HIP_ERROR(
            hipMemcpy(ell_width, workspace, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsr2ell(rocsparse_handle handle,
                                               rocsparse_int m,
                                               const rocsparse_mat_descr csr_descr,
                                               const float* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int ell_width,
                                               float* ell_val,
                                               rocsparse_int* ell_col_ind)
{
    return rocsparse_csr2ell_template<float>(handle,
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

extern "C" rocsparse_status rocsparse_dcsr2ell(rocsparse_handle handle,
                                               rocsparse_int m,
                                               const rocsparse_mat_descr csr_descr,
                                               const double* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int ell_width,
                                               double* ell_val,
                                               rocsparse_int* ell_col_ind)
{
    return rocsparse_csr2ell_template<double>(handle,
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
