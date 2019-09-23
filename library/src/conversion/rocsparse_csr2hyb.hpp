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

#pragma once
#ifndef ROCSPARSE_CSR2HYB_HPP
#define ROCSPARSE_CSR2HYB_HPP

#include "csr2ell_device.h"
#include "csr2hyb_device.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_csr2hyb_template(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const rocsparse_mat_descr descr,
                                            const T*                  csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            rocsparse_hyb_mat         hyb,
                                            rocsparse_int             user_ell_width,
                                            rocsparse_hyb_partition   partition_type)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(hyb == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2hyb"),
              m,
              n,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)hyb,
              user_ell_width,
              partition_type);

    log_bench(handle, "./rocsparse-bench -f csr2hyb -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    // Check partition type
    if(partition_type != rocsparse_hyb_partition_max
       && partition_type != rocsparse_hyb_partition_user
       && partition_type != rocsparse_hyb_partition_auto)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
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

    // Stream
    hipStream_t stream = handle->stream;

    // Get number of CSR non-zeros
    rocsparse_int csr_nnz;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Correct by index base
    csr_nnz -= descr->base;

    // Maximum ELL row width allowed
    rocsparse_int max_row_nnz = 2 * (csr_nnz - 1) / m + 1;

    // Check user_ell_width
    if(partition_type == rocsparse_hyb_partition_user)
    {
        // ELL width cannot be 0 or negative
        if(user_ell_width < 0)
        {
            return rocsparse_status_invalid_value;
        }

        if(user_ell_width > max_row_nnz)
        {
            return rocsparse_status_invalid_value;
        }
    }

    // Clear HYB structure if already allocated
    hyb->m         = m;
    hyb->n         = n;
    hyb->partition = partition_type;
    hyb->ell_nnz   = 0;
    hyb->ell_width = 0;
    hyb->coo_nnz   = 0;

    if(hyb->ell_col_ind)
    {
        RETURN_IF_HIP_ERROR(hipFree(hyb->ell_col_ind));
    }
    if(hyb->ell_val)
    {
        RETURN_IF_HIP_ERROR(hipFree(hyb->ell_val));
    }
    if(hyb->coo_row_ind)
    {
        RETURN_IF_HIP_ERROR(hipFree(hyb->coo_row_ind));
    }
    if(hyb->coo_col_ind)
    {
        RETURN_IF_HIP_ERROR(hipFree(hyb->coo_col_ind));
    }
    if(hyb->coo_val)
    {
        RETURN_IF_HIP_ERROR(hipFree(hyb->coo_val));
    }

    // Determine ELL width

#define CSR2ELL_DIM 512
    // Workspace size
    rocsparse_int blocks = (m - 1) / CSR2ELL_DIM + 1;

    if(partition_type == rocsparse_hyb_partition_user)
    {
        // ELL width given by user
        hyb->ell_width = user_ell_width;
    }
    else if(partition_type == rocsparse_hyb_partition_auto)
    {
        // ELL width determined by average nnz per row
        hyb->ell_width = (csr_nnz - 1) / m + 1;
    }
    else
    {
        // Allocate workspace
        rocsparse_int* workspace = nullptr;
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&workspace, sizeof(rocsparse_int) * blocks));

        // HYB == ELL - no COO part - compute maximum nnz per row
        hipLaunchKernelGGL((ell_width_kernel_part1<CSR2ELL_DIM>),
                           dim3(blocks),
                           dim3(CSR2ELL_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr,
                           workspace);

        hipLaunchKernelGGL((ell_width_kernel_part2<CSR2ELL_DIM>),
                           dim3(1),
                           dim3(CSR2ELL_DIM),
                           0,
                           stream,
                           blocks,
                           workspace);
        // Copy ell width back to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &hyb->ell_width, workspace, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        RETURN_IF_HIP_ERROR(hipFree(workspace));
    }

    // Re-check ELL width
    if(hyb->ell_width > max_row_nnz)
    {
        return rocsparse_status_invalid_value;
    }

    // Compute ELL non-zeros
    hyb->ell_nnz = hyb->ell_width * m;

    // Allocate ELL part
    if(hyb->ell_nnz > 0)
    {
        RETURN_IF_HIP_ERROR(
            hipMalloc((void**)&hyb->ell_col_ind, sizeof(rocsparse_int) * hyb->ell_nnz));
        RETURN_IF_HIP_ERROR(hipMalloc(&hyb->ell_val, sizeof(T) * hyb->ell_nnz));
    }

    // Allocate workspace
    rocsparse_int* workspace = NULL;
    RETURN_IF_HIP_ERROR(hipMalloc((void**)&workspace, sizeof(rocsparse_int) * (m + 1)));

    // If there is a COO part, compute the COO non-zero elements per row
    if(partition_type != rocsparse_hyb_partition_max)
    {
        // If there is no ELL part, its easy...
        if(hyb->ell_nnz == 0)
        {
            hyb->coo_nnz = csr_nnz;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(workspace,
                                               csr_row_ptr,
                                               sizeof(rocsparse_int) * (m + 1),
                                               hipMemcpyDeviceToDevice,
                                               stream));
        }
        else
        {
            hipLaunchKernelGGL((hyb_coo_nnz<CSR2ELL_DIM>),
                               dim3((m - 1) / CSR2ELL_DIM + 1),
                               dim3(CSR2ELL_DIM),
                               0,
                               stream,
                               m,
                               hyb->ell_width,
                               csr_row_ptr,
                               workspace,
                               descr->base);

            // Inclusive sum on workspace
            void*  d_temp_storage     = nullptr;
            size_t temp_storage_bytes = 0;

            // Obtain rocprim buffer size
            RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(d_temp_storage,
                                                        temp_storage_bytes,
                                                        workspace,
                                                        workspace,
                                                        m + 1,
                                                        rocprim::plus<rocsparse_int>(),
                                                        stream));

            // Allocate rocprim buffer
            RETURN_IF_HIP_ERROR(hipMalloc(&d_temp_storage, temp_storage_bytes));

            // Do inclusive sum
            RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(d_temp_storage,
                                                        temp_storage_bytes,
                                                        workspace,
                                                        workspace,
                                                        m + 1,
                                                        rocprim::plus<rocsparse_int>(),
                                                        stream));

            // Clear rocprim buffer
            RETURN_IF_HIP_ERROR(hipFree(d_temp_storage));

            // Obtain coo nnz from workspace
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hyb->coo_nnz,
                                               workspace + m,
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               stream));

            // Wait for host transfer to finish
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

            hyb->coo_nnz -= descr->base;
        }
    }

    // Allocate COO part
    if(hyb->coo_nnz > 0)
    {
        RETURN_IF_HIP_ERROR(
            hipMalloc((void**)&hyb->coo_row_ind, sizeof(rocsparse_int) * hyb->coo_nnz));
        RETURN_IF_HIP_ERROR(
            hipMalloc((void**)&hyb->coo_col_ind, sizeof(rocsparse_int) * hyb->coo_nnz));
        RETURN_IF_HIP_ERROR(hipMalloc(&hyb->coo_val, sizeof(T) * hyb->coo_nnz));
    }

    dim3 csr2ell_blocks((m - 1) / CSR2ELL_DIM + 1);
    dim3 csr2ell_threads(CSR2ELL_DIM);

    hipLaunchKernelGGL((csr2hyb_kernel<T>),
                       csr2ell_blocks,
                       csr2ell_threads,
                       0,
                       stream,
                       m,
                       csr_val,
                       csr_row_ptr,
                       csr_col_ind,
                       hyb->ell_width,
                       hyb->ell_col_ind,
                       (T*)hyb->ell_val,
                       hyb->coo_row_ind,
                       hyb->coo_col_ind,
                       (T*)hyb->coo_val,
                       workspace,
                       descr->base);

    RETURN_IF_HIP_ERROR(hipFree(workspace));
#undef CSR2ELL_DIM

    return rocsparse_status_success;
}

#endif // ROCSPARSE_CSR2HYB_HPP
