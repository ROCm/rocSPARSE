/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_bsrgeam.hpp"
#include "definitions.h"
#include "rocsparse_csrgeam.hpp"
#include "utility.h"

#include "bsrgeam_device.h"
#include <rocprim/rocprim.hpp>

template <unsigned int BLOCKSIZE,
          unsigned int BLOCKDIM,
          unsigned int WFSIZE,
          typename T,
          typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void bsrgeam_wf_per_row_multipass_2_3_kernel(rocsparse_direction dir,
                                             rocsparse_int       mb,
                                             rocsparse_int       nb,
                                             rocsparse_int       block_dim,
                                             U                   alpha_device_host,
                                             const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                             const rocsparse_int* __restrict__ bsr_col_ind_A,
                                             const T* __restrict__ bsr_val_A,
                                             U beta_device_host,
                                             const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                             const rocsparse_int* __restrict__ bsr_col_ind_B,
                                             const T* __restrict__ bsr_val_B,
                                             const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                             rocsparse_int* __restrict__ bsr_col_ind_C,
                                             T* __restrict__ bsr_val_C,
                                             rocsparse_index_base idx_base_A,
                                             rocsparse_index_base idx_base_B,
                                             rocsparse_index_base idx_base_C)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    bsrgeam_wf_per_row_multipass_2_3_device<BLOCKSIZE, BLOCKDIM, WFSIZE>(dir,
                                                                         mb,
                                                                         nb,
                                                                         block_dim,
                                                                         alpha,
                                                                         bsr_row_ptr_A,
                                                                         bsr_col_ind_A,
                                                                         bsr_val_A,
                                                                         beta,
                                                                         bsr_row_ptr_B,
                                                                         bsr_col_ind_B,
                                                                         bsr_val_B,
                                                                         bsr_row_ptr_C,
                                                                         bsr_col_ind_C,
                                                                         bsr_val_C,
                                                                         idx_base_A,
                                                                         idx_base_B,
                                                                         idx_base_C);
}

template <unsigned int BLOCKSIZE, unsigned int BLOCKDIM, typename T, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void bsrgeam_wf_per_row_multipass_kernel(rocsparse_direction dir,
                                         rocsparse_int       mb,
                                         rocsparse_int       nb,
                                         rocsparse_int       block_dim,
                                         U                   alpha_device_host,
                                         const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                         const rocsparse_int* __restrict__ bsr_col_ind_A,
                                         const T* __restrict__ bsr_val_A,
                                         U beta_device_host,
                                         const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                         const rocsparse_int* __restrict__ bsr_col_ind_B,
                                         const T* __restrict__ bsr_val_B,
                                         const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                         rocsparse_int* __restrict__ bsr_col_ind_C,
                                         T* __restrict__ bsr_val_C,
                                         rocsparse_index_base idx_base_A,
                                         rocsparse_index_base idx_base_B,
                                         rocsparse_index_base idx_base_C)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    bsrgeam_wf_per_row_multipass_device<BLOCKSIZE, BLOCKDIM>(dir,
                                                             mb,
                                                             nb,
                                                             block_dim,
                                                             alpha,
                                                             bsr_row_ptr_A,
                                                             bsr_col_ind_A,
                                                             bsr_val_A,
                                                             beta,
                                                             bsr_row_ptr_B,
                                                             bsr_col_ind_B,
                                                             bsr_val_B,
                                                             bsr_row_ptr_C,
                                                             bsr_col_ind_C,
                                                             bsr_val_C,
                                                             idx_base_A,
                                                             idx_base_B,
                                                             idx_base_C);
}

template <unsigned int BLOCKSIZE, unsigned int BLOCKDIM, typename T, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void bsrgeam_block_per_row_multipass_kernel(rocsparse_direction dir,
                                            rocsparse_int       mb,
                                            rocsparse_int       nb,
                                            rocsparse_int       block_dim,
                                            U                   alpha_device_host,
                                            const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                            const rocsparse_int* __restrict__ bsr_col_ind_A,
                                            const T* __restrict__ bsr_val_A,
                                            U beta_device_host,
                                            const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                            const rocsparse_int* __restrict__ bsr_col_ind_B,
                                            const T* __restrict__ bsr_val_B,
                                            const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                            rocsparse_int* __restrict__ bsr_col_ind_C,
                                            T* __restrict__ bsr_val_C,
                                            rocsparse_index_base idx_base_A,
                                            rocsparse_index_base idx_base_B,
                                            rocsparse_index_base idx_base_C)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    bsrgeam_block_per_row_multipass_device<BLOCKSIZE, BLOCKDIM>(dir,
                                                                mb,
                                                                nb,
                                                                block_dim,
                                                                alpha,
                                                                bsr_row_ptr_A,
                                                                bsr_col_ind_A,
                                                                bsr_val_A,
                                                                beta,
                                                                bsr_row_ptr_B,
                                                                bsr_col_ind_B,
                                                                bsr_val_B,
                                                                bsr_row_ptr_C,
                                                                bsr_col_ind_C,
                                                                bsr_val_C,
                                                                idx_base_A,
                                                                idx_base_B,
                                                                idx_base_C);
}

template <unsigned int BLOCKSIZE, unsigned int BLOCKDIM, typename T, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void bsrgeam_block_per_row_multipass_kernel2(rocsparse_direction dir,
                                             rocsparse_int       mb,
                                             rocsparse_int       nb,
                                             rocsparse_int       block_dim,
                                             U                   alpha_device_host,
                                             const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                             const rocsparse_int* __restrict__ bsr_col_ind_A,
                                             const T* __restrict__ bsr_val_A,
                                             U beta_device_host,
                                             const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                             const rocsparse_int* __restrict__ bsr_col_ind_B,
                                             const T* __restrict__ bsr_val_B,
                                             const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                             rocsparse_int* __restrict__ bsr_col_ind_C,
                                             T* __restrict__ bsr_val_C,
                                             rocsparse_index_base idx_base_A,
                                             rocsparse_index_base idx_base_B,
                                             rocsparse_index_base idx_base_C)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    bsrgeam_block_per_row_multipass_device2<BLOCKSIZE, BLOCKDIM>(dir,
                                                                 mb,
                                                                 nb,
                                                                 block_dim,
                                                                 alpha,
                                                                 bsr_row_ptr_A,
                                                                 bsr_col_ind_A,
                                                                 bsr_val_A,
                                                                 beta,
                                                                 bsr_row_ptr_B,
                                                                 bsr_col_ind_B,
                                                                 bsr_val_B,
                                                                 bsr_row_ptr_C,
                                                                 bsr_col_ind_C,
                                                                 bsr_val_C,
                                                                 idx_base_A,
                                                                 idx_base_B,
                                                                 idx_base_C);
}

template <typename T, typename U>
rocsparse_status rocsparse_bsrgeam_dispatch(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_int             mb,
                                            rocsparse_int             nb,
                                            rocsparse_int             block_dim,
                                            U                         alpha_device_host,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnzb_A,
                                            const T*                  bsr_val_A,
                                            const rocsparse_int*      bsr_row_ptr_A,
                                            const rocsparse_int*      bsr_col_ind_A,
                                            U                         beta_device_host,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnzb_B,
                                            const T*                  bsr_val_B,
                                            const rocsparse_int*      bsr_row_ptr_B,
                                            const rocsparse_int*      bsr_col_ind_B,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        bsr_val_C,
                                            const rocsparse_int*      bsr_row_ptr_C,
                                            rocsparse_int*            bsr_col_ind_C)
{
    // Stream
    hipStream_t stream = handle->stream;

    if(block_dim == 1)
    {
        return rocsparse_csrgeam_dispatch(handle,
                                          mb,
                                          nb,
                                          alpha_device_host,
                                          descr_A,
                                          nnzb_A,
                                          bsr_val_A,
                                          bsr_row_ptr_A,
                                          bsr_col_ind_A,
                                          beta_device_host,
                                          descr_B,
                                          nnzb_B,
                                          bsr_val_B,
                                          bsr_row_ptr_B,
                                          bsr_col_ind_B,
                                          descr_C,
                                          bsr_val_C,
                                          bsr_row_ptr_C,
                                          bsr_col_ind_C);
    }
    else if(block_dim == 2)
    {
#define BSRGEAM_DIM 256
        if(handle->wavefront_size == 32)
        {
            hipLaunchKernelGGL((bsrgeam_wf_per_row_multipass_2_3_kernel<BSRGEAM_DIM, 2, 32>),
                               dim3((mb - 1) / (BSRGEAM_DIM / 32) + 1),
                               dim3(BSRGEAM_DIM),
                               0,
                               stream,
                               dir,
                               mb,
                               nb,
                               block_dim,
                               alpha_device_host,
                               bsr_row_ptr_A,
                               bsr_col_ind_A,
                               bsr_val_A,
                               beta_device_host,
                               bsr_row_ptr_B,
                               bsr_col_ind_B,
                               bsr_val_B,
                               bsr_row_ptr_C,
                               bsr_col_ind_C,
                               bsr_val_C,
                               descr_A->base,
                               descr_B->base,
                               descr_C->base);
        }
        else
        {
            hipLaunchKernelGGL((bsrgeam_wf_per_row_multipass_2_3_kernel<BSRGEAM_DIM, 2, 64>),
                               dim3((mb - 1) / (BSRGEAM_DIM / 64) + 1),
                               dim3(BSRGEAM_DIM),
                               0,
                               stream,
                               dir,
                               mb,
                               nb,
                               block_dim,
                               alpha_device_host,
                               bsr_row_ptr_A,
                               bsr_col_ind_A,
                               bsr_val_A,
                               beta_device_host,
                               bsr_row_ptr_B,
                               bsr_col_ind_B,
                               bsr_val_B,
                               bsr_row_ptr_C,
                               bsr_col_ind_C,
                               bsr_val_C,
                               descr_A->base,
                               descr_B->base,
                               descr_C->base);
        }
#undef BSRGEAM_DIM
    }
    else if(block_dim == 3)
    {
#define BSRGEAM_DIM 256
        if(handle->wavefront_size == 32)
        {
            hipLaunchKernelGGL((bsrgeam_wf_per_row_multipass_2_3_kernel<BSRGEAM_DIM, 3, 32>),
                               dim3((mb - 1) / (BSRGEAM_DIM / 32) + 1),
                               dim3(BSRGEAM_DIM),
                               0,
                               stream,
                               dir,
                               mb,
                               nb,
                               block_dim,
                               alpha_device_host,
                               bsr_row_ptr_A,
                               bsr_col_ind_A,
                               bsr_val_A,
                               beta_device_host,
                               bsr_row_ptr_B,
                               bsr_col_ind_B,
                               bsr_val_B,
                               bsr_row_ptr_C,
                               bsr_col_ind_C,
                               bsr_val_C,
                               descr_A->base,
                               descr_B->base,
                               descr_C->base);
        }
        else
        {
            hipLaunchKernelGGL((bsrgeam_wf_per_row_multipass_2_3_kernel<BSRGEAM_DIM, 3, 64>),
                               dim3((mb - 1) / (BSRGEAM_DIM / 64) + 1),
                               dim3(BSRGEAM_DIM),
                               0,
                               stream,
                               dir,
                               mb,
                               nb,
                               block_dim,
                               alpha_device_host,
                               bsr_row_ptr_A,
                               bsr_col_ind_A,
                               bsr_val_A,
                               beta_device_host,
                               bsr_row_ptr_B,
                               bsr_col_ind_B,
                               bsr_val_B,
                               bsr_row_ptr_C,
                               bsr_col_ind_C,
                               bsr_val_C,
                               descr_A->base,
                               descr_B->base,
                               descr_C->base);
        }
#undef BSRGEAM_DIM
    }
    else if(block_dim == 4)
    {
#define BSRGEAM_DIM 64
        hipLaunchKernelGGL((bsrgeam_wf_per_row_multipass_kernel<BSRGEAM_DIM, 4>),
                           dim3((mb - 1) / (BSRGEAM_DIM / (4 * 4)) + 1),
                           dim3(BSRGEAM_DIM),
                           0,
                           stream,
                           dir,
                           mb,
                           nb,
                           block_dim,
                           alpha_device_host,
                           bsr_row_ptr_A,
                           bsr_col_ind_A,
                           bsr_val_A,
                           beta_device_host,
                           bsr_row_ptr_B,
                           bsr_col_ind_B,
                           bsr_val_B,
                           bsr_row_ptr_C,
                           bsr_col_ind_C,
                           bsr_val_C,
                           descr_A->base,
                           descr_B->base,
                           descr_C->base);
#undef BSRGEAM_DIM
    }
    else if(block_dim <= 8)
    {
#define BSRGEAM_DIM 64
        hipLaunchKernelGGL((bsrgeam_wf_per_row_multipass_kernel<BSRGEAM_DIM, 8>),
                           dim3((mb - 1) / (BSRGEAM_DIM / (8 * 8)) + 1),
                           dim3(BSRGEAM_DIM),
                           0,
                           stream,
                           dir,
                           mb,
                           nb,
                           block_dim,
                           alpha_device_host,
                           bsr_row_ptr_A,
                           bsr_col_ind_A,
                           bsr_val_A,
                           beta_device_host,
                           bsr_row_ptr_B,
                           bsr_col_ind_B,
                           bsr_val_B,
                           bsr_row_ptr_C,
                           bsr_col_ind_C,
                           bsr_val_C,
                           descr_A->base,
                           descr_B->base,
                           descr_C->base);
#undef BSRGEAM_DIM
    }
    else if(block_dim <= 16)
    {
#define BSRGEAM_DIM 256
        hipLaunchKernelGGL((bsrgeam_block_per_row_multipass_kernel2<BSRGEAM_DIM, 16>),
                           dim3(mb),
                           dim3(BSRGEAM_DIM),
                           0,
                           stream,
                           dir,
                           mb,
                           nb,
                           block_dim,
                           alpha_device_host,
                           bsr_row_ptr_A,
                           bsr_col_ind_A,
                           bsr_val_A,
                           beta_device_host,
                           bsr_row_ptr_B,
                           bsr_col_ind_B,
                           bsr_val_B,
                           bsr_row_ptr_C,
                           bsr_col_ind_C,
                           bsr_val_C,
                           descr_A->base,
                           descr_B->base,
                           descr_C->base);
#undef BSRGEAM_DIM
    }
    else if(block_dim <= 32)
    {
#define BSRGEAM_DIM 256
        hipLaunchKernelGGL((bsrgeam_block_per_row_multipass_kernel2<BSRGEAM_DIM, 32>),
                           dim3(mb),
                           dim3(BSRGEAM_DIM),
                           0,
                           stream,
                           dir,
                           mb,
                           nb,
                           block_dim,
                           alpha_device_host,
                           bsr_row_ptr_A,
                           bsr_col_ind_A,
                           bsr_val_A,
                           beta_device_host,
                           bsr_row_ptr_B,
                           bsr_col_ind_B,
                           bsr_val_B,
                           bsr_row_ptr_C,
                           bsr_col_ind_C,
                           bsr_val_C,
                           descr_A->base,
                           descr_B->base,
                           descr_C->base);
#undef BSRGEAM_DIM
    }
    else
    {
        return rocsparse_status_not_implemented;
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_bsrgeam_template(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_int             mb,
                                            rocsparse_int             nb,
                                            rocsparse_int             block_dim,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnzb_A,
                                            const T*                  bsr_val_A,
                                            const rocsparse_int*      bsr_row_ptr_A,
                                            const rocsparse_int*      bsr_col_ind_A,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnzb_B,
                                            const T*                  bsr_val_B,
                                            const rocsparse_int*      bsr_row_ptr_B,
                                            const rocsparse_int*      bsr_col_ind_B,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        bsr_val_C,
                                            const rocsparse_int*      bsr_row_ptr_C,
                                            rocsparse_int*            bsr_col_ind_C)
{
    // Check for valid handle, alpha, beta and descriptors
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(alpha == nullptr || beta == nullptr || descr_A == nullptr || descr_B == nullptr
       || descr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrgeam"),
              dir,
              mb,
              nb,
              block_dim,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr_A,
              nnzb_A,
              (const void*&)bsr_val_A,
              (const void*&)bsr_row_ptr_A,
              (const void*&)bsr_col_ind_A,
              LOG_TRACE_SCALAR_VALUE(handle, beta),
              (const void*&)descr_B,
              nnzb_B,
              (const void*&)bsr_val_B,
              (const void*&)bsr_row_ptr_B,
              (const void*&)bsr_col_ind_B,
              (const void*&)descr_C,
              (const void*&)bsr_val_C,
              (const void*&)bsr_row_ptr_C,
              (const void*&)bsr_col_ind_C);

    // Check direction
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general
       || descr_B->type != rocsparse_matrix_type_general
       || descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr_A->storage_mode != rocsparse_storage_mode_sorted
       || descr_B->storage_mode != rocsparse_storage_mode_sorted
       || descr_C->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check valid sizes
    if(mb < 0 || nb < 0 || nnzb_A < 0 || nnzb_B < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0 || (nnzb_A == 0 && nnzb_B == 0))
    {
        return rocsparse_status_success;
    }

    // Check valid pointers
    if(mb > 0)
    {
        if(bsr_row_ptr_A == nullptr || bsr_row_ptr_B == nullptr || bsr_row_ptr_C == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if(nnzb_A > 0)
    {
        if((bsr_val_A == nullptr && bsr_col_ind_A != nullptr)
           || (bsr_val_A != nullptr && bsr_col_ind_A == nullptr))
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if(nnzb_B > 0)
    {
        if((bsr_val_B == nullptr && bsr_col_ind_B != nullptr)
           || (bsr_val_B != nullptr && bsr_col_ind_B == nullptr))
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((bsr_val_C == nullptr && bsr_col_ind_C != nullptr)
       || (bsr_val_C != nullptr && bsr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnzb_A != 0 && (bsr_col_ind_A == nullptr && bsr_val_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnzb_B != 0 && (bsr_col_ind_B == nullptr && bsr_val_B == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(bsr_col_ind_C == nullptr && bsr_val_C == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                           &bsr_row_ptr_C[mb],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                           &bsr_row_ptr_C[0],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        rocsparse_int nnzb_C = (end - start);

        if(nnzb_C != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Pointer mode device
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_bsrgeam_dispatch(handle,
                                          dir,
                                          mb,
                                          nb,
                                          block_dim,
                                          alpha,
                                          descr_A,
                                          nnzb_A,
                                          bsr_val_A,
                                          bsr_row_ptr_A,
                                          bsr_col_ind_A,
                                          beta,
                                          descr_B,
                                          nnzb_B,
                                          bsr_val_B,
                                          bsr_row_ptr_B,
                                          bsr_col_ind_B,
                                          descr_C,
                                          bsr_val_C,
                                          bsr_row_ptr_C,
                                          bsr_col_ind_C);
    }
    else
    {
        return rocsparse_bsrgeam_dispatch(handle,
                                          dir,
                                          mb,
                                          nb,
                                          block_dim,
                                          *alpha,
                                          descr_A,
                                          nnzb_A,
                                          bsr_val_A,
                                          bsr_row_ptr_A,
                                          bsr_col_ind_A,
                                          *beta,
                                          descr_B,
                                          nnzb_B,
                                          bsr_val_B,
                                          bsr_row_ptr_B,
                                          bsr_col_ind_B,
                                          descr_C,
                                          bsr_val_C,
                                          bsr_row_ptr_C,
                                          bsr_col_ind_C);
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_bsrgeam_nnzb(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             block_dim,
                                                   const rocsparse_mat_descr descr_A,
                                                   rocsparse_int             nnzb_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   rocsparse_int             nnzb_B,
                                                   const rocsparse_int*      bsr_row_ptr_B,
                                                   const rocsparse_int*      bsr_col_ind_B,
                                                   const rocsparse_mat_descr descr_C,
                                                   rocsparse_int*            bsr_row_ptr_C,
                                                   rocsparse_int*            nnzb_C)
{
    // Check direction
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    // Check block dimension
    if(block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    return rocsparse_csrgeam_nnz(handle,
                                 mb,
                                 nb,
                                 descr_A,
                                 nnzb_A,
                                 bsr_row_ptr_A,
                                 bsr_col_ind_A,
                                 descr_B,
                                 nnzb_B,
                                 bsr_row_ptr_B,
                                 bsr_col_ind_B,
                                 descr_C,
                                 bsr_row_ptr_C,
                                 nnzb_C);
}

#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_direction       dir,           \
                                     rocsparse_int             mb,            \
                                     rocsparse_int             nb,            \
                                     rocsparse_int             block_dim,     \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr_A,       \
                                     rocsparse_int             nnzb_A,        \
                                     const TYPE*               bsr_val_A,     \
                                     const rocsparse_int*      bsr_row_ptr_A, \
                                     const rocsparse_int*      bsr_col_ind_A, \
                                     const TYPE*               beta,          \
                                     const rocsparse_mat_descr descr_B,       \
                                     rocsparse_int             nnzb_B,        \
                                     const TYPE*               bsr_val_B,     \
                                     const rocsparse_int*      bsr_row_ptr_B, \
                                     const rocsparse_int*      bsr_col_ind_B, \
                                     const rocsparse_mat_descr descr_C,       \
                                     TYPE*                     bsr_val_C,     \
                                     const rocsparse_int*      bsr_row_ptr_C, \
                                     rocsparse_int*            bsr_col_ind_C) \
    {                                                                         \
        return rocsparse_bsrgeam_template(handle,                             \
                                          dir,                                \
                                          mb,                                 \
                                          nb,                                 \
                                          block_dim,                          \
                                          alpha,                              \
                                          descr_A,                            \
                                          nnzb_A,                             \
                                          bsr_val_A,                          \
                                          bsr_row_ptr_A,                      \
                                          bsr_col_ind_A,                      \
                                          beta,                               \
                                          descr_B,                            \
                                          nnzb_B,                             \
                                          bsr_val_B,                          \
                                          bsr_row_ptr_B,                      \
                                          bsr_col_ind_B,                      \
                                          descr_C,                            \
                                          bsr_val_C,                          \
                                          bsr_row_ptr_C,                      \
                                          bsr_col_ind_C);                     \
    }

C_IMPL(rocsparse_sbsrgeam, float);
C_IMPL(rocsparse_dbsrgeam, double);
C_IMPL(rocsparse_cbsrgeam, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrgeam, rocsparse_double_complex);

#undef C_IMPL
