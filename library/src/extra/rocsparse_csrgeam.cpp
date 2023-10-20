/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/extra/rocsparse_csrgeam.h"
#include "definitions.h"
#include "rocsparse_csrgeam.hpp"
#include "utility.h"

#include "csrgeam_device.h"
#include <rocprim/rocprim.hpp>

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T, typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void csrgeam_fill_multipass_kernel(rocsparse_int m,
                                   rocsparse_int n,
                                   U             alpha_device_host,
                                   const rocsparse_int* __restrict__ csr_row_ptr_A,
                                   const rocsparse_int* __restrict__ csr_col_ind_A,
                                   const T* __restrict__ csr_val_A,
                                   U beta_device_host,
                                   const rocsparse_int* __restrict__ csr_row_ptr_B,
                                   const rocsparse_int* __restrict__ csr_col_ind_B,
                                   const T* __restrict__ csr_val_B,
                                   const rocsparse_int* __restrict__ csr_row_ptr_C,
                                   rocsparse_int* __restrict__ csr_col_ind_C,
                                   T* __restrict__ csr_val_C,
                                   rocsparse_index_base idx_base_A,
                                   rocsparse_index_base idx_base_B,
                                   rocsparse_index_base idx_base_C)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    csrgeam_fill_multipass_device<BLOCKSIZE, WFSIZE>(m,
                                                     n,
                                                     alpha,
                                                     csr_row_ptr_A,
                                                     csr_col_ind_A,
                                                     csr_val_A,
                                                     beta,
                                                     csr_row_ptr_B,
                                                     csr_col_ind_B,
                                                     csr_val_B,
                                                     csr_row_ptr_C,
                                                     csr_col_ind_C,
                                                     csr_val_C,
                                                     idx_base_A,
                                                     idx_base_B,
                                                     idx_base_C);
}

template <typename T, typename U>
rocsparse_status rocsparse_csrgeam_dispatch(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            U                         alpha_device_host,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const T*                  csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            U                         beta_device_host,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const T*                  csr_val_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        csr_val_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Pointer mode device
#define CSRGEAM_DIM 256
    if(handle->wavefront_size == 32)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrgeam_fill_multipass_kernel<CSRGEAM_DIM, 32>),
                                           dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                                           dim3(CSRGEAM_DIM),
                                           0,
                                           stream,
                                           m,
                                           n,
                                           alpha_device_host,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_val_A,
                                           beta_device_host,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_val_B,
                                           csr_row_ptr_C,
                                           csr_col_ind_C,
                                           csr_val_C,
                                           descr_A->base,
                                           descr_B->base,
                                           descr_C->base);
    }
    else
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrgeam_fill_multipass_kernel<CSRGEAM_DIM, 64>),
                                           dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                                           dim3(CSRGEAM_DIM),
                                           0,
                                           stream,
                                           m,
                                           n,
                                           alpha_device_host,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_val_A,
                                           beta_device_host,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_val_B,
                                           csr_row_ptr_C,
                                           csr_col_ind_C,
                                           csr_val_C,
                                           descr_A->base,
                                           descr_B->base,
                                           descr_C->base);
    }

#undef CSRGEAM_DIM

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_csrgeam_template(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const T*                  csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const T*                  csr_val_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        csr_val_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C)
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
              replaceX<T>("rocsparse_Xcsrgeam"),
              m,
              n,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr_A,
              nnz_A,
              (const void*&)csr_val_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              LOG_TRACE_SCALAR_VALUE(handle, beta),
              (const void*&)descr_B,
              nnz_B,
              (const void*&)csr_val_B,
              (const void*&)csr_row_ptr_B,
              (const void*&)csr_col_ind_B,
              (const void*&)descr_C,
              (const void*&)csr_val_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)csr_col_ind_C);

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
        return rocsparse_status_requires_sorted_storage;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0))
    {
        return rocsparse_status_success;
    }

    // Check valid pointers
    if(csr_row_ptr_A == nullptr || csr_row_ptr_B == nullptr || csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_A == nullptr && csr_col_ind_A != nullptr)
       || (csr_val_A != nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_B == nullptr && csr_col_ind_B != nullptr)
       || (csr_val_B != nullptr && csr_col_ind_B == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_C == nullptr && csr_col_ind_C != nullptr)
       || (csr_val_C != nullptr && csr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && (csr_col_ind_A == nullptr && csr_val_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_B != 0 && (csr_col_ind_B == nullptr && csr_val_B == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_col_ind_C == nullptr && csr_val_C == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr_C[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                           &csr_row_ptr_C[0],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        rocsparse_int nnz_C = (end - start);

        if(nnz_C != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Pointer mode device
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_csrgeam_dispatch(handle,
                                          m,
                                          n,
                                          alpha,
                                          descr_A,
                                          nnz_A,
                                          csr_val_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          beta,
                                          descr_B,
                                          nnz_B,
                                          csr_val_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          descr_C,
                                          csr_val_C,
                                          csr_row_ptr_C,
                                          csr_col_ind_C);
    }
    else
    {
        return rocsparse_csrgeam_dispatch(handle,
                                          m,
                                          n,
                                          *alpha,
                                          descr_A,
                                          nnz_A,
                                          csr_val_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          *beta,
                                          descr_B,
                                          nnz_B,
                                          csr_val_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          descr_C,
                                          csr_val_C,
                                          csr_row_ptr_C,
                                          csr_col_ind_C);
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_csrgeam_nnz(rocsparse_handle          handle,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr descr_A,
                                                  rocsparse_int             nnz_A,
                                                  const rocsparse_int*      csr_row_ptr_A,
                                                  const rocsparse_int*      csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  rocsparse_int             nnz_B,
                                                  const rocsparse_int*      csr_row_ptr_B,
                                                  const rocsparse_int*      csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_C,
                                                  rocsparse_int*            csr_row_ptr_C,
                                                  rocsparse_int*            nnz_C)
try
{
    // Check for valid handle and descriptors
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(descr_A == nullptr || descr_B == nullptr || descr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csrgeam_nnz",
              m,
              n,
              (const void*&)descr_A,
              nnz_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              (const void*&)descr_B,
              nnz_B,
              (const void*&)csr_row_ptr_B,
              (const void*&)csr_col_ind_B,
              (const void*&)descr_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)nnz_C);

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
        return rocsparse_status_requires_sorted_storage;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid nnz_C pointer
    if(nnz_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0))
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            *nnz_C = 0;
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(rocsparse_int), handle->stream));
        }

        if(nnz_A == 0 && nnz_B == 0)
        {
            if(csr_row_ptr_C == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((set_array_to_value<256>),
                                                   dim3(m / 256 + 1),
                                                   dim3(256),
                                                   0,
                                                   handle->stream,
                                                   m + 1,
                                                   csr_row_ptr_C,
                                                   static_cast<rocsparse_int>(descr_C->base));
            }
        }

        return rocsparse_status_success;
    }

    // Check valid pointers
    if(csr_row_ptr_A == nullptr || csr_row_ptr_B == nullptr || csr_row_ptr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && csr_col_ind_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_B != 0 && csr_col_ind_B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define CSRGEAM_DIM 256
    if(handle->wavefront_size == 32)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrgeam_nnz_multipass_device<CSRGEAM_DIM, 32>),
                                           dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                                           dim3(CSRGEAM_DIM),
                                           0,
                                           stream,
                                           m,
                                           n,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_row_ptr_C,
                                           descr_A->base,
                                           descr_B->base);
    }
    else
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrgeam_nnz_multipass_device<CSRGEAM_DIM, 64>),
                                           dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                                           dim3(CSRGEAM_DIM),
                                           0,
                                           stream,
                                           m,
                                           n,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_row_ptr_C,
                                           descr_A->base,
                                           descr_B->base);
    }
#undef CSRGEAM_DIM

    // Exclusive sum to obtain row pointers of C
    size_t rocprim_size;
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                rocprim_size,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                static_cast<rocsparse_int>(descr_C->base),
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));

    bool  rocprim_alloc;
    void* rocprim_buffer;

    if(handle->buffer_size >= rocprim_size)
    {
        rocprim_buffer = handle->buffer;
        rocprim_alloc  = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&rocprim_buffer, rocprim_size, handle->stream));
        rocprim_alloc = true;
    }

    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                rocprim_size,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                static_cast<rocsparse_int>(descr_C->base),
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));

    if(rocprim_alloc == true)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(rocprim_buffer, handle->stream));
    }

    // Extract the number of non-zero elements of C
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        // Blocking mode
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(nnz_C,
                                           csr_row_ptr_C + m,
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        // Adjust index base of nnz_C
        *nnz_C -= descr_C->base;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz_C, csr_row_ptr_C + m, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));

        // Adjust index base of nnz_C
        if(descr_C->base == rocsparse_index_base_one)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (csrgeam_index_base<1>), dim3(1), dim3(1), 0, stream, nnz_C);
        }
    }

    return rocsparse_status_success;
}
catch(...)
{
    return exception_to_rocsparse_status();
}

#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_int             m,             \
                                     rocsparse_int             n,             \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr_A,       \
                                     rocsparse_int             nnz_A,         \
                                     const TYPE*               csr_val_A,     \
                                     const rocsparse_int*      csr_row_ptr_A, \
                                     const rocsparse_int*      csr_col_ind_A, \
                                     const TYPE*               beta,          \
                                     const rocsparse_mat_descr descr_B,       \
                                     rocsparse_int             nnz_B,         \
                                     const TYPE*               csr_val_B,     \
                                     const rocsparse_int*      csr_row_ptr_B, \
                                     const rocsparse_int*      csr_col_ind_B, \
                                     const rocsparse_mat_descr descr_C,       \
                                     TYPE*                     csr_val_C,     \
                                     const rocsparse_int*      csr_row_ptr_C, \
                                     rocsparse_int*            csr_col_ind_C) \
    try                                                                       \
    {                                                                         \
        return rocsparse_csrgeam_template(handle,                             \
                                          m,                                  \
                                          n,                                  \
                                          alpha,                              \
                                          descr_A,                            \
                                          nnz_A,                              \
                                          csr_val_A,                          \
                                          csr_row_ptr_A,                      \
                                          csr_col_ind_A,                      \
                                          beta,                               \
                                          descr_B,                            \
                                          nnz_B,                              \
                                          csr_val_B,                          \
                                          csr_row_ptr_B,                      \
                                          csr_col_ind_B,                      \
                                          descr_C,                            \
                                          csr_val_C,                          \
                                          csr_row_ptr_C,                      \
                                          csr_col_ind_C);                     \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        return exception_to_rocsparse_status();                               \
    }

C_IMPL(rocsparse_scsrgeam, float);
C_IMPL(rocsparse_dcsrgeam, double);
C_IMPL(rocsparse_ccsrgeam, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrgeam, rocsparse_double_complex);

#undef C_IMPL
