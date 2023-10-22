/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_ell2csr.h"
#include "definitions.h"
#include "rocsparse_ell2csr.hpp"
#include "utility.h"

#include "ell2csr_device.h"
#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_ell2csr_template(rocsparse_handle          handle, //0
                                            rocsparse_int             m, //1
                                            rocsparse_int             n, //2
                                            const rocsparse_mat_descr ell_descr, //3
                                            rocsparse_int             ell_width, //4
                                            const T*                  ell_val, //5
                                            const rocsparse_int*      ell_col_ind, //6
                                            const rocsparse_mat_descr csr_descr, //7
                                            T*                        csr_val, //8
                                            const rocsparse_int*      csr_row_ptr, //9
                                            rocsparse_int*            csr_col_ind) //10
{

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xell2csr"),
              m,
              n,
              (const void*&)ell_descr,
              ell_width,
              (const void*&)ell_val,
              (const void*&)ell_col_ind,
              (const void*&)csr_descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_POINTER(3, ell_descr);
    ROCSPARSE_CHECKARG(3,
                       ell_descr,
                       (ell_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3,
                       ell_descr,
                       (ell_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_SIZE(4, ell_width);
    ROCSPARSE_CHECKARG_POINTER(7, csr_descr);
    ROCSPARSE_CHECKARG(7,
                       csr_descr,
                       (csr_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(7,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(5, ell_val);

    ROCSPARSE_CHECKARG_POINTER(6, ell_col_ind);

    ROCSPARSE_CHECKARG_POINTER(8, csr_val);

    ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr);

    ROCSPARSE_CHECKARG_POINTER(10, csr_col_ind);

    // Stream
    hipStream_t stream = handle->stream;

#define ELL2CSR_DIM 256
    dim3 ell2csr_blocks((m - 1) / ELL2CSR_DIM + 1);
    dim3 ell2csr_threads(ELL2CSR_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((ell2csr_fill<ELL2CSR_DIM>),
                                       ell2csr_blocks,
                                       ell2csr_threads,
                                       0,
                                       stream,
                                       m,
                                       n,
                                       ell_width,
                                       ell_col_ind,
                                       ell_val,
                                       ell_descr->base,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       csr_descr->base);
#undef ELL2CSR_DIM
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_ell2csr_nnz(rocsparse_handle          handle, //0
                                                  rocsparse_int             m, //1
                                                  rocsparse_int             n, //2
                                                  const rocsparse_mat_descr ell_descr, //3
                                                  rocsparse_int             ell_width, //4
                                                  const rocsparse_int*      ell_col_ind, //5
                                                  const rocsparse_mat_descr csr_descr, //6
                                                  rocsparse_int*            csr_row_ptr, //7
                                                  rocsparse_int*            csr_nnz) //8
try
{
    // Logging
    log_trace(handle,
              "rocsparse_ell2csr_nnz",
              m,
              n,
              (const void*&)ell_descr,
              ell_width,
              (const void*&)ell_col_ind,
              (const void*&)csr_descr,
              (const void*&)csr_row_ptr,
              (const void*&)csr_nnz);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_POINTER(3, ell_descr);
    ROCSPARSE_CHECKARG(3,
                       ell_descr,
                       (ell_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3,
                       ell_descr,
                       (ell_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_SIZE(4, ell_width);
    ROCSPARSE_CHECKARG_POINTER(6, csr_descr);
    ROCSPARSE_CHECKARG(6,
                       csr_descr,
                       (csr_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(6,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(8, csr_nnz);

    // Stream
    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(csr_nnz, 0, sizeof(rocsparse_int), stream));
        }
        else
        {
            *csr_nnz = 0;
        }
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(5, ell_col_ind);

    ROCSPARSE_CHECKARG_ARRAY(7, m, csr_row_ptr);

// Count nnz per row
#define ELL2CSR_DIM 256
    dim3 ell2csr_blocks((m + 1) / ELL2CSR_DIM + 1);
    dim3 ell2csr_threads(ELL2CSR_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((ell2csr_nnz_per_row<ELL2CSR_DIM>),
                                       ell2csr_blocks,
                                       ell2csr_threads,
                                       0,
                                       stream,
                                       m,
                                       n,
                                       ell_width,
                                       ell_col_ind,
                                       ell_descr->base,
                                       csr_row_ptr,
                                       csr_descr->base);
#undef ELL2CSR_DIM

    // Exclusive sum to obtain csr_row_ptr array and number of non-zero elements
    size_t temp_storage_bytes = 0;

    // Obtain rocprim buffer size
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                temp_storage_bytes,
                                                csr_row_ptr,
                                                csr_row_ptr,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));

    // Get rocprim buffer
    bool  d_temp_alloc;
    void* d_temp_storage;

    // Device buffer should be sufficient for rocprim in most cases
    if(handle->buffer_size >= temp_storage_bytes)
    {
        d_temp_storage = handle->buffer;
        d_temp_alloc   = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&d_temp_storage, temp_storage_bytes, handle->stream));
        d_temp_alloc = true;
    }

    // Perform actual inclusive sum
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(d_temp_storage,
                                                temp_storage_bytes,
                                                csr_row_ptr,
                                                csr_row_ptr,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));

    // Extract and adjust nnz
    if(csr_descr->base == rocsparse_index_base_one)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));

            // Adjust nnz according to index base
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (ell2csr_index_base<1>), dim3(1), dim3(1), 0, stream, csr_nnz);
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

            // Adjust nnz according to index base
            *csr_nnz -= csr_descr->base;
        }
    }
    else
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
        }
    }

    // Free rocprim buffer, if allocated
    if(d_temp_alloc == true)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(d_temp_storage, handle->stream));
    }

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_sell2csr(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int             ell_width,
                                               const float*              ell_val,
                                               const rocsparse_int*      ell_col_ind,
                                               const rocsparse_mat_descr csr_descr,
                                               float*                    csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               rocsparse_int*            csr_col_ind)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ell2csr_template(handle,
                                                         m,
                                                         n,
                                                         ell_descr,
                                                         ell_width,
                                                         ell_val,
                                                         ell_col_ind,
                                                         csr_descr,
                                                         csr_val,
                                                         csr_row_ptr,
                                                         csr_col_ind));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_dell2csr(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int             ell_width,
                                               const double*             ell_val,
                                               const rocsparse_int*      ell_col_ind,
                                               const rocsparse_mat_descr csr_descr,
                                               double*                   csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               rocsparse_int*            csr_col_ind)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ell2csr_template(handle,
                                                         m,
                                                         n,
                                                         ell_descr,
                                                         ell_width,
                                                         ell_val,
                                                         ell_col_ind,
                                                         csr_descr,
                                                         csr_val,
                                                         csr_row_ptr,
                                                         csr_col_ind));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_cell2csr(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               rocsparse_int                  n,
                                               const rocsparse_mat_descr      ell_descr,
                                               rocsparse_int                  ell_width,
                                               const rocsparse_float_complex* ell_val,
                                               const rocsparse_int*           ell_col_ind,
                                               const rocsparse_mat_descr      csr_descr,
                                               rocsparse_float_complex*       csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               rocsparse_int*                 csr_col_ind)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ell2csr_template(handle,
                                                         m,
                                                         n,
                                                         ell_descr,
                                                         ell_width,
                                                         ell_val,
                                                         ell_col_ind,
                                                         csr_descr,
                                                         csr_val,
                                                         csr_row_ptr,
                                                         csr_col_ind));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_zell2csr(rocsparse_handle                handle,
                                               rocsparse_int                   m,
                                               rocsparse_int                   n,
                                               const rocsparse_mat_descr       ell_descr,
                                               rocsparse_int                   ell_width,
                                               const rocsparse_double_complex* ell_val,
                                               const rocsparse_int*            ell_col_ind,
                                               const rocsparse_mat_descr       csr_descr,
                                               rocsparse_double_complex*       csr_val,
                                               const rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*                  csr_col_ind)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ell2csr_template(handle,
                                                         m,
                                                         n,
                                                         ell_descr,
                                                         ell_width,
                                                         ell_val,
                                                         ell_col_ind,
                                                         csr_descr,
                                                         csr_val,
                                                         csr_row_ptr,
                                                         csr_col_ind));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
