/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "definitions.h"
#include "utility.h"
#include "rocsparse.h"
#include "ell2csr_device.h"
#include "rocsparse_ell2csr.hpp"

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_ell2csr_nnz(rocsparse_handle handle,
                                                  rocsparse_int m,
                                                  rocsparse_int n,
                                                  const rocsparse_mat_descr ell_descr,
                                                  rocsparse_int ell_width,
                                                  const rocsparse_int* ell_col_ind,
                                                  const rocsparse_mat_descr csr_descr,
                                                  rocsparse_int* csr_row_ptr,
                                                  rocsparse_int* csr_nnz)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(ell_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

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

    // Check index base
    if(ell_descr->base != rocsparse_index_base_zero && ell_descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(csr_descr->base != rocsparse_index_base_zero && csr_descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(ell_descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    if(csr_descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || ell_width < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(ell_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_nnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemset(csr_nnz, 0, sizeof(rocsparse_int)));
        }
        else
        {
            *csr_nnz = 0;
        }
        return rocsparse_status_success;
    }

    hipStream_t stream = handle->stream;

// Count nnz per row
#define ELL2CSR_DIM 256
    dim3 ell2csr_blocks((m + 1) / ELL2CSR_DIM + 1);
    dim3 ell2csr_threads(ELL2CSR_DIM);

    hipLaunchKernelGGL((ell2csr_nnz_per_row),
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
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    // Obtain hipcub buffer size
    RETURN_IF_HIP_ERROR(hipcub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, csr_row_ptr, csr_row_ptr, m + 1));

    // Allocate hipcub buffer
    RETURN_IF_HIP_ERROR(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Perform actual inclusive sum
    RETURN_IF_HIP_ERROR(hipcub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, csr_row_ptr, csr_row_ptr, m + 1));

    // Free hipcub buffer
    RETURN_IF_HIP_ERROR(hipFree(d_temp_storage));

    // Extract and adjust nnz according to index base
    rocsparse_int nnz;
    RETURN_IF_HIP_ERROR(
        hipMemcpy(&nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

    nnz -= csr_descr->base;

    // Set nnz
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(hipMemcpy(csr_nnz, &nnz, sizeof(rocsparse_int), hipMemcpyHostToDevice));
    }
    else
    {
        *csr_nnz = nnz;
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_sell2csr(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int ell_width,
                                               const float* ell_val,
                                               const rocsparse_int* ell_col_ind,
                                               const rocsparse_mat_descr csr_descr,
                                               float* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               rocsparse_int* csr_col_ind)
{
    return rocsparse_ell2csr_template<float>(handle,
                                             m,
                                             n,
                                             ell_descr,
                                             ell_width,
                                             ell_val,
                                             ell_col_ind,
                                             csr_descr,
                                             csr_val,
                                             csr_row_ptr,
                                             csr_col_ind);
}

extern "C" rocsparse_status rocsparse_dell2csr(rocsparse_handle handle,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int ell_width,
                                               const double* ell_val,
                                               const rocsparse_int* ell_col_ind,
                                               const rocsparse_mat_descr csr_descr,
                                               double* csr_val,
                                               const rocsparse_int* csr_row_ptr,
                                               rocsparse_int* csr_col_ind)
{
    return rocsparse_ell2csr_template<double>(handle,
                                              m,
                                              n,
                                              ell_descr,
                                              ell_width,
                                              ell_val,
                                              ell_col_ind,
                                              csr_descr,
                                              csr_val,
                                              csr_row_ptr,
                                              csr_col_ind);
}
