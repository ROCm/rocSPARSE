/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_ELL2CSR_HPP
#define ROCSPARSE_ELL2CSR_HPP

#include "rocsparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"
#include "ell2csr_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_ell2csr_template(rocsparse_handle handle,
                                            rocsparse_int m,
                                            rocsparse_int n,
                                            const rocsparse_mat_descr ell_descr,
                                            rocsparse_int ell_width,
                                            const T* ell_val,
                                            const rocsparse_int* ell_col_ind,
                                            const rocsparse_mat_descr csr_descr,
                                            T* csr_val,
                                            const rocsparse_int* csr_row_ptr,
                                            rocsparse_int* csr_col_ind)
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

    log_bench(handle,
              "./rocsparse-bench -f ell2csr -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

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
    if(ell_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_val == nullptr)
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

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define ELL2CSR_DIM 256
    dim3 ell2csr_blocks((m - 1) / ELL2CSR_DIM + 1);
    dim3 ell2csr_threads(ELL2CSR_DIM);

    hipLaunchKernelGGL((ell2csr_fill<T>),
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

#endif // ROCSPARSE_ELL2CSR_HPP
