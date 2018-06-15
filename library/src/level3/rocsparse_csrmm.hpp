/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_CSRMM_HPP
#define ROCSPARSE_CSRMM_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "csrmm_device.h"

#include <hip/hip_runtime.h>

template <typename T>
__global__ void csrmmn_kernel_host_pointer()
{
    csrmmn_general_device<T>();
}

template <typename T>
__global__ void csrmmn_kernel_device_pointer()
{
    csrmmn_general_device<T>();
}

template <typename T>
rocsparse_status rocsparse_csrmm_template(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int k,
                                          rocsparse_int nnz,
                                          const T* alpha,
                                          const rocsparse_mat_descr descr,
                                          const T* csr_val,
                                          const rocsparse_int* csr_row_ptr,
                                          const rocsparse_int* csr_col_ind,
                                          const T* B,
                                          rocsparse_int ldb,
                                          const T* beta,
                                          T* C,
                                          rocsparse_int ldc)
{
    return rocsparse_status_not_implemented;
}

#endif // ROCSPARSE_CSRMM_HPP
