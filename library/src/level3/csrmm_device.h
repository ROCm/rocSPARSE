#pragma once
#ifndef CSRMM_DEVICE_H
#define CSRMM_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T>
static __device__ void csrmmn_general_device(rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int k,
                                             rocsparse_int nnz,
                                             T alpha,
                                             const rocsparse_int* csr_row_ptr,
                                             const rocsparse_int* csr_col_ind,
                                             const T* csr_val,
                                             const T* B,
                                             rocsparse_int ldb,
                                             T beta,
                                             T* C,
                                             rocsparse_int ldc,
                                             rocsparse_index_base idx_base)
{
}

#endif // CSRMM_DEVICE_H
