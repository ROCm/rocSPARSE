/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "context.h"
#include "matrix.h"

template <typename T>
rocsparseStatus_t rocsparseTcsrmv(rocsparseHandle_t handle,
                                  rocsparseOperation_t transA, 
                                  int m, 
                                  int n, 
                                  int nnz,
                                  const T *alpha,
                                  const rocsparseMatDescr_t descrA, 
                                  const T *csrValA, 
                                  const int *csrRowPtrA, 
                                  const int *csrColIndA, 
                                  const T *x, 
                                  const T *beta, 
                                  T *y)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr)
    {
        return ROCSPARSE_STATUS_NOT_INITIALIZED;
    }
    else if (descrA == nullptr)
    {
        return ROCSPARSE_STATUS_NOT_INITIALIZED;
    }

    // Logging
    if (handle->pointer_mode == ROCSPARSE_POINTER_MODE_HOST)
    {
        // TODO
    }

    // Check matrix type
    if (descrA->base != ROCSPARSE_INDEX_BASE_ZERO)
    {
        // TODO
        return ROCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (descrA->type != ROCSPARSE_MATRIX_TYPE_GENERAL)
    {
        // TODO
        return ROCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }


    // Check sizes
    if (m < 0)
    {
        return ROCSPARSE_STATUS_INVALID_VALUE;
    }
    else if (n < 0)
    {
        return ROCSPARSE_STATUS_INVALID_VALUE;
    }
    else if (nnz < 0)
    {
        return ROCSPARSE_STATUS_INVALID_VALUE;
    }

    // Check pointer arguments
    if (csrValA == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else if (csrRowPtrA == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else if (csrColIndA == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else if (x == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else if (y == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else if (alpha == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else if (beta == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }

    // Quick return if possible
    if (m == 0 || n == 0 || nnz == 0)
    {
        return ROCSPARSE_STATUS_SUCCESS;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different csrmv kernels
    if (transA == ROCSPARSE_OPERATION_NON_TRANSPOSE)
    {
        // TODO
    }
    else
    {
        // TODO
        return ROCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    return ROCSPARSE_STATUS_SUCCESS;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparseStatus_t rocsparseScsrmv(rocsparseHandle_t handle,
                                             rocsparseOperation_t transA, 
                                             int m, 
                                             int n, 
                                             int nnz,
                                             const float *alpha,
                                             const rocsparseMatDescr_t descrA, 
                                             const float *csrValA, 
                                             const int *csrRowPtrA, 
                                             const int *csrColIndA, 
                                             const float *x, 
                                             const float *beta, 
                                             float *y)
{
    return rocsparseTcsrmv<float>(handle, transA, m, n, nnz, alpha, descrA,
                                  csrValA, csrRowPtrA, csrColIndA, x, beta, y);
}
    
extern "C" rocsparseStatus_t rocsparseDcsrmv(rocsparseHandle_t handle,
                                               rocsparseOperation_t transA, 
                                               int m, 
                                               int n, 
                                               int nnz,
                                               const double *alpha,
                                               const rocsparseMatDescr_t descrA, 
                                               const double *csrValA, 
                                               const int *csrRowPtrA, 
                                               const int *csrColIndA, 
                                               const double *x, 
                                               const double *beta,  
                                               double *y)
{
    return rocsparseTcsrmv<double>(handle, transA, m, n, nnz, alpha, descrA,
                                   csrValA, csrRowPtrA, csrColIndA, x, beta, y);
}
