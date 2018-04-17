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
