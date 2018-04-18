/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSPARSE_H_
#define ROCSPARSE_H_

/* !\file
 *  \brief rocsparse.h exposes a common interface that provides Basic Linear
 *   Algebra Subroutines for sparse computation using HIP optimized AMD HCC-
 *   based GPU hardware. This library can also run on CUDA-based NVIDIA GPUs.
 */

#include "rocsparse_version.h"
#include "rocsparse_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief rocsparse status codes definition. */
typedef enum {
    ROCSPARSE_STATUS_SUCCESS                   =  0,
    ROCSPARSE_STATUS_NOT_INITIALIZED           =  1,
    ROCSPARSE_STATUS_ALLOC_FAILED              =  2,
    ROCSPARSE_STATUS_INVALID_VALUE             =  3,
    ROCSPARSE_STATUS_ARCH_MISMATCH             =  4,
    ROCSPARSE_STATUS_MAPPING_ERROR             =  5,
    ROCSPARSE_STATUS_EXECUTION_FAILED          =  6,
    ROCSPARSE_STATUS_INTERNAL_ERROR            =  7,
    ROCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED =  8,
    ROCSPARSE_STATUS_ZERO_PIVOT                =  9,
    ROCSPARSE_STATUS_INVALID_POINTER           = 10,
    ROCSPARSE_STATUS_INVALID_SIZE              = 11,
    ROCSPARSE_STATUS_MEMORY_ERROR              = 12,
    ROCSPARSE_STATUS_INVALID_HANDLE            = 13
} rocsparseStatus_t;

struct rocsparseContext;
typedef struct rocsparseContext *rocsparseHandle_t;

struct rocsparseMatDescr;
typedef struct rocsparseMatDescr *rocsparseMatDescr_t;

/*! \brief Used to specify whether the matrix is to be transposed or not. */
typedef enum {
    ROCSPARSE_OPERATION_NON_TRANSPOSE       = 0,
    ROCSPARSE_OPERATION_TRANSPOSE           = 1,
    ROCSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} rocsparseOperation_t;

/*! \brief Indicates wether the pointer is device or host pointer. */
typedef enum {
    ROCSPARSE_POINTER_MODE_HOST   = 0,
    ROCSPARSE_POINTER_MODE_DEVICE = 1
} rocsparsePointerMode_t;

/*! \brief Used to specify the matrix index base. */
typedef enum {
    ROCSPARSE_INDEX_BASE_ZERO = 0,
    ROCSPARSE_INDEX_BASE_ONE  = 1
} rocsparseIndexBase_t;

/*! \brief Used to specify the matrix type. */
typedef enum {
    ROCSPARSE_MATRIX_TYPE_GENERAL   = 0,
    ROCSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
    ROCSPARSE_MATRIX_TYPE_HERMITIAN = 2
} rocsparseMatrixType_t;

/*! \brief Indicates if layer is active with bitmask. */
typedef enum {
    ROCSPARSE_LAYER_MODE_NONE      = 0b0000000000,
    ROCSPARSE_LAYER_MODE_LOG_TRACE = 0b0000000001,
    ROCSPARSE_LAYER_MODE_LOG_BENCH = 0b0000000010
} rocsparseLayerMode_t;




/********************************************************************************
 * \brief rocsparseHandle_t is a structure holding the rocsparse library context.
 * It must be initialized using rocsparseCreate()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparseDestroy().
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparseStatus_t rocsparseCreate(rocsparseHandle_t *handle);

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparseStatus_t rocsparseDestroy(rocsparseHandle_t handle);

/********************************************************************************
 * \brief rocsparseCreateMatDescr_t is a structure holding the rocsparse matrix
 * descriptor. It must be initialized using rocsparseCreateMatDescr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using rocsparseDestroyMatDescr().
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparseStatus_t rocsparseCreateMatDescr(rocsparseMatDescr_t *descrA);

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparseStatus_t rocsparseDestroyMatDescr(rocsparseMatDescr_t descrA);

/********************************************************************************
 * \brief Indicates whether the scalar value pointers are on the host or device.
 * Set pointer mode, can be host or device
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparseStatus_t rocsparseSetPointerMode(rocsparseHandle_t handle,
                                          rocsparsePointerMode_t mode);
/********************************************************************************
 * \brief Get pointer mode, can be host or device.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparseStatus_t rocsparseGetPointerMode(rocsparseHandle_t handle,
                                          rocsparsePointerMode_t *mode);

/********************************************************************************
 * \brief Get rocSPARSE version
 * version % 100        = patch level
 * version / 100 % 1000 = minor version
 * version / 100000     = major version
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparseStatus_t rocsparseGetVersion(rocsparseHandle_t handle, int *version);

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */



/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */

/*! \brief SPARSE Level 2 API

    \details
    csrmv  multiplies the dense vector x[i] with scalar alpha and sparse m x n
    matrix A that is defined in CSR storage format and add the result to y[i]
    that is multiplied by beta, for  i = 1 , … , n

        y := alpha * op(A) * x + beta * y,

    @param[in]
    handle      rocsparse_handle.
                handle to the rocsparse library context queue.
    @param[in]
    transA      operation type of A.
    @param[in]
    m           number of rows of A.
    @param[in]
    n           number of columns of A.
    @param[in]
    nnz         number of non-zero entries of A.
    @param[in]
    alpha       scalar alpha.
    @param[in]
    descrA      descriptor of A.
    @param[in]
    csrValA     array of nnz elements of A.
    @param[in]
    csrRowPtrA  array of m+1 elements that point to the start
                of every row of A.
    @param[in]
    csrColIndA  array of nnz elements containing the column indices of A.
    @param[in]
    x           array of n elements (op(A) = A) or m elements (op(A) = A^T or
                op(A) = A^H).
    @param[in]
    beta        scalar beta.
    @param[inout]
    y           array of m elements (op(A) = A) or n elements (op(A) = A^T or
                op(A) = A^H).

    ********************************************************************/
ROCSPARSE_EXPORT
rocsparseStatus_t rocsparseScsrmv(rocsparseHandle_t handle,
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
                                  float *y);

ROCSPARSE_EXPORT
rocsparseStatus_t rocsparseDcsrmv(rocsparseHandle_t handle,
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
                                  double *y);

#ifdef __cplusplus
}
#endif

#endif // ROCSPARSE_H_
