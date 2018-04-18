/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "context.h"
#include "utility.h"
#include "matrix.h"
#include "csrmv_device.h"

#include <hip/hip_runtime.h>

template <typename T, int SUBWAVE_SIZE, int WG_SIZE>
__global__
void csrmvn_kernel_host_pointer(int m, T alpha, const int *ptr, const int *col,
                                const T *val, const T *x, T beta, T *y)
{
    csrmvn_general_device<T, SUBWAVE_SIZE, WG_SIZE>(
        m, alpha, ptr, col, val, x, beta, y);
}

template <typename T, int SUBWAVE_SIZE, int WG_SIZE>
__global__
void csrmvn_kernel_device_pointer(int m, const T *alpha, const int *ptr, const int *col,
                                const T *val, const T *x, const T *beta, T *y)
{
    csrmvn_general_device<T, SUBWAVE_SIZE, WG_SIZE>(
        m, *alpha, ptr, col, val, x, *beta, y);
}

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

    // Logging TODO bench logging
    if (handle->pointer_mode == ROCSPARSE_POINTER_MODE_HOST)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmv"),
                  transA,
                  m, n, nnz,
                  *alpha,
                  (const void*&) descrA,
                  (const void*&) csrValA,
                  (const void*&) csrRowPtrA,
                  (const void*&) csrColIndA,
                  (const void*&) x,
                  *beta,
                  (const void*&) y);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xcsrmv"),
                  transA,
                  m, n, nnz,
                  (const void*&) alpha,
                  (const void*&) descrA,
                  (const void*&) csrValA,
                  (const void*&) csrRowPtrA,
                  (const void*&) csrColIndA,
                  (const void*&) x,
                  (const void*&) beta,
                  (const void*&) y);
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
#define CSRMVN_DIM 512

        int nnz_per_row = nnz / m;

        dim3 csrmvn_blocks((m-1)/CSRMVN_DIM+1);
        dim3 csrmvn_threads(CSRMVN_DIM);

        if (handle->pointer_mode == ROCSPARSE_POINTER_MODE_DEVICE)
        {
            if (handle->warp_size == 32)
            {
                if (nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 2, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
                else if (nnz_per_row < 8)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 4, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
                else if (nnz_per_row < 16)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 8, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
                else if (nnz_per_row < 32)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 16, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
                else
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 32, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
            }
            else if (handle->warp_size == 64)
            {
                if (nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 2, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
                else if (nnz_per_row < 8)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 4, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
                else if (nnz_per_row < 16)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 8, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
                else if (nnz_per_row < 32)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 16, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
                else if (nnz_per_row < 64)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 32, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
                else
                {
                    hipLaunchKernelGGL((csrmvn_kernel_device_pointer<T, 64, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, alpha, csrRowPtrA, csrColIndA, csrValA, x, beta, y);
                }
            }
            else
            {
                return ROCSPARSE_STATUS_ARCH_MISMATCH;
            }
        }
        else
        {
            if (*alpha == 0.0 && *beta == 1.0)
            {
                return ROCSPARSE_STATUS_SUCCESS;
            }

            if (handle->warp_size == 32)
            {
                if (nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 2, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
                else if (nnz_per_row < 8)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 4, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
                else if (nnz_per_row < 16)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 8, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
                else if (nnz_per_row < 32)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 16, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
                else
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 32, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
            }
            else if (handle->warp_size == 64)
            {
                if (nnz_per_row < 4)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 2, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
                else if (nnz_per_row < 8)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 4, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
                else if (nnz_per_row < 16)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 8, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
                else if (nnz_per_row < 32)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 16, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
                else if (nnz_per_row < 64)
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 32, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
                else
                {
                    hipLaunchKernelGGL((csrmvn_kernel_host_pointer<T, 64, CSRMVN_DIM>),
                                       csrmvn_blocks, csrmvn_threads, 0, stream,
                                       m, *alpha, csrRowPtrA, csrColIndA, csrValA, x, *beta, y);
                }
            }
            else
            {
                return ROCSPARSE_STATUS_ARCH_MISMATCH;
            }
        }
#undef CSRMVN_DIM
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
