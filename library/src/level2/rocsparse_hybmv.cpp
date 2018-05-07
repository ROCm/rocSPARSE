/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "ellmv_device.h"

#include <hip/hip_runtime.h>

template <typename T>
__global__
void ellmvn_kernel_host_pointer(rocsparse_int m,
                                rocsparse_int n,
                                rocsparse_int ell_width,
                                T alpha,
                                const rocsparse_int *ell_col_ind,
                                const T *ell_val,
                                const T *x,
                                T beta,
                                T *y)
{
    ellmvn_device(m, n, ell_width, alpha, ell_col_ind, ell_val, x, beta, y);
}

template <typename T>
__global__
void ellmvn_kernel_device_pointer(rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int ell_width,
                                  const T *alpha,
                                  const rocsparse_int *ell_col_ind,
                                  const T *ell_val,
                                  const T *x,
                                  const T *beta,
                                  T *y)
{
    ellmvn_device(m, n, ell_width, *alpha, ell_col_ind, ell_val, x, *beta, y);
}

template <typename T>
rocsparse_status rocsparse_hybmv_template(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          const T *alpha,
                                          const rocsparse_mat_descr descr,
                                          const rocsparse_hyb_mat hyb,
                                          const T *x,
                                          const T *beta,
                                          T *y)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if (descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (hyb == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging TODO bench logging
    if (handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xhybmv"),
                  trans,
                  *alpha,
                  (const void*&) descr,
                  (const void*&) hyb,
                  (const void*&) x,
                  *beta,
                  (const void*&) y);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xhybmv"),
                  trans,
                  (const void*&) alpha,
                  (const void*&) descr,
                  (const void*&) hyb,
                  (const void*&) x,
                  (const void*&) beta,
                  (const void*&) y);
    }

    // Check matrix type
    if (descr->base != rocsparse_index_base_zero)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    if (descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    if (hyb->partition != rocsparse_hyb_partition_max)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if (hyb->m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if (hyb->n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if (hyb->ell_nnz + hyb->coo_nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check ELL-HYB structure
    if (hyb->ell_nnz > 0)
    {
        if (hyb->ell_width < 0)
        {
            return rocsparse_status_invalid_size;
        }
        else if (hyb->ell_col_ind == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
        else if (hyb->ell_val == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check COO-HYB structure
    if (hyb->coo_nnz > 0)
    {
        if (hyb->coo_row_ind == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
        else if (hyb->coo_col_ind == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
        else if (hyb->coo_val == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check pointer arguments
    if (x == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if (hyb->m == 0 || hyb->n == 0 ||
        hyb->ell_nnz + hyb->coo_nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different hybmv kernels
    if (trans == rocsparse_operation_none)
    {
#define ELLMVN_DIM 512
        dim3 ellmvn_blocks((hyb->m-1)/ELLMVN_DIM+1);
        dim3 ellmvn_threads(ELLMVN_DIM);

        if (handle->pointer_mode == rocsparse_pointer_mode_device)
        {
        }
        else
        {
            if (*alpha == 0.0 && *beta == 1.0)
            {
                return rocsparse_status_success;
            }

            // ELL part
            if (hyb->ell_nnz > 0)
            {
                hipLaunchKernelGGL((ellmvn_kernel_host_pointer<T>),
                                   ellmvn_blocks, ellmvn_threads, 0, stream,
                                   hyb->m, hyb->n, hyb->ell_width, *alpha,
                                   hyb->ell_col_ind, (T*) hyb->ell_val,
                                   x, *beta, y);
            }

        }
#undef ELLMVN_DIM
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C"
rocsparse_status rocsparse_shybmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  const float *alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat hyb,
                                  const float *x,
                                  const float *beta,
                                  float *y)
{
    return rocsparse_hybmv_template(handle, trans, alpha,
                                    descr, hyb, x, beta, y);
}

extern "C"
rocsparse_status rocsparse_dhybmv(rocsparse_handle handle,
                                  rocsparse_operation trans,
                                  const double *alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat hyb,
                                  const double *x,
                                  const double *beta,
                                  double *y)
{
    return rocsparse_hybmv_template(handle, trans, alpha,
                                    descr, hyb, x, beta, y);
}
