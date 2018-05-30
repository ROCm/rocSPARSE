/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_HYBMV_HPP
#define ROCSPARSE_HYBMV_HPP

#include "rocsparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"
#include "rocsparse_coomv.hpp"
#include "rocsparse_ellmv.hpp"

#include <hip/hip_runtime_api.h>

template <typename T>
rocsparse_status rocsparse_hybmv_template(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          const T* alpha,
                                          const rocsparse_mat_descr descr,
                                          const rocsparse_hyb_mat hyb,
                                          const T* x,
                                          const T* beta,
                                          T* y)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(hyb == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging TODO bench logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xhybmv"),
                  trans,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)hyb,
                  (const void*&)x,
                  *beta,
                  (const void*&)y);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xhybmv"),
                  trans,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)hyb,
                  (const void*&)x,
                  (const void*&)beta,
                  (const void*&)y);
    }

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    // Check partition type
    if(hyb->partition != rocsparse_hyb_partition_max &&
       hyb->partition != rocsparse_hyb_partition_auto &&
       hyb->partition != rocsparse_hyb_partition_user)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(hyb->m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(hyb->n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(hyb->ell_nnz + hyb->coo_nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check ELL-HYB structure
    if(hyb->ell_nnz > 0)
    {
        if(hyb->ell_width < 0)
        {
            return rocsparse_status_invalid_size;
        }
        else if(hyb->ell_col_ind == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
        else if(hyb->ell_val == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check COO-HYB structure
    if(hyb->coo_nnz > 0)
    {
        if(hyb->coo_row_ind == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
        else if(hyb->coo_col_ind == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
        else if(hyb->coo_val == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check pointer arguments
    if(x == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(hyb->m == 0 || hyb->n == 0 || hyb->ell_nnz + hyb->coo_nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Run different hybmv kernels
    if(trans == rocsparse_operation_none)
    {
        // ELL part
        if(hyb->ell_nnz > 0)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_ellmv_template(handle,
                                                               trans,
                                                               hyb->m,
                                                               hyb->n,
                                                               alpha,
                                                               descr,
                                                               (T*)hyb->ell_val,
                                                               hyb->ell_col_ind,
                                                               hyb->ell_width,
                                                               x,
                                                               beta,
                                                               y));
        }

        // COO part
        if(hyb->coo_nnz > 0)
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                // Beta is applied by ELL part, IF ell_nnz > 0
                if(hyb->ell_nnz > 0)
                {
                    T one = static_cast<T>(1);
                    T* coo_beta;

                    RETURN_IF_HIP_ERROR(hipMalloc((void**)&coo_beta, sizeof(T)));
                    RETURN_IF_HIP_ERROR(
                        hipMemcpy(coo_beta, &one, sizeof(T), hipMemcpyHostToDevice));
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_coomv_template(handle,
                                                                       trans,
                                                                       hyb->m,
                                                                       hyb->n,
                                                                       hyb->coo_nnz,
                                                                       alpha,
                                                                       descr,
                                                                       (T*)hyb->coo_val,
                                                                       hyb->coo_row_ind,
                                                                       hyb->coo_col_ind,
                                                                       x,
                                                                       coo_beta,
                                                                       y));
                    RETURN_IF_HIP_ERROR(hipFree(coo_beta));
                }
                else
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_coomv_template(handle,
                                                                       trans,
                                                                       hyb->m,
                                                                       hyb->n,
                                                                       hyb->coo_nnz,
                                                                       alpha,
                                                                       descr,
                                                                       (T*)hyb->coo_val,
                                                                       hyb->coo_row_ind,
                                                                       hyb->coo_col_ind,
                                                                       x,
                                                                       beta,
                                                                       y));
                }
            }
            else
            {
                if(*alpha == 0.0 && *beta == 1.0)
                {
                    return rocsparse_status_success;
                }

                // Beta is applied by ELL part, IF ell_nnz > 0
                T coo_beta = (hyb->ell_nnz > 0) ? 1.0 : *beta;

                RETURN_IF_ROCSPARSE_ERROR(rocsparse_coomv_template(handle,
                                                                   trans,
                                                                   hyb->m,
                                                                   hyb->n,
                                                                   hyb->coo_nnz,
                                                                   alpha,
                                                                   descr,
                                                                   (T*)hyb->coo_val,
                                                                   hyb->coo_row_ind,
                                                                   hyb->coo_col_ind,
                                                                   x,
                                                                   &coo_beta,
                                                                   y));
            }
        }
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_HYBMV_HPP
