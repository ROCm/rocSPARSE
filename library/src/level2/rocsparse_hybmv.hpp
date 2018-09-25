/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
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

    // Logging
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

        log_bench(handle,
                  "./rocsparse-bench -f hybmv -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx> "
                  "--alpha",
                  *alpha,
                  "--beta",
                  *beta);
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
                    T* coo_beta = NULL;
                    rocsparse_one(handle, &coo_beta);

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
