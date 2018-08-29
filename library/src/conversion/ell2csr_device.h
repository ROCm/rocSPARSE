/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef ELL2CSR_DEVICE_H
#define ELL2CSR_DEVICE_H

#include "handle.h"

#include <hip/hip_runtime.h>

__global__ void ell2csr_nnz_per_row(rocsparse_int m,
                                    rocsparse_int n,
                                    rocsparse_int ell_width,
                                    const rocsparse_int* __restrict__ ell_col_ind,
                                    rocsparse_index_base ell_base,
                                    rocsparse_int* __restrict__ csr_row_ptr,
                                    rocsparse_index_base csr_base)
{
    rocsparse_int ai = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(ai >= m)
    {
        return;
    }

    if(ai == 0)
    {
        csr_row_ptr[0] = csr_base;
    }

    rocsparse_int nnz = 0;

    for(rocsparse_int p = 0; p < ell_width; ++p)
    {
        rocsparse_int idx = ELL_IND(ai, p, m, ell_width);
        rocsparse_int col = ell_col_ind[idx] - ell_base;

        if(col >= 0 && col < n)
        {
            ++nnz;
        }
        else
        {
            break;
        }
    }

    csr_row_ptr[ai + 1] = nnz;
}

template <typename T>
__global__ void ell2csr_fill(rocsparse_int m,
                             rocsparse_int n,
                             rocsparse_int ell_width,
                             const rocsparse_int* __restrict__ ell_col_ind,
                             const T* __restrict__ ell_val,
                             rocsparse_index_base ell_base,
                             const rocsparse_int* __restrict__ csr_row_ptr,
                             rocsparse_int* __restrict__ csr_col_ind,
                             T* __restrict__ csr_val,
                             rocsparse_index_base csr_base)
{
    rocsparse_int ai = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(ai >= m)
    {
        return;
    }

    rocsparse_int csr_idx = csr_row_ptr[ai] - csr_base;

    for(rocsparse_int p = 0; p < ell_width; ++p)
    {
        rocsparse_int ell_idx = ELL_IND(ai, p, m, ell_width);
        rocsparse_int ell_col = ell_col_ind[ell_idx] - ell_base;

        if(ell_col >= 0 && ell_col < n)
        {
            csr_col_ind[csr_idx] = ell_col + csr_base;
            csr_val[csr_idx]     = ell_val[ell_idx];
            ++csr_idx;
        }
        else
        {
            break;
        }
    }
}

#endif // ELL2CSR_DEVICE_H
